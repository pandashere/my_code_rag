# llama_index_integration/kg_query_engine.py
"""
代码知识图谱查询引擎（渐进多跳 + 预算控制）
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import asyncio

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.property_graph.base import PropertyGraphIndex
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.indices.property_graph.sub_retrievers.vector import (
    VectorContextRetriever,
)

from code_rag.utils.llm_funcs import normalize_query_for_retrieval


class CodeKGQueryEngine(BaseQueryEngine):
    """代码知识图谱查询引擎"""

    def __init__(
        self,
        index: PropertyGraphIndex,
        llm: Any = None,
        graph_depth: int = 3,
        similarity_top_k: int = 5,
        node_budget: int = 80,
        per_module_limit: int = 12,
        evidence_score_threshold: float = 0.62,
        marginal_gain_threshold: float = 0.06,
    ):
        super().__init__(
            callback_manager=(
                llm.callback_manager
                if llm and hasattr(llm, "callback_manager")
                else None
            )
        )
        self.index = index
        self.llm = llm
        self.graph_depth = max(1, graph_depth)
        self.similarity_top_k = max(1, similarity_top_k)
        self.node_budget = max(self.similarity_top_k, node_budget)
        self.per_module_limit = max(1, per_module_limit)
        self.evidence_score_threshold = evidence_score_threshold
        self.marginal_gain_threshold = marginal_gain_threshold

    def _query(self, query_bundle: QueryBundle) -> Response:
        """同步查询实现"""
        return self._execute_query(query_bundle.query_str)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """异步查询实现（必需）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_query, query_bundle.query_str)

    def _get_prompt_modules(self) -> PromptDictType:
        """Prompt 模块定义（必需）"""
        return {}

    def _normalize_query(self, query_str: str) -> str:
        """将查询标准化；当无可用 LLM 时退化为原始查询。"""
        if not self.llm or not hasattr(self.llm, "complete"):
            return query_str
        try:
            return normalize_query_for_retrieval(query_str, self.llm, True)
        except Exception:
            return query_str

    def _infer_query_intent(self, query_str: str) -> str:
        q = query_str.lower()
        if any(k in q for k in ("调用", "call", "callee", "caller", "依赖", "dependency")):
            return "relationship"
        if any(k in q for k in ("影响", "impact", "blast radius", "受影响", "哪些文件")):
            return "impact"
        return "lookup"

    def _make_retriever(self, depth: int) -> VectorContextRetriever:
        kwargs = {
            "graph_store": self.index.property_graph_store,
            "embed_model": self.index._embed_model,
            "similarity_top_k": self.similarity_top_k,
            "path_depth": max(1, depth),
        }
        if getattr(self.index, "vector_store", None) is not None:
            kwargs["vector_store"] = self.index.vector_store
        return VectorContextRetriever(**kwargs)

    def _dedupe_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        seen = set()
        deduped = []
        for nws in nodes:
            metadata = nws.node.metadata or {}
            key = metadata.get("qualified_name") or nws.node.node_id or nws.node.id_
            if key in seen:
                continue
            deduped.append(nws)
            seen.add(key)
        return deduped

    def _apply_budget(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        per_module_count: Dict[str, int] = {}
        selected: List[NodeWithScore] = []
        for nws in nodes:
            module = (nws.node.metadata or {}).get("module", "unknown")
            if per_module_count.get(module, 0) >= self.per_module_limit:
                continue
            selected.append(nws)
            per_module_count[module] = per_module_count.get(module, 0) + 1
            if len(selected) >= self.node_budget:
                break
        return selected

    def _filter_nodes_by_intent(self, nodes: List[NodeWithScore], intent: str) -> List[NodeWithScore]:
        if intent == "lookup":
            preferred = {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD", "CLASS", "MODULE"}
        elif intent == "relationship":
            preferred = {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD", "CLASS"}
        else:
            preferred = {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD", "CLASS", "MODULE", "VARIABLE"}

        primary = [n for n in nodes if (n.node.metadata or {}).get("node_type") in preferred]
        fallback = [n for n in nodes if n not in primary]
        return primary + fallback

    def _compute_evidence_score(self, nodes: List[NodeWithScore], intent: str) -> Dict[str, float]:
        if not nodes:
            return {"score": 0.0, "relevance": 0.0, "type_cov": 0.0, "module_div": 0.0}

        def _clip(v: float) -> float:
            return max(0.0, min(1.0, v))

        relevance = _clip(sum(float(n.score or 0.0) for n in nodes) / len(nodes))
        node_types = {(n.node.metadata or {}).get("node_type", "UNKNOWN") for n in nodes}
        modules = {(n.node.metadata or {}).get("module", "unknown") for n in nodes}

        if intent == "relationship":
            target_types = {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD"}
        elif intent == "impact":
            target_types = {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD", "CLASS", "MODULE"}
        else:
            target_types = {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD", "CLASS", "MODULE"}

        type_cov = _clip(len(node_types & target_types) / max(1, len(target_types) / 2))
        module_div = _clip(len(modules) / 4.0)

        score = 0.45 * relevance + 0.35 * type_cov + 0.20 * module_div
        return {
            "score": round(score, 4),
            "relevance": round(relevance, 4),
            "type_cov": round(type_cov, 4),
            "module_div": round(module_div, 4),
        }

    def _has_minimum_slots(self, nodes: List[NodeWithScore], intent: str) -> bool:
        if not nodes:
            return False
        metadata_list = [n.node.metadata or {} for n in nodes]
        func_like_count = sum(
            1 for m in metadata_list
            if m.get("node_type") in {"FUNCTION", "METHOD", "ASYNC_FUNCTION", "ASYNC_METHOD"}
        )
        code_span_count = sum(1 for m in metadata_list if m.get("code_span"))
        module_count = len({m.get("module", "unknown") for m in metadata_list})

        if intent == "relationship":
            return func_like_count >= 2
        if intent == "impact":
            return func_like_count >= 2 and module_count >= 2
        return func_like_count >= 1 and code_span_count >= 1

    def _should_stop(
        self,
        depth: int,
        max_depth: int,
        score_info: Dict[str, float],
        gain: float,
        novelty: float,
        slot_ok: bool,
    ) -> bool:
        if depth >= max_depth:
            return True
        if not slot_ok:
            return False
        if score_info["score"] < self.evidence_score_threshold:
            return False
        return gain < self.marginal_gain_threshold and novelty < 0.2

    def _progressive_retrieve(
        self,
        normalized_query: str,
        intent: str,
    ) -> Tuple[List[NodeWithScore], List[Dict[str, Any]]]:
        merged_nodes: List[NodeWithScore] = []
        diagnostics: List[Dict[str, Any]] = []
        prev_score = 0.0
        prev_count = 0

        for depth in range(1, self.graph_depth + 1):
            retriever = self._make_retriever(depth)
            depth_nodes = retriever.retrieve(normalized_query)
            merged_nodes = self._dedupe_nodes(merged_nodes + depth_nodes)
            merged_nodes = sorted(merged_nodes, key=lambda n: float(n.score or 0.0), reverse=True)
            merged_nodes = self._filter_nodes_by_intent(merged_nodes, intent)
            merged_nodes = self._apply_budget(merged_nodes)

            score_info = self._compute_evidence_score(merged_nodes, intent)
            slot_ok = self._has_minimum_slots(merged_nodes, intent)
            gain = score_info["score"] - prev_score
            novelty = (
                max(0, len(merged_nodes) - prev_count) / max(1, len(merged_nodes))
                if merged_nodes else 0.0
            )

            diagnostics.append({
                "depth": depth,
                "node_count": len(merged_nodes),
                "slot_ok": slot_ok,
                "novelty": round(novelty, 4),
                "gain": round(gain, 4),
                **score_info,
            })

            if self._should_stop(
                depth=depth,
                max_depth=self.graph_depth,
                score_info=score_info,
                gain=gain,
                novelty=novelty,
                slot_ok=slot_ok,
            ):
                break

            prev_score = score_info["score"]
            prev_count = len(merged_nodes)

        return merged_nodes, diagnostics

    def _build_context(self, nodes: List[NodeWithScore]) -> List[str]:
        context_lines = []
        for i, node in enumerate(nodes[:self.similarity_top_k], 1):
            metadata = node.node.metadata if node.node and node.node.metadata else {}
            qualified_name = metadata.get("qualified_name", "N/A")
            node_type = metadata.get("node_type", "UNKNOWN")
            summary = metadata.get("llm_summary", "")
            text = node.node.text[:300] if node.node and node.node.text else "N/A"
            context_lines.append(
                f"[{i}] {qualified_name} ({node_type})\n"
                f"Summary: {summary or 'N/A'}\n"
                f"Snippet: {text}"
            )
        return context_lines

    def _synthesize_with_llm(self, query_str: str, context_lines: List[str]) -> str:
        if not self.llm or not hasattr(self.llm, "complete"):
            return ""

        prompt = (
            "你是代码知识图谱问答助手。基于给定证据回答问题。\n"
            "要求：\n"
            "1) 仅使用证据内容，不要编造。\n"
            "2) 答案尽量简洁。\n"
            "3) 最后附上你引用的证据编号，如 [1][3]。\n\n"
            f"问题：{query_str}\n\n"
            "证据：\n"
            + "\n\n".join(context_lines)
            + "\n\n回答："
        )
        try:
            return self.llm.complete(prompt).text.strip()
        except Exception:
            return ""

    def _build_response(
        self,
        original_query: str,
        normalized_query: str,
        nodes: List[NodeWithScore],
        diagnostics: List[Dict[str, Any]],
    ) -> str:
        if not nodes:
            return "未找到相关知识图谱信息。"

        context_lines = self._build_context(nodes)
        synthesized = self._synthesize_with_llm(original_query, context_lines)

        parts = [f"查询：{original_query}"]
        if normalized_query != original_query:
            parts.append(f"标准化查询：{normalized_query}")

        if diagnostics:
            parts.append("检索诊断：")
            for d in diagnostics:
                parts.append(
                    f"- depth={d['depth']}, nodes={d['node_count']}, score={d['score']}, "
                    f"gain={d['gain']}, novelty={d['novelty']}, slot_ok={d['slot_ok']}"
                )

        if synthesized:
            parts.append(f"\n答案：\n{synthesized}")

        parts.append(f"\n找到 {len(nodes)} 个相关节点（展示前 {len(context_lines)} 个）:")
        parts.extend(context_lines)
        return "\n".join(parts)

    def _execute_query(self, query_str: str) -> Response:
        normalized_query = self._normalize_query(query_str)
        intent = self._infer_query_intent(normalized_query)
        nodes, diagnostics = self._progressive_retrieve(normalized_query, intent)
        response_text = self._build_response(query_str, normalized_query, nodes, diagnostics)
        return Response(response=response_text, source_nodes=nodes)
