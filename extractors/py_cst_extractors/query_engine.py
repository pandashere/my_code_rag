# llama_index_integration/kg_query_engine.py
"""
代码知识图谱查询引擎
"""
from typing import Any, List
import asyncio

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.property_graph.base import PropertyGraphIndex
from llama_index.core.prompts.mixin import PromptDictType
from code_rag.utils.llm_funcs import normalize_query_for_retrieval

class CodeKGQueryEngine(BaseQueryEngine):
    """代码知识图谱查询引擎"""
    
    def __init__(
        self,
        index: PropertyGraphIndex,
        llm: Any = None,
        graph_depth: int = 3,
        similarity_top_k: int = 5,
    ):
        super().__init__(callback_manager=llm.callback_manager if llm and hasattr(llm, "callback_manager") else None)
        self.index = index
        self.llm = llm
        self.graph_depth = max(1, graph_depth)
        self.similarity_top_k = similarity_top_k
        
        # 🔧 使用 index 提供的标准 retriever
        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            VectorContextRetriever,
        )
        self._retriever = VectorContextRetriever(
            graph_store=index.property_graph_store,
            embed_model=index._embed_model,
            similarity_top_k=similarity_top_k,
            path_depth=self.graph_depth,
        )
    
    def _query(self, query_bundle: QueryBundle) -> Response:
        """同步查询实现"""
        return self._execute_query(query_bundle.query_str)
    
    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """异步查询实现（必需）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._execute_query, 
            query_bundle.query_str
        )
    
    def _get_prompt_modules(self) -> PromptDictType:
        """Prompt 模块定义（必需）"""
        return {}
    
    
    
    def _execute_query(self, query_str: str) -> Response:
        """执行实际查询逻辑"""
        normalized_query = self._normalize_query(query_str)
        nodes = self._retriever.retrieve(normalized_query)

        # 构建响应
        response_text = self._build_response(query_str, normalized_query, nodes)
        
        return Response(
            response=response_text,
            source_nodes=nodes,
        )
    
    def _normalize_query(self, query_str: str) -> str:
        """将查询标准化；当无可用 LLM 时退化为原始查询。"""
        if not self.llm or not hasattr(self.llm, "complete"):
            return query_str
        try:
            return normalize_query_for_retrieval(query_str, self.llm, True)
        except Exception:
            return query_str

    def _build_context(self, nodes: List) -> List[str]:
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

    def _build_response(self, original_query: str, normalized_query: str, nodes: List) -> str:
        """构建查询响应"""
        if not nodes:
            return "未找到相关知识图谱信息。"

        context_lines = self._build_context(nodes)
        synthesized = self._synthesize_with_llm(original_query, context_lines)

        parts = [f"查询：{original_query}"]
        if normalized_query != original_query:
            parts.append(f"标准化查询：{normalized_query}")

        if synthesized:
            parts.append(f"\n答案：\n{synthesized}")

        parts.append(f"\n找到 {len(nodes)} 个相关节点（展示前 {len(context_lines)} 个）:")
        parts.extend(context_lines)
        return "\n".join(parts)
