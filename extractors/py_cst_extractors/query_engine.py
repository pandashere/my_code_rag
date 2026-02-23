# llama_index_integration/kg_query_engine.py
"""
代码知识图谱查询引擎
"""
from typing import Any, Dict, List, Optional
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
        llm: Any,
        graph_depth: int = 3,
        similarity_top_k: int = 5,
    ):
        super().__init__(callback_manager=llm.callback_manager if hasattr(llm, 'callback_manager') else None)
        self.index = index
        self.llm = llm
        self.graph_depth = graph_depth
        self.similarity_top_k = similarity_top_k
        
        # 🔧 使用 index 提供的标准 retriever
        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            VectorContextRetriever,
        )
        self._retriever = VectorContextRetriever(
            graph_store=index.property_graph_store,
            embed_model=index._embed_model,
            similarity_top_k=similarity_top_k,
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
        # 🔧 使用 retriever 检索，不直接访问 graph_store
        query_str = normalize_query_for_retrieval(query_str, self.llm, True)
        nodes = self._retriever.retrieve(query_str)
        
        # 构建响应
        response_text = self._build_response(query_str, nodes)
        
        return Response(
            response=response_text,
            source_nodes=nodes,
        )
    
    def _build_response(self, query_str: str, nodes: List) -> str:
        """构建查询响应"""
        if not nodes:
            return "未找到相关知识图谱信息。"
        
        parts = [f"查询：{query_str}\n"]
        parts.append(f"找到 {len(nodes)} 个相关节点:\n")
        
        for i, node in enumerate(nodes[:self.similarity_top_k], 1):
            text = node.node.text[:300] if node.node.text else "N/A"
            parts.append(f"{i}. {text}...\n")
        
        return "\n".join(parts)
