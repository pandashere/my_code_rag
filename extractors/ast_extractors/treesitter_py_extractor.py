from tree_sitter import Language, Parser
import tree_sitter_python as tspy

from extractors.extractor_base import *
from llama_index.core.text_splitter import CodeSplitter
from llama_index.core.graph_stores import PropertyGraphStore

class TSPythonExtractor(BaseExtractor):
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)
    
    @classmethod
    def class_name(cls):
        return "TSPythonExtractor"
    
    async def _agenerate_node_summary(self, node:BaseNode) -> str:
        pass

    async def aextract(self, nodes):
        return await super().aextract(nodes)
    
    