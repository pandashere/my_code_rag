# llama_index_integration/index_builder.py

from llama_index.core.indices.property_graph import PropertyGraphIndex
from .kg_extractor import CodeKGExtractor


class CodeKGIndexBuilder:
    def __init__(
        self,
        source_root: str,
        llm=None,
        embed_model=None,
        enable_llm_summary: bool = False,
        summary_strategy: str = "hybrid",
    ):
        self.source_root = source_root
        self.llm = llm
        self.embed_model = embed_model
        self.enable_llm_summary = enable_llm_summary
        self.summary_strategy = summary_strategy
        
        # ✅ CodeKGExtractor 本身就是 TransformComponent
        self.kg_extractor = CodeKGExtractor(
            source_root=source_root,
            enable_llm_summary=enable_llm_summary,
            summary_model=self.llm,
            summary_strategy=summary_strategy,
        )
        
        # ✅ Neo4j Graph Store
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        self.graph_store = Neo4jPropertyGraphStore(
            username="neo4j",
            password="your_password",
            url="bolt://localhost:7687",
        )
    
    def build_index(self, pattern: str = "**/*.py") -> PropertyGraphIndex:
        """构建索引"""
        # ✅ 1. 提取代码图谱（同时 cache 到 kg_extractor 中）
        print("🔍 正在提取代码图谱...")
        nodes = self.kg_extractor.extract(pattern)
        print(f"📦 获取了 {len(nodes)} 个图谱节点")
        
        # ✅ 2. 创建 storage context
        # storage_context = StorageContext.from_defaults(
        #     graph_store=self.graph_store,
        #     persist_dir="/home/zhaochen/code-reg/dbs"
        # )
        
        # ✅ 3. 创建索引 - kg_extractors 直接传 kg_extractor
        print("🔨 正在构建索引...")
        index = PropertyGraphIndex(
            nodes=nodes,
            # storage_context=storage_context,
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
            kg_extractors=[self.kg_extractor],  # ✅ 直接用 CodeKGExtractor
            show_progress=True,
        )
        
        print("✅ 索引构建完成")
        return index
    
    def close(self):
        """关闭连接"""
        self.graph_store.close()
