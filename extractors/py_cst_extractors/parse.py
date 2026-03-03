"""
Neo4j 图存储封装
"""

from typing import List, Optional, Dict, Any

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from .py_relations import CodeRelation
from .extractor import ExtractionResult, extract_directory, SymbolInfo


class CodeGraphStore:
    """代码知识图谱存储封装"""
    
    def __init__(
        self,
        url: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        database: str = "neo4j",
    ):
        self.store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url,
            database=database,
            refresh_schema=True,
            create_indexes=True,
        )
        
        self.global_symbol_table: Dict[str, SymbolInfo] = {}
        self.cross_file_relations: List[CodeRelation] = []  # 🔧 新增：存储跨文件关系
    
    def insert_extraction_result(self, result: ExtractionResult) -> None:
        """插入提取结果"""
        if result.nodes:
            self.store.upsert_nodes(result.nodes)
        
        if result.relations:
            self.store.upsert_relations(result.relations)
        
        if result.errors:
            print(f"⚠️ 提取错误：{result.errors}")
    
    def insert_directory(
        self,
        dir_path: str,
        pattern: str = "**/*.py",
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """提取并插入整个目录"""
        # 🔧 获取跨文件关系
        results, global_symbol_table, cross_file_relations = extract_directory(
            dir_path, pattern, exclude_patterns
        )
        
        self.global_symbol_table = global_symbol_table
        self.cross_file_relations = cross_file_relations  # 🔧 保存跨文件关系
        
        total_nodes = 0
        total_relations = 0
        total_errors = 0
        
        for result in results:
            self.insert_extraction_result(result)
            total_nodes += result.node_count
            total_relations += result.relation_count
            total_errors += len(result.errors)
        
        return {
            "files_processed": len(results),
            "total_nodes": total_nodes,
            "total_relations": total_relations,
            "total_errors": total_errors,
            "global_symbols": len(global_symbol_table),
            "cross_file_relations": len(cross_file_relations),  # 🔧 新增：跨文件关系数
        }
    
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """执行 Cypher 查询"""
        return self.store.structured_query(cypher, params)
    
    def close(self) -> None:
        """关闭连接"""
        self.store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # 方式 2: 提取并存储到 Neo4j
    with CodeGraphStore(
        url="bolt://localhost:7687",
        username="neo4j",
        password="your_password",
    ) as store:
        # 导入整个项目
        stats = store.insert_directory(
            "/home/zhaochen/code-reg/code-venv/lib64/python3.12/site-packages/aiohttp",
            # "/home/zhaochen/code-reg/code_rag",
            exclude_patterns=["**/tests/**", "**/__pycache__/**"],
        )
        print(f"导入完成：{stats}")

        # 查询示例
        results = store.query(
            """
            MATCH (f:FUNCTION)-[:CALLS]->(g:FUNCTION)
            RETURN f.name as caller, g.name as callee
            LIMIT 10
            """
        )
        for row in results:
            print(f"{row['caller']} -> {row['callee']}")
