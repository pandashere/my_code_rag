# llama_index_integration/kg_extractor.py
"""
代码知识图谱提取器 - 修复版 (v2)

✅ 从 results 提取所有文件内实体和关系
✅ 保留跨文件关系
✅ 添加 LLM summary 支持
✅ 修复 CodeEntityNode 属性访问
✅ 【修复】Relation 表示 Entity 之间的关系，不是 TextNode 之间的关系
"""
from typing import List, Optional, Dict, Any, Sequence, Tuple
from pathlib import Path
import hashlib

from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, RelatedNodeInfo, TransformComponent
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from pydantic import Field, PrivateAttr

from .extractor import extract_directory, ExtractionResult
from .symbol_info import SymbolInfo
from .py_relations import CodeEntityNode, CodeRelation


class CodeKGExtractor(TransformComponent):
    """代码图谱提取器 - 修复版 (v2)"""
    
    source_root: str = Field(default=".", description="代码根目录")
    code_max_chars: int = Field(default=2000, description="代码段最大字符数")
    code_max_lines: int = Field(default=50, description="代码段最大行数")
    
    # ✅ LLM Summary 配置
    enable_llm_summary: bool = Field(default=False, description="是否启用 LLM 生成摘要")
    summary_model: Optional[Any] = Field(default=None, description="LLM 模型实例")
    
    _cache: Dict[str, Tuple[List[EntityNode], List[Relation]]] = PrivateAttr(default_factory=dict)
    _all_nodes: List[BaseNode] = PrivateAttr(default_factory=list)
    _file_results: List[ExtractionResult] = PrivateAttr(default_factory=list)
    _global_symbol_table: Dict[str, SymbolInfo] = PrivateAttr(default_factory=dict)
    _cross_file_relations: List[CodeRelation] = PrivateAttr(default_factory=list)
    _node_id_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    
    # ✅ 新增：全局 Entity 和 Relation 池
    _entity_pool: Dict[str, EntityNode] = PrivateAttr(default_factory=dict)  # qualified_name → EntityNode
    _relation_pool: List[Relation] = PrivateAttr(default_factory=list)  # 所有 Relation
    
    @classmethod
    def class_name(cls) -> str:
        return "CodeKGExtractor"
    
    def __init__(
        self, 
        source_root: str = ".", 
        enable_llm_summary: bool = False,
        summary_model: Optional[Any] = None,
        **kwargs: Any
    ):
        """初始化"""
        super().__init__(
            source_root=source_root, 
            enable_llm_summary=enable_llm_summary,
            summary_model=summary_model,
            **kwargs
        )
        self._cache = {}
        self._all_nodes = []
        self._file_results = []
        self._global_symbol_table = {}
        self._cross_file_relations = []
        self._node_id_map = {}
        self._entity_pool = {}
        self._relation_pool = []
    
    def extract(self, pattern: str = "**/*.py") -> List[BaseNode]:
        """
        提取代码图谱
        
        ✅ 从 results 提取所有文件内实体和关系
        ✅ 合并跨文件关系
        ✅ 可选生成 LLM summary
        ✅ 返回 TextNode 列表供 PropertyGraphIndex 使用
        """
        # 1. 解析代码，获取所有 results
        self._initialize_from_results(pattern)
        
        # 2. 构建全局 Entity 池和 Relation 池
        self._build_entity_and_relation_pools()
        
        # 3. 从 results 构建 TextNode 列表
        nodes = self._build_nodes_from_results()
        
        # 4. ✅ 可选：生成 LLM summary
        if self.enable_llm_summary and self.summary_model:
            nodes = self._generate_llm_summaries(nodes)
        
        # 5. Cache KG 数据（每个 TextNode 关联的 EntityNode + Relation）
        self._cache_kg_data(nodes)
        
        # 6. 存储所有 nodes
        self._all_nodes = nodes
        
        print(f"✅ extract 完成：{len(nodes)} 个 TextNode")
        print(f"   - 全局 Entity 池：{len(self._entity_pool)} 个实体")
        print(f"   - 全局 Relation 池：{len(self._relation_pool)} 个关系")
        return nodes
    
    def _initialize_from_results(self, pattern: str = "**/*.py") -> None:
        """从 extract_directory 获取所有数据"""
        print(f"🔍 扫描代码目录：{self.source_root}")
        
        results, global_symbols, cross_file_rels = extract_directory(
            str(self.source_root),
            pattern=pattern,
        )
        
        self._file_results = results
        self._global_symbol_table = global_symbols
        self._cross_file_relations = cross_file_rels
        
        total_nodes = sum(len(r.nodes) for r in results)
        total_rels = sum(len(r.relations) for r in results)
        
        print(f"✅ 提取完成:")
        print(f"   - {len(results)} 个文件")
        print(f"   - {total_nodes} 个文件内实体")
        print(f"   - {total_rels} 个文件内关系")
        print(f"   - {len(cross_file_rels)} 个跨文件关系")
        print(f"   - {len(global_symbols)} 个全局符号")
    
    def _build_entity_and_relation_pools(self) -> None:
        """
        ✅ 构建全局 Entity 池和 Relation 池
        
        Entity 使用 qualified_name 作为 id（保证全局唯一）
        Relation 的 source_id/target_id 使用 Entity 的 qualified_name
        """
        # 1. 构建 Entity 池
        for result in self._file_results:
            for code_node in result.nodes:
                entity = self._code_node_to_entity(code_node)
                self._entity_pool[entity.name] = entity  # name = qualified_name
        
        # 2. 构建 Relation 池（文件内关系）
        for result in self._file_results:
            for code_rel in result.relations:
                relation = self._code_relation_to_relation(code_rel)
                self._relation_pool.append(relation)
        
        # 3. 添加跨文件关系
        for code_rel in self._cross_file_relations:
            relation = self._code_relation_to_relation(code_rel)
            self._relation_pool.append(relation)
        
        print(f"📦 全局池构建完成：{len(self._entity_pool)} 实体，{len(self._relation_pool)} 关系")
    
    def _code_node_to_entity(self, code_node: CodeEntityNode) -> EntityNode:
        """将 CodeEntityNode 转换为 EntityNode"""
        qualified_name = code_node.properties.get("qualified_name", code_node.id)
        
        return EntityNode(
            name=qualified_name,  # ✅ 使用 qualified_name 作为 Entity 的 id
            label=code_node.properties.get("node_type", "UNKNOWN"),
            properties={
                "file_path": code_node.properties.get("file_path", ""),
                "line_number": code_node.properties.get("line_number", 0),
                "module": code_node.properties.get("module", ""),
                "scope": code_node.properties.get("scope", ""),
                "code_span": self._truncate_code(code_node.properties.get("code_span", "")),
            },
        )
    
    def _code_relation_to_relation(self, code_rel: CodeRelation) -> Relation:
        """
        将 CodeRelation 转换为 Relation
        
        ✅ source_id/target_id 使用 Entity 的 qualified_name，不是 TextNode 的 id
        """
        return Relation(
            source_id=code_rel.source_id,      # ✅ Entity 的 qualified_name
            target_id=code_rel.target_id,      # ✅ Entity 的 qualified_name
            label=code_rel.label,
            properties={
                "is_cross_file": code_rel.properties.get("is_cross_file", False),
                **code_rel.properties,
            },
        )
    
    def _build_nodes_from_results(self) -> List[BaseNode]:
        """从 results 构建 TextNode 列表"""
        nodes = []
        node_id_map = {}
        
        for result in self._file_results:
            for code_node in result.nodes:
                text_node = self._code_node_to_text_node(code_node)
                nodes.append(text_node)
                node_id_map[code_node.id] = text_node.id_
        
        self._node_id_map = node_id_map
        
        return nodes
    
    def _code_node_to_text_node(self, code_node: CodeEntityNode) -> TextNode:
        """
        将 CodeEntityNode 转换为 TextNode
        
        ✅ 所有属性从 properties 字典获取
        """
        # ✅ 生成唯一 ID
        node_id = self._generate_text_node_id(code_node)
        
        # ✅ 构建元数据
        metadata = {
            "qualified_name": code_node.properties.get("qualified_name", ""),
            "node_type": code_node.properties.get("node_type", ""),
            "file_path": code_node.properties.get("file_path", ""),
            "line_number": code_node.properties.get("line_number", 0),
            "module": code_node.properties.get("module", ""),
            "scope": code_node.properties.get("scope", ""),
        }
        
        # ✅ 添加代码段
        if "code_span" in code_node.properties:
            metadata["code_span"] = self._truncate_code(
                code_node.properties["code_span"]
            )
        
        # ✅ 构建文本内容
        text = self._build_node_text(code_node)
        
        return TextNode(
            id_=node_id,
            text=text,
            metadata=metadata,
        )
    
    def _generate_llm_summaries(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        ✅ 使用 LLM 为每个节点生成摘要
        
        摘要会添加到：
        - node.metadata["llm_summary"]
        - node.text (追加到末尾)
        """
        print(f"🤖 开始生成 LLM 摘要 ({len(nodes)} 个节点)...")
        
        # 构建 prompt
        prompt_template = """
请为以下代码实体生成简洁的功能摘要（50 字以内）：

实体名称：{name}
实体类型：{node_type}
代码内容：
```python
{code}"""
        # 批量生成（避免 API 限制）
        batch_size = 10
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            for node in batch:
                try:
                    code = node.metadata.get("code_span", "")
                    if not code:
                        continue
                    
                    prompt = prompt_template.format(
                        name=node.metadata.get("qualified_name", ""),
                        node_type=node.metadata.get("node_type", ""),
                        code=code[:1000],  # 限制代码长度
                    )
                    
                    # 调用 LLM
                    response = self.summary_model.complete(prompt)
                    summary = response.text.strip()
                    
                    # ✅ 存储摘要
                    node.metadata["llm_summary"] = summary
                    
                except Exception as e:
                    print(f"⚠️ 生成摘要失败：{e}")
                    node.metadata["llm_summary"] = ""
            
            print(f"   进度：{min(i + batch_size, len(nodes))}/{len(nodes)}")
        
        print(f"✅ LLM 摘要生成完成")
        return nodes

    def _cache_kg_data(self, nodes: List[BaseNode]) -> None:
        """
        ✅ 将每个 TextNode 关联的 KG 数据 cache 起来
        
        每个 TextNode 的 metadata 包含：
        - KG_NODES_KEY: 与该 TextNode 相关的 EntityNode 列表（通常是 1 个）
        - KG_RELATIONS_KEY: 与该 TextNode 相关的 Relation 列表（Entity 之间的关系）
        """
        total_relations = 0
        
        for node in nodes:
            qualified_name = node.metadata.get("qualified_name", "")
            
            # 1. 找到该 TextNode 对应的 EntityNode
            related_entities = []
            if qualified_name in self._entity_pool:
                related_entities = [self._entity_pool[qualified_name]]
            
            # 2. 找到与该 Entity 相关的所有 Relation
            related_relations = []
            for rel in self._relation_pool:
                if rel.source_id == qualified_name or rel.target_id == qualified_name:
                    related_relations.append(rel)
                    total_relations += 1
            
            # 3. Cache 起来
            self._cache[node.id_] = (related_entities, related_relations)
        
        print(f"📦 Cache: {len(nodes)} 个 TextNode，共 {total_relations} 个关系")

    def __call__(
        self, 
        nodes: Sequence[BaseNode], 
        show_progress: bool = False, 
        **kwargs: Any
    ) -> Sequence[BaseNode]:
        """将 TextNode 转换为 LlamaIndex KG 格式"""
        if show_progress:
            print(f"📋 处理 {len(nodes)} 个节点...")
        
        for node in nodes:
            if node.id_ in self._cache:
                kg_nodes, kg_relations = self._cache[node.id_]
            else:
                kg_nodes, kg_relations = [], []
            
            node.metadata[KG_NODES_KEY] = kg_nodes
            node.metadata[KG_RELATIONS_KEY] = kg_relations
        
        if show_progress:
            print(f"✅ 处理完成")
        
        return nodes

    async def acall(
        self, 
        nodes: Sequence[BaseNode], 
        show_progress: bool = False, 
        **kwargs: Any
    ) -> Sequence[BaseNode]:
        """异步版本"""
        return self.__call__(nodes, show_progress=show_progress, **kwargs)

    def _map_relation_type(self, label: str) -> Optional[NodeRelationship]:
        """映射自定义关系到 NodeRelationship 枚举"""
        mapping = {
            "CALLS": NodeRelationship.NEXT,
            "INHERITS": NodeRelationship.PARENT,
            "CONTAINS": NodeRelationship.CHILD,
            "IMPORTS": NodeRelationship.PREVIOUS,
            "USES": NodeRelationship.NEXT,
            "ASSIGNS": NodeRelationship.CHILD,
            "HAS_PARAM": NodeRelationship.CHILD,
            "HAS_TYPE": NodeRelationship.CHILD,
        }
        return mapping.get(label)

    def _generate_text_node_id(self, code_node: CodeEntityNode) -> str:
        """
        ✅ 从 properties 获取属性
        """
        qname = code_node.properties.get("qualified_name", code_node.id)
        file_path = code_node.properties.get("file_path", "unknown")
        unique_str = f"{qname}_{file_path}"
        hash_id = hashlib.md5(unique_str.encode()).hexdigest()[:16]
        return f"code_node_{hash_id}"

    def _build_node_text(self, code_node: CodeEntityNode) -> str:
        """
        ✅ 从 properties 获取属性
        ✅ 添加：LLM summary 占位符（会在生成后填充）
        """
        # ✅ 修复：使用 .get() 从 properties 字典获取
        parts = [
            f"# {code_node.name}",
            f"Type: {code_node.properties.get('node_type', 'UNKNOWN')}",
            f"Qualified: {code_node.properties.get('qualified_name', '')}",
            f"File: {code_node.properties.get('file_path', 'unknown')}",
            f"Line: {code_node.properties.get('line_number', 0)}",
        ]
        
        # ✅ 添加代码段
        if "code_span" in code_node.properties:
            code = self._truncate_code(code_node.properties["code_span"])
            parts.append(f"\n```python\n{code}\n```")
        
        return "\n".join(parts)

    def _truncate_code(self, code: str) -> str:
        """截断代码段"""
        if len(code) > self.code_max_chars:
            code = code[:self.code_max_chars] + "..."
        lines = code.split("\n")
        if len(lines) > self.code_max_lines:
            code = "\n".join(lines[:self.code_max_lines]) + "\n..."
        return code
