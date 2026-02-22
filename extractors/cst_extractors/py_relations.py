"""
代码知识图谱专用 Relation 和 EntityNode 类
"""

from typing import Any, Dict, Optional, List
from llama_index.core.graph_stores.types import (
    Relation as BaseRelation,
    EntityNode as BaseEntityNode,
)


class CodeRelation(BaseRelation):
    """
    代码知识图谱专用 Relation
    
    使用复合 ID (source__label__target) 避免相同 label 的关系冲突
    """
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            source_id=source_id,
            target_id=target_id,
            label=label,
            properties=properties or {},
        )
    
    @property
    def id(self) -> str:
        """生成唯一 ID：source__label__target"""
        return f"{self.source_id}__{self.label}__{self.target_id}"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CodeRelation):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.label == other.label
        )
    
    def __repr__(self) -> str:
        return f"CodeRelation({self.source_id})-[{self.label}]->({self.target_id})"


class CodeEntityNode(BaseEntityNode):
    """
    代码知识图谱专用 EntityNode
    
    简化创建流程，自动处理 label 和 properties
    """
    
    def __init__(
        self,
        name: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ):
        # EntityNode 的构造函数签名
        super().__init__(
            name=name,
            label=label,
            properties=properties or {},
            embedding=embedding,
        )
    
    @classmethod
    def create(
        cls,
        name: str,
        node_type: str,
        qualified_name: Optional[str] = None,
        module: Optional[str] = None,
        file_path: Optional[str] = None,
        scope: Optional[str] = None,
        extra_properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> "CodeEntityNode":
        """
        创建代码实体节点
        
        Args:
            name: 节点名称（如函数名、类名）
            node_type: 节点类型（如 "FUNCTION", "CLASS", "MODULE"）- 会作为 Neo4j Label
            qualified_name: 限定名（如 "module.submodule.Class.method"）
            module: 模块名
            file_path: 文件路径
            scope: 作用域 ID
            extra_properties: 其他属性
            embedding: 嵌入向量
            
        Returns:
            CodeEntityNode 实例
        """
        # 构建 qualified_name
        qname = qualified_name if qualified_name else name
        
        # 构建 properties
        properties = {
            "qualified_name": qname,
            "node_type": node_type,
        }
        
        if module:
            properties["module"] = module
        if file_path:
            properties["file_path"] = file_path
        if scope:
            properties["scope"] = scope
        if extra_properties:
            properties.update(extra_properties)
        
        return cls(
            name=name,
            label=node_type,  # label 作为 Neo4j 的额外 Label
            properties=properties,
            embedding=embedding,
        )
    
    @property
    def id(self) -> str:
        """获取节点的唯一 ID（使用 qualified_name）"""
        return self.properties.get("qualified_name", self.name)
    
    def __repr__(self) -> str:
        return f"CodeEntityNode({self.name}:{self.label})"
