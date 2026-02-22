"""
代码符号提取器 - 类型常量定义模块
统一管理所有实体类型和关系类型
"""
from dataclasses import dataclass
from typing import Set


@dataclass(frozen=True)
class TYPE:
    """
    所有类型常量统一管理类
    
    包含：
    - 实体类型（ENTITY_*）
    - 非实体类型（CHUNK_*）
    - 关系类型（REL_*）
    """
    
    # ==================== 实体类型 ====================
    ENTITY_MODULE: str = "MODULE"
    ENTITY_CLASS: str = "CLASS"
    ENTITY_FUNCTION: str = "FUNCTION"
    ENTITY_METHOD: str = "METHOD"
    ENTITY_ASYNC_FUNCTION: str = "ASYNC_FUNCTION"
    ENTITY_ASYNC_METHOD: str = "ASYNC_METHOD"
    ENTITY_VARIABLE: str = "VARIABLE"
    ENTITY_PARAMETER: str = "PARAMETER"
    ENTITY_MODULE_IMPORT: str = "MODULE_IMPORT"
    
    # ==================== 非实体类型 ====================
    CHUNK: str = "CHUNK"
    STATEMENT: str = "STATEMENT"
    
    # ==================== 关系类型 ====================
    REL_CALLS: str = "CALLS"
    REL_CONTAINS: str = "CONTAINS"
    REL_ASSIGNS: str = "ASSIGNS"
    REL_INHERITS: str = "INHERITS"
    REL_OVERRIDES: str = "OVERRIDES"
    REL_IMPORTS: str = "IMPORTS"
    REL_HAS_TYPE: str = "HAS_TYPE"
    REL_HAS_PARAM: str = "HAS_PARAM"
    REL_USES: str = "USES"
    
    # ==================== 实体类型集合 ====================
    @property
    def ENTITY_TYPES(self) -> Set[str]:
        """获取所有实体类型集合"""
        return {
            self.ENTITY_MODULE,
            self.ENTITY_CLASS,
            self.ENTITY_FUNCTION,
            self.ENTITY_METHOD,
            self.ENTITY_ASYNC_FUNCTION,
            self.ENTITY_ASYNC_METHOD,
            self.ENTITY_VARIABLE,
            self.ENTITY_PARAMETER,
        }
    
    @property
    def CHUNK_TYPES(self) -> Set[str]:
        """获取所有非实体类型集合"""
        return {self.CHUNK, self.STATEMENT}
    
    @property
    def RELATION_TYPES(self) -> Set[str]:
        """获取所有关系类型集合"""
        return {
            self.REL_CALLS,
            self.REL_CONTAINS,
            self.REL_ASSIGNS,
            self.REL_INHERITS,
            self.REL_OVERRIDES,
            self.REL_IMPORTS,
            self.REL_HAS_TYPE,
            self.REL_HAS_PARAM,
            self.REL_USES,
        }
    
    @property
    def ALL_NODE_TYPES(self) -> Set[str]:
        """获取所有节点类型（实体 + 非实体）"""
        return self.ENTITY_TYPES | self.CHUNK_TYPES
    
    # ==================== 判断方法 ====================
    def is_entity_type(self, node_type: str) -> bool:
        """判断是否为实体类型"""
        return node_type in self.ENTITY_TYPES
    
    def is_chunk_type(self, node_type: str) -> bool:
        """判断是否为非实体类型"""
        return node_type in self.CHUNK_TYPES
    
    def is_valid_node_type(self, node_type: str) -> bool:
        """判断是否为有效的节点类型"""
        return node_type in self.ALL_NODE_TYPES
    
    def is_valid_relation_type(self, relation_type: str) -> bool:
        """判断是否为有效的关系类型"""
        return relation_type in self.RELATION_TYPES

TypeManager = TYPE()
