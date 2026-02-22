"""
代码符号提取器 - 使用 libcst 解析 Python 代码并构建知识图谱
"""
from libcst import MaybeSentinel 
import libcst as cst
from libcst.metadata import (
    MetadataWrapper,
    ScopeProvider,
    QualifiedNameProvider,
    ParentNodeProvider,
    PositionProvider,
)
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import json

from py_relations import CodeRelation, CodeEntityNode


# ==================== 新增：符号信息数据类 ====================
@dataclass
class SymbolInfo:
    """全局符号表中的符号信息"""
    name: str
    qualified_name: str
    node_type: str
    scope: str
    line_number: int
    file_path: str
    node_id: str
    references: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    # 🔧 新增：存储额外属性（如基类、装饰器等）
    extra_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "node_type": self.node_type,
            "scope": self.scope,
            "line_number": self.line_number,
            "file_path": self.file_path,
            "node_id": self.node_id,
            "references": self.references,
            "definitions": self.definitions,
            "extra_properties": self.extra_properties,
        }



def _serialize_property(value: Any) -> Any:
    """将复杂对象序列化为 Neo4j 兼容的格式"""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


@dataclass
class ExtractionResult:
    """提取结果容器"""
    nodes: List[CodeEntityNode] = field(default_factory=list)
    relations: List[CodeRelation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    symbol_table: Dict[str, SymbolInfo] = field(default_factory=dict)
    file_path: Optional[str] = None
    # 🔧 新增：存储未解析的引用（用于跨文件解析）
    unresolved_calls: List[Dict[str, str]] = field(default_factory=list)
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def relation_count(self) -> int:
        return len(self.relations)
    
    @staticmethod
    def merge_symbol_tables(results: List['ExtractionResult']) -> Dict[str, SymbolInfo]:
        """合并多个 ExtractionResult 的符号表"""
        merged = {}
        for result in results:
            for qname, symbol in result.symbol_table.items():
                if qname in merged:
                    existing = merged[qname]
                    if existing.file_path != symbol.file_path:
                        print(f"⚠️ 符号冲突：{qname} 存在于 {existing.file_path} 和 {symbol.file_path}")
                else:
                    merged[qname] = symbol
        return merged

    @staticmethod
    def resolve_and_create_cross_file_relations(
        results: List['ExtractionResult'], 
        global_symbol_table: Dict[str, SymbolInfo]
    ) -> List[CodeRelation]:
        """解析跨文件引用并创建关系"""
        new_relations = []
        seen_relations: Set[Tuple[str, str, str]] = set()
        
        qname_to_symbol = global_symbol_table
        
        # 构建 class_name -> SymbolInfo 映射（用于 super 解析）
        class_to_symbol: Dict[str, SymbolInfo] = {}
        for qname, symbol in qname_to_symbol.items():
            if symbol.node_type == CodeSymbolExtractor.TYPE_CLASS:
                class_to_symbol[symbol.name] = symbol
        
        for result in results:
            for call_info in result.unresolved_calls:
                caller_id = call_info['caller_id']
                func_name = call_info['func_name']
                call_file = call_info.get('file_path', result.file_path)
                is_super_call = call_info.get('is_super_call', False)
                
                callee_symbol = None
                
                # 🔧 处理 super() 调用
                if is_super_call and func_name.startswith("super."):
                    method_name = func_name.split(".", 1)[1]
                    callee_symbol = CodeSymbolExtractor._resolve_super_call_cross_file(
                        method_name, caller_id, qname_to_symbol, class_to_symbol
                    )
                
                # 普通调用解析
                if not callee_symbol:
                    # 尝试 1: 直接匹配 qualified_name
                    if func_name in qname_to_symbol:
                        callee_symbol = qname_to_symbol[func_name]
                    
                    # 尝试 2: 按方法名匹配（排除自调用）
                    if not callee_symbol:
                        callee_symbol = CodeSymbolExtractor._resolve_by_method_name_cross_file(
                            func_name, caller_id, qname_to_symbol
                        )
                
                if callee_symbol and callee_symbol.node_id != caller_id:
                    # 🔧 确保不是自调用
                    rel_key = (caller_id, callee_symbol.node_id, "CALLS")
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        rel = CodeRelation(
                            source_id=caller_id,
                            target_id=callee_symbol.node_id,
                            label="CALLS",
                            properties={
                                "call_site": call_file,
                                "is_cross_file": callee_symbol.file_path != call_file,
                                "is_super_call": is_super_call,
                            }
                        )
                        new_relations.append(rel)
        
        return new_relations


class CodeSymbolExtractor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)
    
    # ========== 实体类型（Entity）==========
    TYPE_MODULE = "MODULE"
    TYPE_CLASS = "CLASS"
    TYPE_FUNCTION = "FUNCTION"
    TYPE_METHOD = "METHOD"
    TYPE_ASYNC_FUNCTION = "ASYNC_FUNCTION"
    TYPE_ASYNC_METHOD = "ASYNC_METHOD"
    TYPE_VARIABLE = "VARIABLE"
    TYPE_PARAMETER = "PARAMETER"
    # 🔧 新增：导入模块类型
    TYPE_MODULE_IMPORT = "MODULE_IMPORT"
    
    # ========== 非实体类型（Chunk/其他）==========
    TYPE_CHUNK = "CHUNK"
    TYPE_STATEMENT = "STATEMENT"
    
    # ========== 关系类型 ==========
    REL_CALLS = "CALLS"
    REL_CONTAINS = "CONTAINS"
    REL_ASSIGNS = "ASSIGNS"
    REL_INHERITS = "INHERITS"
    REL_OVERRIDES = "OVERRIDES"
    REL_IMPORTS = "IMPORTS"
    REL_HAS_TYPE = "HAS_TYPE"
    REL_HAS_PARAM = "HAS_PARAM"
    # 🔧 新增：使用关系
    REL_USES = "USES"

    
    def __init__(self, file_path: str, module_name: Optional[str] = None):
        self.file_path = file_path
        self.module_name = module_name or self._infer_module_name(file_path)
        
        self.nodes: List[CodeEntityNode] = []
        self.relations: List[CodeRelation] = []
        
        
        self._scope_stack: List[Dict[str, Any]] = []
        self._seen_relations: Set[Tuple[str, str, str]] = set()
        self._import_map: Dict[str, str] = {}
        self._local_vars: Dict[str, str] = {}
        self._current_class_id: Optional[str] = None
        self._current_function_id: Optional[str] = None
        self._metadata = {}
        self.errors: List[str] = []
        
        # ==================== 全局符号表 ====================
        self.symbol_table: Dict[str, SymbolInfo] = {}
        self._global_symbols: Set[str] = set()
        self.unresolved_calls: List[Dict[str, str]] = [] # ← 需要这个属性
    
    # ==================== 符号表管理方法 ====================
    @staticmethod
    def is_entity_type(node_type: str) -> bool:
        """判断是否为实体类型（可建立关系的节点）"""
        entity_types = {
            CodeSymbolExtractor.TYPE_MODULE,
            CodeSymbolExtractor.TYPE_CLASS,
            CodeSymbolExtractor.TYPE_FUNCTION,
            CodeSymbolExtractor.TYPE_METHOD,
            CodeSymbolExtractor.TYPE_ASYNC_FUNCTION,
            CodeSymbolExtractor.TYPE_ASYNC_METHOD,
            CodeSymbolExtractor.TYPE_VARIABLE,
            CodeSymbolExtractor.TYPE_PARAMETER,
        }
        return node_type in entity_types

    @staticmethod
    def _resolve_super_call_cross_file(
        method_name: str,
        caller_id: str,
        qname_to_symbol: Dict[str, SymbolInfo],
        class_to_symbol: Dict[str, SymbolInfo]
    ) -> Optional[SymbolInfo]:
        """跨文件解析 super() 调用"""
        parts = caller_id.split(".")
        if len(parts) < 2:
            return None
        
        class_name = parts[-2]
        
        if class_name not in class_to_symbol:
            return None
        
        current_class = class_to_symbol[class_name]
        base_classes = current_class.extra_properties.get("base_classes", [])
        
        for base_name in base_classes:
            # 尝试直接匹配
            method_qname = f"{base_name}.{method_name}"
            if method_qname in qname_to_symbol:
                return qname_to_symbol[method_qname]
            
            # 尝试模糊匹配
            for qname, symbol in qname_to_symbol.items():
                if symbol.name == method_name and qname.startswith(f"{base_name}."):
                    return symbol
        
        return None
    
    @staticmethod
    def _resolve_by_method_name_cross_file(
        method_name: str,
        caller_id: str,
        qname_to_symbol: Dict[str, SymbolInfo]
    ) -> Optional[SymbolInfo]:
        """跨文件按方法名解析（避免自调用）"""
        parts = caller_id.split(".")
        current_class = parts[-2] if len(parts) >= 2 else None
        
        for qname, symbol in qname_to_symbol.items():
            if symbol.name == method_name and symbol.node_type in (
                CodeSymbolExtractor.TYPE_FUNCTION,
                CodeSymbolExtractor.TYPE_METHOD,
                CodeSymbolExtractor.TYPE_ASYNC_FUNCTION,
            ):
                # 跳过当前类的方法
                if current_class and qname.startswith(f"{current_class}."):
                    continue
                return symbol
        
        return None

    def _register_symbol(self, name: str, node_type: str, node: cst.CSTNode, 
                        node_id: str, extra_properties: Optional[Dict[str, Any]] = None) -> str:
        """注册符号到全局符号表"""
        if self._scope_stack:
            qualified_name = f"{self._scope_stack[-1]['id']}.{name}"
        else:
            qualified_name = f"{self.module_name}.{name}"
            self._global_symbols.add(qualified_name)
        
        symbol = SymbolInfo(
            name=name,
            qualified_name=qualified_name,
            node_type=node_type,
            scope=self._scope_stack[-1]['id'] if self._scope_stack else self.module_name,
            line_number=self._get_line_number(node),
            file_path=self.file_path,
            node_id=node_id,
            extra_properties=extra_properties or {},  # 🔧 传入额外属性
        )
        
        self.symbol_table[qualified_name] = symbol
        return qualified_name

    def _extract_base_classes(self, bases) -> List[str]:
        """提取类的基类名称"""
        base_classes = []
        for base in bases:
            base_name = self._get_annotation_string(base.value)
            if base_name:
                base_classes.append(base_name)
        return base_classes


    def _add_symbol_reference(self, ref_name: str, from_scope: str) -> Optional[str]:
        """添加符号引用关系"""
        # 尝试 1: 在当前作用域链中查找
        for scope in reversed(self._scope_stack):
            candidate = f"{scope['id']}.{ref_name}"
            if candidate in self.symbol_table:
                self.symbol_table[candidate].references.append(from_scope)
                return candidate
        
        # 尝试 2: 在全局符号中查找
        if ref_name in self.symbol_table:
            self.symbol_table[ref_name].references.append(from_scope)
            return ref_name
        
        # 尝试 3: 模糊匹配
        for qname, symbol in self.symbol_table.items():
            if symbol.name == ref_name:
                symbol.references.append(from_scope)
                return qname
        
        return None

    def _resolve_super_call(self, method_name: str, caller_id: str) -> Optional[str]:
        """
        解析 super() 调用到父类方法
        
        Args:
            method_name: 方法名（如 "__init__"）
            caller_id: 调用者 ID（如 "module.Child.my_method"）
        
        Returns:
            父类方法的 node_id，如果无法解析则返回 None
        """
        # 从 caller_id 提取类名
        parts = caller_id.split(".")
        if len(parts) < 2:
            return None
        
        class_name = parts[-2]  # Child
        current_class_symbol = None
        
        # 查找当前类的符号信息
        for qname, symbol in self.symbol_table.items():
            if symbol.node_type == self.TYPE_CLASS and symbol.name == class_name:
                current_class_symbol = symbol
                break
        
        if not current_class_symbol:
            return None
        
        # 获取基类列表
        base_classes = current_class_symbol.extra_properties.get("base_classes", [])
        
        # 在基类中查找方法
        for base_name in base_classes:
            # 尝试 1: 直接匹配基类符号
            if base_name in self.symbol_table:
                base_symbol = self.symbol_table[base_name]
                # 在基类中查找方法
                method_qname = f"{base_name}.{method_name}"
                if method_qname in self.symbol_table:
                    return self.symbol_table[method_qname].node_id
            
            # 尝试 2: 模糊匹配
            for qname, symbol in self.symbol_table.items():
                if symbol.node_type in (self.TYPE_FUNCTION, self.TYPE_METHOD):
                    if symbol.name == method_name and qname.startswith(f"{base_name}."):
                        return symbol.node_id
        
        return None

    def _resolve_by_method_name(self, method_name: str, caller_id: str) -> Optional[str]:
        """
        按方法名解析（避免自调用）
        
        关键：优先查找父类/其他类的方法，而不是当前作用域
        """
        # 获取当前类名（如果调用者是方法）
        parts = caller_id.split(".")
        current_class = parts[-2] if len(parts) >= 2 else None
        
        # 🔧 优先查找其他类的方法（避免找到当前类的方法导致递归）
        for qname, symbol in self.symbol_table.items():
            if symbol.name == method_name and symbol.node_type in (
                self.TYPE_FUNCTION, self.TYPE_METHOD, self.TYPE_ASYNC_FUNCTION
            ):
                # 跳过当前类的方法（避免递归）
                if current_class and qname.startswith(f"{current_class}."):
                    continue
                return symbol.node_id
        
        # 如果没找到其他类的，再查找当前类（可能是静态方法等）
        if current_class:
            method_qname = f"{current_class}.{method_name}"
            if method_qname in self.symbol_table:
                return self.symbol_table[method_qname].node_id
        
        return None

    def _is_cross_file(self, callee_id: str) -> bool:
        """判断调用是否跨文件"""
        if callee_id in self.symbol_table:
            return self.symbol_table[callee_id].file_path != self.file_path
        
        # 通过 module 前缀判断
        if not callee_id.startswith(self.module_name):
            return True
        
        return False


    # ==================== 工具方法 ====================
    
    def _get_line_number(self, node: cst.CSTNode) -> int:
        """获取节点的起始行号"""
        try:
            position = self._metadata.get(node)
            if position:
                return position.start.line
        except Exception:
            pass
        return -1
    
    def _is_not_sentinel(self, value) -> bool:
        return value is not MaybeSentinel.DEFAULT and value is not None

    def _infer_module_name(self, file_path: str) -> str:
        """从文件路径推断模块名"""
        path = Path(file_path)
        parts = []
        for part in path.parts:
            if part.endswith('.py'):
                parts.append(part[:-3])
            elif part != '__init__':
                parts.append(part)
        return '.'.join(parts)
    
    def _create_node_id(self, name: str, scope_type: str = "") -> str:
        """创建唯一的节点 ID"""
        if scope_type and self._scope_stack:
            parent_id = self._scope_stack[-1].get('id', '')
            if parent_id:
                return f"{parent_id}.{name}"
        return f"{self.module_name}.{name}"
    
    def _create_node(
        self,
        name: str,
        node_type: str,
        extra_properties: Optional[Dict[str, Any]] = None,
    ) -> CodeEntityNode:
        """创建实体节点"""
        node_id = self._create_node_id(name, node_type)
        
        properties = {
            "qualified_name": node_id,
            "module": self.module_name,
            "file_path": self.file_path,
            "node_type": node_type,
        }
        
        if self._scope_stack:
            properties["scope"] = self._scope_stack[-1].get('id', '')
        
        if extra_properties:
            properties.update(extra_properties)
        
        return CodeEntityNode.create(
            name=name,
            node_type=node_type,
            qualified_name=node_id,
            module=self.module_name,
            file_path=self.file_path,
            scope=self._scope_stack[-1].get('id') if self._scope_stack else None,
            extra_properties=extra_properties,
        )
    
    def _add_relation(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加关系（带去重）"""
        rel_key = (source_id, target_id, label)
        if rel_key in self._seen_relations:
            return False
        
        self._seen_relations.add(rel_key)
        
        rel = CodeRelation(
            source_id=source_id,
            target_id=target_id,
            label=label,
            properties=properties or {}
        )
        self.relations.append(rel)
        return True
    
    def _get_annotation_string(self, node) -> str:
        """将注解节点转换为字符串"""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, cst.Subscript):
            base = self._get_annotation_string(node.value)
            params = []
            for item in node.slice:
                if isinstance(item, cst.SubscriptElement):
                    params.append(self._get_annotation_string(item.slice.value))
            return f"{base}[{', '.join(params)}]"
        return ""

    def _extract_type_annotation(self, annotation) -> Optional[str]:
        """提取类型注解"""
        if annotation is None or annotation is MaybeSentinel.DEFAULT:
            return None
        if isinstance(annotation, cst.Annotation):
            return self._get_annotation_string(annotation.annotation)
        return None
    
    def _extract_decorators(self, decorators) -> List[str]:
        """提取装饰器列表"""
        result = []
        for dec in decorators:
            if isinstance(dec, cst.Decorator):
                dec_name = self._get_decorator_name(dec.decorator)
                if dec_name:
                    result.append(dec_name)
        return result
    
    def _get_decorator_name(self, decorator) -> str:
        """获取装饰器名称"""
        if isinstance(decorator, cst.Name):
            return decorator.value
        elif isinstance(decorator, cst.Attribute):
            return self._get_attribute_name(decorator)
        elif isinstance(decorator, cst.Call):
            return self._get_decorator_name(decorator.func)
        return ""
    
    def _get_attribute_name(self, attr: cst.Attribute) -> str:
        """获取属性访问的完整名称"""
        parts = []
        current = attr
        while isinstance(current, cst.Attribute):
            parts.append(current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.append(current.value)
        return '.'.join(reversed(parts))

    def _resolve_name(self, name: str) -> str:
        """解析名称到节点 ID"""
        if name in self._import_map:
            return self._import_map[name]
        if name in self._local_vars:
            return self._local_vars[name]
        return self._create_node_id(name)
    
    # ==================== 作用域管理 ====================
    
    def _push_scope(self, scope_type: str, name: str, node_id: str) -> None:
        """压入新的作用域"""
        self._scope_stack.append({
            'type': scope_type,
            'name': name,
            'id': node_id,
        })
    
    def _pop_scope(self) -> None:
        """弹出当前作用域"""
        if self._scope_stack:
            self._scope_stack.pop()
    
    # ==================== 访问器方法 ====================
    
    def visit_Module(self, node: cst.Module) -> None:
        """处理模块节点"""
        module_node = self._create_node(
            name=self.module_name.split('.')[-1],
            node_type=self.TYPE_MODULE,
            extra_properties={
                "is_package": self.module_name.endswith('__init__'),
            }
        )
        self.nodes.append(module_node)
        
        self._register_symbol(
            name=self.module_name.split('.')[-1],
            node_type=self.TYPE_MODULE,
            node=node,
            node_id=module_node.id,
        )
        
        self._push_scope(self.TYPE_MODULE, module_node.name, module_node.id)
    
    def leave_Module(self, original_node: cst.Module) -> None:
        self._pop_scope()
    

    def _resolve_base_class_id(self, base_name: str, child_class_id: str) -> str:
        """
        解析基类名称到节点 ID
        
        Args:
            base_name: 基类名称（如 "ParentClass" 或 "module.ParentClass"）
            child_class_id: 子类 ID（用于上下文解析）
        
        Returns:
            基类的节点 ID
        """
        # 尝试 1: 直接匹配符号表
        if base_name in self.symbol_table:
            return self.symbol_table[base_name].node_id
        
        # 尝试 2: 在当前模块中查找
        candidate = f"{self.module_name}.{base_name}"
        if candidate in self.symbol_table:
            return self.symbol_table[candidate].node_id
        
        # 尝试 3: 从子类作用域推断
        parts = child_class_id.split(".")
        if len(parts) >= 2:
            module_prefix = ".".join(parts[:-2])  # 去掉类名和方法名
            candidate = f"{module_prefix}.{base_name}"
            if candidate in self.symbol_table:
                return self.symbol_table[candidate].node_id
        
        # 尝试 4: 使用导入映射
        if base_name in self._import_map:
            return self._import_map[base_name]
        
        # 回退：返回原始名称作为 ID
        return base_name

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """处理类定义"""
        class_name = node.name.value
        
        if self._scope_stack:
            qualified_name = f"{self._scope_stack[-1]['id']}.{class_name}"
        else:
            qualified_name = f"{self.module_name}.{class_name}"
        
        # 🔧 提取基类信息
        base_classes = self._extract_base_classes(node.bases)
        
        class_node = CodeEntityNode.create(
            name=class_name,
            node_type=self.TYPE_CLASS,
            qualified_name=qualified_name,
            extra_properties={
                "decorators": self._extract_decorators(node.decorators),
                "line_number": self._get_line_number(node),
                "base_classes": base_classes,
            }
        )
        self.nodes.append(class_node)
        
        self._register_symbol(
            name=class_name,
            node_type=self.TYPE_CLASS,
            node=node,
            node_id=qualified_name,
            extra_properties={"base_classes": base_classes},
        )
        
        # 🔧 创建 CONTAINS 关系（父作用域 CONTAINS 当前类）
        if self._scope_stack:
            parent_id = self._scope_stack[-1]['id']
            self._add_relation(parent_id, qualified_name, self.REL_CONTAINS)
        
        # 🔧🔧🔧 关键修复：创建 INHERITS 关系（子类 INHERITS 父类）
        for base_name in base_classes:
            # 跳过 object 等内置基类
            if base_name == "object":
                continue
            
            # 🔧 解析基类名称到节点 ID
            base_id = self._resolve_base_class_id(base_name, qualified_name)
            
            # 创建 INHERITS 关系
            self._add_relation(
                qualified_name,  # source: 子类
                base_id,         # target: 父类
                self.REL_INHERITS,  # ← 使用 INHERITS，不是 CALLS
                properties={"inheritance_type": "class"}
            )
        
        self._scope_stack.append({
            'id': qualified_name,
            'type': self.TYPE_CLASS,
            'name': class_name,
        })
        
        # 🔧 更新当前类 ID（用于判断方法）
        self._current_class_id = qualified_name




    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """离开类定义"""
        self._pop_scope()
        # 🔧 恢复父作用域的类 ID（处理嵌套类）
        if self._scope_stack:
            for scope in reversed(self._scope_stack):
                if scope['type'] == self.TYPE_CLASS:
                    self._current_class_id = scope['id']
                    break
            else:
                self._current_class_id = None
        else:
            self._current_class_id = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """处理函数定义"""
        is_async = self._is_not_sentinel(node.asynchronous)
        func_name = node.name.value
        is_method = self._current_class_id is not None
        
        # 🔧 正确设置节点类型
        if is_method:
            node_type = self.TYPE_ASYNC_METHOD if is_async else self.TYPE_METHOD
        else:
            node_type = self.TYPE_ASYNC_FUNCTION if is_async else self.TYPE_FUNCTION
        
        # 构建 qualified_name
        if self._scope_stack:
            qualified_name = f"{self._scope_stack[-1]['id']}.{func_name}"
        else:
            qualified_name = f"{self.module_name}.{func_name}"
        
        func_node = CodeEntityNode.create(
            name=func_name,
            node_type=node_type,  # ✅ 实体类型
            qualified_name=qualified_name,
            extra_properties={
                "is_entity": True,  # ✅ 明确标记
                "is_async": is_async,
                "is_method": is_method,
                "decorators": self._extract_decorators(node.decorators),
                "parameters": self._extract_parameters(node.params),
                "return_type": self._extract_type_annotation(node.returns),
                "line_number": self._get_line_number(node),
            }
        )
        self.nodes.append(func_node)
        
        # 注册到符号表
        self._register_symbol(
            name=func_name,
            node_type=node_type,
            node=node,
            node_id=qualified_name,
        )
        
        # 创建 CONTAINS 关系
        if self._scope_stack:
            parent_id = self._scope_stack[-1]['id']
            self._add_relation(parent_id, qualified_name, self.REL_CONTAINS)
        
        # 更新作用域栈
        self._scope_stack.append({
            'id': qualified_name,
            'type': node_type,
            'name': func_name,
        })
        
        self._current_function_id = qualified_name


    
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """离开函数定义"""
        self._pop_scope()
        # 🔧 恢复父作用域的函数 ID（处理嵌套函数）
        if self._scope_stack:
            for scope in reversed(self._scope_stack):
                if scope['type'] in (self.TYPE_FUNCTION, self.TYPE_METHOD, self.TYPE_ASYNC_FUNCTION):
                    self._current_function_id = scope['id']
                    break
            else:
                self._current_function_id = None
        else:
            self._current_function_id = None

    
    
    def visit_Param(self, node: cst.Param) -> None:
        """处理函数参数"""
        param_name = node.name.value
        
        # 🔧 先创建节点，使用节点的真实 ID
        param_node = self._create_node(
            name=param_name,
            node_type=self.TYPE_PARAMETER,
            extra_properties={
                "has_default": node.default is not None,
                "is_star": node.star in ('*', '**'),
            }
        )
        self.nodes.append(param_node)
        
        # 🔧 使用节点的真实 ID
        param_id = param_node.id
        
        # 🔧 添加 CONTAINS 关系（函数 CONTAINS 参数）
        if self._current_function_id:
            self._add_relation(self._current_function_id, param_id, self.REL_CONTAINS)
            # 原有的 HAS_PARAM 关系保留
            self._add_relation(self._current_function_id, param_id, "HAS_PARAM")
        
        self._local_vars[param_name] = param_id

    def _get_name_string(self, node: Union[cst.Name, cst.Attribute, None]) -> str:
        """🔧 从 libcst Name 或 Attribute 节点提取字符串名称"""
        if node is None:
            return ""
        
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            # 递归处理嵌套属性 (如 a.b.c)
            value = self._get_name_string(node.value)
            return f"{value}.{node.attr.value}"
        else:
            return str(node)

    def _ensure_module_node(self, module_name: str) -> str:
        """确保模块节点存在，返回节点 ID"""
        # 检查是否已存在
        for node in self.nodes:
            if node.properties.get("module_name") == module_name:
                return node.id
        
        # 创建新节点
        module_node = CodeEntityNode.create(
            name=module_name.split('.')[-1],
            node_type=self.TYPE_MODULE_IMPORT,
            qualified_name=module_name,
            extra_properties={
                "module_name": module_name,
                "is_external": not module_name.startswith(self.module_name.split('.')[0]),
            }
        )
        self.nodes.append(module_node)
        
        # 注册到符号表
        self._register_symbol(
            name=module_name.split('.')[-1],
            node_type=self.TYPE_MODULE_IMPORT,
            node=None,  # 外部模块没有 CST 节点
            node_id=module_node.id,
            extra_properties={"module_name": module_name},
        )
        
        return module_node.id



    def visit_Import(self, node: cst.Import) -> None:
        """处理 import 语句 - 创建导入关系"""
        for alias in node.names:
            if isinstance(alias, cst.ImportStar):
                continue
            
            module_name = self._get_name_string(alias.name)
            
            if alias.asname:
                alias_name = alias.asname.name.value
            else:
                alias_name = module_name.split('.')[-1]
            
            self._import_map[alias_name] = module_name
            
            # ✅ 创建模块节点（如果不存在）
            module_node_id = self._ensure_module_node(module_name)
            
            # ✅ 使用标准 _add_relation 方法
            self._add_relation(
                source_id=self.module_name,      # 当前模块 ID
                target_id=module_node_id,         # 被导入模块 ID
                label=self.REL_IMPORTS,
                properties={
                    "alias": alias_name,
                    "is_star": False,
                }
            )


    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """处理 from ... import 语句"""
        module_name = self._get_name_string(node.module) if node.module else ""
        
        if isinstance(node.names, cst.ImportStar):
            self._import_map["*"] = module_name
            module_node_id = self._ensure_module_node(module_name)
            
            self._add_relation(
                source_id=self.module_name,
                target_id=module_node_id,
                label=self.REL_IMPORTS,
                properties={
                    "alias": "*",
                    "is_star": True,
                }
            )
            return
        
        for alias in node.names:
            imported_name = alias.name.value
            
            if alias.asname:
                alias_name = alias.asname.name.value
            else:
                alias_name = imported_name
            
            full_name = f"{module_name}.{alias_name}" if module_name else alias_name
            self._import_map[alias_name] = full_name
            
            module_node_id = self._ensure_module_node(module_name)
            
            self._add_relation(
                source_id=self.module_name,
                target_id=module_node_id,
                label=self.REL_IMPORTS,
                properties={
                    "alias": alias_name,
                    "imported_name": imported_name,
                    "is_star": False,
                }
            )

    def visit_Assign(self, node: cst.Assign) -> None:
        """处理赋值语句"""
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                var_name = target.target.value
                
                # 🔧 先创建节点，使用节点的真实 ID
                var_node = self._create_node(
                    name=var_name,
                    node_type=self.TYPE_VARIABLE,
                    extra_properties={
                        "value_hint": self._get_value_hint(node.value),
                    }
                )
                self.nodes.append(var_node)
                
                # 🔧 使用节点的真实 ID
                var_id = var_node.id
                
                self._local_vars[var_name] = var_id
                
                # 🔧 添加 CONTAINS 关系（父作用域 CONTAINS 变量）
                if self._scope_stack:
                    parent_id = self._scope_stack[-1]['id']
                    self._add_relation(parent_id, var_id, self.REL_CONTAINS)
                
                # 原有的 ASSIGNS 关系
                if self._current_function_id:
                    self._add_relation(self._current_function_id, var_id, self.REL_ASSIGNS)
                elif self._current_class_id:
                    self._add_relation(self._current_class_id, var_id, self.REL_ASSIGNS)


    
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        """处理带类型注解的赋值"""
        if isinstance(node.target, cst.Name):
            var_name = node.target.value
            type_annotation = self._extract_type_annotation(node.annotation)
            
            # 🔧 先创建节点，使用节点的真实 ID
            var_node = self._create_node(
                name=var_name,
                node_type=self.TYPE_VARIABLE,
                extra_properties={
                    "type": type_annotation,
                    "value_hint": self._get_value_hint(node.value) if node.value else None,
                }
            )
            self.nodes.append(var_node)
            
            # 🔧 使用节点的真实 ID
            var_id = var_node.id
            
            self._local_vars[var_name] = var_id
            
            # 🔧 添加 CONTAINS 关系（父作用域 CONTAINS 变量）
            if self._scope_stack:
                parent_id = self._scope_stack[-1]['id']
                self._add_relation(parent_id, var_id, self.REL_CONTAINS)
            
            # 类型关系
            if type_annotation:
                type_id = self._resolve_name(type_annotation)
                self._add_relation(var_id, type_id, "HAS_TYPE")
            
            # 原有的 ASSIGNS 关系
            if self._current_function_id:
                self._add_relation(self._current_function_id, var_id, self.REL_ASSIGNS)


    
    def visit_Call(self, node: cst.Call) -> None:
        """处理函数调用"""
        func_name = self._get_call_name(node.func)
        if not func_name:
            return
        
        caller_id = self._current_function_id or self._current_class_id
        if not caller_id:
            return
        is_resolved = False
        is_cross_file = False
        callee_id = None
        # 🔧 关键：super() 调用应该创建 CALLS 关系，不是 INHERITS
        if func_name.startswith("super."):
            method_name = func_name.split(".", 1)[1]
            callee_id = self._resolve_super_call(method_name, caller_id)
            if callee_id:
                is_cross_file = self._is_cross_file(callee_id)
                
                # ✅ super() 调用是 CALLS 关系（调用父类方法）
                self._add_relation(caller_id, callee_id, self.REL_CALLS, {
                    "call_site": self.file_path,
                    "is_cross_file": is_cross_file,
                    "is_super_call": True,  # ← 标记为 super 调用
                })
                return
        # ==================== 尝试 2: 限定名解析 (ParentClass.method) ====================
        elif "." in func_name:
            # 直接匹配符号表
            if func_name in self.symbol_table:
                callee_id = self.symbol_table[func_name].node_id
                is_resolved = True
                is_cross_file = self._is_cross_file(callee_id)
            else:
                # 尝试匹配末尾部分
                method_name = func_name.split(".")[-1]
                callee_id = self._resolve_by_method_name(method_name, caller_id)
                if callee_id:
                    is_resolved = True
                    is_cross_file = self._is_cross_file(callee_id)
        
        # ==================== 尝试 3: 简单名称解析 ====================
        else:
            # 导入映射
            if func_name in self._import_map:
                callee_id = self._import_map[func_name]
                is_resolved = True
            # 局部变量
            elif func_name in self._local_vars:
                callee_id = self._local_vars[func_name]
                is_resolved = True
            # 符号表精确匹配
            elif func_name in self.symbol_table:
                callee_id = self.symbol_table[func_name].node_id
                is_resolved = True
                is_cross_file = self._is_cross_file(callee_id)
            # 符号表模糊匹配
            else:
                callee_id = self._resolve_by_method_name(func_name, caller_id)
                if callee_id:
                    is_resolved = True
                    is_cross_file = self._is_cross_file(callee_id)
        
        # ==================== 创建关系 ====================
        if is_resolved and callee_id:
            # 🔧 关键：避免自调用（递归）
            if callee_id == caller_id:
                # 记录为未解析，等待跨文件解析
                self.unresolved_calls.append({
                    'caller_id': caller_id,
                    'func_name': func_name,
                    'file_path': self.file_path,
                    'line_number': self._get_line_number(node),
                    'is_super_call': func_name.startswith("super."),
                })
            else:
                self._add_relation(caller_id, callee_id, self.REL_CALLS, {
                    "call_site": self.file_path,
                    "is_cross_file": is_cross_file,
                    "is_super_call": func_name.startswith("super."),
                })
                
                # 更新符号表引用
                for qname, symbol in self.symbol_table.items():
                    if symbol.node_id == callee_id:
                        symbol.references.append(caller_id)
                        break
        else:
            # 记录为未解析调用
            self.unresolved_calls.append({
                'caller_id': caller_id,
                'func_name': func_name,
                'file_path': self.file_path,
                'line_number': self._get_line_number(node),
                'is_super_call': func_name.startswith("super."),
            })


    def visit_Attribute(self, node: cst.Attribute) -> None:
        """处理属性访问"""
        if self._current_function_id:
            attr_name = node.attr.value
            if isinstance(node.value, cst.Name):
                obj_name = node.value.value
                obj_id = self._resolve_name(obj_name)
                self._add_relation(
                    self._current_function_id,
                    obj_id,
                    self.REL_USES,
                    {"attribute": attr_name}
                )
    
    # ==================== 辅助方法 ====================
    
    def _extract_parameters(self, params: cst.Parameters) -> List[Dict[str, Any]]:
        """提取函数参数信息"""
        result = []
        
        # 1. 普通位置参数
        for param in params.params:
            result.append({
                "name": param.name.value,
                "type": self._extract_type_annotation(param.annotation),
                "has_default": param.default is not None,
            })
        
        # 2. 🔧 修复：正确处理 *args (ParamStar 情况)
        if params.star_arg is not MaybeSentinel.DEFAULT and params.star_arg is not None:
            # 🔧 检查是否为 ParamStar (单独的 *)
            if isinstance(params.star_arg, cst.ParamStar):
                # 单独的 *，没有名字，跳过或标记
                result.append({
                    "name": "*",
                    "type": None,
                    "is_vararg": True,
                    "is_star_only": True,  # 标记这是单独的 *
                })
            elif isinstance(params.star_arg, cst.Param):
                # *args 形式
                result.append({
                    "name": params.star_arg.name.value,
                    "type": self._extract_type_annotation(params.star_arg.annotation),
                    "is_vararg": True,
                })
        
        # 3. 关键字参数
        for param in params.kwonly_params:
            result.append({
                "name": param.name.value,
                "type": self._extract_type_annotation(param.annotation),
                "has_default": param.default is not None,
                "is_kwonly": True,
            })
        
        # 4. 🔧 修复：正确处理 **kwargs
        if params.star_kwarg is not MaybeSentinel.DEFAULT and params.star_kwarg is not None:
            if isinstance(params.star_kwarg, cst.Param):
                result.append({
                    "name": params.star_kwarg.name.value,
                    "type": self._extract_type_annotation(params.star_kwarg.annotation),
                    "is_kwarg": True,
                })
        
        # 5. 仅位置参数 (Python 3.8+)
        for param in params.posonly_params:
            result.append({
                "name": param.name.value,
                "type": self._extract_type_annotation(param.annotation),
                "has_default": param.default is not None,
                "is_posonly": True,
            })
        
        return result


    def _get_call_name(self, func: cst.CSTNode) -> Optional[str]:
        """获取被调用函数的完整名称（包含对象信息）"""
        if isinstance(func, cst.Name):
            return func.value
        
        elif isinstance(func, cst.Attribute):
            # 🔧 保留完整的对象.方法名
            attr_name = func.attr.value
            
            # 处理 super().__init__()
            if isinstance(func.value, cst.Call):
                call_func = func.value.func
                if isinstance(call_func, cst.Name) and call_func.value == "super":
                    # 🔧 标记为 super 调用，后续解析时查找父类
                    return f"super.{attr_name}"
            
            # 处理 ParentClass.__init__()
            elif isinstance(func.value, cst.Name):
                obj_name = func.value.value
                return f"{obj_name}.{attr_name}"
            
            # 处理 self.method()
            elif isinstance(func.value, cst.Attribute):
                obj_name = self._get_attribute_name(func.value)
                return f"{obj_name}.{attr_name}"
            
            # 默认只返回方法名
            return attr_name
        
        elif isinstance(func, cst.Call):
            return self._get_call_name(func.func)
        
        return None

    
    def _get_value_hint(self, value: Optional[cst.CSTNode]) -> Optional[str]:
        """获取赋值值的提示"""
        if value is None:
            return None
        if isinstance(value, cst.SimpleString):
            return f"str: {value.value[:50]}"
        elif isinstance(value, cst.Integer):
            return f"int: {value.value}"
        elif isinstance(value, cst.Float):
            return f"float: {value.value}"
        elif isinstance(value, cst.Name):
            return f"ref: {value.value}"
        elif isinstance(value, cst.Call):
            return self._get_call_name(value.func)
        return value.code[:50] if hasattr(value, 'code') else None
    
    # ==================== 主入口 ====================
    
    def extract(self, source_code: str) -> ExtractionResult:
        """提取代码符号"""
        self.module = cst.parse_module(source_code)
        
        wrapper = MetadataWrapper(self.module, unsafe_skip_copy=True)
        wrapper.visit(self)
        
        # 序列化
        for node in self.nodes:
            if hasattr(node, 'properties') and node.properties:
                node.properties = {
                    key: _serialize_property(value) 
                    for key, value in node.properties.items()
                }
        
        for rel in self.relations:
            if hasattr(rel, 'properties') and rel.properties:
                rel.properties = {
                    key: _serialize_property(value) 
                    for key, value in rel.properties.items()
                }
        
        return ExtractionResult(
            nodes=self.nodes,
            relations=self.relations,
            errors=self.errors,
            symbol_table=self.symbol_table,
            file_path=self.file_path,
            unresolved_calls=self.unresolved_calls,  # 🔧 返回未解析调用
        )



def extract_file(file_path: str, module_name: Optional[str] = None) -> ExtractionResult:
    """提取单个文件的符号"""
    path = Path(file_path)
    if not path.exists():
        return ExtractionResult(errors=[f"文件不存在：{file_path}"])
    
    source_code = path.read_text(encoding='utf-8')
    
    extractor = CodeSymbolExtractor(
        file_path=str(path.absolute()),
        module_name=module_name,
    )
    
    return extractor.extract(source_code)


def extract_directory(
    dir_path: str,
    pattern: str = "**/*.py",
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[ExtractionResult], Dict[str, SymbolInfo], List[CodeRelation]]:  # 🔧 返回类型修改
    """
    提取目录下所有 Python 文件的符号
    
    Returns:
        Tuple: (单个文件结果列表，合并后的全局符号表，跨文件关系列表)
    """
    from fnmatch import fnmatch
    
    path = Path(dir_path)
    if not path.is_dir():
        raise ValueError(f"不是目录：{dir_path}")
    
    exclude_patterns = exclude_patterns or [
        "**/__pycache__/**",
        "**/.git/**",
        "**/venv/**",
        "**/.venv/**",
        "**/node_modules/**",
        "**/*.pyc",
    ]
    
    results = []
    py_files = list(path.glob(pattern))
    
    for py_file in py_files:
        should_exclude = False
        for exclude in exclude_patterns:
            if fnmatch(str(py_file), exclude):
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        result = extract_file(str(py_file))
        results.append(result)
    
    # 合并符号表
    global_symbol_table = ExtractionResult.merge_symbol_tables(results)
    
    # 🔧 解析跨文件引用并创建关系
    cross_file_relations = ExtractionResult.resolve_and_create_cross_file_relations(
        results, 
        global_symbol_table
    )
    
    # 🔧 将跨文件关系添加到对应结果的 relations 中
    for rel in cross_file_relations:
        for result in results:
            if rel.properties.get('call_site') == result.file_path:
                result.relations.append(rel)
                break
    
    return results, global_symbol_table, cross_file_relations  # 🔧 返回跨文件关系