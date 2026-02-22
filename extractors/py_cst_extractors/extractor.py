"""
代码符号提取器 - 主提取器模块
"""
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from fnmatch import fnmatch

import libcst as cst
from libcst import MaybeSentinel
from libcst.metadata import (
    MetadataWrapper,
    PositionProvider,
    ScopeProvider,
)

from py_relations import CodeEntityNode
from cst_types import TypeManager

from utils import (
    serialize_property,
    infer_module_name,
    get_annotation_string,
    get_attribute_name,
    get_name_string,
    extract_decorators,
    extract_type_annotation,
    get_value_hint
)
from node_parsers import (
    extract_base_classes,
    extract_parameters,
    get_call_name,
    determine_node_type,
    is_not_sentinel,
)
from relation_parsers import (
    resolve_base_class_id,
    resolve_super_call,
    resolve_by_method_name,
    is_cross_file,
)
from symbol_info import ExtractionResult, SymbolInfo

class CodeSymbolExtractor(cst.CSTVisitor):
    """
    代码符号提取器 - 使用 libcst 解析 Python 代码并构建知识图谱
    """
    
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, file_path: str, module_name: Optional[str] = None):
        """
        初始化提取器
        
        Args:
            file_path: 文件路径
            module_name: 模块名（可选）
        """
        self.file_path = file_path
        self.module_name = module_name or infer_module_name(file_path)
        
        self.nodes: List[CodeEntityNode] = []
        self.relations: List = []
        
        self._scope_stack: List[Dict[str, Any]] = []
        self._seen_relations: Set[Tuple[str, str, str]] = set()
        self._import_map: Dict[str, str] = {}
        self._local_vars: Dict[str, str] = {}
        self._current_class_id: Optional[str] = None
        self._current_function_id: Optional[str] = None
        self._metadata = {}
        self.errors: List[str] = []
        
        self.symbol_table: Dict[str, SymbolInfo] = {}
        self._global_symbols: Set[str] = set()
        self.unresolved_calls: List[Dict[str, str]] = []

    def _get_line_number(self, node: cst.CSTNode) -> int:
        """
        获取节点的起始行号
        
        Args:
            node: libcst 节点
            
        Returns:
            行号
        """
        try:
            position = self._metadata.get(node)
            if position:
                return position.start.line
        except Exception:
            pass
        return -1

    def _create_node_id(self, name: str, scope_type: str = "") -> str:
        """
        创建唯一的节点 ID
        
        Args:
            name: 节点名称
            scope_type: 作用域类型
            
        Returns:
            节点 ID
        """
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
        """
        创建实体节点
        
        Args:
            name: 节点名称
            node_type: 节点类型
            extra_properties: 额外属性
            
        Returns:
            CodeEntityNode 实例
        """
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
        """
        添加关系（带去重）
        
        Args:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            label: 关系标签
            properties: 关系属性
            
        Returns:
            是否成功添加
        """
        rel_key = (source_id, target_id, label)
        if rel_key in self._seen_relations:
            return False
        
        self._seen_relations.add(rel_key)
        
        from py_relations import CodeRelation
        rel = CodeRelation(
            source_id=source_id,
            target_id=target_id,
            label=label,
            properties=properties or {}
        )
        self.relations.append(rel)
        return True

    def _register_symbol(
        self, 
        name: str, 
        node_type: str, 
        node: cst.CSTNode, 
        node_id: str, 
        extra_properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        注册符号到全局符号表
        
        Args:
            name: 符号名
            node_type: 节点类型
            node: libcst 节点
            node_id: 节点 ID
            extra_properties: 额外属性
            
        Returns:
            限定名
        """
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
            extra_properties=extra_properties or {},
        )
        
        self.symbol_table[qualified_name] = symbol
        return qualified_name

    def _push_scope(self, scope_type: str, name: str, node_id: str) -> None:
        """
        压入新的作用域
        
        Args:
            scope_type: 作用域类型
            name: 作用域名
            node_id: 节点 ID
        """
        self._scope_stack.append({
            'type': scope_type,
            'name': name,
            'id': node_id,
        })
    
    def _pop_scope(self) -> None:
        """弹出当前作用域"""
        if self._scope_stack:
            self._scope_stack.pop()

    def _resolve_name(self, name: str) -> str:
        """
        解析名称到节点 ID
        
        Args:
            name: 名称
            
        Returns:
            节点 ID
        """
        if name in self._import_map:
            return self._import_map[name]
        if name in self._local_vars:
            return self._local_vars[name]
        return self._create_node_id(name)

    def _ensure_module_node(self, module_name: str) -> str:
        """
        确保模块节点存在
        
        Args:
            module_name: 模块名
            
        Returns:
            节点 ID
        """
        for node in self.nodes:
            if node.properties.get("module_name") == module_name:
                return node.id
        
        module_node = CodeEntityNode.create(
            name=module_name.split('.')[-1],
            node_type=TypeManager.ENTITY_MODULE_IMPORT,
            qualified_name=module_name,
            extra_properties={
                "module_name": module_name,
                "is_external": not module_name.startswith(self.module_name.split('.')[0]),
            }
        )
        self.nodes.append(module_node)
        
        self._register_symbol(
            name=module_name.split('.')[-1],
            node_type=TypeManager.ENTITY_MODULE_IMPORT,
            node=None,
            node_id=module_node.id,
            extra_properties={"module_name": module_name},
        )
        
        return module_node.id

    def visit_Module(self, node: cst.Module) -> None:
        """
        处理模块节点
        
        Args:
            node: 模块节点
        """
        module_node = self._create_node(
            name=self.module_name.split('.')[-1],
            node_type=TypeManager.ENTITY_MODULE,
            extra_properties={
                "is_package": self.module_name.endswith('__init__'),
            }
        )
        self.nodes.append(module_node)
        
        self._register_symbol(
            name=self.module_name.split('.')[-1],
            node_type=TypeManager.ENTITY_MODULE,
            node=node,
            node_id=module_node.id,
        )
        
        self._push_scope(TypeManager.ENTITY_MODULE, module_node.name, module_node.id)
    
    def leave_Module(self, original_node: cst.Module) -> None:
        """离开模块节点"""
        self._pop_scope()

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """
        处理类定义
        
        Args:
            node: 类定义节点
        """
        class_name = node.name.value
        
        if self._scope_stack:
            qualified_name = f"{self._scope_stack[-1]['id']}.{class_name}"
        else:
            qualified_name = f"{self.module_name}.{class_name}"
        
        base_classes = extract_base_classes(node.bases, get_annotation_string)
        
        class_node = CodeEntityNode.create(
            name=class_name,
            node_type=TypeManager.ENTITY_CLASS,
            qualified_name=qualified_name,
            extra_properties={
                "decorators": extract_decorators(node.decorators),
                "line_number": self._get_line_number(node),
                "base_classes": base_classes,
            }
        )
        self.nodes.append(class_node)
        
        self._register_symbol(
            name=class_name,
            node_type=TypeManager.ENTITY_CLASS,
            node=node,
            node_id=qualified_name,
            extra_properties={"base_classes": base_classes},
        )
        
        if self._scope_stack:
            parent_id = self._scope_stack[-1]['id']
            self._add_relation(parent_id, qualified_name, TypeManager.REL_CONTAINS)
        
        for base_name in base_classes:
            if base_name == "object":
                continue
            
            base_id = resolve_base_class_id(
                base_name, qualified_name, 
                self.symbol_table, self.module_name, self._import_map
            )
            
            self._add_relation(
                qualified_name,
                base_id,
                TypeManager.REL_INHERITS,
                properties={"inheritance_type": "class"}
            )
        
        self._scope_stack.append({
            'id': qualified_name,
            'type': TypeManager.ENTITY_CLASS,
            'name': class_name,
        })
        
        self._current_class_id = qualified_name

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """离开类定义"""
        self._pop_scope()
        if self._scope_stack:
            for scope in reversed(self._scope_stack):
                if scope['type'] == TypeManager.ENTITY_CLASS:
                    self._current_class_id = scope['id']
                    break
            else:
                self._current_class_id = None
        else:
            self._current_class_id = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """
        处理函数定义
        
        Args:
            node: 函数定义节点
        """
        is_async = is_not_sentinel(node.asynchronous)
        func_name = node.name.value
        is_method = self._current_class_id is not None
        
        node_type = determine_node_type(is_method, is_async)
        
        if self._scope_stack:
            qualified_name = f"{self._scope_stack[-1]['id']}.{func_name}"
        else:
            qualified_name = f"{self.module_name}.{func_name}"
        
        func_node = CodeEntityNode.create(
            name=func_name,
            node_type=node_type,
            qualified_name=qualified_name,
            extra_properties={
                "is_entity": True,
                "is_async": is_async,
                "is_method": is_method,
                "decorators": extract_decorators(node.decorators),
                "parameters": extract_parameters(node.params),
                "return_type": extract_type_annotation(node.returns),
                "line_number": self._get_line_number(node),
            }
        )
        self.nodes.append(func_node)
        
        self._register_symbol(
            name=func_name,
            node_type=node_type,
            node=node,
            node_id=qualified_name,
        )
        
        if self._scope_stack:
            parent_id = self._scope_stack[-1]['id']
            self._add_relation(parent_id, qualified_name, TypeManager.REL_CONTAINS)
        
        self._scope_stack.append({
            'id': qualified_name,
            'type': node_type,
            'name': func_name,
        })
        
        self._current_function_id = qualified_name

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """离开函数定义"""
        self._pop_scope()
        if self._scope_stack:
            for scope in reversed(self._scope_stack):
                if scope['type'] in (TypeManager.ENTITY_FUNCTION, TypeManager.ENTITY_METHOD, TypeManager.ENTITY_ASYNC_FUNCTION):
                    self._current_function_id = scope['id']
                    break
            else:
                self._current_function_id = None
        else:
            self._current_function_id = None

    def visit_Param(self, node: cst.Param) -> None:
        """
        处理函数参数
        
        Args:
            node: 参数节点
        """
        param_name = node.name.value
        
        param_node = self._create_node(
            name=param_name,
            node_type=TypeManager.ENTITY_PARAMETER,
            extra_properties={
                "has_default": node.default is not None,
                "is_star": node.star in ('*', '**'),
            }
        )
        self.nodes.append(param_node)
        
        param_id = param_node.id
        
        if self._current_function_id:
            self._add_relation(self._current_function_id, param_id, TypeManager.REL_CONTAINS)
            self._add_relation(self._current_function_id, param_id, TypeManager.REL_HAS_PARAM)
        
        self._local_vars[param_name] = param_id

    def visit_Import(self, node: cst.Import) -> None:
        """
        处理 import 语句
        
        Args:
            node: 导入节点
        """
        for alias in node.names:
            if isinstance(alias, cst.ImportStar):
                continue
            
            module_name = get_name_string(alias.name)
            
            if alias.asname:
                alias_name = alias.asname.name.value
            else:
                alias_name = module_name.split('.')[-1]
            
            self._import_map[alias_name] = module_name
            
            module_node_id = self._ensure_module_node(module_name)
            
            self._add_relation(
                source_id=self.module_name,
                target_id=module_node_id,
                label=TypeManager.REL_IMPORTS,
                properties={
                    "alias": alias_name,
                    "is_star": False,
                }
            )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """
        处理 from ... import 语句
        
        Args:
            node: 从导入节点
        """
        module_name = get_name_string(node.module) if node.module else ""
        
        if isinstance(node.names, cst.ImportStar):
            self._import_map["*"] = module_name
            module_node_id = self._ensure_module_node(module_name)
            
            self._add_relation(
                source_id=self.module_name,
                target_id=module_node_id,
                label=TypeManager.REL_IMPORTS,
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
                label=TypeManager.REL_IMPORTS,
                properties={
                    "alias": alias_name,
                    "imported_name": imported_name,
                    "is_star": False,
                }
            )

    def visit_Assign(self, node: cst.Assign) -> None:
        """
        处理赋值语句
        
        Args:
            node: 赋值节点
        """
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                var_name = target.target.value
                
                var_node = self._create_node(
                    name=var_name,
                    node_type=TypeManager.ENTITY_VARIABLE,
                    extra_properties={
                        "value_hint": get_value_hint(node.value),
                    }
                )
                self.nodes.append(var_node)
                
                var_id = var_node.id
                
                self._local_vars[var_name] = var_id
                
                if self._scope_stack:
                    parent_id = self._scope_stack[-1]['id']
                    self._add_relation(parent_id, var_id, TypeManager.REL_CONTAINS)
                
                if self._current_function_id:
                    self._add_relation(self._current_function_id, var_id, TypeManager.REL_ASSIGNS)
                elif self._current_class_id:
                    self._add_relation(self._current_class_id, var_id, TypeManager.REL_ASSIGNS)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        """
        处理带类型注解的赋值
        
        Args:
            node: 注解赋值节点
        """
        if isinstance(node.target, cst.Name):
            var_name = node.target.value
            type_annotation = extract_type_annotation(node.annotation)
            
            var_node = self._create_node(
                name=var_name,
                node_type=TypeManager.ENTITY_VARIABLE,
                extra_properties={
                    "type": type_annotation,
                    "value_hint": get_value_hint(node.value) if node.value else None,
                }
            )
            self.nodes.append(var_node)
            
            var_id = var_node.id
            
            self._local_vars[var_name] = var_id
            
            if self._scope_stack:
                parent_id = self._scope_stack[-1]['id']
                self._add_relation(parent_id, var_id, TypeManager.REL_CONTAINS)
            
            if type_annotation:
                type_id = self._resolve_name(type_annotation)
                self._add_relation(var_id, type_id, TypeManager.REL_HAS_TYPE)
            
            if self._current_function_id:
                self._add_relation(self._current_function_id, var_id, TypeManager.REL_ASSIGNS)

    def visit_Call(self, node: cst.Call) -> None:
        """
        处理函数调用
        
        Args:
            node: 调用节点
        """
        func_name = get_call_name(node.func)
        if not func_name:
            return
        
        caller_id = self._current_function_id or self._current_class_id
        if not caller_id:
            return
        
        is_resolved = False
        is_cross_file_call = False
        callee_id = None
        
        if func_name.startswith("super."):
            method_name = func_name.split(".", 1)[1]
            callee_id = resolve_super_call(method_name, caller_id, self.symbol_table)
            if callee_id:
                is_cross_file_call = is_cross_file(callee_id, self.symbol_table, 
                                                   self.file_path, self.module_name)
                
                self._add_relation(caller_id, callee_id, TypeManager.REL_CALLS, {
                    "call_site": self.file_path,
                    "is_cross_file": is_cross_file_call,
                    "is_super_call": True,
                })
                return
                
        elif "." in func_name:
            if func_name in self.symbol_table:
                callee_id = self.symbol_table[func_name].node_id
                is_resolved = True
                is_cross_file_call = is_cross_file(callee_id, self.symbol_table,
                                                   self.file_path, self.module_name)
            else:
                method_name = func_name.split(".")[-1]
                callee_id = resolve_by_method_name(method_name, caller_id, self.symbol_table)
                if callee_id:
                    is_resolved = True
                    is_cross_file_call = is_cross_file(callee_id, self.symbol_table,
                                                       self.file_path, self.module_name)
        else:
            if func_name in self._import_map:
                callee_id = self._import_map[func_name]
                is_resolved = True
            elif func_name in self._local_vars:
                callee_id = self._local_vars[func_name]
                is_resolved = True
            elif func_name in self.symbol_table:
                callee_id = self.symbol_table[func_name].node_id
                is_resolved = True
                is_cross_file_call = is_cross_file(callee_id, self.symbol_table,
                                                   self.file_path, self.module_name)
            else:
                callee_id = resolve_by_method_name(func_name, caller_id, self.symbol_table)
                if callee_id:
                    is_resolved = True
                    is_cross_file_call = is_cross_file(callee_id, self.symbol_table,
                                                       self.file_path, self.module_name)
        
        if is_resolved and callee_id:
            if callee_id == caller_id:
                self.unresolved_calls.append({
                    'caller_id': caller_id,
                    'func_name': func_name,
                    'file_path': self.file_path,
                    'line_number': self._get_line_number(node),
                    'is_super_call': func_name.startswith("super."),
                })
            else:
                self._add_relation(caller_id, callee_id, TypeManager.REL_CALLS, {
                    "call_site": self.file_path,
                    "is_cross_file": is_cross_file_call,
                    "is_super_call": func_name.startswith("super."),
                })
                
                for qname, symbol in self.symbol_table.items():
                    if symbol.node_id == callee_id:
                        symbol.references.append(caller_id)
                        break
        else:
            self.unresolved_calls.append({
                'caller_id': caller_id,
                'func_name': func_name,
                'file_path': self.file_path,
                'line_number': self._get_line_number(node),
                'is_super_call': func_name.startswith("super."),
            })

    def visit_Attribute(self, node: cst.Attribute) -> None:
        """
        处理属性访问
        
        Args:
            node: 属性节点
        """
        if self._current_function_id:
            attr_name = node.attr.value
            if isinstance(node.value, cst.Name):
                obj_name = node.value.value
                obj_id = self._resolve_name(obj_name)
                self._add_relation(
                    self._current_function_id,
                    obj_id,
                    TypeManager.REL_USES,
                    {"attribute": attr_name}
                )

    def extract(self, source_code: str) -> ExtractionResult:
        """
        提取代码符号
        
        Args:
            source_code: 源代码字符串
            
        Returns:
            提取结果
        """
        self.module = cst.parse_module(source_code)
        
        wrapper = MetadataWrapper(self.module, unsafe_skip_copy=True)
        wrapper.visit(self)
        
        for node in self.nodes:
            if hasattr(node, 'properties') and node.properties:
                node.properties = {
                    key: serialize_property(value) 
                    for key, value in node.properties.items()
                }
        
        for rel in self.relations:
            if hasattr(rel, 'properties') and rel.properties:
                rel.properties = {
                    key: serialize_property(value) 
                    for key, value in rel.properties.items()
                }
        
        return ExtractionResult(
            nodes=self.nodes,
            relations=self.relations,
            errors=self.errors,
            symbol_table=self.symbol_table,
            file_path=self.file_path,
            unresolved_calls=self.unresolved_calls,
        )


def extract_file(file_path: str, module_name: Optional[str] = None) -> ExtractionResult:
    """
    提取单个文件的符号
    
    Args:
        file_path: 文件路径
        module_name: 模块名（可选）
        
    Returns:
        提取结果
    """
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
) -> Tuple[List[ExtractionResult], Dict[str, SymbolInfo], List]:
    """
    提取目录下所有 Python 文件的符号
    
    Args:
        dir_path: 目录路径
        pattern: 文件匹配模式
        exclude_patterns: 排除模式列表
        
    Returns:
        Tuple: (单个文件结果列表，合并后的全局符号表，跨文件关系列表)
    """
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
    
    global_symbol_table = ExtractionResult.merge_symbol_tables(results)
    
    cross_file_relations = ExtractionResult.resolve_and_create_cross_file_relations(
        results, 
        global_symbol_table
    )
    
    for rel in cross_file_relations:
        for result in results:
            if rel.properties.get('call_site') == result.file_path:
                result.relations.append(rel)
                break
    
    return results, global_symbol_table, cross_file_relations
