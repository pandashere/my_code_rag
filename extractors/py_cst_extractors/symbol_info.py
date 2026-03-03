"""
代码符号提取器 - 符号信息数据类模块
"""
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field

from .py_relations import CodeRelation, CodeEntityNode
from .cst_types import TypeManager


@dataclass
class SymbolInfo:
    """
    全局符号表中的符号信息
    """
    name: str
    qualified_name: str
    node_type: str
    scope: str
    line_number: int
    file_path: str
    node_id: str
    references: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
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


@dataclass
class ExtractionResult:
    """
    提取结果容器
    """
    nodes: List[CodeEntityNode] = field(default_factory=list)
    relations: List[CodeRelation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    symbol_table: Dict[str, "SymbolInfo"] = field(default_factory=dict)
    file_path: Optional[str] = None
    unresolved_calls: List[Dict[str, Any]] = field(default_factory=list)
    import_map: Dict[str, str] = field(default_factory=dict)
    module_name: Optional[str] = None
    
    @property
    def node_count(self) -> int:
        """获取节点数量"""
        return len(self.nodes)
    
    @property
    def relation_count(self) -> int:
        """获取关系数量"""
        return len(self.relations)
    
    @staticmethod
    def merge_symbol_tables(results: List["ExtractionResult"]) -> Dict[str, "SymbolInfo"]:
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
        results: List["ExtractionResult"], 
        global_symbol_table: Dict[str, "SymbolInfo"]
    ) -> List[CodeRelation]:
        """解析跨文件引用并创建关系"""
        from .relation_parsers import (
            resolve_super_call_cross_file,
            resolve_by_method_name_cross_file,
        )
        
        new_relations = []
        seen_relations: Set[Tuple[str, str, str]] = set()
        
        qname_to_symbol = global_symbol_table
        name_to_symbols: Dict[str, List["SymbolInfo"]] = {}
        for symbol in qname_to_symbol.values():
            name_to_symbols.setdefault(symbol.name, []).append(symbol)
        
        class_to_symbol: Dict[str, "SymbolInfo"] = {}
        for qname, symbol in qname_to_symbol.items():
            if symbol.node_type == TypeManager.ENTITY_CLASS:
                class_to_symbol[symbol.name] = symbol
        
        for result in results:
            import_map = result.import_map or {}
            module_name = result.module_name or ""
            for call_info in result.unresolved_calls:
                caller_id = call_info["caller_id"]
                func_name = call_info["func_name"]
                call_file = call_info.get("file_path", result.file_path)
                is_super_call = call_info.get("is_super_call", False)
                candidate_id = call_info.get("candidate_id")
                caller_module = call_info.get("caller_module", module_name)
                
                callee_symbol = None
                
                if is_super_call and func_name.startswith("super."):
                    method_name = func_name.split(".", 1)[1]
                    callee_symbol = resolve_super_call_cross_file(
                        method_name, caller_id, qname_to_symbol, class_to_symbol
                    )
                
                if not callee_symbol:
                    candidate_qnames = []
                    if candidate_id:
                        candidate_qnames.append(candidate_id)

                    if func_name in import_map:
                        candidate_qnames.append(import_map[func_name])
                    elif "*" in import_map and "." not in func_name:
                        candidate_qnames.append(f"{import_map['*']}.{func_name}")

                    if "." in func_name:
                        prefix, suffix = func_name.split(".", 1)
                        if prefix in import_map:
                            candidate_qnames.append(f"{import_map[prefix]}.{suffix}")
                        candidate_qnames.append(func_name)
                    else:
                        candidate_qnames.append(func_name)
                        if caller_module:
                            candidate_qnames.append(f"{caller_module}.{func_name}")

                    for qname in candidate_qnames:
                        if qname in qname_to_symbol:
                            callee_symbol = qname_to_symbol[qname]
                            break

                    if not callee_symbol:
                        for qname in candidate_qnames:
                            suffix = f".{qname}"
                            suffix_matches = [
                                s for name, s in qname_to_symbol.items()
                                if name.endswith(suffix)
                            ]
                            if suffix_matches:
                                callee_symbol = suffix_matches[0]
                                break
                    
                    if not callee_symbol:
                        callee_symbol = resolve_by_method_name_cross_file(
                            func_name.split(".")[-1], caller_id, qname_to_symbol
                        )

                    if not callee_symbol:
                        name_key = func_name.split(".")[-1]
                        candidates = name_to_symbols.get(name_key, [])
                        if candidates:
                            if caller_module:
                                same_pkg = [
                                    c for c in candidates
                                    if c.qualified_name.startswith(caller_module.rsplit(".", 1)[0])
                                ]
                                if same_pkg:
                                    callee_symbol = same_pkg[0]
                                else:
                                    callee_symbol = candidates[0]
                            else:
                                callee_symbol = candidates[0]
                
                if callee_symbol and callee_symbol.node_id != caller_id:
                    rel_key = (caller_id, callee_symbol.node_id, TypeManager.REL_CALLS)
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        rel = CodeRelation(
                            source_id=caller_id,
                            target_id=callee_symbol.node_id,
                            label=TypeManager.REL_CALLS,
                            properties={
                                "call_site": call_file,
                                "is_cross_file": callee_symbol.file_path != call_file,
                                "is_super_call": is_super_call,
                            }
                        )
                        new_relations.append(rel)
        
        return new_relations
