"""
代码符号提取器 - 关系解析模块
处理复杂 relation 的解析逻辑
"""
from typing import Dict, Optional, List
from symbol_info import SymbolInfo
from cst_types import TypeManager


def resolve_super_call_cross_file(
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
        method_qname = f"{base_name}.{method_name}"
        if method_qname in qname_to_symbol:
            return qname_to_symbol[method_qname]
        
        for qname, symbol in qname_to_symbol.items():
            if symbol.name == method_name and qname.startswith(f"{base_name}."):
                return symbol
    
    return None


def resolve_by_method_name_cross_file(
    method_name: str,
    caller_id: str,
    qname_to_symbol: Dict[str, SymbolInfo]
) -> Optional[SymbolInfo]:
    """跨文件按方法名解析（避免自调用）"""
    caller_parts = caller_id.split(".")
    caller_module = ".".join(caller_parts[:-1]) if len(caller_parts) > 1 else ""
    
    candidates = []
    for qname, symbol in qname_to_symbol.items():
        if symbol.name == method_name and symbol.node_id != caller_id:
            if symbol.node_type in [TypeManager.ENTITY_FUNCTION, TypeManager.ENTITY_METHOD]:
                candidates.append((qname, symbol))
    
    if not candidates:
        return None
    
    for qname, symbol in candidates:
        if qname.startswith(caller_module):
            return symbol
    
    return candidates[0][1] if candidates else None


def resolve_base_class_id(
    base_name: str,
    child_class_id: str,
    symbol_table: Dict[str, SymbolInfo],
    module_name: str,
    import_map: Dict[str, str]
) -> str:
    """
    解析基类名称到节点 ID
    
    Args:
        base_name: 基类名称
        child_class_id: 子类 ID
        symbol_table: 符号表
        module_name: 模块名
        import_map: 导入映射
        
    Returns:
        基类的节点 ID
    """
    if base_name in symbol_table:
        return symbol_table[base_name].node_id
    
    candidate = f"{module_name}.{base_name}"
    if candidate in symbol_table:
        return symbol_table[candidate].node_id
    
    parts = child_class_id.split(".")
    if len(parts) >= 2:
        module_prefix = ".".join(parts[:-2])
        candidate = f"{module_prefix}.{base_name}"
        if candidate in symbol_table:
            return symbol_table[candidate].node_id
    
    if base_name in import_map:
        return import_map[base_name]
    
    return base_name


def resolve_super_call(
    method_name: str,
    caller_id: str,
    symbol_table: Dict[str, SymbolInfo]
) -> Optional[str]:
    """
    解析 super() 调用到父类方法
    
    Args:
        method_name: 方法名
        caller_id: 调用者 ID
        symbol_table: 符号表
        
    Returns:
        父类方法的 node_id，无法解析返回 None
    """
    parts = caller_id.split(".")
    if len(parts) < 2:
        return None
    
    class_name = parts[-2]
    current_class_symbol = None
    
    for qname, symbol in symbol_table.items():
        if symbol.node_type == TypeManager.ENTITY_CLASS and symbol.name == class_name:
            current_class_symbol = symbol
            break
    
    if not current_class_symbol:
        return None
    
    base_classes = current_class_symbol.extra_properties.get("base_classes", [])
    
    for base_name in base_classes:
        if base_name in symbol_table:
            method_qname = f"{base_name}.{method_name}"
            if method_qname in symbol_table:
                return symbol_table[method_qname].node_id
        
        for qname, symbol in symbol_table.items():
            if symbol.node_type in (TypeManager.ENTITY_FUNCTION, TypeManager.ENTITY_METHOD):
                if symbol.name == method_name and qname.startswith(f"{base_name}."):
                    return symbol.node_id
    
    return None


def resolve_by_method_name(
    method_name: str,
    caller_id: str,
    symbol_table: Dict[str, SymbolInfo]
) -> Optional[str]:
    """
    按方法名解析（避免自调用）
    
    Args:
        method_name: 方法名
        caller_id: 调用者 ID
        symbol_table: 符号表
        
    Returns:
        解析到的 node_id，无法解析返回 None
    """
    parts = caller_id.split(".")
    current_class = parts[-2] if len(parts) >= 2 else None
    
    for qname, symbol in symbol_table.items():
        if symbol.name == method_name and symbol.node_type in (
            TypeManager.ENTITY_FUNCTION,
            TypeManager.ENTITY_METHOD,
            TypeManager.ENTITY_ASYNC_FUNCTION
        ):
            if current_class and qname.startswith(f"{current_class}."):
                continue
            return symbol.node_id
    
    if current_class:
        method_qname = f"{current_class}.{method_name}"
        if method_qname in symbol_table:
            return symbol_table[method_qname].node_id
    
    return None


def is_cross_file(callee_id: str, symbol_table: Dict[str, SymbolInfo], 
                  file_path: str, module_name: str) -> bool:
    """
    判断调用是否跨文件
    
    Args:
        callee_id: 被调用者 ID
        symbol_table: 符号表
        file_path: 当前文件路径
        module_name: 模块名
        
    Returns:
        是否跨文件
    """
    if callee_id in symbol_table:
        return symbol_table[callee_id].file_path != file_path
    
    if not callee_id.startswith(module_name):
        return True
    
    return False
