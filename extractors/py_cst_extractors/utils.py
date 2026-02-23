"""
代码符号提取器 - 工具函数模块
"""
import json
from typing import Any, Optional
from pathlib import Path


def serialize_property(value: Any) -> Any:
    """
    将复杂对象序列化为 Neo4j 兼容的格式
    
    Args:
        value: 待序列化的值
        
    Returns:
        序列化后的值
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def infer_module_name(file_path: str) -> str:
    """
    从文件路径推断模块名
    
    Args:
        file_path: 文件路径
        
    Returns:
        模块名
    """
    path = Path(file_path)
    parts = []
    for part in path.parts:
        if part.endswith('.py'):
            parts.append(part[:-3])
        elif part != '__init__':
            parts.append(part)
    return '.'.join(parts)


def get_annotation_string(node) -> str:
    """
    将注解节点转换为字符串
    
    Args:
        node: libcst 注解节点
        
    Returns:
        注解字符串
    """
    import libcst as cst
    
    if isinstance(node, cst.Name):
        return node.value
    elif isinstance(node, cst.Attribute):
        return get_attribute_name(node)
    elif isinstance(node, cst.Subscript):
        base = get_annotation_string(node.value)
        params = []
        for item in node.slice:
            if isinstance(item, cst.SubscriptElement):
                params.append(get_annotation_string(item.slice.value))
        return f"{base}[{', '.join(params)}]"
    return ""


def get_attribute_name(attr) -> str:
    """
    获取属性访问的完整名称
    
    Args:
        attr: libcst Attribute 节点
        
    Returns:
        属性完整名称
    """
    import libcst as cst
    
    parts = []
    current = attr
    while isinstance(current, cst.Attribute):
        parts.append(current.attr.value)
        current = current.value
    if isinstance(current, cst.Name):
        parts.append(current.value)
    return '.'.join(reversed(parts))


def get_name_string(node) -> str:
    """
    从 libcst Name 或 Attribute 节点提取字符串名称
    
    Args:
        node: libcst Name 或 Attribute 节点
        
    Returns:
        名称字符串
    """
    import libcst as cst
    
    if node is None:
        return ""
    
    if isinstance(node, cst.Name):
        return node.value
    elif isinstance(node, cst.Attribute):
        value = get_name_string(node.value)
        return f"{value}.{node.attr.value}"
    else:
        return str(node)


def extract_decorators(decorators) -> list:
    """
    提取装饰器列表
    
    Args:
        decorators: 装饰器列表
        
    Returns:
        装饰器名称列表
    """
    import libcst as cst
    
    result = []
    for dec in decorators:
        if isinstance(dec, cst.Decorator):
            dec_name = get_decorator_name(dec.decorator)
            if dec_name:
                result.append(dec_name)
    return result


def get_decorator_name(decorator) -> str:
    """
    获取装饰器名称
    
    Args:
        decorator: 装饰器节点
        
    Returns:
        装饰器名称
    """
    import libcst as cst
    
    if isinstance(decorator, cst.Name):
        return decorator.value
    elif isinstance(decorator, cst.Attribute):
        return get_attribute_name(decorator)
    elif isinstance(decorator, cst.Call):
        return get_decorator_name(decorator.func)
    return ""


def extract_type_annotation(annotation) -> Optional[str]:
    """
    提取类型注解
    
    Args:
        annotation: 注解节点
        
    Returns:
        类型注解字符串
    """
    import libcst as cst
    from libcst import MaybeSentinel
    
    if annotation is None or annotation is MaybeSentinel.DEFAULT:
        return None
    if isinstance(annotation, cst.Annotation):
        return get_annotation_string(annotation.annotation)
    return None


def get_value_hint(value) -> Optional[str]:
    """
    获取赋值值的提示
    
    Args:
        value: 赋值值节点
        
    Returns:
        值提示字符串
    """
    import libcst as cst
    
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
        from .node_parsers import get_call_name
        return get_call_name(value.func)
    return value.code[:50] if hasattr(value, 'code') else None
