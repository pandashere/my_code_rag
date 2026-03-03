"""
代码符号提取器 - 节点解析模块
处理复杂 node 的解析逻辑
"""
from typing import Dict, List, Optional, Any
from libcst import MaybeSentinel
import libcst as cst

from .utils import (
    get_annotation_string,
    extract_decorators,
    extract_type_annotation,
    get_value_hint,
    get_name_string,
)
from .cst_types import TypeManager


def extract_base_classes(bases, get_annotation_string_func) -> List[str]:
    """提取类的基类名称"""
    base_classes = []
    for base in bases:
        base_name = get_annotation_string_func(base.value)
        if base_name:
            base_classes.append(base_name)
    return base_classes


def extract_parameters(params: cst.Parameters) -> List[Dict[str, Any]]:
    """提取函数参数信息"""
    result = []
    
    for param in params.params:
        result.append({
            "name": param.name.value,
            "type": extract_type_annotation(param.annotation),
            "has_default": param.default is not None,
        })
    
    if params.star_arg is not MaybeSentinel.DEFAULT and params.star_arg is not None:
        if isinstance(params.star_arg, cst.ParamStar):
            result.append({
                "name": "*",
                "type": None,
                "is_vararg": True,
                "is_star_only": True,
            })
        elif isinstance(params.star_arg, cst.Param):
            result.append({
                "name": params.star_arg.name.value,
                "type": extract_type_annotation(params.star_arg.annotation),
                "is_vararg": True,
            })
    
    for param in params.kwonly_params:
        result.append({
            "name": param.name.value,
            "type": extract_type_annotation(param.annotation),
            "has_default": param.default is not None,
            "is_kwonly": True,
        })
    
    if params.star_kwarg is not MaybeSentinel.DEFAULT and params.star_kwarg is not None:
        if isinstance(params.star_kwarg, cst.Param):
            result.append({
                "name": params.star_kwarg.name.value,
                "type": extract_type_annotation(params.star_kwarg.annotation),
                "is_kwarg": True,
            })
    
    for param in params.posonly_params:
        result.append({
            "name": param.name.value,
            "type": extract_type_annotation(param.annotation),
            "has_default": param.default is not None,
            "is_posonly": True,
        })
    
    return result


def get_call_name(func: cst.CSTNode) -> Optional[str]:
    """获取被调用函数的完整名称（包含对象信息）"""
    if isinstance(func, cst.Name):
        return func.value
    
    elif isinstance(func, cst.Attribute):
        attr_name = func.attr.value
        
        if isinstance(func.value, cst.Call):
            call_func = func.value.func
            if isinstance(call_func, cst.Name) and call_func.value == "super":
                return f"super.{attr_name}"
        
        elif isinstance(func.value, cst.Name):
            obj_name = func.value.value
            return f"{obj_name}.{attr_name}"
        
        elif isinstance(func.value, cst.Attribute):
            from .utils import get_attribute_name
            obj_name = get_attribute_name(func.value)
            return f"{obj_name}.{attr_name}"
        
        return attr_name
    
    elif isinstance(func, cst.Call):
        return get_call_name(func.func)
    
    return None


def determine_node_type(is_method: bool, is_async: bool) -> str:
    """确定节点类型"""
    if is_method:
        return TypeManager.ENTITY_ASYNC_METHOD if is_async else TypeManager.ENTITY_METHOD
    else:
        return TypeManager.ENTITY_ASYNC_FUNCTION if is_async else TypeManager.ENTITY_FUNCTION


def is_not_sentinel(value) -> bool:
    """判断值是否不是 Sentinel"""
    return value is not MaybeSentinel.DEFAULT and value is not None
