"""
代码符号提取器 - 使用 libcst 解析 Python 代码并构建知识图谱

模块结构:
    - utils.py: 工具函数
    - symbol_info.py: 符号信息数据类
    - node_parsers.py: 节点解析逻辑
    - relation_parsers.py: 关系解析逻辑
    - extractor.py: 主提取器类
"""

from extractor import (
    CodeSymbolExtractor,
    extract_file,
    extract_directory,
)
from symbol_info import (
    SymbolInfo,
    ExtractionResult,
)

__all__ = [
    'CodeSymbolExtractor',
    'extract_file',
    'extract_directory',
    'SymbolInfo',
    'ExtractionResult',
]
