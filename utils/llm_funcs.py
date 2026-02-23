import json
import re
from typing import Any, Optional

def _parse_structured_query(raw_output: str) -> str:
    """
    从 LLM 的输出中提取标准化的英文查询字符串。
    
    LLM 应被提示输出如：
        {"query": "How does the DFS function work in ast_traversal.py?"}
    """
    if not raw_output or not isinstance(raw_output, str):
        return ""

    # 尝试从 ```json ... ``` 中提取
    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", raw_output, re.DOTALL)
    candidate = json_match.group(1) if json_match else raw_output.strip()

    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "query" in data and isinstance(data["query"], str):
            query = data["query"].strip()
            if query:
                return query
    except (json.JSONDecodeError, TypeError):
        pass

    # 回退：如果输出本身就是合理英文句子（无中文、非 JSON 残片）
    stripped = raw_output.strip()
    if (
        stripped
        and any(c.isalpha() for c in stripped)
        and not stripped.startswith("{")
        and not re.search(r'[\u4e00-\u9fff]', stripped)  # 无中文
    ):
        return stripped

    return ""


def normalize_query_for_retrieval(
    original_query: str,
    llm: Any,
    enable_refinement: bool = True,
) -> str:
    """
    将原始用户查询（可能含中文、模糊表述）转换为标准化的英文检索查询。
    
    流程：
      1. （可选）用 LLM 精炼原始查询
      2. 让 LLM 生成结构化的英文查询（JSON 格式）
      3. 解析并返回纯英文查询字符串
    
    Args:
        original_query: 用户输入（如 "ast遍历怎么实现的？"）
        llm: LLM 实例（需支持 .complete()）
        enable_refinement: 是否启用查询精炼步骤
    
    Returns:
        str: 标准化英文查询，如 "How is AST traversal implemented in the codebase?"
             若失败，返回原始查询的英文翻译或原样（由 LLM 决定）
    """
    if not enable_refinement:
        # 直接要求 LLM 输出结构化英文查询
        effective_input = original_query
    else:
        # Step 1: 先精炼查询（提升语义清晰度）
        refine_prompt = (
            "You are a technical assistant. Refine the following user question to be clear, precise, "
            "and suitable for code knowledge retrieval. Keep it concise.\n"
            f"Original: {original_query}\n"
            "Refined (English):"
        )
        refined = llm.complete(refine_prompt).text.strip()
        effective_input = refined if refined else original_query

    # Step 2: 生成结构化英文查询
    struct_prompt = (
        "You are an expert in code understanding. Based on the question below, output a precise, "
        "standalone English search query suitable for retrieving relevant code documentation or implementation details.\n"
        "Respond ONLY with a JSON object containing a 'query' field (string).\n"
        "Do not include explanations.\n\n"
        f"Question: {effective_input}\n"
        "Search query (JSON):"
    )
    raw_response = llm.complete(struct_prompt).text.strip()

    # Step 3: 解析
    normalized = _parse_structured_query(raw_response)

    # 最终兜底：如果解析失败，用精炼后的结果或原始输入
    if not normalized:
        normalized = effective_input if isinstance(effective_input, str) else str(original_query)

    # 确保返回非空字符串
    return normalized.strip() or "code implementation details"
