"""OpenAI Compatible Embedding - 支持任意兼容端点"""

from typing import Any, Dict, List, Optional
import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Any, Dict, Optional
import httpx
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI as BaseOpenAI
from openai import OpenAI as SyncOpenAI, AsyncOpenAI, OpenAI
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    ChatResponseGen,
    MessageRole,
    LLMMetadata,
)

class OpenAICompatibleEmbedding(BaseEmbedding):
    """
    通用 OpenAI Compatible Embedding 类
    
    支持任何 OpenAI 兼容的 API 端点（vLLM、Ollama、LocalAI 等）
    
    Args:
        model_name: 模型名称
        api_base: API 基础 URL
        api_key: API 密钥
        embed_batch_size: 批处理大小
        timeout: 请求超时（秒）
        max_retries: 最大重试次数
        dimensions: 输出向量维度（可选）
    
    使用示例：
        embed_model = OpenAICompatibleEmbedding(
            model_name="BAAI/bge-m3",
            api_base="http://localhost:8000/v1",
            api_key="sk-no-key-required",
            embed_batch_size=10,
        )
    """
    
    model_name: str = Field(description="模型名称")
    api_base: str = Field(description="API 基础 URL")
    api_key: str = Field(default="sk-no-key-required", description="API 密钥")
    timeout: float = Field(default=60.0, description="请求超时（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")
    dimensions: Optional[int] = Field(default=None, description="输出向量维度")
    
    _client: Optional[OpenAI] = PrivateAttr()
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()
    _async_http_client: Optional[httpx.AsyncClient] = PrivateAttr()
    
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "sk-no-key-required",
        embed_batch_size: int = 10,
        timeout: float = 60.0,
        max_retries: int = 3,
        dimensions: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            embed_batch_size=embed_batch_size,
            timeout=timeout,
            max_retries=max_retries,
            dimensions=dimensions,
            callback_manager=callback_manager,
            num_workers=num_workers,
            **kwargs,
        )
        
        self._client = None
        self._aclient = None
        self._http_client = http_client
        self._async_http_client = async_http_client
    
    @classmethod
    def class_name(cls) -> str:
        return "OpenAICompatibleEmbedding"
    
    def _get_client(self) -> OpenAI:
        """获取同步客户端"""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=self._http_client,
            )
        return self._client
    
    def _get_aclient(self) -> AsyncOpenAI:
        """获取异步客户端"""
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=self._async_http_client,
            )
        return self._aclient
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        client = self._get_client()
        text = text.replace("\n", " ")
        
        kwargs = {"model": self.model_name}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        
        response = client.embeddings.create(
            input=[text],
            **kwargs,
        )
        return response.data[0].embedding
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取单个文本的 embedding"""
        aclient = self._get_aclient()
        text = text.replace("\n", " ")
        
        kwargs = {"model": self.model_name}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        
        response = await aclient.embeddings.create(
            input=[text],
            **kwargs,
        )
        return response.data[0].embedding
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本 embeddings"""
        client = self._get_client()
        texts = [text.replace("\n", " ") for text in texts]
        
        kwargs = {"model": self.model_name}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        
        response = client.embeddings.create(
            input=texts,
            **kwargs,
        )
        return [d.embedding for d in response.data]
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本 embeddings"""
        aclient = self._get_aclient()
        texts = [text.replace("\n", " ") for text in texts]
        
        kwargs = {"model": self.model_name}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        
        response = await aclient.embeddings.create(
            input=texts,
            **kwargs,
        )
        return [d.embedding for d in response.data]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询 embedding（与文本 embedding 相同）"""
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询 embedding"""
        return await self._aget_text_embedding(query)

class CustomOpenAILLM(BaseOpenAI):
    """
    支持自定义 OpenAI 端点的 LLM
    
    适用于：
    - Azure OpenAI
    - 本地部署 (vLLM, Ollama, LM Studio 等)
    - 第三方 OpenAI 兼容 API (DeepSeek, Moonshot, 智谱等)
    - 企业私有化部署
    
    示例：
        # 本地 vLLM
        llm = CustomOpenAILLM(
            api_base="http://localhost:8000/v1",
            api_key="not-needed",
            model="local-model"
        )
        
        # Azure OpenAI
        llm = CustomOpenAILLM(
            api_base="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
            api_key="your-azure-key",
            api_version="2024-02-15-preview",
            model="gpt-4"
        )
        
        # DeepSeek
        llm = CustomOpenAILLM(
            api_base="https://api.deepseek.com/v1",
            api_key="your-deepseek-key",
            model="deepseek-chat"
        )
    """
    
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="自定义 OpenAI API 端点"
    )
    
    api_version: Optional[str] = Field(
        default=None,
        description="API 版本（Azure 需要）"
    )
    
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="自定义请求头"
    )
    
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        api_version: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            default_headers=custom_headers,
            http_client=http_client,
            async_http_client=async_http_client,
            **kwargs,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "custom_openai_llm"
    
    @property
    def metadata(self) -> LLMMetadata:
        """
        覆盖 metadata 属性 - 完全绕过模型名验证
        """
        return LLMMetadata(
            context_window=128000,
            num_output=10000,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )