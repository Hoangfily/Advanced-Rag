from typing import Any

from google import genai
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr


class LLMClient(LLM):
    """Gemini-backed completion model exposed as a LangChain Runnable for use in LCEL chains."""

    model_name: str
    api_key: str
    _client: genai.Client = PrivateAttr()

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self._client = genai.Client(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        response = self._client.models.generate_content(model=self.model_name, contents=prompt)
        return response.text

    def complete(self, prompt: str) -> str:
        return self.invoke(prompt)
