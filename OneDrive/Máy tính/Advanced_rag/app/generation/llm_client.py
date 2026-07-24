from google import genai


class LLMClient:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return response.text
