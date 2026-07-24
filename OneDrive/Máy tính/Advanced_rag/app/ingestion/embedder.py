from google import genai


class Embedder:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = self.client.models.embed_content(model=self.model_name, contents=texts)
        return [embedding.values for embedding in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
