from google import genai

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    for model in client.models.list():
        actions = getattr(model, "supported_actions", None)
        print(model.name, "-", actions)


if __name__ == "__main__":
    main()
