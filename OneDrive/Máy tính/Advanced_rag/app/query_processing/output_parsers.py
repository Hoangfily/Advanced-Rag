import re

from langchain_core.output_parsers import BaseOutputParser


def _clean_line(line: str) -> str:
    cleaned = re.sub(r"^\s*(\d+[.)]|[-*>]|#+)\s*", "", line).strip()
    return cleaned.strip("*").strip('"').strip()


class QuestionListOutputParser(BaseOutputParser[list[str]]):
    """Parses one-question-per-line LLM output, stripping bullet/number/quote/markdown prefixes and keeping only lines that look like questions."""

    def parse(self, text: str) -> list[str]:
        questions = []
        for line in text.splitlines():
            cleaned = _clean_line(line)
            if cleaned.endswith("?"):
                questions.append(cleaned)
        return questions

    @property
    def _type(self) -> str:
        return "question_list"


class SingleQuestionOutputParser(BaseOutputParser[str]):
    """Extracts the first question-like line from LLM output, discarding any leading commentary,
    multiple alternatives, or markdown formatting the model may add despite being asked for one line."""

    def parse(self, text: str) -> str:
        for line in text.splitlines():
            cleaned = _clean_line(line)
            if cleaned.endswith("?"):
                return cleaned
        return text.strip().strip('"')

    @property
    def _type(self) -> str:
        return "single_question"
