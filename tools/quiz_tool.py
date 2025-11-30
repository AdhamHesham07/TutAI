from langchain_google_genai import ChatGoogleGenerativeAI
from llm.prompt_templates import QUIZ_PROMPT
from config.settings import LLM_MODEL_NAME
import os

# -----------------------------
# Initialize Gemini LLM internally
# -----------------------------
api_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)

def generate_quiz_from_text(passage: str, n: int = 5) -> str:
    """
    Generate a quiz from the given passage using Gemini LLM.

    Args:
        passage (str): Text to generate quiz from.
        n (int): Number of questions to generate.

    Returns:
        str: Generated quiz text.
    """
    prompt = QUIZ_PROMPT.format(passage=passage, n=n)
    result = api_llm.invoke(prompt)  # uses the internal default LLM
    return result.content
