from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import LLM_MODEL_NAME
import os

# Initialize the LLM for intent detection
intent_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.0
)

# Merged prompt with file-awareness
INTENT_PROMPT = """
You are an assistant that detects the user's intent. 
A file may or may not have been uploaded by the user. 
Based on the input and whether a file exists, choose the correct intent from the list:

- greeting: The user is saying hello.
- summarize_file: The user wants a summary of the uploaded file.
- explain_file: The user wants an explanation of the uploaded file.
- direct_question: The user wants to ask a question (using uploaded file if present, otherwise RAG) or an explanation about something with no file uploaded.
- quiz: The user wants a quiz based on content (file if uploaded, otherwise web).
- search: The user wants information from the web.

User input: "{user_text}"
Uploaded file exists: {uploaded_file_exists}

Return only **one** intent from the list above.
"""

def decide_intent(user_text: str, file_path=None, explicit_intent=None) -> str:
    """
    Determine the user's intent based on input text or explicit override.
    Fully LLM-driven with file-awareness.

    Args:
        user_text (str): The input text from the user.
        file_path (str, optional): Path to uploaded file, if any.
        explicit_intent (str, optional): Force a specific intent.

    Returns:
        str: One of the recognized intent strings.
    """
    # Use explicit intent if provided
    if explicit_intent:
        return explicit_intent

    # Determine if a file exists
    uploaded_file_exists = bool(file_path)

    # Build prompt for LLM
    prompt = INTENT_PROMPT.format(
        user_text=user_text,
        uploaded_file_exists=uploaded_file_exists
    )

    # Call LLM to classify intent
    response = intent_llm.invoke(prompt)

    # Return the intent exactly as LLM responds (strip extra whitespace)
    return response.content.strip()
