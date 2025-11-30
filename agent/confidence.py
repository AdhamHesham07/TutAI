from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import LLM_MODEL_NAME, CONFIDENCE_THRESHOLD
import os

confidence_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,                    
    google_api_key=os.environ["GEMINI_API_KEY"],  
    temperature=0.0
)

CONFIDENCE_PROMPT = """
You are an expert educational assistant. Evaluate the following answer and determine its reliability based on the provided context chunks.
- If the answer fully and accurately covers the content in the context, respond: high
- If the answer is partially correct, vague, or unsupported by context, respond: low

Answer: {answer}
Context chunks: {chunks}
Threshold (for similarity understanding, informational only): {threshold}
"""

def evaluate_confidence(answer: str, retrieved_chunks: list, threshold: float = CONFIDENCE_THRESHOLD):
    if not retrieved_chunks:
        return {"confidence": "low", "max_similarity": 0.0, "similarities": []}

    prompt = CONFIDENCE_PROMPT.format(answer=answer, chunks=retrieved_chunks, threshold=threshold)
    result = confidence_llm.invoke(prompt)
    confidence_text = result.content.strip().lower()
    confidence = "high" if confidence_text == "high" else "low"

    return {"confidence": confidence, "max_similarity": 1.0 if confidence=="high" else 0.0, "similarities": []}
