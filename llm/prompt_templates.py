RAG_PROMPT = """
You are an educational assistant.

Answer the question ONLY using the context below.
Do NOT repeat the question, context, or instructions.
If the answer is not found in the context, reply with: "I don't know."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Final Answer:
""".strip()

RAG_PROMPT_MEMORY = """
You are an educational assistant.

Here is the recent conversation:
{chat_memory}

Here is additional context retrieved from documents:
{context}

Use BOTH the recent conversation and the retrieved context to answer the question.
If the answer is not found, reply with: "I don't know."

Question: {question}

Final Answer:
""".strip()


SEARCH_SYNTHESIS_PROMPT = """You are an educational assistant.
Use the RAG answer and web search snippets below to produce a concise, accurate answer.
If information conflicts, prefer reliable web sources and cite briefly.
If neither source answers, say "I don't know".

Question: {question}

RAG answer:
{rag}

Web snippets:
{web}

Final Answer:"""


QUIZ_PROMPT = """Generate {n} multiple choice questions (A-D) from the passage below.
Each question should include the correct option and a 1-2 sentence explanation.
Passage:
{passage}"""