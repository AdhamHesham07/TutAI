import re
import os

from agent.intents import decide_intent
from agent.router import default_rag_flow, synthesize_with_search

from tools.search_tool import web_search_snippets
from tools.parser_tool import parse_file
from tools.quiz_tool import generate_quiz_from_text
from tools.memory_tool import add_to_memory, build_chat_context

from llm.model_loader import generate_with_local_llm
from llm.prompt_templates import RAG_PROMPT, RAG_PROMPT_MEMORY

from config.settings import LLM_MODEL_NAME

import faiss

from langchain_classic import LLMChain, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory


# --------------------------
# Gemini (for synthesis only)
# --------------------------
api_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.0
)

SYNTH_PROMPT = PromptTemplate(
    template="""You are an educational assistant.
Use the RAG answer and web search snippets below to produce a concise, accurate answer.
Question: {question}
RAG answer:
{rag}
Web snippets:
{web}
Final Answer:""",
    input_variables=['question','rag','web']
)

api_llm_chain = LLMChain(llm=api_llm, prompt=SYNTH_PROMPT)


def create_agent(rag_system, local_tokenizer, local_model, api_llm_chain_local=None):

    if api_llm_chain_local is None:
        api_llm_chain_local = api_llm_chain
    
    uploaded_file = None  # holds latest uploaded file state

    session_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # --------------------------
    # Helper: batch large files
    # --------------------------
    def batch_file_handler(file_text, rag_system, chunk_size=200, overlap=30, batch_size=5):
        chunks = rag_system.chunk_text_by_tokens(file_text, max_tokens=chunk_size, overlap=overlap)
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        
        emb = rag_system.embedder.encode(chunks, convert_to_numpy=True)
        temp_index = faiss.IndexFlatL2(emb.shape[1])
        temp_index.add(emb)
        
        return {"chunks": chunks, "batches": batches, "faiss_index": temp_index}

    # --------------------------
    # Extract number of questions for quiz
    # --------------------------
    def extract_requested_quiz_number(user_text: str, default: int = 10) -> int:
        match = re.search(r"(\d+)", user_text)
        return int(match.group(1)) if match else default

    # --------------------------
    # MAIN HANDLER
    # --------------------------
    def handle(user_text: str, file_path: str = None, explicit_intent: str = None):

        nonlocal uploaded_file

        intent = decide_intent(user_text, file_path=file_path, explicit_intent=explicit_intent)

        # --------------------------
        # UNKNOWN INTENT FALLBACK
        # --------------------------
        VALID_INTENTS = {
            "greeting", "summarize_file", "explain_file",
            "direct_question", "quiz", "search"
        }
        if intent not in VALID_INTENTS and intent is not None:
            return {'answer': "I'm not sure what you mean. Try rephrasing.", 'source': 'unknown_intent'}

        # --------------------------
        # GREETING
        # --------------------------
        if intent == "greeting":
            msg = "Hello! I'm your educational assistant. How can I help you today?"
            add_to_memory(session_memory, user_text, msg)
            return {'answer': msg, 'source': 'greeting'}

        # --------------------------
        # FILE UPLOAD
        # --------------------------
        if file_path:
            full_text = parse_file(file_path)

            # --------------------------
            # SIMPLE FILE SIZE GUARD
            # --------------------------
            if len(full_text) > 200_000:  
                return {"answer": "File too large for this chatbot to process.", "source": "file_error"}

            uploaded_file = batch_file_handler(full_text, rag_system)

        # --------------------------
        # FILE SUMMARY
        # --------------------------
        # NO FILE UPLOADED HANDLING
        if intent == "summarize_file" and not uploaded_file:
            return {'answer': "Please upload a file first.", 'source': 'error'}

        if intent == "summarize_file" and uploaded_file:
            summaries = []
            for batch in uploaded_file["batches"]:
                batch_text = "\n\n".join(batch)
                summary = api_llm_chain_local.run({
                    "question": "Summarize the following content clearly.",
                    "rag": batch_text,
                    "web": "No web search was used for this query."
                })
                summaries.append(summary)

            final_summary = "\n\n".join(summaries)
            add_to_memory(session_memory, user_text, final_summary)
            return {'answer': final_summary, 'source': 'file_summary'}


        # --------------------------
        # FILE EXPLANATION
        # --------------------------
        if intent == "explain_file" and uploaded_file:
            explanations = []
            for batch in uploaded_file["batches"]:
                batch_text = "\n\n".join(batch)
                explanation = api_llm_chain_local.run({
                    "question": "Explain the following content clearly and in detail.",
                    "rag": batch_text,
                    "web": "No web search was used for this query."
                })
                explanations.append(explanation)

            final_explanation = "\n\n".join(explanations)
            add_to_memory(session_memory, user_text, final_explanation)
            return {'answer': final_explanation, 'source': 'file_explanation'}

        # --------------------------
        # DIRECT QUESTION / FILE OR RAG
        # --------------------------
        if intent == "direct_question":
            if uploaded_file:
                # Answer from uploaded file
                q_emb = rag_system.embedder.encode([user_text], convert_to_numpy=True)
                distances, indices = uploaded_file["faiss_index"].search(q_emb, k=5)

                context = "\n\n".join(uploaded_file["chunks"][i] for i in indices[0])
                chat_memory = build_chat_context(session_memory)

                prompt = RAG_PROMPT_MEMORY.format(chat_memory=chat_memory, context=context, question=user_text)
                answer = generate_with_local_llm(local_tokenizer, local_model, prompt)
            
                add_to_memory(session_memory, user_text, answer)

                return {'answer': answer, 'source': 'file_rag'}
            
            else:
                # Fallback to global RAG + web
                out = default_rag_flow(
                    user_text,
                    rag_system,
                    local_tokenizer,
                    local_model,
                    api_llm_chain_local,
                    web_search_snippets,
                    memory=session_memory,
                    is_memory=True
                )
                add_to_memory(session_memory, user_text, out['answer'])
                return out              

        # --------------------------
        # QUIZ
        # --------------------------
        if intent == "quiz":
            number = extract_requested_quiz_number(user_text, default=10)
            if uploaded_file:
                passage = "\n\n".join([chunk for batch in uploaded_file["batches"][:6] for chunk in batch])
            else:
                web_snippets = web_search_snippets(user_text)
                passage = "\n\n".join(web_snippets)
            quiz = generate_quiz_from_text(passage, n=number)
            add_to_memory(session_memory, user_text, quiz)
            return {'answer': quiz, 'source': 'quiz'}

        # --------------------------
        # SEARCH ONLY
        # --------------------------
        if intent == "search":
            web_snippets = web_search_snippets(user_text)
            synth = synthesize_with_search(api_llm_chain_local, user_text, '', web_snippets)
            add_to_memory(session_memory, user_text, synth)
            return {'answer': synth, 'source': 'web'}

    return handle
