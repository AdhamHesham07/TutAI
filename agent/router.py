from llm.prompt_templates import RAG_PROMPT, RAG_PROMPT_MEMORY
from llm.model_loader import generate_with_local_llm 
from agent.confidence import evaluate_confidence 
from tools.memory_tool import build_chat_context

def synthesize_with_search(api_llm_chain, question: str, rag_answer: str, web_snippets: list) -> str:
     web_text = '\n\n'.join(web_snippets)
     return api_llm_chain.run({'question': question, 'rag': rag_answer, 'web': web_text}) 

def default_rag_flow(question: str, rag_system, local_tokenizer, local_model, api_llm_chain, search_fn, memory=None, k: int = 5, is_memory: bool = False): 
    """ 
    Default RAG-based question answering flow. - Uses the updated RagSystem instance for retrieval. - Keeps generate_with_local_llm() for generating answers.
    """ 
    retrieved, _ = rag_system.query(question, k=k)
    context = '\n\n'.join(retrieved) 

    if is_memory and memory is not None :
        chat_memory = build_chat_context(memory)
        prompt = RAG_PROMPT_MEMORY.format(chat_memory=chat_memory, context=context, question=question)
    else:
        prompt = RAG_PROMPT.format(context=context, question=question)
    
    # Generate answer using your fine-tuned local LLM
    rag_out = generate_with_local_llm(local_tokenizer, local_model, prompt)

    # Evaluate confidence
    conf = evaluate_confidence(rag_out, retrieved) 
    
    if conf['confidence'] == 'high':
        return {'answer': rag_out, 'source': 'rag', 'confidence': conf}
    
    # Fall back to web synthesis if confidence is low
    web = search_fn(question)
    synth = synthesize_with_search(api_llm_chain, question, rag_out, web) 
    
    return {'answer': synth, 'source': 'rag+web', 'confidence': conf}