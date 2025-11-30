import os
from dotenv import load_dotenv
load_dotenv()  # Load local .env AND HF secrets

import gradio as gr
from rag.rag_system import RagSystem
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from agent.orchestrator import create_agent
from config.settings import TORCH_DEVICE, LLM_MODEL_NAME, MODEL_PATH



# -----------------------------
# Load Tokenizer + Model (HF Hub)
# -----------------------------
MODEL_HF_ID = MODEL_PATH  # replace with your repo ID

tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID, trust_remote_code=True)

model_kwargs = {"trust_remote_code": True}
if TORCH_DEVICE.startswith("cuda"):
    model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
else:
    model_kwargs.update({"torch_dtype": torch.float32})

model = AutoModelForCausalLM.from_pretrained(MODEL_HF_ID, **model_kwargs)
model.to(TORCH_DEVICE)

# -----------------------------
# Initialize RAG System
# -----------------------------
rag_system = RagSystem(tokenizer)

# -----------------------------
# Create Agent
# -----------------------------
agent = create_agent(rag_system, tokenizer, model)

# -----------------------------
#  Gradio States
# -----------------------------
with gr.Blocks() as demo:
    
    # Chat display
    chatbot = gr.Chatbot()
    
    # Hidden states
    chat_history_state = gr.State([])
    file_path_state = gr.State(None)
    
    # Input row
    with gr.Row():
        file_input = gr.File(label="Upload File", file_types=[".pdf", ".docx", ".txt"])
        user_input = gr.Textbox(label="Your message", placeholder="Type your question here...", lines=1)
        send_button = gr.Button("Send")
    
    # Optional control buttons
    with gr.Row():
        clear_chat_button = gr.Button("Clear Chat")
        clear_file_button = gr.Button("Clear File")
    
    # -----------------------------
    # Callbacks
    # -----------------------------
    def handle_send(user_message, uploaded_file, chat_history, file_state):
        # Update file path if new file uploaded
        if uploaded_file is not None:
            file_state = uploaded_file.name
        
        # Call agent
        response = agent(user_message, file_state)
        answer = response.get("answer", "Sorry, no answer.")
        
        # Update chat history
        chat_history = chat_history + [[user_message, answer]]
        
        # Clear file state after first use
        if file_state is not None:
            file_state = None
        
        return chat_history, file_state, ""
    
    send_button.click(
        handle_send,
        inputs=[user_input, file_input, chat_history_state, file_path_state],
        outputs=[chatbot, file_path_state, user_input]
    )
    
    # Clear chat
    def clear_chat(chat_history):
        return [], chat_history  # reset display and state
    
    clear_chat_button.click(
        clear_chat,
        inputs=[chat_history_state],
        outputs=[chatbot, chat_history_state]
    )
    
    # Clear file
    def clear_file(file_state):
        return None, file_state
    
    clear_file_button.click(
        clear_file,
        inputs=[file_path_state],
        outputs=[file_path_state, file_input]
    )

# -----------------------------
# Launch
# -----------------------------
demo.launch()