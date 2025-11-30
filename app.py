import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from rag.rag_system import RagSystem
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from agent.orchestrator import create_agent
from config.settings import TORCH_DEVICE, LLM_MODEL_NAME, MODEL_PATH


# -----------------------------
# Load Tokenizer + Model
# -----------------------------
MODEL_HF_ID = MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID, trust_remote_code=True)

model_kwargs = {"trust_remote_code": True}
if TORCH_DEVICE.startswith("cuda"):
    model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
else:
    model_kwargs.update({"torch_dtype": torch.float32})

model = AutoModelForCausalLM.from_pretrained(MODEL_HF_ID, **model_kwargs)
model.to(TORCH_DEVICE)

# -----------------------------
# Initialize RAG
# -----------------------------
rag_system = RagSystem(tokenizer)

# -----------------------------
# Create Agent
# -----------------------------
agent = create_agent(rag_system, tokenizer, model)


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(
    analytics_enabled=False,
    fill_height=True,
    allow_flagging="never"
) as demo:

    chatbot = gr.Chatbot()
    chat_history_state = gr.State([])
    file_path_state = gr.State(None)

    with gr.Row():
        file_input = gr.File(label="Upload File", file_types=[".pdf", ".docx", ".txt"])
        user_input = gr.Textbox(label="Your message", lines=1)
        send_button = gr.Button("Send")

    with gr.Row():
        clear_chat_button = gr.Button("Clear Chat")
        clear_file_button = gr.Button("Clear File")

    # ---- Callback ----
    def handle_send(user_message, uploaded_file, chat_history, file_state):

        # Fix HF Spaces message structure
        if isinstance(user_message, list):
            if len(user_message) > 0 and isinstance(user_message[0], dict):
                user_message = user_message[0].get("content", "")
            else:
                user_message = ""

        if uploaded_file is not None:
            file_state = uploaded_file.name

        response = agent(user_message, file_state)
        answer = response.get("answer", "Sorry, no answer.")

        chat_history = chat_history + [[user_message, answer]]

        if file_state is not None:
            file_state = None

        return chat_history, file_state, ""

    send_button.click(
        handle_send,
        inputs=[user_input, file_input, chat_history_state, file_path_state],
        outputs=[chatbot, file_path_state, user_input]
    )

    clear_chat_button.click(
        lambda _: [],
        inputs=[chat_history_state],
        outputs=[chatbot]
    )

    clear_file_button.click(
        lambda _: None,
        inputs=[file_path_state],
        outputs=[file_input]
    )

# -----------------------------
# Launch
# -----------------------------
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)