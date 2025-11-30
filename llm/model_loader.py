from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config.settings import MODEL_PATH, TORCH_DEVICE


# ======================================================
#  Echo Cleaner (internal only)
# ======================================================
def _smart_strip_echo(output: str) -> str:
    """
    Extract the final answer from LLM output using 'Final Answer:' as marker.
    Removes repeated occurrences of the same answer.
    """
    marker = "Final Answer:"
    if marker in output:
        # Take everything after the first 'Final Answer:'
        answer_part = output.split(marker, 1)[1].strip()
        
        # Remove repeated occurrences
        # Split by lines, join first occurrence of each line sequence
        lines = answer_part.splitlines()
        seen = set()
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return "\n".join(unique_lines)
    
    # fallback if marker not found
    return output.strip()


# ======================================================
#  Load Local LLM
# ======================================================
def load_local_llm(model_path: str = MODEL_PATH):
    """
    Load a local LLM model from disk.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    kwargs = {"local_files_only": True, "trust_remote_code": True}
    device = TORCH_DEVICE

    if device.startswith("cuda"):
        kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
    else:
        kwargs.update({"torch_dtype": torch.float32})

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    return tokenizer, model



# ======================================================
#  Generate With Local LLM (with auto-cleaning)
# ======================================================
def generate_with_local_llm(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.0
):
    """
    Generate text from the local LLM.
    Automatically removes prompt echoes.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature
    )

    raw_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)

    # Clean
    cleaned = _smart_strip_echo(raw_text)

    return cleaned
