def add_to_memory(memory, user: str, assistant: str):
    memory.save_context({'input': user}, {'output': assistant})

def get_memory_messages(memory):
    return memory.load_memory_variables({})

def build_chat_context(memory, max_turns=6, max_total_chars=20000):
    """
    Convert chat memory into a formatted conversation block.
    Dynamically include as many recent messages as possible
    without exceeding max_total_chars.
    """

    msgs = memory.load_memory_variables({}).get("chat_history", [])
    
    # Start from the last message, prepend messages until max_total_chars
    block = ""
    total_chars = 0

    for m in reversed(msgs[-max_turns:]):

        role = "user" if m.type == "user" else "assistant"
        content = f"{role}: {m.content}\n\n"

        if total_chars + len(content) > max_total_chars:
            break

        block = content + block  # prepend to keep chronological order
        total_chars += len(content)

    if not block:
        return "<No prior conversation>"

    return block.strip()

