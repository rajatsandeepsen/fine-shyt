def parse_response(text: str) -> dict:
    """
    custom parsing logic.
    """
    marker = "assistant\n"
    idx = text.rfind(marker)
    if idx == -1:
        return {"role": "assistant", "content": text}

    content = text[idx + len(marker) :].strip()
    return {"role": "assistant", "content": content}
