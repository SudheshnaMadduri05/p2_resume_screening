import re

def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    - Lowercasing
    - Removing special characters
    - Removing extra spaces
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
