import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s'áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Training normalization: NFC + whitespace collapse. Keeps punctuation and capitalisation."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_wer(text: str) -> str:
    """Benchmark normalization: lowercase + strip punctuation. Apply to both ref and hyp."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace(" ", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
