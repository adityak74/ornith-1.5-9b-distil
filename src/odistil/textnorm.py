"""Answer normalization + n-gram decontamination shared by dataset and eval."""

from __future__ import annotations

import re
import string

ANSWER_LINE = re.compile(r"answer\s*[::]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
LETTER = re.compile(r"\b([A-Z])\b")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def final_answer(text: str) -> str | None:
    m = ANSWER_LINE.findall(text.strip())
    return m[-1].strip() if m else None


def final_letter(text: str, n_choices: int = 26) -> str | None:
    """Last 'Answer: X' line, else the last standalone letter in the tail."""
    ans = final_answer(text)
    valid = set(string.ascii_uppercase[:n_choices])
    if ans:
        m = LETTER.findall(ans.upper())
        if m and m[0] in valid:
            return m[0]
        # 'Answer: (C)' / 'Answer: C. foo'
        head = ans.strip().lstrip("(*[ ")[:1].upper()
        if head in valid:
            return head
    m = LETTER.findall(text[-400:].upper())
    return m[-1] if m and m[-1] in valid else None


def qa_match(prediction: str, aliases: list[str]) -> bool:
    pred = normalize(final_answer(prediction) or prediction)
    if not pred:
        return False
    return any(normalize(a) and normalize(a) in pred for a in aliases)


def ngrams(text: str, n: int) -> set[str]:
    toks = normalize(text).split()
    return {" ".join(toks[i : i + n]) for i in range(max(len(toks) - n + 1, 0))}


class Decontaminator:
    """Drops any training prompt sharing an n-gram with an eval item."""

    def __init__(self, n: int = 13):
        self.n = n
        self.index: set[str] = set()

    def add(self, text: str) -> None:
        self.index |= ngrams(text, self.n)

    def is_contaminated(self, text: str) -> bool:
        return bool(ngrams(text, self.n) & self.index)
