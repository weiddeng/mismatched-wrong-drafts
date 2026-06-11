"""
Reward functions for GRPO training on MATH-style datasets.
Lenient answer extraction with code-block stripping.

Public API
----------
- extract_boxed(text)                  — strict: last \\boxed{...} only
- extract_mathematical_answer(text)    — lenient: boxed, then "answer is X" patterns
- math_verify_match(pred, gold)        — symbolic + string-fallback comparison
- mathematically_quasi_correct(text, gold) — extract + math_verify_match
"""

from __future__ import annotations

import logging
import re
import signal
from typing import Optional

from math_verify import parse, verify


class _Timeout:
    """Context manager that raises TimeoutError after *seconds* (Unix only)."""
    def __init__(self, seconds: int = 5):
        self.seconds = seconds
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self
    def __exit__(self, *args):
        signal.alarm(0)
    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("math_verify timed out")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Extract content of the last ``\\boxed{...}``, handling nested braces.

    Also handles the bare form ``\\boxed X`` (no braces, just a space then a
    token) which appears in some upstream MATH solutions.
    """
    results: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed", i)
        if idx == -1:
            break

        after = idx + len("\\boxed")

        if after < len(text) and text[after] == "{":
            # Standard form: \boxed{...}
            depth, start = 0, after + 1
            for j in range(start, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    if depth == 0:
                        results.append(text[start:j].strip())
                        i = j + 1
                        break
                    depth -= 1
            else:
                break
        elif after < len(text) and text[after] in (" ", "\t"):
            # Bare form: \boxed 2, \boxed 9, \boxed{-free}
            # Grab the next non-whitespace token up to a delimiter.
            start = after + 1
            while start < len(text) and text[start] in (" ", "\t"):
                start += 1
            if start < len(text):
                # Collect until whitespace, period, comma, $, or end.
                end = start
                while end < len(text) and text[end] not in (" ", "\t", ".", ",", "$", "\n", ")"):
                    end += 1
                token = text[start:end].strip()
                if token:
                    results.append(token)
            i = max(after + 1, end if start < len(text) else after + 1)
        else:
            # Not followed by { or space — skip.
            i = after + 1

    return results[-1] if results else None


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def math_verify_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Symbolic comparison via math-verify, with string fallback.

    Wraps the symbolic check in a 5-second timeout to avoid hangs on
    pathological inputs.
    """
    if pred is None or gold is None:
        return False
    try:
        with _Timeout(5):
            if verify(parse(f"${gold}$"), parse(f"${pred}$")):
                return True
    except TimeoutError:
        logger.warning("math_verify timed out (5s) on pred=%r gold=%r — falling back to string match", pred[:100], gold[:100])
    except Exception:
        pass

    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("$", "")

    return norm(pred) == norm(gold)


def _strip_var_assignment(s: str) -> str:
    """Strip leading 'var = ' from extracted answers like 'x = 2' → '2'."""
    m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s.strip())
    return m.group(1).strip() if m else s.strip()


def extract_mathematical_answer(text: str) -> Optional[str]:
    """Extract the single most likely final-answer string from a solution.

    Tries strategies in priority order and returns on first hit:
      1. Last ``\\boxed{...}`` via :func:`extract_boxed` (preferred).
      2. Last "(final) answer is/:/= X" phrase.
      3. Last math expression in the text (whichever appears last):
         - ``$...=X$`` or ``\\(...=X\\)`` — value after last ``=``
         - ``$...$`` or ``\\(...\\)`` — full content
         - Bare ``var = VALUE`` on its own line

    Any extracted result has a leading ``var = `` stripped (e.g. ``x = 2`` → ``2``).

    Returns ``None`` if no strategy matches.  This function never returns
    multiple candidates — committing to one extraction avoids pass@n-style
    cheating when the result is later compared against gold.
    """
    if not text:
        return None

    # Strip Python/code blocks before matching (they pollute patterns).
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 1. Strict \boxed{...} — already returns the last occurrence.
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    # 2. "The answer is X" / "Answer: X" / "Answer = X" / "Final answer: X"
    matches = re.findall(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?([^\n\.\$]+?)\$?(?=\.|\n|$)",
        text, re.IGNORECASE,
    )
    if not matches:
        matches = re.findall(
            r"\b(?:final\s+)?answer\s*[=:]\s*\$?([^\n\.\$]+?)\$?(?=\.|\n|$)",
            text, re.IGNORECASE,
        )
    if matches:
        return _strip_var_assignment(matches[-1])

    # 4. Last math expression in the text — combines inline math with '=',
    #    inline math without '=', and bare plaintext assignments.
    #    Collects all candidates with positions, returns the last one.
    candidates = []  # list of (position, extracted_value)

    # 4a. $...=...$ or \(...=...\) — extract value after last '='
    for m in re.finditer(r"\$([^\$]*=[^\$]+)\$", text):
        val = m.group(1).rsplit("=", 1)[1].strip()
        candidates.append((m.end(), val))
    for m in re.finditer(r"\\\(([^)]*=[^)]+)\\\)", text):
        val = m.group(1).rsplit("=", 1)[1].strip()
        candidates.append((m.end(), val))

    # 4b. $...$ or \(...\) — no '=' required
    for m in re.finditer(r"\$([^\$\n]+)\$", text):
        candidates.append((m.end(), m.group(1).strip()))
    for m in re.finditer(r"\\\(([^)]+)\\\)", text):
        candidates.append((m.end(), m.group(1).strip()))

    # 4c. Bare "var = VALUE" on its own line (no LaTeX delimiters)
    for m in re.finditer(r"^[-\s]*[a-zA-Z]\s*=\s*(.+?)\s*$", text, re.MULTILINE):
        candidates.append((m.end(), m.group(1).strip()))

    if candidates:
        # Return the candidate that appears last in the text.
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return None


def mathematically_quasi_correct(text: str, gold: str) -> bool:
    """Whether the text's extracted answer matches gold via math-verify."""
    return math_verify_match(extract_mathematical_answer(text), gold)
