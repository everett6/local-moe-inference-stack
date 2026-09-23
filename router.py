"""Prompt-bucket routing -- kept in spirit with your original LM Studio
script, minus anything that talked to LM Studio's REST API."""
import json
import os
import re
from typing import Any, Dict

from config import Paths

# Whole words only, with the inflections each term actually takes. The original
# checked `term in text` as a substring, so "no" matched know/now/another/cannot,
# "what" matched whatever, "cli" matched client, "plan" matched explanation --
# each of which sent a prompt to the 0.5B draft instead of the 30B -- while
# "profile" failed to match "profiling". tests/test_router.py pins all of these.
# The second group of terms was added after a live session: "Now change that
# function so it removes duplicates from the merged result" -- a follow-up to a
# code request -- came back as "quick", because none of the words above appear in
# it and it is 12 words long. The list errs towards matching on purpose: "code"
# really means "do not hand this to the 0.5B draft", so a false positive costs a
# label, while a miss costs an answer. Words that are ordinary English as often
# as they are technical (class, test, library, write) are still left out, so a
# yoga class does not get routed as code.
_CODE = re.compile(
    r"\b(?:debug\w*|code|codes|coding|codebase|fix|fixes|fixed|fixing|bugs?|buggy"
    r"|errors?|trace|traces|traced|tracing|traceback|optimi[sz]\w*|benchmark\w*"
    r"|profil(?:e|es|ed|er|ers|ing)|cli|apis?|python|rust|cuda|gpus?"
    r"|functions?|refactor\w*|implement\w*|compil\w*|scripts?|scripting"
    r"|algorithms?|syntax|regexe?s?|regexp|unittest\w*|pytest"
    r"|git|github|docker|kubernetes|sql|json|yaml|xml|html|css|bash"
    r"|javascript|typescript|java|golang|numpy|pandas|pytorch|tensorflow)\b"
    r"|(?<![\w+])c\+\+(?![\w+])"
)
_ANALYSIS = re.compile(
    r"\b(?:plan|plans|planned|planning|design\w*|architect\w*|compar\w*|analy[sz]\w*"
    r"|write|writes|writing|rewrite|summar\w*|explain\w*|explanations?|drafts?|drafting|proposals?)\b"
)
_QUESTION = re.compile(r"\b(?:what|why|who|when|where|yes|no)\b")

QUICK_MAX_WORDS = 12


def prompt_bucket(prompt: str) -> str:
    """Route a prompt: "quick" goes to the 0.5B draft's fast path; every other
    bucket goes to the 30B.

    Code and analysis requests are checked BEFORE the quick rule. Previously the
    quick rule ran first, so the keyword lists never applied to a prompt of 12
    words or fewer -- "Write a Python LRU cache class" was answered by the 0.5B
    model. What still counts as quick is unchanged: short prompts and
    who/what/why-style questions that aren't code or analysis requests.
    """
    text = (prompt or "").strip().lower()
    if not text:
        return "quick"
    word_count = len(text.split())
    if _CODE.search(text):
        return "code"
    if _ANALYSIS.search(text):
        return "analysis"
    if word_count <= QUICK_MAX_WORDS or _QUESTION.search(text):
        return "quick"
    if word_count >= 80:
        return "long"
    return "analysis"


def load_router_state(paths: Paths) -> Dict[str, Any]:
    if not os.path.exists(paths.router_state_path):
        return {"buckets": {}}
    try:
        with open(paths.router_state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"buckets": {}}


def save_router_state(paths: Paths, state: Dict[str, Any]) -> None:
    try:
        with open(paths.router_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass
