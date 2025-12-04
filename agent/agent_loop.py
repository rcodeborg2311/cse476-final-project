import re
from typing import Optional, List
from .api import call_model_chat_completions

# extract a short answer or number
def extract_answer(t: str) -> str:
    if not t:
        return ""
    s = t.strip()
    tag = re.search(r"answer\s*[:=]\s*([-\d\.]+)", s, re.I)
    if tag:
        return tag.group(1)
    nums = re.findall(r"-?\d+\.?\d*", s)
    if nums:
        return nums[0]
    for ln in reversed(s.splitlines()):
        if ln.strip():
            return ln.strip()
    return s

# basic call wrapper for consistency
def safe_call(p: str, sys_msg: str,
              temperature: float = 0.0,
              timeout: int = 8) -> str:
    r = call_model_chat_completions(p, system=sys_msg,
                                    temperature=temperature,
                                    timeout=timeout)
    return extract_answer(r.get("text") or "")

# generic short-answer mode
def call_generic(p: str) -> str:
    return safe_call(p, system="give only the final short answer.", temperature=0.0)

# math single-number mode
def call_math(p: str) -> str:
    return safe_call(p, system="respond with only a single number.", temperature=0.0)

# hidden chain-of-thought reasoning
def call_hidden_thoughts(q: str) -> str:
    prm = (
        f"work through the problem carefully:\n{q}\n\n"
        "think step by step, but do not give a final answer yet."
    )
    sys_msg = "you are reasoning internally. produce detailed thoughts but stop before the answer."
    return safe_call(prm, sys_msg, temperature=0.7)

# convert internal reasoning to final answer
def call_final_answer(r: str, q: str) -> str:
    prm = (
        f"{r}\n\n"
        "using the reasoning above, now provide the final answer to:\n"
        f"{q}\n"
        "return only one word or one number."
    )
    return safe_call(prm, "return only the final answer.", temperature=0.0)

# yes/no correctness check
def call_critique(q: str, g: str) -> str:
    prm = (
        f"question: {q}\n"
        f"proposed answer: {g}\n"
        "is this likely correct? answer only yes or no."
    )
    return safe_call(prm, "reply strictly with yes or no.", temperature=0.0)

# generate next-step thoughts
def call_thought_candidates(q: str, ctx: str, count: int = 3) -> List[str]:
    prm = (
        f"problem: {q}\n"
        f"current reasoning:\n{ctx}\n\n"
        f"list {count} plausible next steps as numbered items."
    )
    r = call_model_chat_completions(prm, "suggest next steps.", temperature=0.7)
    txt = r.get("text") or ""
    return re.findall(r"\d+\.\s*(.+)", txt)

# evaluate usefulness of a candidate step
def call_evaluate_thought(st: str, q: str) -> str:
    prm = (
        f"problem: {q}\n"
        f"proposed step: {st}\n"
        "does this step seem useful? reply with sure, maybe, or impossible."
    )
    return safe_call(prm, "classify usefulness of the step.", temperature=0.0)

# choose best among several answers
def call_vote(c_list: List[str], q: str) -> str:
    if not c_list:
        return ""
    lbl = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(c_list))
    prm = (
        f"{q}\n\n"
        f"possible answers:\n{lbl}\n\n"
        "which option is most likely correct? reply only with the letter."
    )
    ch = safe_call(prm, "pick the best answer.", temperature=0.0).strip()
    idx = ord(ch.upper()[0]) - 65
    return c_list[idx] if 0 <= idx < len(c_list) else c_list[0]

# short tree-of-thoughts search
def solve_tree_of_thoughts(q: str) -> str:
    ctx = ""
    for _ in range(2):
        opts = call_thought_candidates(q, ctx)
        good = []
        for idea in opts:
            v = call_evaluate_thought(idea, q).lower()
            if v in ("sure", "maybe"):
                good.append(idea)
        if not good:
            break
        ctx += "\n" + good[0]
    return call_final_answer(ctx.strip(), q)

# hidden-step reasoning solver
def solve_with_reasoning(q: str) -> str:
    try:
        r = call_hidden_thoughts(q)
        return call_final_answer(r, q)
    except Exception:
        return call_generic(q)

# ensemble sampling solver
def solve_with_ensemble(q: str, d: Optional[str], k: int = 5) -> str:
    g_list = []
    for _ in range(k):
        try:
            r = call_hidden_thoughts(q)
            ans = call_final_answer(r, q)
            g_list.append(ans)
        except Exception:
            pass
    return call_vote(g_list, q)

# main entrypoint
def solve_one(q: str, d: Optional[str] = None) -> str:
    try:
        if d == "coding":
            return call_generic(q)
        if d == "math":
            first = solve_with_reasoning(q)
            crit = call_critique(q, first)
            if crit.lower() == "yes":
                return first
            return solve_tree_of_thoughts(q)
        return solve_with_ensemble(q, d)
    except Exception:
        return call_generic(q)
