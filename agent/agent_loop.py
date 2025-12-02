from .api import call_model_chat_completions
import re

def extract_final(text: str) -> str:
    text = text.strip()
    m = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    return text

def solve_math(prompt: str) -> str:
    system = (
        "You are an expert math solver. "
        "You may reason silently, but you MUST end your output with: FINAL_ANSWER: <answer>"
    )
    r = call_model_chat_completions(prompt, system=system, temperature=0.2)
    return extract_final(r["text"] or "")

def solve_coding(prompt: str) -> str:
    system = (
        "You are an expert Python programmer. "
        "Output ONLY the code that goes inside the function body, "
        "starting with exactly 4 spaces of indent. No comments, no fences."
    )
    r = call_model_chat_completions(prompt, system=system)
    txt = (r["text"] or "").strip()
    lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join("    " + ln.lstrip() for ln in lines)

def solve_generic(prompt: str) -> str:
    system = "Reply only with the final answer. No explanation."
    r = call_model_chat_completions(prompt, system=system)
    return extract_final(r["text"] or "")

def solve_one(question: str, domain: str | None = None) -> str:
    if domain == "math":
        return solve_math(question)
    if domain == "coding":
        return solve_coding(question)
    return solve_generic(question)
