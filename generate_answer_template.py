#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from agent.agent_loop import solve_one

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        return json.load(fp)

def show_progress(i, total, start):
    done = i / total
    bar_len = 30
    filled = int(done * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    elapsed = time.time() - start
    rate = i / elapsed if elapsed > 0 else 0
    left = (total - i) / rate if rate > 0 else 0
    sys.stdout.write(
        f"\r[{bar}] {done*100:5.1f}%  "
        f"({i}/{total})  "
        f"elapsed: {elapsed:6.1f}s  "
        f"left: {left:6.1f}s  "
        f"speed: {rate:4.1f}/s"
    )
    sys.stdout.flush()

def build_answers(questions):
    out = []
    total = len(questions)
    start = time.time()
    for i, q in enumerate(questions, start=1):
        inp = q["input"]
        domain = q.get("domain")
        res = solve_one(inp, domain)
        out.append({"output": res.strip()})
        show_progress(i, total, start)
    print()
    return out

def validate_results(questions, answers):
    if len(questions) != len(answers):
        raise ValueError("length mismatch")
    for idx, ans in enumerate(answers):
        if "output" not in ans:
            raise ValueError(f"missing output at {idx}")
        if not isinstance(ans["output"], str):
            raise TypeError(f"output not string at {idx}")
        if len(ans["output"]) >= 5000:
            raise ValueError(f"output too long at {idx}")

def main():
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)
    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    with OUTPUT_PATH.open("r") as fp:
        saved = json.load(fp)
    validate_results(questions, saved)
    print(f"wrote {len(saved)} answers to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
