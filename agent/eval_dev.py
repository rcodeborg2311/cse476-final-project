import json
import sys
import time
from .agent_loop import solve_one

DEV_PATH = "data/cse476_final_project_dev_data.json"


def norm(s):
    s = str(s or "").strip().lower()
    return " ".join(s.split())


def print_progress(current, total, start_time):
    percent = current / total
    bar_len = 30
    filled = int(percent * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    remaining = (total - current) / rate if rate > 0 else 0

    sys.stdout.write(
        f"\r[{bar}] {percent*100:5.1f}% "
        f"({current}/{total})  "
        f"Elapsed: {elapsed:6.1f}s  "
        f"Left: {remaining:6.1f}s  "
        f"Speed: {rate:4.1f}/s"
    )
    sys.stdout.flush()


def main():
    with open(DEV_PATH, "r") as f:
        dev = json.load(f)

    correct = 0
    total = len(dev)
    start = time.time()

    for i, ex in enumerate(dev, start=1):
        inp = ex["input"]
        gold = ex["output"]
        domain = ex.get("domain")

        try:
            pred = solve_one(inp, domain)
        except Exception as e:
            pred = ""
            print(f"\nError on example {i}: {e}")

        if norm(pred) == norm(gold):
            correct += 1

        print_progress(i, total, start)

    print()
    print(f"Accuracy: {correct}/{total}")


if __name__ == "__main__":
    main()
