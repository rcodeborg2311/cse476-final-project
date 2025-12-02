import json
from .agent_loop import solve_one

DEV_PATH = "data/cse476_final_project_dev_data.json"

def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def main():
    with open(DEV_PATH, "r") as f:
        dev = json.load(f)

    correct = 0
    total = len(dev)

    for ex in dev:
        inp = ex["input"]
        gold = ex["output"]
        domain = ex.get("domain")

        pred = solve_one(inp, domain)

        if norm(pred) == norm(gold):
            correct += 1

    print(f"Accuracy: {correct}/{total}")

if __name__ == "__main__":
    main()
