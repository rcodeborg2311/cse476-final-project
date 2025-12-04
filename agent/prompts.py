from agent.api import call_model_chat_completions

r = call_model_chat_completions(
    "What is 2+2? Respond only with a number.",
    system="Return only the final answer.",
    temperature=0.0
)

print("RAW:", r)
