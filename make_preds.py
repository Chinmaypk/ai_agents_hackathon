import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent_tools import DetectiveTools


def build_prompt(case: Dict[str, Any]) -> str:
    suspects = "\n".join(f"- {s}" for s in case.get("suspects", []))
    return (
        "You are a meticulous detective agent. Use the provided tools to gather facts, "
        "stay concise, and minimize tool calls. Only use tools when inputs are valid.\n"
        "Case details:\n"
        f"  - case_id: {case['case_id']}\n"
        f"  - description: {case.get('description', '').strip()}\n"
        f"  - initial_clue: {case.get('initial_clue', '').strip()}\n"
        f"  - suspects:\n{suspects}\n\n"
        "Rules:\n"
        "- Tool input rules: full names (>=4 letters), proper time ranges (HH:MM-HH:MM or h[:mm]am/pm-h[:mm]am/pm), "
        "plates/phones ignore spacing.\n"
        "- Return a final line as: FINAL CULPRIT: <full name from suspects>. "
        "Keep other text minimal.\n"
    )


def extract_culprit(output: str, suspects: List[str]) -> str:
    lines = [l.strip() for l in output.splitlines() if l.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if lower.startswith("final culprit"):
            return line.split(":", 1)[-1].strip()
        if lower.startswith("culprit"):
            return line.split(":", 1)[-1].strip()
    # Fallback: return first suspect-like match
    for s in suspects:
        if s.lower() in output.lower():
            return s
    return output.strip()


def run_case(
    case: Dict[str, Any],
    model_name: str,
    temperature: float,
) -> Dict[str, Any]:
    tools_client = DetectiveTools(case_id=case["case_id"], match_mode="smart", raise_errors=False)
    tools = tools_client.as_langchain_tools()

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are Detective LangChain. Always end with 'FINAL CULPRIT: <name>'.",
    )

    user_msg = HumanMessage(content=build_prompt(case))
    result = agent.invoke({"messages": [user_msg]}, config={"recursion_limit": 30})

    if isinstance(result, dict):
        messages: List[BaseMessage] = result.get("messages", [])  # type: ignore[assignment]
    elif isinstance(result, list):
        messages = result  # type: ignore[assignment]
    else:
        messages = []
    # Extract steps from tool calls in the transcript
    steps: List[Dict[str, Any]] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            if not isinstance(args, dict):
                args = {"input": str(args)}
            steps.append({"action": tc.get("name") or tc.get("tool"), "args": args})

    final_text = ""
    if messages:
        # choose last AI message content as output
        for msg in reversed(messages):
            if msg.type == "ai":
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
    culprit = extract_culprit(final_text.strip(), case.get("suspects", []))
    return {"culprit": culprit, "steps": steps}


def main():
    ap = argparse.ArgumentParser(description="Generate predictions for reported cases using LangChain + Gemini.")
    ap.add_argument("--cases", default="reported_cases.json", help="Path to cases JSON.")
    ap.add_argument("--out", default="preds_gen.json", help="Where to write predictions JSON.")
    ap.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model name.")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set. Export it or add to a .env file.")

    cases_data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    preds: Dict[str, Any] = {}

    for bucket, case_list in (cases_data.get("cases") or {}).items():
        for case in case_list:
            print(f"\n=== Running case: {case['case_id']} (bucket: {bucket}) ===")
            preds[case["case_id"]] = run_case(case, model_name=args.model, temperature=args.temperature)
            time.sleep(60)

    Path(args.out).write_text(json.dumps(preds, indent=2), encoding="utf-8")
    print(f"\nWrote predictions to {args.out}")


if __name__ == "__main__":
    main()
