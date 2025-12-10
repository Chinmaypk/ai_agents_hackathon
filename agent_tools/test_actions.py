import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_tools import DetectiveTools
from agent_tools.actions import ToolException


def main():
    tools = DetectiveTools(case_id="canteen_cashbox_theft", match_mode="smart")

    print("== Available tool names ==")
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
        lc_tools = tools.as_langchain_tools()
        print(", ".join(sorted(t.name for t in lc_tools)))
    except Exception as exc:
        print(f"[skip] langchain-core not available ({exc}); falling back to declared list.")
        print(", ".join(sorted(DetectiveTools._TOOL_METHOD_NAMES)))

    print("\n== Happy-path samples ==")
    print("interview_witness:", tools.interview_witness("Nisha"))
    print("review_traffic_cctv:", tools.review_traffic_cctv("Parking B", "20:10-20:20"))

    print("\n== Error path (raise_errors) ==")
    bad = DetectiveTools(case_id="unknown_case", raise_errors=True)
    try:
        bad.interview_witness("Nisha")
    except ToolException as exc:
        print("caught ToolException:", exc)


if __name__ == "__main__":
    main()
