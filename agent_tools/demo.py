
from pathlib import Path
from agent_tools import DetectiveTools

def main():
    path = Path("easy_cases.agt")
    if not path.exists():
        path = Path("easy_cases_with_aliases.json")
    if not path.exists():
        path = Path("easy_cases.json")

    tools = DetectiveTools(dataset_path=path, case_id="canteen_cashbox_theft", match_mode="smart")
    print(tools.review_traffic_cctv("parking-b", "8:10pm to 8:18 pm"))
    print(tools.check_vehicle_registration("gj05-xy-7788"))
    print(tools.trace_mobile_number("+91 98765 43210"))
    print(tools.interrogate_suspect("Niraj the Volunteer"))
    print("solution:", tools.solution())
    print("summary:", tools.summary())

if __name__ == "__main__":
    main()
