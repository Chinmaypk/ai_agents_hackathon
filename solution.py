# run_three_cases.py
from pathlib import Path
from agent_tools import DetectiveTools

def run_hostel(tools: DetectiveTools):
    print("\n=== CASE: hostel_laptop_missing ===")
    print(tools.review_wifi_logs("Hall", "00:30-01:00"))
    print(tools.interrogate_suspect_final("Jaggu the Senior"))

def run_temple(tools: DetectiveTools):
    print("\n=== CASE: temple_donation_box_theft ===")
    print(tools.interview_witness("Ram"))
    print(tools.review_traffic_cctv("Hall", "00:00-00:10"))
    print(tools.interrogate_suspect("Ramesh the Priest"))

if __name__ == "__main__":

    tools = DetectiveTools(case_id="hostel_laptop_missing", match_mode="smart")
    run_hostel(tools)

    tools.set_case("temple_donation_box_theft")
    run_temple(tools)
