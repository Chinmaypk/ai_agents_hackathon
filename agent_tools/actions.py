from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional, Union

# DB stays internal; students never import it
from .db import CaseDB
from .constants import FIELD_ALIAS_CATEGORY

try:  # LangChain is optional; we still offer a fallback error type
    from langchain_core.tools import ToolException  # type: ignore
except Exception:  # pragma: no cover - runtime fallback when langchain_core is absent
    class ToolException(Exception):
        """Fallback error type mirroring langchain_core.tools.ToolException."""
        pass


class DetectiveTools:
    """
    Student-facing API.
    Loads the dataset internally and never reveals scripted tuples or answers.

    Input rules (strict):
    - Names must include a full token (>=4 letters); no prefixes or typos.
    - Timeframes: HH:MM-HH:MM or h[:mm]am/pm-h[:mm]am/pm (overnight allowed).
    - Plates / phones ignore spacing & punctuation.
    """

    _TOOL_METHOD_NAMES: List[str] = [
        "interview_witness",
        "review_traffic_cctv",
        "check_vehicle_registration",
        "collect_evidence",
        "analyze_fingerprints",
        "trace_mobile_number",
        "review_access_logs",
        "review_wifi_logs",
        "check_upi_transactions",
        "interrogate_suspect",
        "interrogate_suspect_final",
        "interrogate_suspect_3rd_degree",
    ]

    def __init__(
        self,
        case_id: str,
        match_mode: str = "smart",
        dataset_path: Union[str, Path] = "agent_tools/config.agt",
        raise_errors: bool = False,
    ):
        self._db = CaseDB.from_file(dataset_path)
        self.case_id = case_id
        self.match_mode = match_mode
        self.raise_errors = raise_errors

    def set_case(self, case_id: str):
        self.case_id = case_id

    # ---------- Core call wrapper (ground-truth safe) ----------
    def _call(self, action_name: str, *, raise_errors: Optional[bool] = None, **kwargs) -> str:
        use_raise = self.raise_errors if raise_errors is None else raise_errors

        def _maybe_error(msg: str) -> str:
            if use_raise:
                raise ToolException(msg)
            return msg

        if not self._db.case_exists(self.case_id):
            return _maybe_error("[error] Unknown case_id.")

        # Validate action exists in catalog
        try:
            expected_order = self._db.input_arg_order(action_name)
        except KeyError:
            return _maybe_error("[error] Unknown action.")

        # Validate action is enabled for this case (do not list enabled actions)
        if action_name not in self._db.actions_for_case(self.case_id):
            return _maybe_error("[error] This action is not available for the current case.")

        # Build ordered args (with alias canonicalization)
        try:
            args_in_order = []
            for name in expected_order:
                v = str(kwargs[name])
                cat = FIELD_ALIAS_CATEGORY.get(name)
                args_in_order.append(self._db.canonicalize(cat, v) if cat else v)
        except KeyError as e:
            # Arg names are not ground-truth; safe to show
            return _maybe_error(f"[error] Missing required argument '{e.args[0]}'. Required args: {expected_order}")

        # 1) Exact match
        resp = self._db.lookup_exact(self.case_id, action_name, args_in_order)
        if resp is not None or self.match_mode == "exact":
            if resp is None:
                # Do NOT print known tuples/examples
                return _maybe_error("[no-match] Inputs not recognized. Check spelling, use full names, and standard time ranges (e.g., '20:10-20:20').")
            return resp

        # 2) Smart (fuzzy) fallback -- do NOT surface debug info
        resp, _dbg = self._db.lookup_fuzzy(self.case_id, action_name, args_in_order)
        if resp is None:
            # Keep generic; no hints that expose candidates
            return _maybe_error("[no-match] Could not confidently match your inputs. Try exact location names and full person names.")
        return resp

    # ---------- Allowed tool functions ----------
    def interview_witness(self, witness_name: str) -> str:
        """Get a short witness statement. Input: full witness name (>=4-letter token, no typos/prefixes)."""
        return self._call("interview_witness", witness_name=witness_name)

    def review_traffic_cctv(self, location: str, timeframe: str) -> str:
        """Traffic/CCTV summary for an area in a time window. Timeframe e.g., '20:10-20:20' or '8pm-8:20pm'."""
        return self._call("review_traffic_cctv", location=location, timeframe=timeframe)

    def check_vehicle_registration(self, vehicle_number: str) -> str:
        """Vehicle/owner lookup by plate; spacing/dashes are ignored."""
        return self._call("check_vehicle_registration", vehicle_number=vehicle_number)

    def collect_evidence(self, location: str, evidence_type: str) -> str:
        """Collect or report physical evidence at a location."""
        return self._call("collect_evidence", location=location, evidence_type=evidence_type)

    def analyze_fingerprints(self, sample_id: str) -> str:
        """Fingerprint analysis outcome for a given sample id."""
        return self._call("analyze_fingerprints", sample_id=sample_id)

    def trace_mobile_number(self, mobile_number: str) -> str:
        """Basic trace (subscriber / last tower) for a mobile number; punctuation ignored."""
        return self._call("trace_mobile_number", mobile_number=mobile_number)

    def review_access_logs(self, facility_or_room: str, timeframe: str) -> str:
        """Badge/door access logs for a room or facility in a time window."""
        return self._call("review_access_logs", facility_or_room=facility_or_room, timeframe=timeframe)

    def review_wifi_logs(self, area: str, timeframe: str) -> str:
        """Devices seen by Wi-Fi near an area in a time window."""
        return self._call("review_wifi_logs", area=area, timeframe=timeframe)

    def check_upi_transactions(self, party_name: str, timeframe: str) -> str:
        """UPI activity summary for a party in a given timeframe."""
        return self._call("check_upi_transactions", party_name=party_name, timeframe=timeframe)

    def interrogate_suspect(self, suspect_name: str) -> str:
        """Question a suspect (may be evasive). Names must match a full token exactly (>=4 letters)."""
        return self._call("interrogate_suspect", suspect_name=suspect_name)

    def interrogate_suspect_final(self, suspect_name: str) -> str:
        """Final interrogation round for a suspect; strict name matching (>=4-letter token)."""
        return self._call("interrogate_suspect_final", suspect_name=suspect_name)

    def interrogate_suspect_3rd_degree(self, suspect_name: str) -> str:
        """Extreme interrogation (case-dependent). Only the true suspect confesses; others may pass out."""
        return self._call("interrogate_suspect_3rd_degree", suspect_name=suspect_name)

    # ---------- LangChain helper ----------
    def as_langchain_tools(self, *, handle_tool_error: bool = True):
        """
        Return the available tools as LangChain StructuredTool instances.
        Requires langchain-core; errors/no-match are surfaced as ToolException.
        """
        try:
            from langchain_core.tools import StructuredTool  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("Install langchain-core>=0.2 to use as_langchain_tools().") from exc

        available_actions = set(getattr(self._db, "actions_catalog", {}).keys())
        tools = []
        for name in self._TOOL_METHOD_NAMES:
            if name not in available_actions:
                continue  # silently skip actions missing from the loaded catalog
            fn = getattr(self, name)

            @functools.wraps(fn)
            def _wrapped(*args, __fn=fn, **kwargs):
                prev = self.raise_errors
                try:
                    self.raise_errors = True
                    return __fn(*args, **kwargs)
                finally:
                    self.raise_errors = prev

            tools.append(
                StructuredTool.from_function(
                    func=_wrapped,
                    name=name,
                    description=(fn.__doc__ or "").strip() or name.replace("_", " "),
                    handle_tool_error=handle_tool_error,
                )
            )
        return tools
