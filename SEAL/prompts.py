from labels import STAGE_DEFINITIONS, STRATEGY_DEFINITIONS
from utils import clean_text


EXEC_TOKEN = "<execution>"


def build_stage_prompt(context_text):
    stages = "\n".join(f"- {label}: {definition}" for label, definition in STAGE_DEFINITIONS.items())
    strategies = "\n".join(f"- {label}: {definition}" for label, definition in STRATEGY_DEFINITIONS.items())
    return (
        "First identify the current support stage using the ESC Framework. Then rank the three most suitable, "
        "distinct therapist strategies for the next response. Infer the stage without seeing the next therapist response. "
        "Return only JSON with keys stage, strategy_top_1, strategy_top_2, strategy_top_3.\n\n"
        f"ESC STAGES:\n{stages}\n\nSTRATEGIES:\n{strategies}\n\nDIALOGUE:\n{context_text}"
    )


def build_planner_input(context_text, strategy):
    return f"[CONTEXT]\n{clean_text(context_text)}\n\n[THERAPIST_STRATEGY]\n{clean_text(strategy)}"
