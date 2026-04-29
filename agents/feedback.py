import os
import json
import pandas as pd
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ══════════════════════════════════════════════════════════════════════
# FEEDBACK STORE
# ══════════════════════════════════════════════════════════════════════
# All user feedback is stored in a single JSON file.
# Structure:
# {
#   "Adobe Creative": {
#     "overrides": ["keep", "keep"],
#     "last_updated": "2025-04-29",
#     "user_note": "I use this for freelance work"
#   }
# }

def load_feedback(feedback_path: str) -> dict:
    """
    Loads the feedback store from disk.
    If the file doesn't exist yet, returns an empty dict.
    This happens on first run — perfectly normal.
    """
    if not os.path.exists(feedback_path):
        return {}
    # os.path.exists() checks if a file is present before trying to open it
    # Avoids FileNotFoundError on first run

    with open(feedback_path, "r", encoding="utf-8") as f:
        return json.load(f)
    # json.load() reads a JSON file and converts it to a Python dict
    # Different from json.loads() which reads from a string


def save_feedback(feedback: dict, feedback_path: str) -> None:
    """
    Writes the feedback dict back to disk as formatted JSON.
    Called every time a new override is recorded.
    """
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    # os.path.dirname() extracts the folder from a full path
    # e.g. "data/feedback.json" → "data"
    # exist_ok=True means don't crash if folder already exists

    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)


def record_override(
    merchant: str,
    agent_decision: str,
    user_decision: str,
    user_note: str,
    feedback_path: str
) -> dict:
    """
    Records a single user override into the feedback store.

    merchant:       which subscription (e.g. "Adobe Creative")
    agent_decision: what the agent said ("cancel", "review", "keep")
    user_decision:  what the user corrected it to ("keep", "cancel", "review")
    user_note:      optional explanation from the user
    feedback_path:  where to save the feedback JSON
    """

    feedback = load_feedback(feedback_path)
    # Load existing feedback first so we don't overwrite previous entries

    if merchant not in feedback:
        # First time we're getting feedback on this merchant — create its entry
        feedback[merchant] = {
            "overrides":      [],
            "agent_decisions": [],
            "last_updated":   None,
            "user_note":      ""
        }

    # Append this override to the history list
    feedback[merchant]["overrides"].append(user_decision)
    # We keep a full history so we can spot patterns
    # e.g. user overrode "cancel" to "keep" 3 times = strong signal

    feedback[merchant]["agent_decisions"].append(agent_decision)
    # Also track what the agent said each time — useful for measuring accuracy

    feedback[merchant]["last_updated"] = datetime.today().strftime("%Y-%m-%d")
    feedback[merchant]["user_note"]    = user_note
    # user_note overwrites each time — we keep the most recent explanation

    save_feedback(feedback, feedback_path)

    print(f"  ✅ Recorded: {merchant} → agent said '{agent_decision}', "
          f"user says '{user_decision}'")

    return feedback


# ══════════════════════════════════════════════════════════════════════
# CONFIDENCE ADJUSTER
# ══════════════════════════════════════════════════════════════════════

def adjust_confidence(
    merchant: str,
    original_confidence: float,
    original_decision: str,
    feedback: dict
) -> tuple:
    """
    Takes the agent's original decision + confidence and adjusts them
    based on accumulated user feedback for that merchant.

    Returns: (adjusted_decision, adjusted_confidence, adjustment_note)

    How the adjustment works:
    - Each override in the same direction adds weight
    - 1 override = small nudge
    - 2+ overrides in same direction = strong signal, decision may flip
    - If user note exists, the LLM reads it to factor in context
    """

    if merchant not in feedback:
        # No feedback for this merchant yet — return original unchanged
        return original_decision, original_confidence, "No user feedback on record."

    merchant_feedback = feedback[merchant]
    overrides         = merchant_feedback.get("overrides", [])
    user_note         = merchant_feedback.get("user_note", "")

    if not overrides:
        return original_decision, original_confidence, "No overrides recorded."

    # Count how many times user chose each option
    keep_count   = overrides.count("keep")
    cancel_count = overrides.count("cancel")
    review_count = overrides.count("review")
    total        = len(overrides)

    # The dominant user preference
    dominant = max(
        [("keep", keep_count), ("cancel", cancel_count), ("review", review_count)],
        key=lambda x: x[1]
        # key=lambda x: x[1] means sort by the count (second element of each tuple)
    )[0]
    # [0] gets just the decision string from the winning tuple

    dominant_ratio = overrides.count(dominant) / total
    # e.g. if user said "keep" 3 out of 4 times, ratio = 0.75

    # Calculate confidence adjustment
    # The more consistently the user overrides, the bigger the adjustment
    if dominant == original_decision:
        # User agrees with the agent — boost confidence
        adjustment = 0.05 * total
        # Each agreement adds 5% confidence, up to a cap
        adjusted_confidence = min(0.99, original_confidence + adjustment)
        adjusted_decision   = original_decision
        note = f"User confirmed '{dominant}' {total}x — confidence boosted."

    else:
        # User disagrees with the agent — reduce confidence and possibly flip decision
        adjustment = 0.15 * total * dominant_ratio
        # Each disagreement reduces confidence by up to 15% × how consistent they are
        adjusted_confidence = max(0.10, original_confidence - adjustment)
        # max(0.10) ensures confidence never drops to zero — we always have some uncertainty

        if adjusted_confidence < 0.50 or total >= 2:
            # If confidence dropped below 50% OR user overrode at least twice
            # → flip the decision to match user preference
            adjusted_decision = dominant
            note = (f"User overrode to '{dominant}' {total}x "
                    f"(ratio: {dominant_ratio:.0%}) — decision flipped.")
        else:
            adjusted_decision = original_decision
            note = (f"User overrode once to '{dominant}' — "
                    f"confidence reduced but decision held.")

    # If user left a note, use LLM to factor it into the reasoning
    if user_note:
        note += f" User note: \"{user_note}\""

    return adjusted_decision, round(adjusted_confidence, 2), note


def apply_feedback_to_report(
    agent_report: pd.DataFrame,
    feedback_path: str
) -> pd.DataFrame:
    """
    Takes the agent's report and returns an adjusted version
    where every subscription's decision and confidence reflects
    accumulated user feedback.

    This is what gets displayed in the dashboard —
    the agent's raw output PLUS user corrections.
    """

    feedback = load_feedback(feedback_path)

    if not feedback:
        # No feedback exists yet — return report as-is
        agent_report["feedback_adjusted"] = False
        agent_report["adjustment_note"]   = "No feedback on record."
        return agent_report

    adjusted_report = agent_report.copy()
    # .copy() so we don't modify the original DataFrame

    adjusted_decisions   = []
    adjusted_confidences = []
    adjustment_notes     = []
    was_adjusted         = []

    for _, row in agent_report.iterrows():
        merchant   = row["merchant"]
        orig_dec   = row["decision"]
        orig_conf  = row["confidence"]

        new_dec, new_conf, note = adjust_confidence(
            merchant, orig_conf, orig_dec, feedback
        )

        adjusted_decisions.append(new_dec)
        adjusted_confidences.append(new_conf)
        adjustment_notes.append(note)
        was_adjusted.append(new_dec != orig_dec or new_conf != orig_conf)
        # True if anything changed, False if feedback left it unchanged

    adjusted_report["decision"]          = adjusted_decisions
    adjusted_report["confidence"]        = adjusted_confidences
    adjusted_report["feedback_adjusted"] = was_adjusted
    adjusted_report["adjustment_note"]   = adjustment_notes

    return adjusted_report


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT — interactive demo
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    base_dir      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feedback_path = os.path.join(base_dir, "data", "feedback.json")
    sys.path.insert(0, os.path.join(base_dir, "utils"))
    sys.path.insert(0, os.path.join(base_dir, "agents"))

    from detect_subscriptions import load_transactions, detect_recurring_payments #type:ignore
    from infer_usage import infer_subscription_usage #type:ignore
    from subscription_agent import run_agent_analysis

    print("=" * 60)
    print("  FEEDBACK LOOP — INTERACTIVE DEMO")
    print("=" * 60)

    # Run the full pipeline to get agent decisions
    transactions  = load_transactions(os.path.join(base_dir, "data", "transactions.csv"))
    subscriptions = detect_recurring_payments(transactions)
    usage_report  = infer_subscription_usage(subscriptions, transactions)
    agent_report  = run_agent_analysis(usage_report, transactions)

    print("\n  Agent decisions BEFORE feedback:")
    for _, row in agent_report.iterrows():
        icon = "🔴" if row["decision"] == "cancel" else "🟢"
        print(f"    {icon} {row['merchant']} → {row['decision']} ({row['confidence']:.0%})")

    # Simulate user overriding Adobe Creative from cancel → keep
    print("\n  Simulating user override: Adobe Creative → keep")
    print("  User note: 'I use this for client design work'")
    record_override(
        merchant       = "Adobe Creative",
        agent_decision = "cancel",
        user_decision  = "keep",
        user_note      = "I use this for client design work",
        feedback_path  = feedback_path
    )

    # Simulate a second override on same merchant — strengthens the signal
    print("\n  Simulating second override: Adobe Creative → keep")
    record_override(
        merchant       = "Adobe Creative",
        agent_decision = "cancel",
        user_decision  = "keep",
        user_note      = "I use this for client design work",
        feedback_path  = feedback_path
    )

    # Apply feedback and show adjusted report
    adjusted = apply_feedback_to_report(agent_report, feedback_path)

    print("\n  Agent decisions AFTER feedback:")
    for _, row in adjusted.iterrows():
        icon    = "🔴" if row["decision"] == "cancel" else "🟢"
        changed = " ← ADJUSTED" if row["feedback_adjusted"] else ""
        print(f"    {icon} {row['merchant']} → {row['decision']} "
              f"({row['confidence']:.0%}){changed}")
        if row["feedback_adjusted"]:
            print(f"       Note: {row['adjustment_note']}")

    print("\n  Feedback saved to data/feedback.json")