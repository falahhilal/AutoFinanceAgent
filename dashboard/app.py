import os
import sys
import json
import pandas as pd
import streamlit as st

# Add project root to path so we can import from utils/ and agents/
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "utils"))
sys.path.insert(0, os.path.join(base_dir, "agents"))

from detect_subscriptions import load_transactions, detect_recurring_payments #type:ignore
from infer_usage import infer_subscription_usage  #type:ignore
from subscription_agent import run_agent_analysis    #type:ignore
from action_layer import simulate_ledger, whatif_simulation    #type:ignore
from feedback import load_feedback, record_override, apply_feedback_to_report  #type:ignore

#PAGE CONFIG
st.set_page_config(
    page_title="AutoFinanceAgent",
    page_icon="💳",
    layout="wide"
    # layout="wide" uses the full browser width instead of a narrow centered column
)

# ── PATHS ──────────────────────────────────────────────────────────────────────
TRANSACTIONS_PATH = os.path.join(base_dir, "data", "transactions.csv")
FEEDBACK_PATH     = os.path.join(base_dir, "data", "feedback.json")
LETTERS_DIR       = os.path.join(base_dir, "data", "letters")


# ── PIPELINE RUNNER (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
# @st.cache_data caches the function's return value
# ttl=300 means the cache expires after 300 seconds (5 minutes)
# Without caching, the full pipeline reruns every time the user clicks anything
# With caching, it runs once and reuses the result for 5 minutes
def run_pipeline():
    """
    Runs the full Phase 1→2→3 pipeline and returns the agent report.
    Cached so it doesn't re-call the LLM on every UI interaction.
    """
    transactions  = load_transactions(TRANSACTIONS_PATH)
    subscriptions = detect_recurring_payments(transactions)
    usage_report  = infer_subscription_usage(subscriptions, transactions)
    agent_report  = run_agent_analysis(usage_report, transactions)
    return agent_report, transactions


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.title("💳 AutoFinanceAgent")
st.caption("AI-powered subscription manager — detects, scores, and acts on unused subscriptions")

st.divider()
# st.divider() draws a horizontal line — visual separator


# ── LOAD DATA ──────────────────────────────────────────────────────────────────
with st.spinner("Running pipeline..."):
    # st.spinner() shows a loading animation while the indented block runs
    agent_report, transactions = run_pipeline()

# Apply any stored user feedback on top of agent decisions
feedback     = load_feedback(FEEDBACK_PATH)
final_report = apply_feedback_to_report(agent_report, FEEDBACK_PATH)


# ── TOP METRICS ROW ────────────────────────────────────────────────────────────
# st.columns() divides the page into side-by-side columns
# The list [1,1,1,1] means 4 equal-width columns
col1, col2, col3, col4 = st.columns(4)

total_monthly  = final_report["monthly_cost"].sum()
cancel_report  = final_report[final_report["decision"] == "cancel"]
monthly_saving = cancel_report["monthly_cost"].sum()
annual_saving  = monthly_saving * 12
cancel_count   = len(cancel_report)

with col1:
    st.metric("Total Subscriptions", len(final_report))
    # st.metric() shows a number with a label — designed for KPI cards

with col2:
    st.metric("Monthly Spend", f"{total_monthly:,.0f} PKR")

with col3:
    st.metric("Flagged for Cancellation", cancel_count)

with col4:
    st.metric("Potential Annual Savings", f"{annual_saving:,.0f} PKR",
              delta=f"-{annual_saving:,.0f} PKR",
              delta_color="inverse")
    # delta= shows a change indicator below the main number
    # delta_color="inverse" makes negative values green (saving money is good)

st.divider()


# ── SUBSCRIPTION CARDS ─────────────────────────────────────────────────────────
st.subheader("📋 Subscription Analysis")

for _, row in final_report.iterrows():

    decision = row["decision"]

    # Pick color and icon based on decision
    if decision == "cancel":
        color = "🔴"
        badge = "CANCEL"
        container_color = "#fff0f0"
        # light red background
    elif decision == "review":
        color = "🟡"
        badge = "REVIEW"
        container_color = "#fffbe6"
    else:
        color = "🟢"
        badge = "KEEP"
        container_color = "#f0fff4"

    # Each subscription gets an expander — collapsed by default, expandable for details
    with st.expander(
        f"{color} **{row['merchant']}** — {row['monthly_cost']:,.0f} PKR/month — [{badge}]",
        expanded=(decision == "cancel")
        # Automatically expand cards that are flagged for cancellation
    ):
        # Two columns inside the expander: details on left, actions on right
        left, right = st.columns([2, 1])

        with left:
            st.markdown(f"**Category:** {row['category']}")
            st.markdown(f"**Annual cost:** {row['annual_cost']:,.0f} PKR")
            st.markdown(f"**Days inactive:** {row['days_inactive']} days")
            st.markdown(f"**Usage score:** {row['usage_score']:.2f} / 1.0")
            st.markdown(f"**Unused probability:** {row['unused_probability']:.0%}")

            # Confidence bar
            st.markdown(f"**AI Confidence:** {row['confidence']:.0%}")
            st.progress(float(row["confidence"]))
            # st.progress() draws a filled progress bar — takes a value 0.0 to 1.0

            st.markdown(f"**AI Reasoning:**")
            st.info(row["reasoning"])
            # st.info() shows text in a blue info box — stands out from regular text

            # Show adjustment note if feedback changed this decision
            if row.get("feedback_adjusted"):
                st.warning(f"⚙ Adjusted by feedback: {row['adjustment_note']}")
                # st.warning() shows text in an orange warning box

        with right:
            if decision == "cancel":
                st.markdown("### 💰 Savings")
                st.metric("Per Month", f"{row['monthly_cost']:,.0f} PKR")
                st.metric("Per Year",  f"{row['annual_cost']:,.0f} PKR")

                # Show cancellation letter if it exists
                letter_path = os.path.join(
                    LETTERS_DIR,
                    f"cancel_{row['merchant'].lower().replace(' ', '_')}.txt"
                )
                if os.path.exists(letter_path):
                    with open(letter_path, "r", encoding="utf-8") as f:
                        letter_text = f.read()
                    st.markdown("**📄 Cancellation Letter:**")
                    st.text_area(
                        label="",
                        value=letter_text,
                        height=200,
                        key=f"letter_{row['merchant']}"
                        # key= must be unique for each widget on the page
                    )
                    # st.text_area() shows a scrollable text box — user can copy from it

            # ── FEEDBACK BUTTONS ───────────────────────────────────────────
            st.markdown("**Your Feedback:**")
            st.caption("Override the AI decision if it's wrong")

            # Unique key for each row of feedback widgets
            merchant_key = row["merchant"].replace(" ", "_")

            user_choice = st.radio(
                label="Your decision:",
                options=["keep", "cancel", "review"],
                index=["keep", "cancel", "review"].index(decision),
                # index= sets the default selected option to match current decision
                key=f"radio_{merchant_key}",
                horizontal=True
                # horizontal=True shows options side by side instead of stacked
            )

            user_note = st.text_input(
                label="Optional note (why?):",
                placeholder="e.g. I use this for work",
                key=f"note_{merchant_key}"
            )

            if st.button("Submit Feedback", key=f"btn_{merchant_key}"):
                # st.button() returns True only when clicked
                record_override(
                    merchant       = row["merchant"],
                    agent_decision = decision,
                    user_decision  = user_choice,
                    user_note      = user_note,
                    feedback_path  = FEEDBACK_PATH
                )
                st.success(f"Feedback recorded for {row['merchant']}!")
                # st.success() shows a green success message
                st.cache_data.clear()
                # Clear the pipeline cache so the next interaction reruns with fresh data
                st.rerun()
                # st.rerun() refreshes the entire Streamlit app
                # This makes the adjusted decision appear immediately after feedback

st.divider()


# ── LEDGER SIMULATION ──────────────────────────────────────────────────────────
st.subheader("📊 12-Month Ledger Projection")

ledger = simulate_ledger(final_report, months_ahead=12)

lcol1, lcol2, lcol3 = st.columns(3)

with lcol1:
    st.metric(
        "Current Monthly Avg",
        f"{ledger['current_monthly_avg']:,.0f} PKR"
    )
with lcol2:
    st.metric(
        "Optimized Monthly Avg",
        f"{ledger['optimized_monthly_avg']:,.0f} PKR",
        delta=f"-{ledger['current_monthly_avg'] - ledger['optimized_monthly_avg']:,.0f} PKR",
        delta_color="inverse"
    )
with lcol3:
    st.metric(
        "12-Month Savings",
        f"{ledger['total_savings']:,.0f} PKR",
        delta=f"-{ledger['total_savings']:,.0f} PKR",
        delta_color="inverse"
    )

# Monthly savings bar chart
savings_data = pd.DataFrame({
    "Month":   list(ledger["monthly_breakdown"].keys()),
    "Savings": list(ledger["monthly_breakdown"].values())
})
# This creates a simple two-column DataFrame from the monthly_breakdown dict

st.bar_chart(savings_data.set_index("Month"))
# .set_index("Month") makes the Month column the row labels
# st.bar_chart() draws a bar chart — x axis = index, y axis = values

st.divider()


# ── WHAT-IF SIMULATOR ──────────────────────────────────────────────────────────
st.subheader("🔮 What-If Simulator")
st.caption("Select subscriptions to cancel and see the projected savings instantly")

all_merchants = final_report["merchant"].tolist()

selected = st.multiselect(
    label="Select subscriptions to cancel:",
    options=all_merchants,
    default=final_report[final_report["decision"] == "cancel"]["merchant"].tolist()
    # default= pre-selects the agent's cancel recommendations
)
# st.multiselect() is a dropdown where user can select multiple options

if selected:
    result = whatif_simulation(final_report, selected)

    w1, w2, w3 = st.columns(3)
    with w1:
        st.metric("Monthly Before", f"{result['monthly_before']:,.0f} PKR")
    with w2:
        st.metric("Monthly After",  f"{result['monthly_after']:,.0f} PKR",
                  delta=f"-{result['monthly_savings']:,.0f} PKR",
                  delta_color="inverse")
    with w3:
        st.metric("Annual Savings", f"{result['annual_savings']:,.0f} PKR")

    st.success(result["message"])
else:
    st.info("Select at least one subscription above to simulate cancellation.")

st.divider()


# ── FEEDBACK HISTORY ───────────────────────────────────────────────────────────
st.subheader("🧠 Feedback History")
st.caption("All user overrides on record — this is what the agent learns from")

if not feedback:
    st.info("No feedback recorded yet. Use the subscription cards above to teach the agent.")
else:
    for merchant, data in feedback.items():
        with st.expander(f"📝 {merchant}"):
            st.markdown(f"**Overrides:** {', '.join(data['overrides'])}")
            st.markdown(f"**Last updated:** {data['last_updated']}")
            if data.get("user_note"):
                st.markdown(f"**User note:** {data['user_note']}")

st.divider()
st.caption("AutoFinanceAgent — Phase 5 complete")