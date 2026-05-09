import os
import sys
import json
import pandas as pd
import streamlit as st

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "utils"))
sys.path.insert(0, os.path.join(base_dir, "agents"))

from detect_subscriptions import load_transactions, detect_recurring_payments      #type: ignore
from infer_usage import infer_subscription_usage   #type: ignore
from subscription_agent import run_agent_analysis  #type: ignore
from action_layer import simulate_ledger, whatif_simulation  #type: ignore
from feedback import load_feedback, record_override, apply_feedback_to_report   #type: ignore
import sys
import os as _os
sys.path.insert(0, _os.path.join(base_dir, "data"))
from generate_transactions import generate_transactions   #type: ignore

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoFinanceAgent",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
    # initial_sidebar_state="expanded" means sidebar is open on first load
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
# st.markdown() with unsafe_allow_html=True lets us inject raw CSS into the page
# We use this to style things Streamlit doesn't expose natively
st.markdown("""
<style>
    /* Hide default Streamlit top decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #2d2d2d;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Page title styling */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }

    /* Metric card override */
    [data-testid="stMetric"] {
        background: #1a1a2e;
        border: 1px solid #2d2d2d;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Subscription card */
    .sub-card {
        background: #1a1a2e;
        border: 1px solid #2d2d2d;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .sub-card-cancel { border-left: 4px solid #ff4b4b; }
    .sub-card-review { border-left: 4px solid #ffa500; }
    .sub-card-keep   { border-left: 4px solid #00c897; }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #2d2d2d;
        text-align: center;
        color: #666666;
        font-size: 0.85rem;
    }
    .footer a {
        color: #888888;
        text-decoration: none;
        margin: 0 0.5rem;
    }
    .footer a:hover { color: #ffffff; }
    .footer-name {
        font-size: 1rem;
        font-weight: 600;
        color: #cccccc;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── PATHS ──────────────────────────────────────────────────────────────────────
TRANSACTIONS_PATH = os.path.join(base_dir, "data", "transactions.csv")
FEEDBACK_PATH     = os.path.join(base_dir, "data", "feedback.json")
LETTERS_DIR       = os.path.join(base_dir, "data", "letters")
METADATA_PATH     = os.path.join(base_dir, "data", "simulation_metadata.json")


# ── PIPELINE (cached) ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def run_pipeline():
    """
    Runs Phase 1 → 2 → 3 once and caches the result for 5 minutes.
    Prevents re-calling the LLM on every single UI interaction.
    """
    transactions  = load_transactions(TRANSACTIONS_PATH)
    subscriptions = detect_recurring_payments(transactions)
    usage_report  = infer_subscription_usage(subscriptions, transactions)
    agent_report  = run_agent_analysis(usage_report, transactions)
    return agent_report, transactions


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # App branding at top of sidebar
    st.markdown("## 💳 AutoFinanceAgent")
    st.markdown("<p style='color:#666; font-size:0.8rem; margin-top:-10px;'>AI Subscription Manager</p>",
                unsafe_allow_html=True)
    st.divider()

    # st.radio() with label_visibility="collapsed" hides the label
    # but the radio options act as our navigation tabs
    st.markdown("**Navigation**")
    page = st.radio(
        label="Navigation",
        options=[
            "📋  Subscriptions",
            "📊  Ledger & Savings",
            "🔮  What-If Simulator",
            "🧠  Feedback History",
        ],
        label_visibility="collapsed"
        # label_visibility="collapsed" hides the "Navigation" label
        # but keeps it accessible for screen readers
    )
    # page now holds the string of whichever option is selected
    # e.g. "📋  Subscriptions" — we'll use this to decide what to render

    st.divider()
    st.markdown("**Simulation**")

    if st.button("🎲 Run New Simulation", use_container_width=True,
                 type="primary"):
        # Generate fresh randomized transaction data
        with st.spinner("Generating new data..."):
            df, metadata = generate_transactions(months=6)
            # Save transactions CSV
            df.to_csv(TRANSACTIONS_PATH, index=False)
            # Save metadata JSON
            import json as _json
            with open(METADATA_PATH, "w") as f:
                _json.dump(metadata, f, indent=2)
        # Clear pipeline cache so it reruns with the new data
        st.cache_data.clear()
        st.success(f"New simulation ready — profile: **{metadata['profile']}**")
        st.rerun()
        # st.rerun() refreshes the whole app with fresh data

    # Show current simulation info if metadata exists
    if os.path.exists(METADATA_PATH):
        import json as _json
        with open(METADATA_PATH) as f:
            meta = _json.load(f)
        st.markdown(f"""
        <div style='font-size:0.75rem; color:#666; margin-top:0.5rem;
                    background:#1a1a2e; border-radius:8px; padding:0.6rem;'>
            <strong style='color:#888'>Current profile:</strong><br>
            {meta['profile'].title()}<br>
            <span style='color:#555'>{meta['description']}</span><br><br>
            <strong style='color:#888'>Generated:</strong><br>
            <span style='color:#555'>{meta['generated_at']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Sidebar footer — developer credit
    st.markdown("""
    <div style='font-size:0.75rem; color:#555; text-align:center; padding-top:0.5rem;'>
        Built by <strong style='color:#888'>Falah Hilal</strong><br>
        <a href='https://github.com/falahhilal' style='color:#555;'>GitHub</a> ·
        <a href='https://www.linkedin.com/in/falahhilal/' style='color:#555;'>LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


# ── LOAD DATA ──────────────────────────────────────────────────────────────────
with st.spinner("Running AI pipeline..."):
    agent_report, transactions = run_pipeline()

feedback     = load_feedback(FEEDBACK_PATH)
final_report = apply_feedback_to_report(agent_report, FEEDBACK_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SUBSCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📋  Subscriptions":

    st.markdown("<div class='page-title'>📋 Subscriptions</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>AI analysis of every detected recurring payment</div>",
                unsafe_allow_html=True)

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    cancel_rows    = final_report[final_report["decision"] == "cancel"]
    monthly_saving = cancel_rows["monthly_cost"].sum()
    annual_saving  = monthly_saving * 12

    with c1:
        st.metric("Total Subscriptions", len(final_report))
    with c2:
        st.metric("Monthly Spend", f"{final_report['monthly_cost'].sum():,.0f} PKR")
    with c3:
        st.metric("Flagged for Cancel", len(cancel_rows))
    with c4:
        st.metric("Potential Annual Saving", f"{annual_saving:,.0f} PKR",
                  delta=f"-{annual_saving:,.0f} PKR", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter bar ────────────────────────────────────────────────────────────
    # Let user filter the subscription list by decision type
    filter_col, _ = st.columns([2, 4])
    with filter_col:
        decision_filter = st.selectbox(
            "Filter by decision:",
            options=["All", "Cancel", "Review", "Keep"],
            index=0
            # index=0 means "All" is selected by default
        )

    # Apply filter
    if decision_filter == "All":
        display_df = final_report
    else:
        display_df = final_report[
            final_report["decision"] == decision_filter.lower()
        ]
    # .lower() converts "Cancel" → "cancel" to match what's stored in the DataFrame

    # ── Subscription cards ────────────────────────────────────────────────────
    for _, row in display_df.iterrows():
        decision = row["decision"]

        if decision == "cancel":
            icon, card_class, badge_color = "🔴", "sub-card-cancel", "#ff4b4b"
        elif decision == "review":
            icon, card_class, badge_color = "🟡", "sub-card-review", "#ffa500"
        else:
            icon, card_class, badge_color = "🟢", "sub-card-keep", "#00c897"

        with st.expander(
            f"{icon} **{row['merchant']}** — {row['monthly_cost']:,.0f} PKR/month",
            expanded=(decision == "cancel")
        ):
            left, right = st.columns([3, 2])

            with left:
                # Decision badge
                st.markdown(
                    f"<span style='background:{badge_color}22; color:{badge_color}; "
                    f"padding:3px 12px; border-radius:20px; font-size:0.8rem; "
                    f"font-weight:600;'>{decision.upper()}</span>",
                    unsafe_allow_html=True
                )
                # The 22 appended to badge_color makes it a semi-transparent background
                # e.g. #ff4b4b22 = red at ~13% opacity

                st.markdown("<br>", unsafe_allow_html=True)

                # Details grid
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Monthly:** {row['monthly_cost']:,.0f} PKR")
                    st.markdown(f"**Annual:** {row['annual_cost']:,.0f} PKR")
                with d2:
                    st.markdown(f"**Days inactive:** {row['days_inactive']}")
                    st.markdown(f"**Usage score:** {row['usage_score']:.2f}")
                    st.markdown(f"**Unused prob:** {row['unused_probability']:.0%}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**AI Confidence**")
                st.progress(float(row["confidence"]),
                            text=f"{row['confidence']:.0%}")
                # text= shows a label next to the progress bar

                st.markdown("**AI Reasoning**")
                st.info(row["reasoning"])

                # Show feedback adjustment note if decision was changed
                if row.get("feedback_adjusted"):
                    st.warning(f"⚙ Feedback adjustment: {row['adjustment_note']}")

            with right:
                # Cancellation savings box
                if decision == "cancel":
                    st.markdown("### 💰 If Cancelled")
                    st.metric("Monthly saving", f"{row['monthly_cost']:,.0f} PKR")
                    st.metric("Annual saving",  f"{row['annual_cost']:,.0f} PKR")

                    # Cancellation letter
                    letter_path = os.path.join(
                        LETTERS_DIR,
                        f"cancel_{row['merchant'].lower().replace(' ', '_')}.txt"
                    )
                    if os.path.exists(letter_path):
                        with open(letter_path, "r", encoding="utf-8") as f:
                            letter_text = f.read()
                        st.markdown("**📄 Cancellation Letter**")
                        st.text_area("", value=letter_text, height=220,
                                     key=f"letter_{row['merchant']}")

                st.markdown("---")

                # Feedback section
                st.markdown("**🧠 Your Feedback**")
                st.caption("Override the AI if it got it wrong")

                mk = row["merchant"].replace(" ", "_")
                # mk = merchant key — used to make widget keys unique per row

                current_index = ["keep", "cancel", "review"].index(
                    decision if decision in ["keep", "cancel", "review"] else "keep"
                )
                user_choice = st.radio(
                    "Your decision:",
                    options=["keep", "cancel", "review"],
                    index=current_index,
                    key=f"radio_{mk}",
                    horizontal=True
                )
                user_note = st.text_input(
                    "Why? (optional)",
                    placeholder="e.g. I use this for work",
                    key=f"note_{mk}"
                )
                if st.button("Submit", key=f"btn_{mk}", use_container_width=True):
                    record_override(
                        merchant       = row["merchant"],
                        agent_decision = decision,
                        user_decision  = user_choice,
                        user_note      = user_note,
                        feedback_path  = FEEDBACK_PATH
                    )
                    st.success("Feedback saved!")
                    st.cache_data.clear()
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LEDGER & SAVINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Ledger & Savings":

    st.markdown("<div class='page-title'>📊 Ledger & Savings</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>12-month projection — current spend vs optimized</div>",
                unsafe_allow_html=True)

    ledger = simulate_ledger(final_report, months_ahead=12)

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Current Monthly Avg",
                  f"{ledger['current_monthly_avg']:,.0f} PKR")
    with m2:
        st.metric("Optimized Monthly Avg",
                  f"{ledger['optimized_monthly_avg']:,.0f} PKR",
                  delta=f"-{ledger['current_monthly_avg'] - ledger['optimized_monthly_avg']:,.0f} PKR",
                  delta_color="inverse")
    with m3:
        st.metric("Total 12-Month Savings",
                  f"{ledger['total_savings']:,.0f} PKR",
                  delta=f"-{ledger['total_savings']:,.0f} PKR",
                  delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Monthly savings chart ─────────────────────────────────────────────────
    st.markdown("**Monthly savings if cancellations are applied**")
    savings_df = pd.DataFrame({
        "Month":        list(ledger["monthly_breakdown"].keys()),
        "Savings (PKR)": list(ledger["monthly_breakdown"].values())
    }).set_index("Month")
    st.bar_chart(savings_df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side by side comparison ───────────────────────────────────────────────
    st.markdown("**Subscription breakdown**")

    keep_col, cancel_col = st.columns(2)

    with keep_col:
        st.markdown("✅ **Keeping**")
        kept = final_report[final_report["decision"] != "cancel"][
            ["merchant", "monthly_cost"]
        ]
        for _, r in kept.iterrows():
            st.markdown(f"- {r['merchant']} — {r['monthly_cost']:,.0f} PKR/mo")
        st.markdown(f"**Subtotal: {kept['monthly_cost'].sum():,.0f} PKR/mo**")

    with cancel_col:
        st.markdown("✂ **Cancelling**")
        cxl = final_report[final_report["decision"] == "cancel"][
            ["merchant", "monthly_cost"]
        ]
        if cxl.empty:
            st.markdown("_No cancellations recommended_")
        else:
            for _, r in cxl.iterrows():
                st.markdown(f"- {r['merchant']} — {r['monthly_cost']:,.0f} PKR/mo")
            st.markdown(f"**Subtotal: {cxl['monthly_cost'].sum():,.0f} PKR/mo**")

    # ── 12-month totals table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**12-month cost comparison**")

    comparison_df = pd.DataFrame({
        "Scenario":       ["Current (keep all)", "Optimized (after cancels)", "Savings"],
        "Monthly Avg":    [
            f"{ledger['current_monthly_avg']:,.0f} PKR",
            f"{ledger['optimized_monthly_avg']:,.0f} PKR",
            f"{ledger['current_monthly_avg'] - ledger['optimized_monthly_avg']:,.0f} PKR"
        ],
        "12-Month Total": [
            f"{ledger['total_current_cost']:,.0f} PKR",
            f"{ledger['total_optimized_cost']:,.0f} PKR",
            f"{ledger['total_savings']:,.0f} PKR"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    # use_container_width=True makes table fill the full column width
    # hide_index=True removes the 0,1,2 row numbers


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  What-If Simulator":

    st.markdown("<div class='page-title'>🔮 What-If Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Pick any combination of subscriptions and see the exact savings</div>",
                unsafe_allow_html=True)

    all_merchants = final_report["merchant"].tolist()
    default_cancel = final_report[
        final_report["decision"] == "cancel"
    ]["merchant"].tolist()

    selected = st.multiselect(
        "Select subscriptions to cancel:",
        options=all_merchants,
        default=default_cancel
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if selected:
        result = whatif_simulation(final_report, selected)

        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric("Monthly Before",
                      f"{result['monthly_before']:,.0f} PKR")
        with w2:
            st.metric("Monthly After",
                      f"{result['monthly_after']:,.0f} PKR",
                      delta=f"-{result['monthly_savings']:,.0f} PKR",
                      delta_color="inverse")
        with w3:
            st.metric("Annual Savings",
                      f"{result['annual_savings']:,.0f} PKR")

        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"💰 {result['message']}")

        # Show a breakdown of what's being cancelled
        st.markdown("**Selected for cancellation:**")
        for m in selected:
            row = final_report[final_report["merchant"] == m].iloc[0]
            # .iloc[0] gets the first (and only) matching row as a Series
            st.markdown(
                f"- **{m}** — {row['monthly_cost']:,.0f} PKR/month "
                f"({row['annual_cost']:,.0f} PKR/year)"
            )

        # Remaining subscriptions
        remaining = [m for m in all_merchants if m not in selected]
        # List comprehension: build a new list of merchants NOT in selected
        if remaining:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Remaining after cancellation:**")
            for m in remaining:
                row = final_report[final_report["merchant"] == m].iloc[0]
                st.markdown(f"- {m} — {row['monthly_cost']:,.0f} PKR/month")
    else:
        st.info("Select at least one subscription above to run the simulation.")

    # ── Preset scenarios ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Quick scenarios**")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**Scenario A** — Cancel AI recommendations only")
        if default_cancel:
            r = whatif_simulation(final_report, default_cancel)
            st.metric("Annual saving", f"{r['annual_savings']:,.0f} PKR")
        else:
            st.caption("No cancellations recommended")

    with p2:
        st.markdown("**Scenario B** — Cancel most expensive subscription")
        most_exp = final_report.loc[
            final_report["monthly_cost"].idxmax(), "merchant"
        ]
        r = whatif_simulation(final_report, [most_exp])
        st.metric(f"Cancel {most_exp}", f"{r['annual_savings']:,.0f} PKR/year")

    with p3:
        st.markdown("**Scenario C** — Cancel everything")
        r = whatif_simulation(final_report, all_merchants)
        st.metric("Annual saving", f"{r['annual_savings']:,.0f} PKR")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEEDBACK HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Feedback History":

    st.markdown("<div class='page-title'>🧠 Feedback History</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Every override you've submitted — this is what the agent learns from</div>",
                unsafe_allow_html=True)

    if not feedback:
        st.info("No feedback recorded yet. Go to the Subscriptions page and submit overrides to teach the agent.")
    else:
        # Summary row
        total_overrides = sum(len(v["overrides"]) for v in feedback.values())
        # Generator expression inside sum(): for each merchant's data dict,
        # count how many overrides it has, then sum them all

        f1, f2 = st.columns(2)
        with f1:
            st.metric("Merchants with feedback", len(feedback))
        with f2:
            st.metric("Total overrides recorded", total_overrides)

        st.markdown("<br>", unsafe_allow_html=True)

        for merchant, data in feedback.items():
            overrides      = data.get("overrides", [])
            agent_decisions = data.get("agent_decisions", [])
            last_updated   = data.get("last_updated", "Unknown")
            user_note      = data.get("user_note", "")

            with st.expander(f"📝 {merchant} — {len(overrides)} override(s)"):
                h1, h2 = st.columns(2)

                with h1:
                    st.markdown(f"**Last updated:** {last_updated}")
                    if user_note:
                        st.markdown(f"**Your note:** {user_note}")

                with h2:
                    st.markdown("**Override history:**")
                    for i, (agent_dec, user_dec) in enumerate(
                        zip(agent_decisions, overrides), start=1
                    ):
                        # zip() pairs agent_decisions and overrides together
                        # e.g. zip(["cancel","cancel"], ["keep","keep"])
                        # → [("cancel","keep"), ("cancel","keep")]
                        arrow = "✅" if agent_dec == user_dec else "⚠"
                        st.markdown(
                            f"{i}. Agent said **{agent_dec}** → "
                            f"You said **{user_dec}** {arrow}"
                        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Clear feedback button
        if st.button("🗑 Clear all feedback", type="secondary"):
            if os.path.exists(FEEDBACK_PATH):
                os.remove(FEEDBACK_PATH)
                # os.remove() deletes a file permanently
            st.success("Feedback cleared.")
            st.cache_data.clear()
            st.rerun()


# ── FOOTER ─────────────────────────────────────────────────────────────────────
# Renders on every page
st.markdown("""
<div class='footer'>
    <div class='footer-name'>Developed by Falah Hilal</div>
    <div>
        <a href='https://github.com/falahhilal' target='_blank'>⌥ GitHub</a>
        <a href='https://www.linkedin.com/in/falahhilal/' target='_blank'>💼 LinkedIn</a>
        <a href='mailto:falahhilal2018@gmail.com'>✉ falahhilal2018@gmail.com</a>
        <a href='tel:+923303261875'>📞 +92 330 3261875</a>
    </div>
</div>
""", unsafe_allow_html=True)