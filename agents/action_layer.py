import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])


#CANCELLATION LETTER GENERATOR

def generate_cancellation_letter(merchant: str, category: str, monthly_cost: float) -> str:
    """
    Uses the LLM to write a formal cancellation request letter
    for a specific subscription.
    """

    prompt = f"""Write a formal, polite cancellation request letter for the following subscription:

- Service name: {merchant}
- Service type: {category}
- Monthly charge: {monthly_cost:,.0f} PKR

The letter should:
- Be addressed to the customer service team of {merchant}
- State clearly that the user wants to cancel their subscription
- Request confirmation of cancellation via email
- Request that no further charges be applied from the next billing cycle
- Be professional and concise — no more than 4 short paragraphs
- End with "Regards," followed by a blank line for the user's name

Return only the letter text. No subject line. No explanation. Just the letter body."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional letter writer. Write clear, formal, concise letters."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3, #so the letters don't all sound identical
    )

    return response.choices[0].message.content.strip()


def generate_all_letters(agent_report: pd.DataFrame) -> dict:
    """
    Generates cancellation letters for every subscription
    the agent recommended cancelling.

    Returns a dictionary: { merchant_name: letter_text }
    """

    # Filter to only the subscriptions marked for cancellation
    cancel_df = agent_report[agent_report["decision"] == "cancel"]

    if cancel_df.empty:
        print("  No cancellations recommended — no letters to generate.")
        return {}

    letters = {}
    # letters is a dict we'll fill

    for _, row in cancel_df.iterrows():
        merchant = row["merchant"]
        print(f"  Generating letter for: {merchant}...")

        letter = generate_cancellation_letter(
            merchant=merchant,
            category=row["category"],
            monthly_cost=row["monthly_cost"]
        )
        letters[merchant] = letter

        # Print a preview
        preview = letter[:100].replace("\n", " ")
        print(f"    Preview: {preview}...")

    return letters


def save_letters(letters: dict, output_dir: str) -> None:
    """
    Saves each cancellation letter as its own .txt file
    in the output directory.

    Why individual files?
    One file per merchant keeps things clean and usable.
    """

    if not letters:
        return

    # Create the output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)
    # exist_ok=True means don't crash if the folder already exists

    for merchant, letter in letters.items():
        # Build a safe filename, replace spaces with underscores, lowercase
        safe_name = merchant.lower().replace(" ", "_")

        filepath = os.path.join(output_dir, f"cancel_{safe_name}.txt")
        # os.path.join builds the full path: "data/letters/cancel_adobe_creative.txt"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"CANCELLATION REQUEST — {merchant.upper()}\n")
            f.write(f"Generated: {datetime.today().strftime('%Y-%m-%d')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(letter)
        # with open(...) as f: opens the file, writes to it, then closes it automatically
        # "w" mode = write (creates file if it doesn't exist, overwrites if it does)
        # encoding="utf-8" handles special characters properly

        print(f"  Saved: {filepath}")


#LEDGER SIMULATOR

def simulate_ledger(agent_report: pd.DataFrame, months_ahead: int = 12) -> dict:
    """
    Simulates the user's subscription expenses for the next N months,
    in two scenarios:
    - Current: keep all subscriptions
    - Optimized: cancel the ones the agent flagged

    Returns a dict with both scenarios and the difference between them.
    months_ahead: how many future months to project (default 12 = one year)
    """

    today = datetime.today()

    # All subscriptions — what you're paying now
    all_subs = agent_report[["merchant", "monthly_cost", "decision"]].copy()

    # Only the ones the agent said to keep or review
    # (cancellations are removed in the optimized scenario)
    kept_subs = agent_report[agent_report["decision"] != "cancel"][["merchant", "monthly_cost"]].copy()

    # Build the monthly projection for both scenarios
    current_schedule   = []  # list of dicts, one per month per subscription
    optimized_schedule = []

    for month_offset in range(months_ahead):
        # Calculate the date for this future month
        future_date = today + timedelta(days=30 * month_offset)
        month_label = future_date.strftime("%Y-%m")
        # strftime("%Y-%m") gives us "2025-05", "2025-06", etc.

        # Current scenario: pay for everything
        for _, sub in all_subs.iterrows():
            current_schedule.append({
                "month":        month_label,
                "merchant":     sub["merchant"],
                "amount":       sub["monthly_cost"],
                "scenario":     "current"
            })

        # Optimized scenario: only pay for kept subscriptions
        for _, sub in kept_subs.iterrows():
            optimized_schedule.append({
                "month":        month_label,
                "merchant":     sub["merchant"],
                "amount":       sub["monthly_cost"],
                "scenario":     "optimized"
            })

    # Convert to DataFrames for easy aggregation
    current_df   = pd.DataFrame(current_schedule)
    optimized_df = pd.DataFrame(optimized_schedule)

    # Calculate monthly totals for each scenario
    current_monthly   = current_df.groupby("month")["amount"].sum()
    optimized_monthly = optimized_df.groupby("month")["amount"].sum()
    # .groupby("month") groups rows by their month label
    # ["amount"].sum() adds up all amounts in each group

    # Calculate savings per month
    monthly_savings = current_monthly - optimized_monthly

    # Summary numbers
    total_current   = current_monthly.sum()
    total_optimized = optimized_monthly.sum()
    total_savings   = total_current - total_optimized

    # Which subscriptions are being cancelled
    cancelled = agent_report[agent_report["decision"] == "cancel"]["merchant"].tolist()
    # .tolist() converts the pandas Series to a plain Python list

    return {
        "months_projected":    months_ahead,
        "cancelled_subs":      cancelled,
        "kept_subs":           kept_subs["merchant"].tolist(),
        "current_monthly_avg": round(total_current / months_ahead, 0),
        "optimized_monthly_avg": round(total_optimized / months_ahead, 0),
        "total_current_cost":  round(total_current, 0),
        "total_optimized_cost": round(total_optimized, 0),
        "total_savings":       round(total_savings, 0),
        "monthly_breakdown":   monthly_savings.to_dict(),
        # .to_dict() converts a pandas Series to { "2025-05": 2530, "2025-06": 2530, ... }
        "current_df":          current_df,
        "optimized_df":        optimized_df,
    }


def print_ledger_report(ledger: dict) -> None:
    """
    Prints the ledger simulation results to the terminal.
    """

    print("\n" + "=" * 70)
    print("  LEDGER SIMULATION — NEXT 12 MONTHS")
    print("=" * 70)

    print(f"\n  Subscriptions being CANCELLED:")
    for m in ledger["cancelled_subs"]:
        print(f"    ✂  {m}")

    print(f"\n  Subscriptions being KEPT:")
    for m in ledger["kept_subs"]:
        print(f"    ✓  {m}")

    print(f"\n  {'Scenario':<30} {'Monthly Avg':>15} {'12-Month Total':>15}")
    print(f"  {'-'*60}")
    # :<30 = left-align in 30 characters
    # :>15 = right-align in 15 characters
    # This creates a clean table layout

    print(f"  {'Current (keep all)':<30} "
          f"{ledger['current_monthly_avg']:>14,.0f} PKR "
          f"{ledger['total_current_cost']:>13,.0f} PKR")

    print(f"  {'Optimized (after cancels)':<30} "
          f"{ledger['optimized_monthly_avg']:>14,.0f} PKR "
          f"{ledger['total_optimized_cost']:>13,.0f} PKR")

    print(f"\n  {'TOTAL SAVINGS':<30} "
          f"{'':>15} "
          f"{ledger['total_savings']:>13,.0f} PKR")

    print("=" * 70 + "\n")


# WHAT-IF CALCULATOR
def whatif_simulation(agent_report: pd.DataFrame, cancel_these: list) -> dict:
    """
    Simulates what happens if you cancel a specific custom list of subscriptions.

    This lets the user ask "what if I only cancel the gym and Adobe?"
    without committing to all of the agent's recommendations.

    cancel_these: a list of merchant name strings to simulate cancelling
                  e.g. ["GymNation", "Adobe Creative"]
    """

    all_monthly = agent_report["monthly_cost"].sum()
    # Total you're paying right now across all subscriptions

    # Find the rows for the subscriptions the user wants to cancel
    cancel_df = agent_report[agent_report["merchant"].isin(cancel_these)]
    # .isin() checks if each merchant value is in our cancel_these list

    if cancel_df.empty:
        return {
            "cancelled":        [],
            "monthly_before":   all_monthly,
            "monthly_after":    all_monthly,
            "monthly_savings":  0,
            "annual_savings":   0,
            "message":          "None of the specified merchants were found."
        }

    cancelled_monthly = cancel_df["monthly_cost"].sum()
    new_monthly       = all_monthly - cancelled_monthly
    annual_savings    = cancelled_monthly * 12

    return {
        "cancelled":         cancel_these,
        "monthly_before":    round(all_monthly, 0),
        "monthly_after":     round(new_monthly, 0),
        "monthly_savings":   round(cancelled_monthly, 0),
        "annual_savings":    round(annual_savings, 0),
        "message":           f"Cancelling {len(cancel_these)} subscription(s) saves "
                             f"{annual_savings:,.0f} PKR/year"
    }


def print_whatif_report(result: dict) -> None:
    """
    Prints the what-if simulation result.
    """

    print("\n" + "─" * 50)
    print("  WHAT-IF SIMULATION")
    print("─" * 50)
    print(f"  Simulating cancellation of: {', '.join(result['cancelled'])}")
    print(f"  Monthly cost before : {result['monthly_before']:,.0f} PKR")
    print(f"  Monthly cost after  : {result['monthly_after']:,.0f} PKR")
    print(f"  Monthly savings     : {result['monthly_savings']:,.0f} PKR")
    print(f"  Annual savings      : {result['annual_savings']:,.0f} PKR")
    print(f"\n  {result['message']}")
    print("─" * 50 + "\n")

# ENTRY POINT
if __name__ == "__main__":

    # Build paths relative to project root
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base_dir, "utils"))

    from detect_subscriptions import load_transactions, detect_recurring_payments #type:ignore
    from infer_usage import infer_subscription_usage   #type: ignore

    # Import the Phase 3 agent runner
    sys.path.insert(0, os.path.join(base_dir, "agents"))
    from subscription_agent import run_agent_analysis

    transactions_path = os.path.join(base_dir, "data", "transactions.csv")
    letters_dir       = os.path.join(base_dir, "data", "letters")

    #Run the full pipeline
    print("=" * 70)
    print("  AUTOFINANCEAGENT — FULL PIPELINE RUN")
    print("=" * 70)

    print("\n[1/4] Loading and detecting subscriptions...")
    transactions  = load_transactions(transactions_path)
    subscriptions = detect_recurring_payments(transactions)
    print(f"  Found {len(subscriptions)} subscriptions")

    print("\n[2/4] Running usage inference...")
    usage_report = infer_subscription_usage(subscriptions, transactions)

    print("\n[3/4] Running AI agent analysis...")
    agent_report = run_agent_analysis(usage_report, transactions)

    # Print what the agent decided
    for _, row in agent_report.iterrows():
        icon = "🔴" if row["decision"] == "cancel" else ("🟡" if row["decision"] == "review" else "🟢")
        print(f"  {icon} {row['merchant']} → {row['decision'].upper()}")

    # Generate cancellation letters
    print("\n[4/4] Generating cancellation letters...")
    letters = generate_all_letters(agent_report)
    save_letters(letters, letters_dir)

    # Ledger simulation 
    print("\nRunning ledger simulation...")
    ledger = simulate_ledger(agent_report, months_ahead=12)
    print_ledger_report(ledger)

    # Save ledger summary as JSON so Phase 5 dashboard can read it
    ledger_summary = {k: v for k, v in ledger.items()
                      if k not in ("current_df", "optimized_df", "monthly_breakdown")}
    # We exclude the DataFrames and the big dict — they don't serialize to JSON cleanly
    # The dashboard will recompute them from the agent_report CSV

    ledger_path = os.path.join(base_dir, "data", "ledger_summary.json")
    with open(ledger_path, "w") as f:
        json.dump(ledger_summary, f, indent=2)
    # json.dump() writes a Python dict to a file as formatted JSON
    # indent=2 makes it human-readable with 2-space indentation
    print(f"  Ledger summary saved to {ledger_path}")

    # What-if simulations 
    print("\nRunning what-if simulations...")

    # Scenario A: cancel only the most expensive flagged subscription
    cancel_decisions = agent_report[agent_report["decision"] == "cancel"]
    if not cancel_decisions.empty:
        most_expensive = cancel_decisions.loc[
            cancel_decisions["monthly_cost"].idxmax(), "merchant"
        ]
        # .idxmax() returns the index of the row with the highest monthly_cost
        # .loc[index, "merchant"] extracts that row's merchant name

        result_a = whatif_simulation(agent_report, [most_expensive])
        print(f"\n  Scenario A — Cancel only most expensive flagged sub:")
        print_whatif_report(result_a)

    # Scenario B: cancel everything the agent flagged
    all_cancel = agent_report[agent_report["decision"] == "cancel"]["merchant"].tolist()
    if all_cancel:
        result_b = whatif_simulation(agent_report, all_cancel)
        print(f"  Scenario B — Cancel all agent recommendations:")
        print_whatif_report(result_b)

    # Scenario C: cancel everything including "review" items
    all_risky = agent_report[
        agent_report["decision"].isin(["cancel", "review"])
    ]["merchant"].tolist()
    if all_risky:
        result_c = whatif_simulation(agent_report, all_risky)
        print(f"  Scenario C — Cancel all cancel + review items:")
        print_whatif_report(result_c)

    print("\nAll outputs saved to data/")