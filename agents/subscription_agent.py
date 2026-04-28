import os
import json
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file into os.environ
load_dotenv()

# Initialize the Anthropic client, automatically reads key from environment variables
client = Anthropic()


#PROMPT BUILDER
def build_analysis_prompt(subscription_row: pd.Series, transactions: pd.DataFrame) -> str:
    """
    Builds the prompt we send to Claude for a single subscription.
    Gives the model everything it needs to reason well:
    - What the subscription is and what it costs
    - The usage signals we computed in Phase 2
    - Context about what the user actually spends money on
    - A clear instruction for what to output

    subscription_row: one row from the usage_report DataFrame (one subscription)
    transactions: the full transaction history for spending context
    """

    merchant     = subscription_row["merchant"]
    category     = subscription_row["category"]
    monthly_cost = subscription_row["monthly_cost"]
    annual_cost  = subscription_row["annual_cost"]
    days_inactive = subscription_row["days_inactive"]
    usage_score  = subscription_row["usage_score"]
    unused_prob  = subscription_row["unused_probability"]
    related_count = subscription_row["related_transaction_count"]

    # Build a spending summary so Claude understands what this user actually does
    # We take the top 5 categories by transaction count
    # Claude gets behavioral context
    spending_summary = (
        transactions.groupby("category")["amount"]
        .agg(count="count", total="sum")
        # count = number of transactions, total = sum of amounts
        .sort_values("total", ascending=False)
        .head(5)
    )

    # Convert the spending summary to a readable string for the prompt
    spending_lines = []
    for cat, row in spending_summary.iterrows():
        spending_lines.append(
            f"  - {cat}: {int(row['count'])} transactions, {row['total']:,.0f} PKR total"
        )
    spending_context = "\n".join(spending_lines)

    # Build the full prompt
    prompt = f"""You are a financial AI agent analyzing subscription usage for a user in Pakistan.

Your job is to reason carefully about whether a subscription is being used, and decide what action to take.

## Subscription Details
- Merchant: {merchant}
- Category: {category}
- Monthly cost: {monthly_cost:,.0f} PKR
- Annual cost: {annual_cost:,.0f} PKR

## Usage Signals (computed from transaction history)
- Days since last related activity: {days_inactive} days
- Related transaction count (last 90 days): {related_count}
- Computed usage score: {usage_score} (0.0 = definitely unused, 1.0 = definitely used)
- Computed unused probability: {unused_prob} (0.0 = definitely used, 1.0 = definitely unused)

## User's Top Spending Categories
{spending_context}

## Your Task
Reason about whether this subscription is being used. Consider:
1. The inactivity duration — how long since any related activity?
2. The user's spending pattern — does it suggest they would use this service?
3. The cost — is the financial impact significant enough to act on?
4. Any alternative explanations — could the user be using it without leaving transaction traces?

Then respond with ONLY a valid JSON object in exactly this format, no other text:

{{
  "merchant": "{merchant}",
  "decision": "cancel" | "review" | "keep",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one to two sentences explaining your decision>",
  "monthly_savings": <number if decision is cancel, else 0>,
  "annual_savings": <number if decision is cancel, else 0>,
  "risk": "low" | "medium" | "high"
}}

The "risk" field means: how risky is it to cancel this?
- low: very safe to cancel, strong evidence of non-use
- medium: probably fine to cancel but some uncertainty
- high: cancelling might be a mistake, weak evidence

Respond with the JSON object only. No markdown, no explanation outside the JSON."""

    return prompt


#SINGLE SUBSCRIPTION ANALYSIS
def analyze_subscription(subscription_row: pd.Series, transactions: pd.DataFrame) -> dict:
    """
    Sends one subscription to Claude and gets back a structured decision.
    Returns a Python dictionary with Claude's analysis.
    """

    prompt = build_analysis_prompt(subscription_row, transactions)

    # Send the prompt to Claude
    message = client.messages.create(
        model="claude-sonnet-4-5",
        # Always use the latest Sonnet model
        max_tokens=1024,
        # max_tokens caps how long the response can be
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # message.content is a list of content blocks, for a text response, we want the first block's text
    raw_response = message.content[0].text

    # Parse the JSON string into a Python dictionary
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        # If Claude returned something that isn't valid JSON we return a fallback dict so the program doesn't crash
        result = {
            "merchant":       subscription_row["merchant"],
            "decision":       "review",
            "confidence":     0.5,
            "reasoning":      f"Could not parse model response: {raw_response[:100]}",
            "monthly_savings": 0,
            "annual_savings":  0,
            "risk":           "medium"
        }

    return result


#FULL ANALYSIS RUN 
def run_agent_analysis(usage_report: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Claude analysis on every subscription in the usage report.
    Returns a new DataFrame with Claude's decisions appended.
    """
    agent_results = []
    total = len(usage_report)

    for i, (_, row) in enumerate(usage_report.iterrows()):
        merchant = row["merchant"]
        print(f"  Analyzing {i+1}/{total}: {merchant}...")
        result = analyze_subscription(row, transactions)
        agent_results.append(result)
        print(f"    → {result['decision'].upper()} (confidence: {result['confidence']:.0%})")
        print(f"    → {result['reasoning']}")

    # Convert list of dicts to DataFrame
    agent_df = pd.DataFrame(agent_results)

    # Merge Claude's decisions back onto the usage report
    # This gives us one complete row per subscription with ALL signals + AI decision
    final_df = usage_report.merge(
        agent_df[["merchant", "decision", "confidence", "reasoning", "risk"]],
        on="merchant",
        # merge on the merchant column, common key between both DataFrames
        how="left"
        # left join to keep all rows from usage_report even if no match in agent_df
    )

    final_df = final_df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return final_df


#REPORT PRINTER
def print_agent_report(final_df: pd.DataFrame) -> None:
    """
    Prints the final combined report-signals,AI decisions.
    """
    cancel_df = final_df[final_df["decision"] == "cancel"]
    total_monthly_savings = cancel_df["monthly_savings_if_cancelled"].sum()
    total_annual_savings  = cancel_df["annual_savings_if_cancelled"].sum()

    print("\n" + "=" * 70)
    print("  AI AGENT ANALYSIS — SUBSCRIPTION DECISIONS")
    print("=" * 70)

    for _, row in final_df.iterrows():
        if row["decision"] == "cancel":
            indicator = "🔴 CANCEL"
        elif row["decision"] == "review":
            indicator = "🟡 REVIEW"
        else:
            indicator = "🟢 KEEP"

        print(f"\n  {indicator}  {row['merchant']}  [{row.get('risk','?').upper()} RISK]")
        print(f"    Cost        : {row['monthly_cost']:,.0f} PKR/month")
        print(f"    Confidence  : {row['confidence']:.0%}")
        print(f"    Reasoning   : {row['reasoning']}")
        print(f"    Days inactive: {row['days_inactive']} days")

        if row["decision"] == "cancel":
            print(f"    💰 Save: {row['annual_savings_if_cancelled']:,.0f} PKR/year if cancelled")

    print("\n" + "=" * 70)
    print(f"  TOTAL POTENTIAL SAVINGS (cancel decisions only):")
    print(f"    Monthly : {total_monthly_savings:,.0f} PKR")
    print(f"    Annual  : {total_annual_savings:,.0f} PKR")
    print("=" * 70 + "\n")


#ENTRY POINT 
if __name__ == "__main__":
    import sys
    import os

    # Build absolute paths from project root regardless of where script is run from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base_dir, "utils"))
    # sys.path is the list of folders Python searches when you do "import something"
    # inserting utils/ at position 0 means Python checks there first
    # This lets us import from detect_subscriptions and infer_usage

    from detect_subscriptions import load_transactions, detect_recurring_payments # type: ignore
    from infer_usage import infer_subscription_usage # type: ignore

    transactions_path   = os.path.join(base_dir, "data", "transactions.csv")
    agent_output_path   = os.path.join(base_dir, "data", "agent_report.csv")

    print("Loading transactions...")
    transactions = load_transactions(transactions_path)

    print("Detecting subscriptions...")
    subscriptions = detect_recurring_payments(transactions)
    print(f"Found {len(subscriptions)} subscriptions")

    print("Running usage inference...")
    usage_report = infer_subscription_usage(subscriptions, transactions)

    print("\nRunning AI agent analysis...")
    final_df = run_agent_analysis(usage_report, transactions)

    print_agent_report(final_df)

    final_df.to_csv(agent_output_path, index=False)
    print(f"Saved to {agent_output_path}")
