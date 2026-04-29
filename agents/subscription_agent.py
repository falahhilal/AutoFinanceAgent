import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load .env so GROQ_API_KEY is available
load_dotenv()

# Initialize Groq client
# Groq runs open-source models (LLaMA 3) on their own fast hardware for free
client = Groq(api_key=os.environ["GROQ_API_KEY"])
# Groq() works identically to OpenAI's client — same SDK pattern

#PROMPT BUILDER
def build_analysis_prompt(subscription_row: pd.Series, transactions: pd.DataFrame) -> str:
    """
    Builds the full context prompt for a single subscription.
    Gives Gemini everything it needs: cost, usage signals, and spending behavior.
    """

    merchant      = subscription_row["merchant"]
    category      = subscription_row["category"]
    monthly_cost  = subscription_row["monthly_cost"]
    annual_cost   = subscription_row["annual_cost"]
    days_inactive = subscription_row["days_inactive"]
    usage_score   = subscription_row["usage_score"]
    unused_prob   = subscription_row["unused_probability"]
    related_count = subscription_row["related_transaction_count"]

    # Summarize the user's top 5 spending categories for behavioral context
    spending_summary = (
        transactions.groupby("category")["amount"]
        .agg(count="count", total="sum")
        .sort_values("total", ascending=False)
        .head(5)
    )

    spending_lines = []
    for cat, row in spending_summary.iterrows():
        spending_lines.append(
            f"  - {cat}: {int(row['count'])} transactions, {row['total']:,.0f} PKR total"
        )
    spending_context = "\n".join(spending_lines)

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

Then respond with ONLY a valid JSON object in exactly this format, with no other text before or after it:

{{
  "merchant": "{merchant}",
  "decision": "cancel",
  "confidence": 0.0,
  "reasoning": "your reasoning here in one to two sentences",
  "monthly_savings": 0,
  "annual_savings": 0,
  "risk": "low"
}}

Rules for each field:
- "decision" must be exactly one of: cancel, review, keep
- "confidence" must be a float between 0.0 and 1.0
- "reasoning" must be one to two sentences, plain English
- "monthly_savings" must be {monthly_cost} if decision is cancel, else 0
- "annual_savings" must be {annual_cost} if decision is cancel, else 0
- "risk" must be exactly one of: low, medium, high
  - low = very safe to cancel, strong evidence of non-use
  - medium = probably fine but some uncertainty
  - high = cancelling might be a mistake

Return the JSON object only. No markdown code blocks. No explanation. Just the raw JSON."""

    return prompt


#SINGLE SUBSCRIPTION ANALYSIS
def analyze_subscription(subscription_row: pd.Series, transactions: pd.DataFrame) -> dict:
    """
    Sends one subscription's context to Gemini and parses the JSON response.
    Returns a Python dictionary with the AI's decision.
    """

    prompt = build_analysis_prompt(subscription_row, transactions)

    # Send the prompt to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # llama-3.3-70b-versatile = current recommended LLaMA 3.3 model on Groq
        # 70b = 70 billion parameters — smarter than the old 8b model
        # versatile = optimized for a wide range of tasks including structured output
        # Still free tier on Groq
        messages=[
            {
                "role": "system",
                "content": "You are a financial AI agent. You always respond with valid JSON only, no markdown, no explanation."
                # System message sets the model's behavior before the user prompt
                # Separating instructions (system) from data (user) improves reliability
            },
            {
                "role": "user",
                "content": prompt
                # The actual subscription analysis prompt goes here
            }
        ],
        temperature=0.1,
        # temperature controls randomness: 0.0 = deterministic, 1.0 = creative
        # 0.1 keeps responses consistent and focused — important for JSON output
    )

    raw_text = response.choices[0].message.content
    # Groq follows OpenAI's response format exactly
    # response.choices is a list of possible responses (we always want the first)
    # .message.content gives us the text string

    # Gemini sometimes wraps its response in markdown code fences
    # even when we tell it not to. Strip them if present.
    cleaned = raw_text.strip()
    # .strip() removes leading and trailing whitespace and newlines

    if cleaned.startswith("```"):
        # If it starts with ``` (markdown code block), remove the first and last lines
        lines = cleaned.split("\n")
        # split("\n") breaks the string into a list of lines
        # e.g. "```json\n{...}\n```" → ["```json", "{...}", "```"]
        cleaned = "\n".join(lines[1:-1])
        # lines[1:-1] skips the first line (```json) and last line (```)
        # "\n".join() joins them back into a single string

    try:
        result = json.loads(cleaned)
        # json.loads() parses the JSON string into a Python dictionary
    except json.JSONDecodeError as e:
        # If parsing still fails, log what we got and return a safe fallback
        print(f"    ⚠ JSON parse error: {e}")
        print(f"    Raw response was: {raw_text[:200]}")
        result = {
            "merchant":        subscription_row["merchant"],
            "decision":        "review",
            "confidence":      0.5,
            "reasoning":       "Could not parse model response — manual review needed.",
            "monthly_savings": 0,
            "annual_savings":  0,
            "risk":            "medium"
        }

    return result


# FULL ANALYSIS RUN
def run_agent_analysis(usage_report: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Loops through every subscription, runs Gemini analysis on each one,
    and merges the AI decisions back onto the usage report DataFrame.
    """

    agent_results = []
    total = len(usage_report)

    for i, (_, row) in enumerate(usage_report.iterrows()):
        merchant = row["merchant"]
        print(f"  Analyzing {i+1}/{total}: {merchant}...")

        result = analyze_subscription(row, transactions)
        agent_results.append(result)

        print(f"    → {result['decision'].upper()} "
              f"(confidence: {result['confidence']:.0%}, risk: {result['risk']})")
        print(f"    → {result['reasoning']}")

    # Convert list of result dicts into a DataFrame
    agent_df = pd.DataFrame(agent_results)

    # Merge AI decisions onto the usage report using merchant as the join key
    final_df = usage_report.merge(
        agent_df[["merchant", "decision", "confidence", "reasoning", "risk"]],
        on="merchant",
        how="left"
    )

    # Sort by confidence so the most certain decisions appear first
    final_df = final_df.sort_values("confidence", ascending=False)
    final_df = final_df.reset_index(drop=True)

    return final_df


#REPORT PRINTER 
def print_agent_report(final_df: pd.DataFrame) -> None:
    """
    Prints the final AI analysis report to the terminal.
    Shows each subscription with its decision, confidence, reasoning, and savings.
    """

    cancel_df = final_df[final_df["decision"] == "cancel"]
    total_monthly = cancel_df["monthly_savings_if_cancelled"].sum()
    total_annual  = cancel_df["annual_savings_if_cancelled"].sum()

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

        risk_label = str(row.get("risk", "?")).upper()

        print(f"\n  {indicator}  {row['merchant']}  [{risk_label} RISK]")
        print(f"    Cost         : {row['monthly_cost']:,.0f} PKR/month")
        print(f"    Confidence   : {row['confidence']:.0%}")
        print(f"    Days inactive: {row['days_inactive']} days")
        print(f"    Reasoning    : {row['reasoning']}")

        if row["decision"] == "cancel":
            print(f"    💰 Cancel → save {row['annual_savings_if_cancelled']:,.0f} PKR/year")

    print("\n" + "=" * 70)
    print(f"  TOTAL POTENTIAL SAVINGS IF CANCELLED:")
    print(f"    Monthly : {total_monthly:,.0f} PKR")
    print(f"    Annual  : {total_annual:,.0f} PKR")
    print("=" * 70 + "\n")


#ENTRY POINT
if __name__ == "__main__":
    import sys

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base_dir, "utils"))

    from detect_subscriptions import load_transactions, detect_recurring_payments # type: ignore
    from infer_usage import infer_subscription_usage # type: ignore

    transactions_path = os.path.join(base_dir, "data", "transactions.csv")
    agent_output_path = os.path.join(base_dir, "data", "agent_report.csv")

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