import pandas as pd
from datetime import datetime, timedelta

# BEHAVIORAL PROXY MAP 
# For each subscription category, we define what "usage" looks like
# Logic: if you have a gym membership, evidence of gym usage would be
# things like transport to get there, sports gear purchases, etc.
# If none of those exist then its most likely unused

# "related_categories" = what transaction categories suggest you're using this
# "related_keywords"   = merchant name keywords that also suggest usage
# "inverse_signal"     = if these categories are HIGH, that's evidence AGAINST usage
USAGE_PROXY_MAP = {
    "fitness": {
        "related_categories": ["transport", "fuel"],
        "related_keywords":   ["gym", "sport", "fitness", "health"],
        "inverse_signal":     [],
        "description":        "gym or fitness membership"
    },
    "streaming": {
        "related_categories": ["food_delivery"],
        "related_keywords":   ["netflix", "youtube", "stream", "prime"],
        "inverse_signal":     [],
        "description":        "video streaming service"
    },
    "music": {
        "related_categories": ["transport", "fuel"],
        # People listening to Spotify are usually commuting
        "related_keywords":   ["spotify", "music", "soundcloud"],
        "inverse_signal":     [],
        "description":        "music streaming service"
    },
    "software": {
        "related_categories": [],
        "related_keywords":   ["adobe", "microsoft", "notion", "figma"],
        "inverse_signal":     [],
        "description":        "software subscription"
    },
    "cloud": {
        "related_categories": [],
        "related_keywords":   ["icloud", "google", "dropbox", "drive"],
        "inverse_signal":     [],
        "description":        "cloud storage service"
    },
}

# Default proxy for categories not in the map 
DEFAULT_PROXY = {
    "related_categories": [],
    "related_keywords":   [],
    "inverse_signal":     [],
    "description":        "subscription service"
}

def get_days_since_related_activity(
    df: pd.DataFrame,
    category: str,
    merchant: str,
    lookback_days: int = 90
) -> int:
    """
    Finds how many days have passed since the last transaction
    that could indicate usage of this subscription.

    Returns the number of days since the most recent related transaction.
    Returns lookback_days if no related activity was found at all
    """

    proxy = USAGE_PROXY_MAP.get(category, DEFAULT_PROXY)
    today = pd.Timestamp(datetime.today().date())
    cutoff = today - timedelta(days=lookback_days)
    recent_df = df[df["date"] >= cutoff].copy()

    if recent_df.empty:
        return lookback_days  

    # Condition 1: transaction category matches one of the related categories
    category_mask = recent_df["category"].isin(proxy["related_categories"])

    # Condition 2: merchant name contains one of the related keywords
    # We check the merchant column of ALL transactions 
    keyword_mask = pd.Series([False] * len(recent_df), index=recent_df.index)

    for keyword in proxy["related_keywords"]:
        keyword_mask = keyword_mask | recent_df["merchant"].str.contains(
            keyword, case=False, na=False
        )

    # Combine both conditions
    combined_mask = category_mask | keyword_mask

    # Also exclude transactions FROM the subscription itself (paying doesn't prove you're using it)
    not_self = ~recent_df["merchant"].str.contains(merchant, case=False, na=False)

    related_df = recent_df[combined_mask & not_self]

    if related_df.empty:
        return lookback_days  

    last_related_date = related_df["date"].max()
    days_since = (today - last_related_date).days
    return max(0, days_since)


def calculate_usage_score(
    days_since_activity: int,
    occurrences: int,
    lookback_days: int = 90
) -> float:
    """
    Converts raw signals into a single usage score between 0.0 and 1.0.
    0.0 = definitely not using it
    1.0 = definitely using it

    Formula logic:
    - Recency score: how recently did we see related activity?
      If activity was yesterday then recency = 1.0
      If no activity in 90 days then recency = 0.0
    - Frequency bonus: subscriptions we've seen more often get slight trust boost
      (more data = more confident the detection is correct)
    """
    # days_since=0  → recency = 1.0  (activity today)
    # days_since=90 → recency = 0.0  (no activity in 90 days)
    recency_score = 1.0 - (days_since_activity / lookback_days)
    recency_score = max(0.0, min(1.0, recency_score))
    frequency_bonus = min(0.1, (occurrences / 60))
    usage_score = recency_score + frequency_bonus
    usage_score = max(0.0, min(1.0, usage_score)) 
    return round(usage_score, 2)


def calculate_unused_probability(usage_score: float) -> float:
    """
    Flips the usage score into an 'unused probability'.
    High usage score → low unused probability (keep it)
    Low usage score  → high unused probability (cancel it)
    """
    return round(1.0 - usage_score, 2)


def infer_subscription_usage(
    subscriptions: pd.DataFrame,
    transactions: pd.DataFrame,
    lookback_days: int = 90
) -> pd.DataFrame:
    """
    Takes the subscription list and transaction history,
    and returns an enriched DataFrame with usage signals for every subscription.
    """
    results = []

    for _, sub in subscriptions.iterrows():
        merchant  = sub["merchant"]
        category  = sub["category"]
        monthly_cost = sub["monthly_cost"]

        # Signal 1: how many days since we saw related activity?
        days_inactive = get_days_since_related_activity(
            transactions, category, merchant, lookback_days
        )

        # Signal 2: how many related transactions exist in the window?
        proxy = USAGE_PROXY_MAP.get(category, DEFAULT_PROXY)
        recent = transactions[
            transactions["date"] >= pd.Timestamp(datetime.today().date()) - timedelta(days=lookback_days)
        ]
        related_count = recent[
            recent["category"].isin(proxy["related_categories"]) |
            recent["merchant"].str.contains(
                "|".join(proxy["related_keywords"]) if proxy["related_keywords"] else "NOMATCH",
                case=False, na=False
            )
        ]
        related_transaction_count = len(related_count)

        # Signal 3: calculate usage score and unused probability
        usage_score = calculate_usage_score(
            days_since_activity=days_inactive,
            occurrences=sub["occurrences"],
            lookback_days=lookback_days
        )
        unused_prob = calculate_unused_probability(usage_score)

        # Signal 4: recommend an action based on unused probability
        if unused_prob >= 0.70:
            recommendation = "cancel"
        elif unused_prob >= 0.40:
            recommendation = "review"
        else:
            recommendation = "keep"

        # Signal 5: what would cancelling this save?
        monthly_savings = monthly_cost
        annual_savings  = monthly_cost * 12

        results.append({
            "merchant":                  merchant,
            "category":                  category,
            "monthly_cost":              monthly_cost,
            "annual_cost":               sub["annual_cost"],
            "days_inactive":             days_inactive,
            "related_transaction_count": related_transaction_count,
            "usage_score":               usage_score,
            "unused_probability":        unused_prob,
            "recommendation":            recommendation,
            "monthly_savings_if_cancelled": monthly_savings,
            "annual_savings_if_cancelled":  annual_savings,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("unused_probability", ascending=False)
        result_df = result_df.reset_index(drop=True)

    return result_df

def print_usage_report(usage_df: pd.DataFrame) -> None:
    """
    detailed usage analysis report to the terminal
    """
    if usage_df.empty:
        print("No subscriptions to analyze.")
        return

    cancel_df = usage_df[usage_df["recommendation"] == "cancel"]
    review_df = usage_df[usage_df["recommendation"] == "review"]
    keep_df   = usage_df[usage_df["recommendation"] == "keep"]

    total_potential_monthly_savings = cancel_df["monthly_savings_if_cancelled"].sum()
    total_potential_annual_savings  = cancel_df["annual_savings_if_cancelled"].sum()

    print("  USAGE ANALYSIS REPORT")

    for _, row in usage_df.iterrows():
        if row["recommendation"] == "cancel":
            indicator = "🔴 CANCEL"
        elif row["recommendation"] == "review":
            indicator = "🟡 REVIEW"
        else:
            indicator = "🟢 KEEP"

        print(f"\n  {indicator}  {row['merchant']}")
        print(f"    Cost              : {row['monthly_cost']:,.0f} PKR/month")
        print(f"    Unused probability: {row['unused_probability']:.0%}")
        print(f"    Usage score       : {row['usage_score']:.2f}")
        print(f"    Days inactive     : {row['days_inactive']} days")
        print(f"    Related activity  : {row['related_transaction_count']} transactions")

        if row["recommendation"] == "cancel":
            print(f"    💰 Cancel → save {row['annual_savings_if_cancelled']:,.0f} PKR/year")

    print(f"  POTENTIAL SAVINGS IF CANCELLED:")
    print(f"    Monthly : {total_potential_monthly_savings:,.0f} PKR")
    print(f"    Annual  : {total_potential_annual_savings:,.0f} PKR")


# ENTRY POINT
if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transactions_path = os.path.join(base_dir, "data", "transactions.csv")
    usage_output_path = os.path.join(base_dir, "data", "usage_report.csv")

    from detect_subscriptions import load_transactions, detect_recurring_payments
    transactions = load_transactions(transactions_path)
    subscriptions = detect_recurring_payments(transactions)
    print(f"Detected {len(subscriptions)} subscriptions")
    usage_report = infer_subscription_usage(subscriptions, transactions)
    print_usage_report(usage_report)
    usage_report.to_csv(usage_output_path, index=False)
    print(f"Saved to {usage_output_path}")