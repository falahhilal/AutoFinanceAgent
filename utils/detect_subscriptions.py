import pandas as pd

def load_transactions(filepath: str) -> pd.DataFrame:
    """
    Reads the CSV file from disk into a DataFrame
    Converts the 'date' column from strings into date objects
    """
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])  #string to datetime object
    return df


def detect_recurring_payments(df: pd.DataFrame,
                               min_occurrences: int = 3,
                               interval_days: int = 30,
                               tolerance_days: int = 7) -> pd.DataFrame:
    """
    Scans the transaction history and identifies recurring payments.
    Logic:
    - Group all transactions by merchant name
    - For each merchant that appears multiple times, check if payments happen
      at a regular interval 
    - If a merchant appears ≥3 times AND the average gap between payments
      is within [23, 37] days → it's a subscription
    Parameters:
    - min_occurrences:  how many times a merchant must appear to be considered
    - interval_days:    expected gap between payments (30)
    - tolerance_days:   how much the gap can vary (±7 days)
    """

    subscriptions_found = []  # one dict per detected subscription

    # Group all rows by merchant name
    grouped = df.groupby("merchant")

    for merchant_name, merchant_df in grouped:
        # look at transactions tagged as 'subscription' 
        sub_txns = merchant_df[merchant_df["type"] == "subscription"]

        if len(sub_txns) < min_occurrences:
            # Not enough occurrences to call it recurring — skip
            continue

        # Sort payments by date to measure gaps in order
        sub_txns = sub_txns.sort_values("date")

        gaps = sub_txns["date"].diff().dt.days.dropna()
        avg_gap = gaps.mean()  

        lower_bound = interval_days - tolerance_days 
        upper_bound = interval_days + tolerance_days  

        if lower_bound <= avg_gap <= upper_bound:
            subscriptions_found.append({
                "merchant":       merchant_name,
                "monthly_cost":   sub_txns["amount"].iloc[0],
                # iloc[0] coz all amounts are same for a subscription
                "occurrences":    len(sub_txns),
                "avg_gap_days":   round(avg_gap, 1),
                "category":       sub_txns["category"].iloc[0],
                "first_seen":     sub_txns["date"].min().strftime("%Y-%m-%d"),
                "last_seen":      sub_txns["date"].max().strftime("%Y-%m-%d"),
                "annual_cost":    sub_txns["amount"].iloc[0] * 12,
            })

    # Convert the list of dicts into a DataFrame, sorted by most expensive first
    result_df = pd.DataFrame(subscriptions_found)

    if not result_df.empty:
        result_df = result_df.sort_values("monthly_cost", ascending=False)
        result_df = result_df.reset_index(drop=True)

    return result_df


def print_subscription_report(subscriptions: pd.DataFrame) -> None:
    """
    Prints summary of all detected subscriptions to the terminal
    """
    if subscriptions.empty:
        print("No recurring subscriptions detected.")
        return

    total_monthly = subscriptions["monthly_cost"].sum()
    total_annual  = subscriptions["annual_cost"].sum()

    print("DETECTED SUBSCRIPTIONS")

    for _, row in subscriptions.iterrows():
        # iterrows() loops through the DataFrame one row at a time
        # _ is the index , row is a Series with the row's data
        print(f"\n  {row['merchant']}")
        print(f"    Cost      : {row['monthly_cost']:,.0f} PKR/month  ({row['annual_cost']:,.0f} PKR/year)")
        print(f"    Category  : {row['category']}")
        print(f"    Occurrences: {row['occurrences']}x  (avg every {row['avg_gap_days']} days)")
        print(f"    Active since: {row['first_seen']}")

    print("\n" + "=" * 60)
    print(f"  TOTAL MONTHLY  : {total_monthly:,.0f} PKR")
    print(f"  TOTAL ANNUAL   : {total_annual:,.0f} PKR")

# ENTRY POINT
if __name__ == "__main__":
    df = load_transactions("data/transactions.csv")
    subscriptions = detect_recurring_payments(df)
    print_subscription_report(subscriptions)

    # Save detected subscriptions to their own CSV 
    subscriptions.to_csv("data/subscriptions.csv", index=False)
    print(f"\nSaved to data/subscriptions.csv")