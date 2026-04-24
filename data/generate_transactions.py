import pandas as pd
import random
from datetime import datetime, timedelta

# subscriptions to simulate
# name, monthly cost in PKR, category tag
SUBSCRIPTIONS = [
    {"merchant": "Netflix",          "amount": 999,  "category": "streaming"},
    {"merchant": "Spotify",          "amount": 499,  "category": "music"},
    {"merchant": "GymNation",        "amount": 3000, "category": "fitness"},
    {"merchant": "Adobe Creative",   "amount": 2200, "category": "software"},
    {"merchant": "iCloud Storage",   "amount": 330,  "category": "cloud"},
]

# one-off, irregular purchases, everyday spending
# simulate what the user actually does with their money
REGULAR_SPENDING = [
    {"merchant": "Careem",           "amount_range": (200, 600),  "category": "transport"},
    {"merchant": "Foodpanda",        "amount_range": (500, 1500), "category": "food_delivery"},
    {"merchant": "Imtiaz Superstore","amount_range": (1000, 4000),"category": "groceries"},
    {"merchant": "Daraz",            "amount_range": (300, 3000), "category": "shopping"},
    {"merchant": "PTCL Internet",    "amount_range": (2000, 2000),"category": "utilities"},
    {"merchant": "Shell Petrol",     "amount_range": (2000, 5000),"category": "fuel"},
]


def generate_transactions(months: int = 6) -> pd.DataFrame:
    """
    Builds a fake transaction history covering the last `months` months.
    Returns a pandas DataFrame — basically an in-memory table with rows and columns.
    """

    transactions = []  #collect every transaction as a dict, then convert to DataFrame

    # date window
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30 * months)

    # PART 1: Generate subscription payments
    # each subscription, one payment per month 
    for sub in SUBSCRIPTIONS:
        current_date = start_date

        while current_date <= end_date:
            # small random noise to the day (±2 days) so it looks realistic
            noise = random.randint(-2, 2)
            payment_date = current_date + timedelta(days=noise)

            transactions.append({
                "date":      payment_date.strftime("%Y-%m-%d"),  #as string "YYYY-MM-DD"
                "merchant":  sub["merchant"],
                "amount":    sub["amount"],
                "category":  sub["category"],
                "type":      "subscription"  
            })

            current_date += timedelta(days=30)  # Move to next month

    # PART 2: Generate regular day-to-day spending 
    # for each day, randomly decide if a purchase happens
    # loop through every day and randomly fire off purchases
    current_date = start_date
    while current_date <= end_date:
        for spend in REGULAR_SPENDING:
            # Each merchant has a 30% chance of appearing on any given day
            if random.random() < 0.30:
                min_amt, max_amt = spend["amount_range"]
                amount = round(random.uniform(min_amt, max_amt), 2)

                transactions.append({
                    "date":      current_date.strftime("%Y-%m-%d"),
                    "merchant":  spend["merchant"],
                    "amount":    amount,
                    "category":  spend["category"],
                    "type":      "regular"  
                })

        current_date += timedelta(days=1)  # Move to next day

    # PART 3: Convert list of dicts to DataFrame and sort by date
    df = pd.DataFrame(transactions)
    # most recent transactions appear last, chronological order
    df = df.sort_values("date").reset_index(drop=True)
    # reset_index re-numbers the rows after sorting

    return df


# only runs when we execute this file directly 
# "if __name__ == '__main__'" is a Python convention for "run this as a script".
if __name__ == "__main__":
    df = generate_transactions(months=6)

    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"Generated {len(df)} transactions")
    print(f"Date range: {df['date'].min()}  →  {df['date'].max()}")
    print(f"\nTransaction type breakdown:")
    print(df["type"].value_counts())
    print(f"\nSample (last 10 rows):")
    print(df.tail(10).to_string(index=False))