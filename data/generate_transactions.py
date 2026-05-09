import pandas as pd
import random
from datetime import datetime, timedelta

# ── SUBSCRIPTION POOL ──────────────────────────────────────────────────────────
# Instead of always using the same 5, we have a pool of 10.
# Each simulation randomly picks 4-6 of them.
# This means every run can have a completely different subscription profile.
SUBSCRIPTION_POOL = [
    {"merchant": "Netflix",         "amount": 999,  "category": "streaming"},
    {"merchant": "Spotify",         "amount": 499,  "category": "music"},
    {"merchant": "GymNation",       "amount": 3000, "category": "fitness"},
    {"merchant": "Adobe Creative",  "amount": 2200, "category": "software"},
    {"merchant": "iCloud Storage",  "amount": 330,  "category": "cloud"},
    {"merchant": "YouTube Premium", "amount": 450,  "category": "streaming"},
    {"merchant": "Headspace",       "amount": 800,  "category": "fitness"},
    {"merchant": "Notion Pro",      "amount": 600,  "category": "software"},
    {"merchant": "Dropbox",         "amount": 750,  "category": "cloud"},
    {"merchant": "Amazon Prime",    "amount": 1200, "category": "streaming"},
]

# ── REGULAR SPENDING POOL ──────────────────────────────────────────────────────
REGULAR_SPENDING = [
    {"merchant": "Careem",            "amount_range": (200, 600),  "category": "transport"},
    {"merchant": "Foodpanda",         "amount_range": (500, 1500), "category": "food_delivery"},
    {"merchant": "Imtiaz Superstore", "amount_range": (1000, 4000),"category": "groceries"},
    {"merchant": "Daraz",             "amount_range": (300, 3000), "category": "shopping"},
    {"merchant": "PTCL Internet",     "amount_range": (2000, 2000),"category": "utilities"},
    {"merchant": "Shell Petrol",      "amount_range": (2000, 5000),"category": "fuel"},
]

# ── USAGE PROFILES ─────────────────────────────────────────────────────────────
# Each profile defines a different kind of user behavior.
# The simulation randomly picks one profile per run.
# This is what makes the AI reach different conclusions each time —
# the same subscription looks used under one profile and unused under another.
USAGE_PROFILES = {
    "homebody": {
        # Stays home, orders food, streams a lot, doesn't go to gym
        "description": "Stays home, heavy food delivery, streaming user",
        "category_weights": {
            "food_delivery": 0.6,   # 60% chance of food delivery transaction per day
            "transport":     0.1,   # rarely uses transport
            "groceries":     0.3,
            "shopping":      0.25,
            "utilities":     0.15,
            "fuel":          0.05,
        }
    },
    "commuter": {
        # Goes to office, uses transport daily, moderate food spend
        "description": "Daily commuter, gym user, moderate spender",
        "category_weights": {
            "food_delivery": 0.2,
            "transport":     0.6,   # high transport usage
            "groceries":     0.35,
            "shopping":      0.15,
            "utilities":     0.15,
            "fuel":          0.45,  # fuels up regularly
        }
    },
    "freelancer": {
        # Works from home, uses software tools, irregular schedule
        "description": "Freelancer, software-heavy, irregular spending",
        "category_weights": {
            "food_delivery": 0.35,
            "transport":     0.2,
            "groceries":     0.25,
            "shopping":      0.4,   # buys equipment/supplies
            "utilities":     0.2,
            "fuel":          0.1,
        }
    },
    "student": {
        # Low spend overall, heavy streaming, no gym
        "description": "Student, low budget, entertainment focused",
        "category_weights": {
            "food_delivery": 0.45,
            "transport":     0.3,
            "groceries":     0.2,
            "shopping":      0.2,
            "utilities":     0.1,
            "fuel":          0.05,
        }
    }
}


def generate_transactions(months: int = 6, seed: int = None) -> pd.DataFrame:
    """
    Generates a randomized transaction history.

    seed: optional random seed for reproducibility
          if None, uses a random seed so every call produces different data
          if set (e.g. seed=42), produces the same data every time — useful for testing
    """

    if seed is not None:
        random.seed(seed)
    # random.seed() sets the starting point for Python's random number generator
    # Same seed = same sequence of random numbers = same output
    # None seed = truly random each time

    # ── Pick this simulation's profile and subscriptions ───────────────────────
    profile_name = random.choice(list(USAGE_PROFILES.keys()))
    profile      = USAGE_PROFILES[profile_name]
    # random.choice() picks one item randomly from a list

    num_subs     = random.randint(4, 7)
    # This run will have between 4 and 7 subscriptions
    active_subs  = random.sample(SUBSCRIPTION_POOL, min(num_subs, len(SUBSCRIPTION_POOL)))
    # random.sample() picks num_subs items WITHOUT replacement
    # (can't pick the same subscription twice)

    # ── Decide which subscriptions are "inactive" for this run ────────────────
    # Between 1 and half the subscriptions will be set as inactive
    # This is what creates the "unused" signal for the AI to detect
    num_inactive = random.randint(1, max(1, len(active_subs) // 2))
    inactive_subs = set(
        s["merchant"] for s in random.sample(active_subs, num_inactive)
    )
    # set() stores unique values — fast to check membership with "in"
    # inactive_subs is now e.g. {"GymNation", "Adobe Creative"}

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=30 * months)

    transactions = []

    # ── Generate subscription payments ────────────────────────────────────────
    for sub in active_subs:
        current_date = start_date
        while current_date <= end_date:
            noise = random.randint(-2, 2)
            payment_date = current_date + timedelta(days=noise)
            transactions.append({
                "date":     payment_date.strftime("%Y-%m-%d"),
                "merchant": sub["merchant"],
                "amount":   sub["amount"],
                "category": sub["category"],
                "type":     "subscription"
            })
            current_date += timedelta(days=30)

    # ── Generate regular spending based on profile ─────────────────────────────
    # For inactive subscriptions, we deliberately reduce related activity
    # so the AI can detect the inactivity signal
    current_date = start_date
    while current_date <= end_date:
        for spend in REGULAR_SPENDING:
            base_prob = profile["category_weights"].get(spend["category"], 0.2)
            # Get this category's probability from the profile
            # If category isn't in profile weights, default to 20%

            # Reduce activity related to inactive subscriptions
            # If this spend category is related to an inactive sub, reduce its probability
            # e.g. GymNation inactive → reduce transport probability
            adjusted_prob = base_prob
            for inactive_merchant in inactive_subs:
                # Find if this inactive merchant affects this spend category
                inactive_sub_data = next(
                    (s for s in SUBSCRIPTION_POOL if s["merchant"] == inactive_merchant),
                    None
                )
                # next() with a generator expression finds the first match
                # Returns None if no match found (the default)

                if inactive_sub_data:
                    if (inactive_sub_data["category"] == "fitness" and
                            spend["category"] == "transport"):
                        adjusted_prob *= 0.3
                        # 70% reduction in transport when gym is inactive
                        # A person not going to gym doesn't commute there

                    elif (inactive_sub_data["category"] == "software" and
                          spend["category"] == "shopping"):
                        adjusted_prob *= 0.4
                        # Reduction in shopping when software sub is unused

            if random.random() < adjusted_prob:
                min_amt, max_amt = spend["amount_range"]
                amount = round(random.uniform(min_amt, max_amt), 2)
                transactions.append({
                    "date":     current_date.strftime("%Y-%m-%d"),
                    "merchant": spend["merchant"],
                    "amount":   amount,
                    "category": spend["category"],
                    "type":     "regular"
                })

        current_date += timedelta(days=1)

    df = pd.DataFrame(transactions)
    df = df.sort_values("date").reset_index(drop=True)

    # Store simulation metadata as a separate small JSON file
    # The dashboard reads this to show what profile was used
    metadata = {
        "profile":       profile_name,
        "description":   profile["description"],
        "active_subs":   [s["merchant"] for s in active_subs],
        "inactive_subs": list(inactive_subs),
        "generated_at":  datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "total_transactions": len(df)
    }

    return df, metadata
    # Now returns TWO values: the DataFrame AND the metadata dict
    # The caller unpacks both: df, metadata = generate_transactions()


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import json

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df, metadata = generate_transactions(months=6)

    output_path   = os.path.join(base_dir, "data", "transactions.csv")
    metadata_path = os.path.join(base_dir, "data", "simulation_metadata.json")

    df.to_csv(output_path, index=False)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(df)} transactions")
    print(f"Profile   : {metadata['profile']} — {metadata['description']}")
    print(f"Active subs ({len(metadata['active_subs'])}): {', '.join(metadata['active_subs'])}")
    print(f"Inactive   ({len(metadata['inactive_subs'])}): {', '.join(metadata['inactive_subs'])}")
    print(f"Saved to {output_path}")