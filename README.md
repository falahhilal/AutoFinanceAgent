# AutoFinanceAgent 💳

An autonomous AI agent that analyzes financial transaction history, detects unused subscriptions, and takes action — generating cancellation letters, simulating 12-month savings projections, and learning from user feedback.

**Live Demo:** [autofinanceagent.streamlit.app](https://autofinanceagent.streamlit.app)  
**Built by:** [Falah Hilal](https://github.com/falahhilal)

---

## What It Does

Most people overpay for subscriptions they forgot they have. AutoFinanceAgent solves this by:

1. **Detecting** recurring payments from raw transaction data using statistical pattern recognition — no labels, no manual input
2. **Scoring** each subscription by likelihood of non-use through behavioral proxy analysis
3. **Reasoning** about each subscription using LLaMA 3.3 70B — producing a structured cancel/review/keep decision with confidence score and plain-English explanation
4. **Acting** — generating formal cancellation letters, simulating 12-month financial projections, and running what-if calculations
5. **Learning** — adjusting confidence scores based on user feedback, flipping decisions after consistent overrides

---

## Pipeline Architecture

```
Raw Transactions
      │
      ▼
Recurrence Detector        ← statistical rules, no pre-labeled data
      │
      ▼
Usage Inference Engine     ← behavioral proxy analysis, inactivity scoring
      │
      ▼
LLM Reasoning Layer        ← LLaMA 3.3 70B via Groq, structured JSON output
      │
      ▼
Action Layer               ← cancellation letters, ledger simulation, what-if calculator
      │
      ▼
Feedback Loop              ← user overrides adjust confidence, system learns
      │
      ▼
Streamlit Dashboard        ← live UI with sidebar navigation
```

---

## Features

| Feature | Details |
|---|---|
| Subscription detection | Blind recurrence detection using occurrence count, amount consistency (≤5% CV), 30-day interval |
| Usage inference | Behavioral proxy map per category, inactivity duration, usage score 0.0–1.0 |
| LLM reasoning | LLaMA 3.3 70B, structured JSON output, confidence score, risk label |
| Cancellation letters | Auto-generated per flagged subscription, saved as `.txt` |
| Ledger simulator | 12-month current vs optimized projection |
| What-if calculator | Custom cancellation combinations with instant savings preview |
| Feedback loop | Per-merchant override history, confidence adjustment, decision flipping |
| Simulation profiles | 4 randomized user profiles (homebody, commuter, freelancer, student) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | LLaMA 3.3 70B via Groq API |
| Dashboard | Streamlit |
| Data processing | Pandas |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
AutoFinanceAgent/
├── .github/workflows/
│   └── ci.yml                    # CI pipeline — validates all modules on every push
├── agents/
│   ├── subscription_agent.py     # Phase 3 — LLM reasoning layer
│   ├── action_layer.py           # Phase 4 — letters, ledger, what-if
│   └── feedback.py               # Phase 5 — feedback loop
├── dashboard/
│   └── app.py                    # Streamlit UI with sidebar navigation
├── data/
│   └── generate_transactions.py  # Randomized transaction simulator
├── utils/
│   ├── detect_subscriptions.py   # Recurrence detector
│   └── infer_usage.py            # Usage inference engine
├── Dockerfile
├── requirements.txt
└── .env                          # GROQ_API_KEY (never committed)
```

---

## Run Locally

**With Docker (recommended):**

```bash
git clone https://github.com/falahhilal/AutoFinanceAgent
cd AutoFinanceAgent

# Create .env file with your Groq API key
echo "GROQ_API_KEY=your-key-here" > .env

# Build and run
docker build -t autofinanceagent .
docker run -p 8501:8501 --env-file .env autofinanceagent
```

Open `http://localhost:8501`

**Without Docker:**

```bash
git clone https://github.com/falahhilal/AutoFinanceAgent
cd AutoFinanceAgent

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your-key-here" > .env

streamlit run dashboard/app.py
```

---

## How the Agent Thinks

For each detected subscription, the agent receives:
- Subscription cost and category
- Days since last related activity
- Related transaction count in the last 90 days
- Computed usage score and unused probability
- User's top 5 spending categories for behavioral context

It then reasons in natural language and returns a structured decision:

```json
{
  "merchant": "GymNation",
  "decision": "cancel",
  "confidence": 0.92,
  "reasoning": "No gym-adjacent transactions in 90 days while spending is concentrated in food delivery — lifestyle inconsistent with gym usage.",
  "monthly_savings": 3000,
  "annual_savings": 36000,
  "risk": "low"
}
```

---

## Dashboard Pages

- **Subscriptions** — AI decision cards with confidence bars, reasoning, cancellation letters, and feedback controls
- **Ledger & Savings** — 12-month projection chart, current vs optimized comparison table
- **What-If Simulator** — pick any combination of subscriptions, see instant savings breakdown
- **Feedback History** — full override log, agent vs user decision comparison

---

## Developer

**Falah Hilal**

[![GitHub](https://img.shields.io/badge/GitHub-falahhilal-181717?logo=github)](https://github.com/falahhilal)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-falahhilal-0077B5?logo=linkedin)](https://www.linkedin.com/in/falahhilal/)
[![Email](https://img.shields.io/badge/Email-falahhilal2018@gmail.com-D14836?logo=gmail)](mailto:falahhilal2018@gmail.com)
[![Phone](https://img.shields.io/badge/Phone-+92_330_3261875-25D366?logo=whatsapp)](tel:+923303261875)