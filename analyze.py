import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
SEED = 123
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "dataset.json"
OUTPUT_FILE = DATA_DIR / "analysis.json"

ANALYSIS_PROMPT = """You are an expert customer support quality analyst. Analyze the following customer support chat dialog and provide a structured assessment.

DIALOG:
{dialog_text}

Analyze this dialog and determine:

1. **intent** — the primary reason the customer contacted support. Choose ONE from:
   - payment_issue: Problem with a specific payment transaction (charge failed, double charge, wrong amount, payment stuck/pending, card declined during checkout)
   - technical_error: Software bug or platform malfunction (app crash, page not loading, button not working, dashboard error, feature broken, notification missing, export failing). This is about the PLATFORM not working correctly, NOT about a payment failing.
   - account_access: Cannot log in, forgot password, account locked, 2FA issues, email verification, session problems
   - pricing_plan: Questions about subscription plans, pricing tiers, features, upgrade/downgrade, billing cycle
   - refund: Customer wants money back for a completed transaction, refund status, partial refund, refund policy
   - other: Does not fit any of the above

   KEY DISTINCTION: If the customer reports a BUG in the app/website/dashboard (crash, broken feature, UI glitch, wrong data displayed), classify as "technical_error" even if the platform is payment-related. Only use "payment_issue" when the customer has a problem with a specific payment/charge/transaction.

2. **satisfaction** — the customer's real satisfaction level. You MUST use all three levels appropriately:
   - satisfied: Customer explicitly expresses happiness or gratitude AND their problem was FULLY resolved. Both conditions must be met.
   - neutral: The customer's reaction is MIXED or MODERATE. Use "neutral" when ANY of these apply:
     * Issue was only partially resolved (agent helped but didn't fully fix it)
     * Customer acknowledges the response but shows no strong emotion either way
     * Customer says something like "okay", "alright", "I see", "I'll try that" without clear enthusiasm or frustration
     * The interaction is professional but the customer doesn't express clear satisfaction or dissatisfaction
     * Agent quality was mediocre (score 3) — the customer is likely neutral, not satisfied
   - unsatisfied: Customer is clearly frustrated, problem not resolved, negative/angry tone, OR hidden dissatisfaction (polite words but issue unresolved)

   CRITICAL: Approximately 20% of support interactions result in "neutral" satisfaction. If you find yourself rarely choosing "neutral", reconsider — many cases fall between satisfied and unsatisfied. When in doubt between "satisfied" and "neutral", prefer "neutral". When in doubt between "neutral" and "unsatisfied", consider whether the customer showed real frustration or just mild disappointment.

3. **quality_score** — rate the support agent's performance from 1 to 5:
   - 5: Excellent — fast, accurate, empathetic resolution
   - 4: Good — resolved with minor issues
   - 3: Average — partially resolved or slow
   - 2: Poor — significant issues in handling
   - 1: Very poor — major failures

4. **agent_mistakes** — list ONLY mistakes that are clearly present. Be conservative — do NOT flag a mistake unless there is strong evidence. Return an empty list if the agent performed reasonably well.
   - ignored_question: Agent completely SKIPPED a specific, direct question the customer asked. Simply not addressing every minor detail does NOT count — the customer must have asked something explicit that went unanswered.
   - incorrect_info: Agent stated something factually WRONG or gave misleading information. Vague or generic responses do NOT count as incorrect info.
   - rude_tone: Agent used dismissive, condescending, sarcastic, or impatient language. Being formal or brief is NOT rude.
   - no_resolution: The customer's PRIMARY issue was clearly NOT resolved by the end. If the agent provided a working solution, next steps, or the issue was fixed, this does NOT apply — even if the resolution took time.
   - unnecessary_escalation: Agent explicitly transferred or escalated to a supervisor/manager/team when they clearly had the ability and information to resolve it themselves.

Return ONLY a valid JSON object:
{{
  "intent": "...",
  "satisfaction": "...",
  "quality_score": N,
  "agent_mistakes": ["...", "..."]
}}"""


def format_dialog(messages):
    lines = []
    for msg in messages:
        role = "Customer" if msg["role"] == "customer" else "Agent"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


def analyze_dialog(dialog):
    dialog_text = format_dialog(dialog["messages"])
    prompt = ANALYSIS_PROMPT.format(dialog_text=dialog_text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a customer support quality analyst. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        seed=SEED,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    analysis = json.loads(content)

    valid_intents = {"payment_issue", "technical_error", "account_access", "pricing_plan", "refund", "other"}
    valid_satisfactions = {"satisfied", "neutral", "unsatisfied"}
    valid_mistakes = {"ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"}

    if analysis.get("intent") not in valid_intents:
        analysis["intent"] = "other"
    if analysis.get("satisfaction") not in valid_satisfactions:
        analysis["satisfaction"] = "neutral"
    if not isinstance(analysis.get("quality_score"), int) or not 1 <= analysis["quality_score"] <= 5:
        analysis["quality_score"] = 3
    analysis["agent_mistakes"] = [m for m in analysis.get("agent_mistakes", []) if m in valid_mistakes]

    return analysis


def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run generate.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} dialogs from {INPUT_FILE}")

    results = []
    for i, dialog in enumerate(dataset):
        print(f"Analyzing dialog {i + 1}/{len(dataset)} (id={dialog['id']})...")
        try:
            analysis = analyze_dialog(dialog)
            results.append({
                "id": dialog["id"],
                "analysis": analysis,
            })
        except Exception as e:
            print(f"  ERROR analyzing dialog {dialog['id']}: {e}")
            results.append({
                "id": dialog["id"],
                "analysis": {
                    "intent": "other",
                    "satisfaction": "neutral",
                    "quality_score": 3,
                    "agent_mistakes": [],
                    "error": str(e),
                },
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete -> {OUTPUT_FILE}")

    intent_counts = {}
    sat_counts = {}
    for r in results:
        a = r["analysis"]
        intent_counts[a["intent"]] = intent_counts.get(a["intent"], 0) + 1
        sat_counts[a["satisfaction"]] = sat_counts.get(a["satisfaction"], 0) + 1

    print("\nPredicted distribution:")
    print(f"  Intents: {json.dumps(intent_counts)}")
    print(f"  Satisfaction: {json.dumps(sat_counts)}")


if __name__ == "__main__":
    main()
