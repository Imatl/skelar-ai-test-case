import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
MAX_WORKERS = 5
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "dataset.json"

INTENTS = ["payment_issue", "technical_error", "account_access", "pricing_plan", "refund"]

AGENT_MISTAKES = [
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
]

SCENARIO_TEMPLATES = [
    {"case_type": "successful",             "satisfaction": "satisfied",   "quality_score": 5, "agent_mistakes": [],                                                      "hidden_dissatisfaction": False},
    {"case_type": "successful",             "satisfaction": "satisfied",   "quality_score": 4, "agent_mistakes": [],                                                      "hidden_dissatisfaction": False},
    {"case_type": "successful",             "satisfaction": "neutral",     "quality_score": 3, "agent_mistakes": [],                                                      "hidden_dissatisfaction": False},
    {"case_type": "successful",             "satisfaction": "satisfied",   "quality_score": 4, "agent_mistakes": [],                                                      "hidden_dissatisfaction": False},
    {"case_type": "problematic",            "satisfaction": "neutral",     "quality_score": 3, "agent_mistakes": ["ignored_question"],                                    "hidden_dissatisfaction": False},
    {"case_type": "problematic",            "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["no_resolution"],                                       "hidden_dissatisfaction": False},
    {"case_type": "problematic",            "satisfaction": "neutral",     "quality_score": 3, "agent_mistakes": ["incorrect_info"],                                      "hidden_dissatisfaction": False},
    {"case_type": "conflict",               "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["rude_tone", "no_resolution"],                          "hidden_dissatisfaction": False},
    {"case_type": "conflict",               "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": ["ignored_question", "rude_tone"],                       "hidden_dissatisfaction": False},
    {"case_type": "conflict",               "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["unnecessary_escalation"],                              "hidden_dissatisfaction": False},
    {"case_type": "conflict",               "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": ["incorrect_info", "no_resolution"],                     "hidden_dissatisfaction": False},
    {"case_type": "agent_error",            "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": ["incorrect_info", "ignored_question", "no_resolution"], "hidden_dissatisfaction": False},
    {"case_type": "agent_error",            "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["rude_tone", "incorrect_info"],                         "hidden_dissatisfaction": False},
    {"case_type": "agent_error",            "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": ["no_resolution", "unnecessary_escalation"],             "hidden_dissatisfaction": False},
    {"case_type": "hidden_dissatisfaction", "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["no_resolution"],                                       "hidden_dissatisfaction": True},
    {"case_type": "hidden_dissatisfaction", "satisfaction": "unsatisfied", "quality_score": 3, "agent_mistakes": ["ignored_question"],                                    "hidden_dissatisfaction": True},
    {"case_type": "hidden_dissatisfaction", "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["incorrect_info"],                                      "hidden_dissatisfaction": True},
    {"case_type": "successful",             "satisfaction": "satisfied",   "quality_score": 5, "agent_mistakes": [],                                                      "hidden_dissatisfaction": False},
    {"case_type": "problematic",            "satisfaction": "neutral",     "quality_score": 3, "agent_mistakes": ["unnecessary_escalation"],                              "hidden_dissatisfaction": False},
    {"case_type": "problematic",            "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["ignored_question", "no_resolution"],                   "hidden_dissatisfaction": False},
]

INTENT_DESCRIPTIONS = {
    "payment_issue": "The customer has a problem with a specific payment transaction (charge failed, double charge, wrong amount, payment stuck/pending, card declined during checkout).",
    "technical_error": "The customer experiences a SOFTWARE BUG or PLATFORM MALFUNCTION — NOT a payment failure. Examples: app crashes, pages not loading, buttons not working, dashboard showing wrong data, features broken, notifications not received, PDF export failing, search not working. The issue must be clearly a software/platform bug, NOT related to a payment transaction.",
    "account_access": "The customer cannot log in, forgot password, account locked, 2FA issues, email verification problems, session expiring unexpectedly.",
    "pricing_plan": "The customer has questions about subscription plans, pricing tiers, feature comparisons, upgrade/downgrade options, billing cycle changes.",
    "refund": "The customer wants their money back for a completed transaction — refund request, refund status check, partial refund, refund policy questions.",
}

CASE_DESCRIPTIONS = {
    "successful": "The agent successfully resolves the customer's issue. The interaction is smooth and professional.",
    "problematic": "The agent struggles with the issue. There are communication problems or partial resolution.",
    "conflict": "The interaction becomes tense or confrontational. The customer is clearly frustrated.",
    "agent_error": "The agent makes significant errors during the interaction.",
    "hidden_dissatisfaction": "The customer appears polite but their issue is not actually resolved.",
}


def build_scenario_matrix():
    configs = []
    id_counter = 1
    for intent in INTENTS:
        for tmpl in SCENARIO_TEMPLATES:
            configs.append({"id": id_counter, "intent": intent, **tmpl})
            id_counter += 1
    return configs


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=16))
def call_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a dataset generator. Output only valid JSON, no extra text."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def generate_dialog(scenario):
    mistakes_str = ", ".join(scenario["agent_mistakes"]) if scenario["agent_mistakes"] else "none"

    hidden_note = ""
    if scenario["hidden_dissatisfaction"]:
        hidden_note = (
            "\nIMPORTANT: The customer has HIDDEN DISSATISFACTION. "
            "The customer should formally thank the agent and appear polite at the end, "
            "but their actual problem must NOT be resolved. The customer should NOT explicitly "
            "complain — they just accept the situation and end the conversation politely "
            "despite being unsatisfied. This should be subtle."
        )

    prompt = f"""Generate a realistic customer support chat dialog for an online payment/fintech platform.

SCENARIO PARAMETERS:
- Customer intent: {scenario["intent"]}
- Intent definition: {INTENT_DESCRIPTIONS[scenario["intent"]]}
- Case type: {scenario["case_type"]} — {CASE_DESCRIPTIONS[scenario["case_type"]]}
- Target satisfaction: {scenario["satisfaction"]}
- Agent quality score: {scenario["quality_score"]}/5
- Agent mistakes to include: {mistakes_str}
{hidden_note}

RULES:
- The dialog should have 4-8 message exchanges (8-16 total messages)
- Messages should feel natural and realistic
- Customer messages should vary in tone and formality
- Agent should use a professional support style
- Include specific details (order numbers, error codes, dates) to make it realistic
- The dialog must clearly demonstrate the specified agent mistakes (if any)
- The dialog topic MUST match the intent definition above — do NOT mix intents
- Do NOT include any labels or annotations in the dialog itself

MISTAKE DEFINITIONS (include ONLY if specified above):
- ignored_question: Agent ignores or skips a direct question from the customer
- incorrect_info: Agent provides wrong or misleading information
- rude_tone: Agent is dismissive, condescending, or impatient
- no_resolution: The customer's issue is not resolved by the end
- unnecessary_escalation: Agent escalates when they could have handled it themselves

Return ONLY a valid JSON object with this exact structure:
{{
  "messages": [
    {{"role": "customer", "text": "..."}},
    {{"role": "agent", "text": "..."}}
  ]
}}"""

    content = call_llm(prompt)
    dialog_data = json.loads(content)

    return {
        "id": scenario["id"],
        "messages": dialog_data["messages"],
        "metadata": {
            "scenario_type": scenario["case_type"],
            "has_hidden_dissatisfaction": scenario["hidden_dissatisfaction"],
        },
        "ground_truth": {
            "intent": scenario["intent"],
            "satisfaction": scenario["satisfaction"],
            "quality_score": scenario["quality_score"],
            "agent_mistakes": scenario["agent_mistakes"],
        },
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenario_matrix()
    print(f"Total scenarios: {len(scenarios)}")

    dataset = []
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_dialog, s): s for s in scenarios}
        for future in as_completed(futures):
            scenario = futures[future]
            try:
                dialog = future.result()
                dataset.append(dialog)
                print(f"  Generated dialog id={dialog['id']} "
                      f"(intent={scenario['intent']}, type={scenario['case_type']})")
            except Exception as e:
                errors += 1
                print(f"  ERROR generating dialog {scenario['id']}: {e}")

    dataset.sort(key=lambda d: d["id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(dataset)} dialogs ({errors} errors) -> {OUTPUT_FILE}")

    intent_counts = {}
    case_counts = {}
    sat_counts = {}
    hidden_count = 0
    for d in dataset:
        gt = d["ground_truth"]
        intent_counts[gt["intent"]] = intent_counts.get(gt["intent"], 0) + 1
        case_counts[d["metadata"]["scenario_type"]] = case_counts.get(d["metadata"]["scenario_type"], 0) + 1
        sat_counts[gt["satisfaction"]] = sat_counts.get(gt["satisfaction"], 0) + 1
        if d["metadata"]["has_hidden_dissatisfaction"]:
            hidden_count += 1

    print("\nDistribution:")
    print(f"  Intents: {json.dumps(intent_counts)}")
    print(f"  Case types: {json.dumps(case_counts)}")
    print(f"  Satisfaction: {json.dumps(sat_counts)}")
    print(f"  Hidden dissatisfaction: {hidden_count}")


if __name__ == "__main__":
    main()