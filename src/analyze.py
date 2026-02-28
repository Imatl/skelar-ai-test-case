import json
import os
import re
import statistics
from collections import Counter
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
VOTING_ROUNDS = 3
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "dataset.json"
OUTPUT_FILE = DATA_DIR / "analysis.json"

VALID_INTENTS = {"payment_issue", "technical_error", "account_access", "pricing_plan", "refund", "other"}
VALID_SATISFACTIONS = {"satisfied", "neutral", "unsatisfied"}
VALID_MISTAKES = {"ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"}

FEW_SHOT_EXAMPLES = """
=== EXAMPLE 1: SATISFIED (resolved, genuine gratitude) ===
Customer: Hi, I was charged twice for order #9912. Can you check?
Agent: Hello! Let me look into that. I can see order #9912 — yes, there are two charges of $45.00 on your account. I've initiated a refund for the duplicate charge. It should appear within 3-5 business days.
Customer: That's great, thank you so much!
Agent: You're welcome! Is there anything else I can help with?
Customer: No, that's all. Thanks again!

REASONING: The customer had a payment issue (double charge). The agent identified the problem quickly and initiated a refund. The customer expressed genuine gratitude and the issue was fully resolved. No agent mistakes.
ANSWER:
{"intent": "payment_issue", "satisfaction": "satisfied", "quality_score": 5, "agent_mistakes": []}

=== EXAMPLE 2: NEUTRAL (partial help, mild disappointment) ===
Customer: The export to PDF feature on my dashboard isn't working. I keep getting a blank page.
Agent: I understand. Have you tried clearing your browser cache?
Customer: Yes, I tried that and also tried a different browser. Same issue. Is this a known bug?
Agent: Let me check. We did have some reports about this. I'll escalate this to our engineering team. You should hear back within 48 hours.
Customer: Okay, I guess I'll wait then.

REASONING: The customer reported a technical error (PDF export broken). The agent acknowledged it but escalated instead of resolving it. The customer's "I guess I'll wait" shows mild disappointment but not anger. The issue is not resolved yet, but the agent provided a path forward. This is neutral — not satisfied (issue unresolved) but not angry either. The escalation was unnecessary as the agent could have provided a workaround.
ANSWER:
{"intent": "technical_error", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": ["unnecessary_escalation"]}

=== EXAMPLE 3: UNSATISFIED (frustrated, unresolved) ===
Customer: I've been trying to log in for 2 hours! I reset my password 3 times and it still doesn't work.
Agent: Please try resetting your password again using the link on the login page.
Customer: I JUST told you I did that 3 times already! Can you actually help me or not?
Agent: I understand your frustration. Unfortunately, password resets are the standard procedure. There's not much else I can do from my end.
Customer: This is ridiculous. I have an important deadline and I can't access my account. Can you at least check if my account is locked?
Agent: I'd recommend waiting 30 minutes and trying again.

REASONING: The customer has an account access issue. The agent ignored the direct question "Can you at least check if my account is locked?" — never addressed it (ignored_question). The agent only repeated the same failed advice and the problem remains unresolved (no_resolution). The customer is clearly frustrated.
ANSWER:
{"intent": "account_access", "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": ["ignored_question", "no_resolution"]}

=== EXAMPLE 4: HIDDEN DISSATISFACTION (polite words, but problem NOT solved) ===
Customer: Hi, I noticed I was charged $99 instead of $49 for my monthly plan. Can you fix this?
Agent: Thank you for reaching out! I can see your account is on the Premium plan at $99/month. This is the correct charge for your current plan.
Customer: But I signed up for the Basic plan at $49. When did it change?
Agent: It looks like the plan was updated on March 15th. If you'd like, I can help you switch back to Basic, but the current charge is valid for the Premium plan.
Customer: I didn't authorize any plan change... Anyway, okay. Can I at least get a refund for this month's difference?
Agent: Unfortunately, charges for the current billing cycle are non-refundable. The switch to Basic would take effect next billing cycle. I've gone ahead and scheduled the downgrade for you.
Customer: Alright, thanks for your help.

REASONING: The customer says "thanks" at the end, but their actual problem is NOT resolved — they were overcharged due to an unauthorized plan change, didn't get a refund, and the agent didn't investigate why the plan changed without authorization. The "thanks" is just politeness, not real satisfaction. The agent provided incorrect info by saying the charge is "valid" when the customer didn't authorize the change. This is HIDDEN DISSATISFACTION — polite tone masking an unresolved issue.
ANSWER:
{"intent": "pricing_plan", "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["incorrect_info", "no_resolution"]}

=== EXAMPLE 5: WHAT IS NOT A MISTAKE (calibration) ===
Customer: My payment of $200 is stuck as pending for 3 days now. Can you check what's going on?
Agent: Let me look into that for you. I can see the transaction — it appears your bank is holding the authorization. This sometimes happens with larger amounts. I've sent a release request from our end, which should clear within 24 hours.
Customer: Is there anything else I can do to speed it up?
Agent: You could also call your bank and ask them to release the hold on their end. That often resolves it faster. Would you like the reference number to give them?
Customer: Yes, please.
Agent: The reference number is TXN-88421. Anything else I can help with?
Customer: No, I think that covers it. I'll call my bank. Thanks.

REASONING: The customer's issue is a pending payment. The agent investigated, took action (sent release request), and provided a concrete next step (call bank with reference number). The customer said "Thanks" and has a clear path forward. Even though the payment isn't instantly resolved, the agent did everything within their power. This is NOT no_resolution — the agent provided a working solution and actionable steps. This is NOT ignored_question — all questions were addressed. Satisfaction is satisfied because the customer has a clear resolution path and expressed thanks genuinely.
ANSWER:
{"intent": "payment_issue", "satisfaction": "satisfied", "quality_score": 4, "agent_mistakes": []}

=== EXAMPLE 6: NO_RESOLUTION vs RESOLVED — critical calibration ===
Customer: I can't download my invoice for order #4420. The button does nothing.
Agent: I'm sorry about that. I've generated the invoice manually and sent it to your email at j.smith@mail.com. Can you check if you received it?
Customer: Yes, got it. But can you also fix the download button?
Agent: I've reported the bug to our engineering team. It should be fixed in the next update. For now, you can always request invoices through support.
Customer: Alright, fair enough. Thanks.

REASONING: The customer's PRIMARY problem was getting the invoice — and the agent SOLVED it by emailing it. The secondary request (fix the button) was acknowledged and reported. The customer said "fair enough, thanks" — genuine acceptance. This IS a resolution. The agent provided a concrete workaround AND escalated the bug appropriately. No mistakes here.
ANSWER:
{"intent": "technical_error", "satisfaction": "satisfied", "quality_score": 4, "agent_mistakes": []}

=== EXAMPLE 7: ACTUAL NO_RESOLUTION ===
Customer: I've been charged $150 but my order was cancelled. Can I get a refund?
Agent: I can see the order was cancelled. For refund requests, you'll need to contact our billing department.
Customer: Can you transfer me or give me their contact info?
Agent: They can be reached during business hours. Is there anything else I can help you with?
Customer: You haven't helped me with anything yet...

REASONING: The customer asked for a refund. The agent redirected to billing without providing contact info, a transfer, or any actionable step. The customer's follow-up question about contact info was partially ignored (no specific contact given). The issue is completely unresolved — the customer is no closer to a refund. This IS no_resolution and unnecessary_escalation (redirected when they could have processed or at least transferred).
ANSWER:
{"intent": "refund", "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": ["no_resolution", "unnecessary_escalation", "ignored_question"]}

=== EXAMPLE 8: UNNECESSARY ESCALATION ===
Customer: I want to switch from the Pro plan ($79/mo) to the Basic plan ($29/mo).
Agent: I understand you'd like to downgrade. Let me transfer you to our billing specialist who handles plan changes.
Customer: Can't you just do it from your end?
Agent: Plan changes need to be processed by our billing team to ensure everything is correct. Please hold while I transfer you.
Customer: Fine...

REASONING: The customer asked for a simple plan downgrade — a routine operation most support agents can handle directly. The agent escalated to a "billing specialist" instead of making the change themselves. The customer even questioned this ("Can't you just do it?") and the agent insisted on transferring. This IS unnecessary_escalation. The issue also remains unresolved at the end of this conversation (no_resolution) since the agent didn't make the change. Satisfaction is unsatisfied — the customer is clearly annoyed ("Fine...").
ANSWER:
{"intent": "pricing_plan", "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": ["unnecessary_escalation", "no_resolution"]}
"""

ANALYSIS_PROMPT = """You are an expert customer support quality analyst.

STUDY THESE REFERENCE EXAMPLES CAREFULLY — they calibrate your judgment:
{few_shot}

---

NOW ANALYZE THIS DIALOG:
{dialog_text}

---

CLASSIFICATION RULES:

**INTENT** — choose the customer's PRIMARY reason for contact:
- payment_issue: Problem with a specific payment/charge/transaction (failed, double charge, wrong amount, pending, declined)
- technical_error: Software bug or platform malfunction (crash, page not loading, broken feature, wrong data displayed). NOT about payments failing.
- account_access: Login problems, password, locked account, 2FA, session issues
- pricing_plan: Subscription plans, pricing, features, upgrade/downgrade, billing cycle
- refund: Wants money back, refund status, refund policy
- other: None of the above

CRITICAL: Classify based on what the customer FIRST asked about. If a customer asks about a refund, it's "refund" even if it relates to a payment. If they ask about pricing/plan changes, it's "pricing_plan" even if money is involved.

**SATISFACTION** — assess the customer's REAL emotional state at the end:
- satisfied: Problem FULLY resolved AND customer shows genuine gratitude/happiness
- neutral: Partial resolution, lukewarm response ("okay", "alright", "I'll try"), no strong emotion
- unsatisfied: Problem NOT resolved, frustration, anger, OR **hidden dissatisfaction**

HIDDEN DISSATISFACTION CHECK — ask yourself these 3 questions:
  1. Was the customer's core problem actually SOLVED by the end? (not just acknowledged)
  2. Did the customer get what they originally asked for?
  3. If you remove the polite words ("thanks", "okay"), is the customer better off than before?
  If answers are NO — the customer is UNSATISFIED regardless of polite language.

**QUALITY SCORE** (1-5):
- 5: Fast, accurate, empathetic, fully resolved
- 4: Resolved with minor delays or imperfections
- 3: Partially resolved, adequate but not impressive
- 2: Significant mistakes, poor handling
- 1: Major failures, hostile, or completely unhelpful

**AGENT MISTAKES** — apply these with HIGH PRECISION (avoid false positives):

- ignored_question: Agent COMPLETELY SKIPPED a specific, direct question. The customer must have explicitly asked something (with "?" or clear request) that the agent never addressed at all. NOT applicable if the agent gave even a partial or indirect answer. NOT applicable if the agent addressed the topic but not in the exact way the customer wanted.

- incorrect_info: Agent stated something FACTUALLY WRONG. The information must be demonstrably false or contradictory to facts in the dialog. Vague answers, generic responses, or unhelpful advice are NOT incorrect info.

- rude_tone: Agent used dismissive, condescending, sarcastic, or hostile language. Being brief, formal, or using templates is NOT rude. The agent must have shown clear disrespect.

- no_resolution: The customer's PRIMARY problem remains UNSOLVED at the end of the conversation.
  IS no_resolution: agent said "contact another department" without transferring or giving details; agent gave only generic advice that doesn't address the specific issue; customer's original request was never fulfilled or started.
  NOT no_resolution: agent provided a workaround that solves the immediate need; agent gave specific actionable steps the customer accepted; agent initiated a fix that just needs time to process (refund in 3-5 days, bug fix deployed, etc.); agent resolved the main issue even if a secondary request remains open.

- unnecessary_escalation: Agent transferred/redirected to another team/manager when they clearly had the information and ability to resolve it themselves.
  IS unnecessary_escalation: agent transfers for routine operations (password reset, plan change, simple refund); agent says "contact billing/another department" for something they could handle.
  NOT unnecessary_escalation: agent escalates a genuine technical bug to engineering; agent transfers to a specialist for a complex compliance/legal issue; simply saying "let me check with the team" is NOT escalation.

RESPOND IN THIS EXACT FORMAT:
REASONING: <2-3 sentences analyzing the dialog>
ANSWER:
{{"intent": "...", "satisfaction": "...", "quality_score": N, "agent_mistakes": [...]}}"""


def format_dialog(messages):
    lines = []
    for msg in messages:
        role = "Customer" if msg["role"] == "customer" else "Agent"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


def extract_json_from_response(content):
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content.strip())

    answer_match = re.search(r'ANSWER:\s*\n?\s*(\{.*\})', content, re.DOTALL)
    if answer_match:
        json_str = answer_match.group(1)
        brace_count = 0
        end_idx = 0
        for i, ch in enumerate(json_str):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        return json.loads(json_str[:end_idx])

    json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', content)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(content)


SYSTEM_PROMPTS = [
    "You are a senior customer support quality analyst with 10 years of experience. You evaluate support interactions with precision. You are especially good at detecting hidden dissatisfaction — when customers use polite words but their problem remains unsolved.",
    "You are an expert customer experience evaluator. You focus on whether the customer's actual problem was solved, not just whether the conversation ended politely. You carefully distinguish between genuine resolution and surface-level politeness.",
    "You are a strict quality auditor for customer support teams. You evaluate each interaction by checking: was the core issue resolved? Did the agent make mistakes? Is the customer truly satisfied or just being polite? You base judgments only on evidence in the dialog.",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=16))
def call_llm(prompt, system_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def validate_analysis(analysis):
    if analysis.get("intent") not in VALID_INTENTS:
        analysis["intent"] = "other"
    if analysis.get("satisfaction") not in VALID_SATISFACTIONS:
        analysis["satisfaction"] = "neutral"
    if not isinstance(analysis.get("quality_score"), int) or not 1 <= analysis["quality_score"] <= 5:
        analysis["quality_score"] = 3
    analysis["agent_mistakes"] = [m for m in analysis.get("agent_mistakes", []) if m in VALID_MISTAKES]
    return analysis


def aggregate_votes(analyses):
    intents = [a["intent"] for a in analyses]
    intent = Counter(intents).most_common(1)[0][0]

    satisfactions = [a["satisfaction"] for a in analyses]
    satisfaction = Counter(satisfactions).most_common(1)[0][0]

    scores = [a["quality_score"] for a in analyses]
    quality_score = int(statistics.median(scores))

    mistake_counts = Counter()
    for a in analyses:
        for m in a["agent_mistakes"]:
            mistake_counts[m] += 1
    threshold = len(analyses) / 2
    agent_mistakes = [m for m, count in mistake_counts.items() if count > threshold]

    return {
        "intent": intent,
        "satisfaction": satisfaction,
        "quality_score": quality_score,
        "agent_mistakes": agent_mistakes,
    }


def analyze_dialog(dialog):
    dialog_text = format_dialog(dialog["messages"])
    prompt = ANALYSIS_PROMPT.format(
        few_shot=FEW_SHOT_EXAMPLES,
        dialog_text=dialog_text,
    )

    analyses = []
    for i in range(VOTING_ROUNDS):
        system_prompt = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
        content = call_llm(prompt, system_prompt)
        analysis = extract_json_from_response(content)
        analyses.append(validate_analysis(analysis))

    return aggregate_votes(analyses)


def main(voting_rounds=None):
    global VOTING_ROUNDS
    if voting_rounds is not None:
        VOTING_ROUNDS = voting_rounds

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run generate.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} dialogs from {INPUT_FILE}")
    print(f"Voting rounds: {VOTING_ROUNDS}")

    results = []
    errors = 0

    cancelled = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_dialog, d): d for d in dataset}
        try:
            for future in as_completed(futures):
                dialog = futures[future]
                try:
                    analysis = future.result()
                    results.append({
                        "id": dialog["id"],
                        "analysis": analysis,
                    })
                    print(f"  Analyzed dialog id={dialog['id']}")
                except Exception as e:
                    errors += 1
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
        except KeyboardInterrupt:
            cancelled = True
            print(f"\n  Interrupted. Cancelling pending tasks...")
            for f in futures:
                f.cancel()

    if cancelled:
        print(f"  Saved {len(results)} completed dialogs before interruption.")
        raise KeyboardInterrupt

    results.sort(key=lambda r: r["id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete ({errors} errors) -> {OUTPUT_FILE}")

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