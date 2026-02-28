import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


client = AzureOpenAI(
    azure_endpoint=os.getenv("MINI_ENDPOINT"),
    api_key=os.getenv("MINI_API_KEY"),
    api_version=os.getenv("MINI_API_VERSION"),
)

MINI_MODEL = os.getenv("MINI_DEPLOYMENT")

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_FILE = DATA_DIR / "dataset.json"
ANALYSIS_FILE = DATA_DIR / "analysis.json"
OUTPUT_FILE = DATA_DIR / "analysis_verified.json"

VALID_INTENTS = {"payment_issue", "technical_error", "account_access", "pricing_plan", "refund", "other"}
VALID_SATISFACTIONS = {"satisfied", "neutral", "unsatisfied"}
VALID_MISTAKES = {"ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"}

VERIFICATION_PROMPT = """You are a precision quality auditor. You will verify an initial analysis of a support dialog and correct any errors.

DIALOG:
{dialog_text}

INITIAL ANALYSIS:
{initial_analysis}

PERFORM EACH CHECK BELOW. For each check, state your finding, then decide if the initial analysis needs correction.

CHECK 1 — no_resolution:
List what the agent CONCRETELY DID to solve the customer's problem (not what they said they'd do):
- Did the agent take an action (refund initiated, account unlocked, setting changed)?
- Did the agent provide specific actionable steps the customer accepted?
- Did the agent provide a workaround that addresses the immediate need?
If YES to any → REMOVE no_resolution from mistakes.
If the agent only gave generic advice, redirected without details, or the core problem remains unaddressed → KEEP no_resolution.

CHECK 2 — hidden dissatisfaction:
Remove all polite words from the customer's last 2 messages ("thanks", "okay", "alright", "appreciate it"). Now re-read:
- Is the customer's original problem actually SOLVED?
- Did the customer get what they specifically asked for?
If the problem is NOT solved despite polite language → satisfaction must be "unsatisfied".
If the problem IS solved and gratitude is genuine → keep "satisfied".

CHECK 3 — ignored_question:
List every direct question (with "?") the customer asked:
{question_list_instruction}
For each question: did the agent address it at all (even partially)?
If ALL questions were addressed → REMOVE ignored_question.
If a specific question was completely skipped → KEEP ignored_question.

CHECK 4 — unnecessary_escalation:
Did the agent transfer/redirect the customer to another team or department?
- If yes: was this a routine task the agent could have handled (plan change, simple refund, password reset)? → KEEP unnecessary_escalation.
- If the issue genuinely requires specialist knowledge (legal, compliance, engineering bug) → REMOVE unnecessary_escalation.
- If no transfer happened → REMOVE unnecessary_escalation.

CHECK 5 — incorrect_info:
Did the agent state something demonstrably false or contradicted by facts in the dialog?
- Vague or unhelpful answers are NOT incorrect_info.
- Only verifiably wrong statements count.

CHECK 6 — quality_score consistency:
- If satisfaction is "satisfied" and no mistakes → score should be 4-5
- If satisfaction is "unsatisfied" and 2+ mistakes → score should be 1-2
- If satisfaction is "neutral" → score should be 2-4

Return ONLY the corrected JSON with this exact structure:
{{"intent": "...", "satisfaction": "...", "quality_score": N, "agent_mistakes": [...]}}"""

QUESTION_LIST_INSTRUCTION = "Count all '?' in customer messages. List each question explicitly."

def validate_analysis(analysis):
    if analysis.get("intent") not in VALID_INTENTS:
        analysis["intent"] = "other"
    if analysis.get("satisfaction") not in VALID_SATISFACTIONS:
        analysis["satisfaction"] = "neutral"
    if not isinstance(analysis.get("quality_score"), int) or not 1 <= analysis["quality_score"] <= 5:
        analysis["quality_score"] = 3
    analysis["agent_mistakes"] = [m for m in analysis.get("agent_mistakes", []) if m in VALID_MISTAKES]
    return analysis


def call_verify_llm(dialog_text, initial_analysis):
    prompt = VERIFICATION_PROMPT.format(
        dialog_text=dialog_text,
        initial_analysis=json.dumps(initial_analysis, ensure_ascii=False),
        question_list_instruction=QUESTION_LIST_INSTRUCTION,
    )

    response = client.chat.completions.create(
        model=MINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a precision quality auditor for customer support analysis. Perform each verification check carefully, then return corrected JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)
    return validate_analysis(result)

def run_verification():
    if not ANALYSIS_FILE.exists():
        print(f"Error: {ANALYSIS_FILE} not found. Run analyze.py first.")
        return

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = {d["id"]: d for d in json.load(f)}
    
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    verified_results = []
    total = len(analysis_data)
    print(f"Starting LLM verification for {total} dialogs using {MINI_MODEL}...")

    for i, item in enumerate(analysis_data):
        dialog_id = item["id"]
        initial_analysis = item["analysis"]
        
        # Форматируем текст диалога для промпта
        messages = dataset[dialog_id]["messages"]
        dialog_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in messages])

        print(f"[{i+1}/{total}] Verifying ID: {dialog_id}...")
        
        try:
            verified_analysis = call_verify_llm(dialog_text, initial_analysis)
            verified_results.append({
                "id": dialog_id,
                "analysis": verified_analysis
            })
        except Exception as e:
            print(f"Error verifying ID {dialog_id}: {e}")
            # Если ошибка — сохраняем оригинальный анализ, чтобы не терять данные
            verified_results.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(verified_results, f, indent=2, ensure_ascii=False)

    print(f"\nVerification complete. Results saved to {OUTPUT_FILE}")

    merge_hybrid(analysis_data, verified_results)


def merge_hybrid(original_results, verified_results):
    verified_map = {item["id"]: item["analysis"] for item in verified_results}
    hybrid_results = []

    changes = 0
    for item in original_results:
        dialog_id = item["id"]
        orig = item["analysis"]
        verif = verified_map.get(dialog_id, orig)

        hybrid = {
            "intent": orig["intent"],
            "satisfaction": orig["satisfaction"],
            "quality_score": orig["quality_score"],
            "agent_mistakes": orig["agent_mistakes"],
        }

        verif_sat = verif.get("satisfaction", orig["satisfaction"])
        verif_mistakes = verif.get("agent_mistakes", [])
        orig_has_no_res = "no_resolution" in orig["agent_mistakes"]
        verif_has_no_res = "no_resolution" in verif_mistakes

        if verif_sat == "unsatisfied" and orig["satisfaction"] != "unsatisfied":
            if orig_has_no_res or (verif_has_no_res and len(orig["agent_mistakes"]) > 0):
                hybrid["satisfaction"] = "unsatisfied"
                if orig["quality_score"] >= 4:
                    hybrid["quality_score"] = min(orig["quality_score"], 3)
                changes += 1

        if verif_sat == "satisfied" and orig["satisfaction"] == "neutral":
            pass

        hybrid_results.append({"id": dialog_id, "analysis": hybrid})

    hybrid_file = DATA_DIR / "analysis_hybrid.json"
    with open(hybrid_file, "w", encoding="utf-8") as f:
        json.dump(hybrid_results, f, indent=2, ensure_ascii=False)

    print(f"Hybrid merge: {changes} satisfaction corrections applied -> {hybrid_file}")


if __name__ == "__main__":
    run_verification()