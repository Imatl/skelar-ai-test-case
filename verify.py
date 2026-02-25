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

DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.json"
ANALYSIS_FILE = DATA_DIR / "analysis.json"
OUTPUT_FILE = DATA_DIR / "analysis_verified.json"

VERIFICATION_PROMPT = """You are a High-Level Quality Auditor. 
Your goal is to fix errors in the initial analysis.

DIALOG:
{dialog_text}

INITIAL ANALYSIS:
{initial_analysis}

CRITICAL VERIFICATION STEPS:
1. no_resolution: 
   - Did the agent provide a clear answer or a working link/fix? 
   - If the agent said "I can't help" or "Contact another department" WITHOUT a warm transfer, it is NO_RESOLUTION.
   - If the customer ended with "ok" but the problem isn't fixed, keep the mistake.

2. Hidden Dissatisfaction (CRITICAL):
   - Look for "Sarcastic Gratitude". If the customer says "Thanks" but their problem was NOT solved, mark satisfaction as "unsatisfied".
   - If the customer just stops responding after a poor agent answer, it is "unsatisfied".
   - Satisfaction is "satisfied" ONLY if the problem is solved AND the customer is clearly happy.

3. ignored_question: 
   - List every "?" in customer messages. 
   - Did the agent address EACH one? If even one was skipped, keep "ignored_question".

Return the corrected JSON ONLY."""

def call_verify_llm(dialog_text, initial_analysis):
    prompt = VERIFICATION_PROMPT.format(
        dialog_text=dialog_text,
        initial_analysis=json.dumps(initial_analysis, ensure_ascii=False)
    )
    
    response = client.chat.completions.create(
        model=MINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise quality auditor. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)

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

if __name__ == "__main__":
    run_verification()