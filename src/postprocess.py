import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "analysis_verified.json"
DATASET_FILE = DATA_DIR / "dataset.json"
OUTPUT_FILE = DATA_DIR / "analysis_final.json"

def postprocess():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        analysis_results = json.load(f)
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = {d["id"]: d for d in json.load(f)}
    
    changes_made = 0
    all_mistakes_found = set() # Для дебага

    for item in analysis_results:
        ans = item["analysis"]
        dialog = dataset[item["id"]]
        
        # Собираем все типы ошибок, которые реально есть в файле
        current_mistakes = ans.get("agent_mistakes", [])
        for m in current_mistakes:
            all_mistakes_found.add(m)

        customer_text = " ".join([m["text"] for m in dialog["messages"] if m["role"] == "customer"])

        # Пробуем найти ошибку через частичное совпадение (на случай если там "Ignored Question")
        new_mistakes = []
        for m in current_mistakes:
            m_lower = m.lower()
            
            # Если в тексте нет "?" и это ошибка про игнор вопроса - удаляем
            if "?" not in customer_text and ("ignore" in m_lower or "игнор" in m_lower):
                changes_made += 1
                continue
            
            # Если оценка 4-5 и это ошибка про решение - удаляем
            if ans.get("quality_score", 0) >= 4 and ("resolution" in m_lower or "решен" in m_lower):
                changes_made += 1
                continue
                
            new_mistakes.append(m)
        
        ans["agent_mistakes"] = new_mistakes

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"--- DEBUG INFO ---")
    print(f"Mistake types found in file: {all_mistakes_found}")
    print(f"Total changes made: {changes_made}")

if __name__ == "__main__":
    postprocess()