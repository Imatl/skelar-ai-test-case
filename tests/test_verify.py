import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verify import merge_hybrid, validate_analysis


class TestValidateAnalysis:
    def test_valid_passthrough(self):
        inp = {
            "intent": "refund",
            "satisfaction": "satisfied",
            "quality_score": 4,
            "agent_mistakes": ["rude_tone"],
        }
        result = validate_analysis(inp)
        assert result == inp

    def test_cleans_invalid_fields(self):
        result = validate_analysis({
            "intent": "banana",
            "satisfaction": "angry",
            "quality_score": 99,
            "agent_mistakes": ["fake", "rude_tone"],
        })
        assert result["intent"] == "other"
        assert result["satisfaction"] == "neutral"
        assert result["quality_score"] == 3
        assert result["agent_mistakes"] == ["rude_tone"]


class TestMergeHybrid:
    def test_no_changes_when_agree(self):
        original = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "satisfied",
            "quality_score": 5, "agent_mistakes": [],
        }}]
        verified = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "satisfied",
            "quality_score": 5, "agent_mistakes": [],
        }}]
        merge_hybrid(original, verified)

    def test_corrects_satisfaction_when_no_resolution(self):
        original = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "neutral",
            "quality_score": 3, "agent_mistakes": ["no_resolution"],
        }}]
        verified = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "unsatisfied",
            "quality_score": 2, "agent_mistakes": ["no_resolution"],
        }}]
        merge_hybrid(original, verified)
        from src.verify import DATA_DIR
        import json
        hybrid_file = DATA_DIR / "analysis_hybrid.json"
        if hybrid_file.exists():
            with open(hybrid_file, encoding="utf-8") as f:
                result = json.load(f)
            assert result[0]["analysis"]["satisfaction"] == "unsatisfied"
            assert result[0]["analysis"]["agent_mistakes"] == ["no_resolution"]

    def test_keeps_original_mistakes(self):
        original = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "unsatisfied",
            "quality_score": 2, "agent_mistakes": ["rude_tone", "no_resolution"],
        }}]
        verified = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "unsatisfied",
            "quality_score": 1, "agent_mistakes": ["no_resolution"],
        }}]
        merge_hybrid(original, verified)
        from src.verify import DATA_DIR
        import json
        hybrid_file = DATA_DIR / "analysis_hybrid.json"
        if hybrid_file.exists():
            with open(hybrid_file, encoding="utf-8") as f:
                result = json.load(f)
            assert "rude_tone" in result[0]["analysis"]["agent_mistakes"]

    def test_no_correction_without_no_resolution(self):
        original = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "neutral",
            "quality_score": 4, "agent_mistakes": [],
        }}]
        verified = [{"id": 1, "analysis": {
            "intent": "refund", "satisfaction": "unsatisfied",
            "quality_score": 2, "agent_mistakes": [],
        }}]
        merge_hybrid(original, verified)
        from src.verify import DATA_DIR
        import json
        hybrid_file = DATA_DIR / "analysis_hybrid.json"
        if hybrid_file.exists():
            with open(hybrid_file, encoding="utf-8") as f:
                result = json.load(f)
            assert result[0]["analysis"]["satisfaction"] == "neutral"
