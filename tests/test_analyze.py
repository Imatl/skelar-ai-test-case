import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyze import (
    validate_analysis,
    aggregate_votes,
    extract_json_from_response,
    format_dialog,
)


class TestValidateAnalysis:
    def test_valid_input_unchanged(self):
        inp = {
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 5,
            "agent_mistakes": ["rude_tone"],
        }
        result = validate_analysis(inp)
        assert result == inp

    def test_invalid_intent_defaults_to_other(self):
        result = validate_analysis({
            "intent": "unknown_stuff",
            "satisfaction": "satisfied",
            "quality_score": 3,
            "agent_mistakes": [],
        })
        assert result["intent"] == "other"

    def test_invalid_satisfaction_defaults_to_neutral(self):
        result = validate_analysis({
            "intent": "refund",
            "satisfaction": "very_happy",
            "quality_score": 3,
            "agent_mistakes": [],
        })
        assert result["satisfaction"] == "neutral"

    def test_quality_score_out_of_range(self):
        result = validate_analysis({
            "intent": "refund",
            "satisfaction": "neutral",
            "quality_score": 10,
            "agent_mistakes": [],
        })
        assert result["quality_score"] == 3

    def test_quality_score_not_int(self):
        result = validate_analysis({
            "intent": "refund",
            "satisfaction": "neutral",
            "quality_score": "high",
            "agent_mistakes": [],
        })
        assert result["quality_score"] == 3

    def test_invalid_mistakes_filtered(self):
        result = validate_analysis({
            "intent": "refund",
            "satisfaction": "neutral",
            "quality_score": 3,
            "agent_mistakes": ["rude_tone", "fake_mistake", "no_resolution"],
        })
        assert result["agent_mistakes"] == ["rude_tone", "no_resolution"]

    def test_missing_mistakes_key(self):
        result = validate_analysis({
            "intent": "refund",
            "satisfaction": "neutral",
            "quality_score": 3,
        })
        assert result["agent_mistakes"] == []


class TestAggregateVotes:
    def test_unanimous_votes(self):
        analyses = [
            {"intent": "refund", "satisfaction": "satisfied", "quality_score": 5, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "satisfied", "quality_score": 5, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "satisfied", "quality_score": 5, "agent_mistakes": []},
        ]
        result = aggregate_votes(analyses)
        assert result["intent"] == "refund"
        assert result["satisfaction"] == "satisfied"
        assert result["quality_score"] == 5
        assert result["agent_mistakes"] == []

    def test_majority_intent(self):
        analyses = [
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": []},
            {"intent": "payment_issue", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": []},
        ]
        result = aggregate_votes(analyses)
        assert result["intent"] == "refund"

    def test_majority_satisfaction(self):
        analyses = [
            {"intent": "refund", "satisfaction": "unsatisfied", "quality_score": 2, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "satisfied", "quality_score": 4, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "unsatisfied", "quality_score": 3, "agent_mistakes": []},
        ]
        result = aggregate_votes(analyses)
        assert result["satisfaction"] == "unsatisfied"

    def test_median_quality_score(self):
        analyses = [
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 1, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 5, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": []},
        ]
        result = aggregate_votes(analyses)
        assert result["quality_score"] == 3

    def test_mistakes_threshold(self):
        analyses = [
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": ["rude_tone", "no_resolution"]},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": ["rude_tone"]},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": ["no_resolution"]},
        ]
        result = aggregate_votes(analyses)
        assert "rude_tone" in result["agent_mistakes"]
        assert "no_resolution" in result["agent_mistakes"]

    def test_mistakes_below_threshold_excluded(self):
        analyses = [
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": ["rude_tone"]},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": []},
            {"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": []},
        ]
        result = aggregate_votes(analyses)
        assert "rude_tone" not in result["agent_mistakes"]


class TestExtractJson:
    def test_with_reasoning_and_answer(self):
        content = """REASONING: The customer was happy.
ANSWER:
{"intent": "payment_issue", "satisfaction": "satisfied", "quality_score": 5, "agent_mistakes": []}"""
        result = extract_json_from_response(content)
        assert result["intent"] == "payment_issue"
        assert result["quality_score"] == 5

    def test_with_code_block(self):
        content = """```json
{"intent": "refund", "satisfaction": "neutral", "quality_score": 3, "agent_mistakes": ["no_resolution"]}
```"""
        result = extract_json_from_response(content)
        assert result["intent"] == "refund"
        assert "no_resolution" in result["agent_mistakes"]

    def test_plain_json(self):
        content = '{"intent": "account_access", "satisfaction": "unsatisfied", "quality_score": 1, "agent_mistakes": []}'
        result = extract_json_from_response(content)
        assert result["intent"] == "account_access"


class TestFormatDialog:
    def test_basic_format(self):
        messages = [
            {"role": "customer", "text": "Hello"},
            {"role": "agent", "text": "Hi there"},
        ]
        result = format_dialog(messages)
        assert "Customer: Hello" in result
        assert "Agent: Hi there" in result
