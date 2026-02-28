# AI Support Chat Quality Analysis

Automated analysis of customer support quality using LLM.

The system generates a dataset of realistic support chat dialogs, then analyzes each dialog to determine customer intent, satisfaction level, agent quality score, and agent mistakes.

## Project Structure

```
├── main.py                # CLI entry point
├── src/
│   ├── generate.py        # Generates 100 chat dialogs with ground truth labels
│   ├── analyze.py         # Analyzes dialogs using LLM (few-shot + CoT + majority voting)
│   ├── verify.py          # Second-pass verification with separate model
│   ├── evaluate.py        # Computes accuracy metrics
│   └── postprocess.py     # Rule-based postprocessing
├── analysis.ipynb         # Jupyter notebook with visualizations
├── requirements.txt       # Python dependencies
├── data/
│   ├── dataset.json       # Generated dialogs (after generate)
│   ├── analysis.json      # LLM analysis results (after analyze)
│   ├── analysis_verified.json  # Verified results (after verify)
│   └── evaluation.json    # Evaluation metrics (after evaluate)
└── .env                   # API credentials
```

## Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment>
AZURE_OPENAI_API_VERSION=<api-version>
MINI_ENDPOINT=<verification-model-endpoint>
MINI_API_KEY=<verification-model-key>
MINI_DEPLOYMENT=<verification-model-deployment>
MINI_API_VERSION=<verification-api-version>
```

## Usage

### Full Pipeline (recommended)

```bash
python main.py run
```

Runs all steps sequentially: generate → analyze → verify → evaluate.

### Individual Steps

```bash
python main.py generate       # Step 1: Generate 100 chat dialogs
python main.py analyze        # Step 2: Analyze with majority voting (3 rounds)
python main.py verify         # Step 3: Verify with second model
python main.py evaluate       # Step 4: Compute metrics
python main.py evaluate --file data/analysis_verified.json
```

## Pipeline Details

### Step 1: Generate Dataset
Generates 100 support chat dialogs with ground truth labels covering:
- **Intents**: payment_issue, technical_error, account_access, pricing_plan, refund
- **Scenarios**: successful, problematic, conflict, agent_error, hidden_dissatisfaction

### Step 2: Analyze Dialogs (Majority Voting)
Each dialog is analyzed 3 times with different system prompts. Results are aggregated:
- `intent`, `satisfaction` — majority vote (mode)
- `quality_score` — median
- `agent_mistakes` — included if detected by 2+ of 3 rounds

### Step 3: Verify with Second Model
A separate model re-checks each analysis with a structured 6-point verification checklist:
1. no_resolution check (concrete agent actions)
2. Hidden dissatisfaction detection
3. ignored_question verification
4. unnecessary_escalation validation
5. incorrect_info fact-check
6. quality_score consistency

### Step 4: Evaluate
Computes accuracy metrics: intent accuracy, satisfaction accuracy, quality MAE, hidden dissatisfaction detection rate, per-mistake precision/recall/F1.

## Key Features

- **Majority voting**: 3 LLM rounds with different perspectives reduce noise
- **Two-model verification**: Second model cross-checks the primary analysis
- **Few-shot + Chain-of-Thought**: 8 calibration examples in the analysis prompt
- **Hidden dissatisfaction detection**: ~15% of dialogs contain polite but unsatisfied customers
- **Comprehensive error taxonomy**: 5 types of agent mistakes tracked and evaluated
- **Parallel execution**: Thread pool for faster processing
- **Automatic retry**: Exponential backoff on API failures

## Tech Stack

- Python 3.12
- Azure OpenAI
- pandas, matplotlib, seaborn, scikit-learn
- tenacity (retry logic)
