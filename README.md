# AI Support Chat Quality Analysis

Automated analysis of customer support quality using LLM (GPT-4.1).

The system generates a dataset of realistic support chat dialogs, then analyzes each dialog to determine customer intent, satisfaction level, agent quality score, and agent mistakes.

## Project Structure

```
├── generate.py        # Generates 100 chat dialogs with ground truth labels
├── analyze.py         # Analyzes dialogs using GPT-4.1
├── analysis.ipynb     # Jupyter notebook with step-by-step analysis
├── requirements.txt   # Python dependencies
├── data/
│   ├── dataset.json   # Generated dialogs (after running generate.py)
│   └── analysis.json  # LLM analysis results (after running analyze.py)
└── .env               # API key (create from .env.example)
```

## Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Azure OpenAI credentials:

```bash
cp .env.example .env
# Edit .env and set your Azure OpenAI endpoint, API key, deployment name, and API version
```

## Usage

### Step 1: Generate Dataset

```bash
python generate.py
```

Generates 100 support chat dialogs with ground truth labels covering:
- **Intents**: payment_issue, technical_error, account_access, pricing_plan, refund
- **Scenarios**: successful, problematic, conflict, agent_error, hidden_dissatisfaction
- **Output**: `data/dataset.json`

### Step 2: Analyze Dialogs

```bash
python analyze.py
```

Sends each dialog to GPT-4.1 for analysis. For each dialog determines:
- `intent` — customer's reason for contacting support
- `satisfaction` — real satisfaction level (satisfied / neutral / unsatisfied)
- `quality_score` — agent performance rating (1-5)
- `agent_mistakes` — list of detected mistakes

Output: `data/analysis.json`

### Step 3: Explore Results in Notebook

```bash
jupyter notebook analysis.ipynb
```

The notebook provides:
1. Dataset overview and distribution statistics
2. Intent classification accuracy (confusion matrix, F1)
3. Satisfaction prediction accuracy
4. Quality score correlation analysis
5. Agent mistake detection precision/recall
6. Hidden dissatisfaction detection rate
7. Disagreement analysis between LLM predictions and ground truth

## Key Features

- **Deterministic results**: Uses `temperature=0` and `seed` parameter for reproducibility
- **Hidden dissatisfaction detection**: ~15% of dialogs contain customers who appear polite but are actually unsatisfied
- **Ground truth labels**: Each generated dialog includes expert-level annotations for validation
- **Comprehensive error taxonomy**: 5 types of agent mistakes tracked and evaluated

## Tech Stack

- Python 3.12
- Azure OpenAI GPT-4.1
- pandas, matplotlib, seaborn, scikit-learn
