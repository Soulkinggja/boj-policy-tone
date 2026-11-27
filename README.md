# Bank of Jamaica Policy Tone Tracker

How hawkish or dovish is the Bank of Jamaica? This project uses an AI language model to read every sentence of the Bank of Jamaica's monetary policy statements and label each one hawkish, dovish or neutral, then tracks the tone against the policy rate.

**Live app:** https://huggingface.co/spaces/devinmcintosha/boj-policy-tone

## How it works

1. `src/fetch.py` downloads every monetary policy press release from the BOJ website through its public WordPress API (February 2020 onward).
2. `src/score.py` splits each statement into sentences, drops procedural lines, and scores each sentence with [mrince/CBRT-RoBERTa-HawkishDovish-Classifier](https://huggingface.co/mrince/CBRT-RoBERTa-HawkishDovish-Classifier), a RoBERTa model trained on the Central Bank of Turkiye's policy statements (87% test accuracy). A statement's tone is the average of (chance hawkish minus chance dovish) times 100.
3. `src/publish.py` ships the results and the Gradio app (`space/app.py`) to Hugging Face Spaces.
4. A GitHub Actions workflow reruns everything after each BOJ decision.

## Data

- `data/raw/statements.jsonl`: statement text, date and link
- `data/processed/sentences.csv`: every sentence with its label and probabilities
- `data/processed/statements.csv`: tone, policy rate and decision for each statement
- `data/processed/summary.json`: headline numbers

## Limits

The model was trained on another central bank's language, so single sentences can be misread. The trend across statements is more reliable than any one label.

Built by Devin McIntosh. Tools: Python, Hugging Face Transformers, Gradio, Plotly, GitHub Actions.
