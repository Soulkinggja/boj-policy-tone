import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = Path(__file__).parent / "data"
stm = pd.read_csv(DATA / "statements.csv", parse_dates=["date"])
sent = pd.read_csv(DATA / "sentences.csv", parse_dates=["date"])
summary = json.loads((DATA / "summary.json").read_text())
MODEL = summary["model"]

HAWK, DOVE, NEUTRAL, INK, MUTED = "#c8553d", "#2a78d6", "#9aa4b8", "#1d2433", "#6b7385"


def tone_chart():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.62, 0.38],
                        subplot_titles=("Tone of each statement (+100 fully hawkish, -100 fully dovish)",
                                        "Policy rate (%)"))
    colors = [HAWK if t >= 0 else DOVE for t in stm["tone"]]
    fig.add_bar(x=stm["date"], y=stm["tone"], marker_color=colors, row=1, col=1,
                customdata=stm[["decision", "rate"]],
                hovertemplate="%{x|%d %b %Y}<br>Tone %{y:.0f}<br>%{customdata[0]} at %{customdata[1]}%<extra></extra>")
    fig.add_scatter(x=stm["date"], y=stm["rate"], mode="lines+markers", line=dict(color=INK, width=2, shape="hv"),
                    marker=dict(size=6), row=2, col=1,
                    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>")
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=1, col=1)
    fig.update_layout(height=620, showlegend=False, plot_bgcolor="white", margin=dict(l=40, r=20, t=60, b=30))
    fig.update_yaxes(gridcolor="#eceef2", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def mix_chart():
    d = stm.copy()
    d["neutral_share"] = 1 - d["hawkish_share"] - d["dovish_share"]
    fig = go.Figure()
    for col, name, c in [("hawkish_share", "Hawkish", HAWK), ("neutral_share", "Neutral", NEUTRAL),
                         ("dovish_share", "Dovish", DOVE)]:
        fig.add_bar(x=d["date"], y=100 * d[col], name=name, marker_color=c,
                    hovertemplate="%{x|%b %Y}: %{y:.0f}% " + name.lower() + "<extra></extra>")
    fig.update_layout(barmode="stack", height=360, plot_bgcolor="white", yaxis_title="Share of sentences (%)",
                      legend=dict(orientation="h", y=1.1), margin=dict(l=40, r=20, t=40, b=30))
    return fig


def statement_view(label):
    date = pd.Timestamp(label.split(" | ")[0])
    s = sent[sent["date"] == date].sort_values("order")
    row = stm[stm["date"] == date].iloc[0]
    head = (f"**{date:%d %B %Y}**: {row['decision']} at {row['rate']:.2f}%. "
            f"Tone **{row['tone']:+.0f}**. [Read the statement]({row['url']})")
    table = s.assign(confidence=(s[["p_hawkish", "p_dovish", "p_neutral"]].max(axis=1) * 100).round(0))
    return head, table[["label", "confidence", "sentence"]]


@lru_cache(maxsize=1)
def classifier():
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL, top_k=None, truncation=True)


def classify(text):
    if not text or not text.strip():
        return {}
    names = {"LABEL_0": "Neutral", "LABEL_1": "Hawkish", "LABEL_2": "Dovish"}
    return {names[x["label"]]: float(x["score"]) for x in classifier()(text.strip())[0]}


choices = [f"{d:%Y-%m-%d} | {dec} at {r:.2f}%" for d, dec, r in
           zip(stm["date"], stm["decision"], stm["rate"])][::-1]

intro = f"""
# Bank of Jamaica Policy Tone Tracker
How hawkish or dovish is the Bank of Jamaica? An AI language model reads every sentence of
{summary['statements']} monetary policy statements ({summary['first'][:4]} to {summary['last'][:4]},
{summary['sentences']:,} sentences) and labels each one hawkish (leaning to higher rates to fight inflation),
dovish (leaning to lower rates to support growth) or neutral.

**Latest statement ({summary['last']}):** tone {summary['latest_tone']:+.0f}, policy rate {summary['latest_rate']:.2f}%.
Before a rate rise the average tone was **{summary['avg_tone_before_raise']:+.0f}**; before a hold,
**{summary['avg_tone_before_hold']:+.0f}**; before a cut, **{summary['avg_tone_before_cut']:+.0f}**.
"""

about = f"""
### How it works
1. Statements come from the Bank of Jamaica website ({summary['first']} onward; earlier ones are not online in the same format).
2. Each statement is split into sentences; procedural lines (dates of the next meeting, links) are dropped.
3. Each sentence is scored by [{MODEL}](https://huggingface.co/{MODEL}), a RoBERTa model trained on statements
   from the Central Bank of the Republic of Turkiye, another inflation-targeting emerging market central bank
   (87% accuracy on its own test set).
4. A statement's tone is the average of (chance hawkish minus chance dovish) across its sentences, times 100.

### Limits
The model was trained on another central bank's language, so read single sentences with care; the trend across
statements is more reliable than any one label. Tone measures words, not intentions.

Built by Devin McIntosh. Code and data: [github.com/Soulkinggja/boj-policy-tone](https://github.com/Soulkinggja/boj-policy-tone)
"""

with gr.Blocks(title="BOJ Policy Tone Tracker") as demo:
    gr.Markdown(intro)
    with gr.Tab("Tone over time"):
        gr.Plot(tone_chart())
        gr.Plot(mix_chart())
    with gr.Tab("Read a statement"):
        pick = gr.Dropdown(choices, value=choices[0], label="Statement")
        head = gr.Markdown()
        table = gr.Dataframe(wrap=True, interactive=False)
        pick.change(statement_view, pick, [head, table])
        demo.load(statement_view, pick, [head, table])
    with gr.Tab("Try a sentence"):
        box = gr.Textbox(label="Paste any central bank sentence", lines=3,
                         value="The Committee is prepared to take further action to bring inflation back within the target range.")
        btn = gr.Button("Score it", variant="primary")
        out = gr.Label(num_top_classes=3, label="Model reading")
        btn.click(classify, box, out)
    gr.Markdown(about)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
