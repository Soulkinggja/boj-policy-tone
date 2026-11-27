"""Score every sentence of every BOJ policy statement as hawkish, dovish or neutral.

Model: mrince/CBRT-RoBERTa-HawkishDovish-Classifier (MIT licence), a RoBERTa model
fine-tuned on monetary policy committee statements from the Central Bank of the
Republic of Turkiye, another inflation-targeting emerging market central bank.
Labels: LABEL_0 neutral, LABEL_1 hawkish, LABEL_2 dovish.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "statements.jsonl"
OUT = ROOT / "data" / "processed"
MODEL = "mrince/CBRT-RoBERTa-HawkishDovish-Classifier"
LABELS = {"LABEL_0": "neutral", "LABEL_1": "hawkish", "LABEL_2": "dovish"}

# Lines that are procedure, not policy stance
BOILERPLATE = re.compile(
    r"(next (policy )?decision|end\s*[-–]|summary of the MPC|published on the Bank|"
    r"www\.|https?://|post author|editorial team|press conference|for further information)",
    re.I)

# A few statements put inflation figures before the rate in the text. Checked by hand.
RATE_OVERRIDE = {"2023-05-19": 7.0}


def policy_rate(text, date):
    if date in RATE_OVERRIDE:
        return RATE_OVERRIDE[date]
    flat = re.sub(r"\s+", " ", text)
    m = re.search(r"POLICY (?:INTEREST )?RATE (?:AT|TO) (\d{1,2}(?:\.\d{1,2})?) PER CENT", flat)
    if m:
        return float(m.group(1))
    for m in re.finditer(r"policy (?:interest )?rate[^.]{0,200}?(\d{1,2}\.\d{1,2})\s*(?:per\s*cent|%)", flat, re.I):
        if "inflation" not in m.group(0).lower():
            return float(m.group(1))
    return np.nan


def sentences(text):
    out = []
    for para in re.split(r"\n+", text):
        para = re.sub(r"\s+", " ", para).strip()
        para = re.sub(r"\.(?=[A-Z][a-z])", ". ", para)  # "range.The" -> "range. The"
        for s in re.split(r"(?<!i\.e\.)(?<!e\.g\.)(?<=[.;!?])\s+(?=[A-Z(\u201c\"])", para):
            s = s.strip(" ;")
            if len(s.split()) < 6 or BOILERPLATE.search(s):
                continue
            out.append(s)
    return out


def main():
    docs = [json.loads(line) for line in RAW.open()]
    clf = pipeline("text-classification", model=MODEL, top_k=None, truncation=True, max_length=256)

    rows = []
    for d in docs:
        sents = sentences(d["text"])
        preds = clf(sents, batch_size=16)
        for i, (s, p) in enumerate(zip(sents, preds)):
            probs = {LABELS[x["label"]]: x["score"] for x in p}
            label = max(probs, key=probs.get)
            rows.append({"date": d["date"], "order": i, "sentence": s, "label": label,
                         "p_hawkish": probs["hawkish"], "p_dovish": probs["dovish"],
                         "p_neutral": probs["neutral"]})
    sent = pd.DataFrame(rows)

    g = sent.groupby("date")
    stm = pd.DataFrame({
        "sentences": g.size(),
        "hawkish_share": g["label"].apply(lambda x: (x == "hawkish").mean()),
        "dovish_share": g["label"].apply(lambda x: (x == "dovish").mean()),
        # Tone index from -100 (fully dovish) to +100 (fully hawkish), probability weighted
        "tone": g.apply(lambda x: 100 * (x["p_hawkish"] - x["p_dovish"]).mean()),
    }).reset_index()
    meta = pd.DataFrame([{"date": d["date"], "url": d["url"], "rate": policy_rate(d["text"], d["date"])}
                         for d in docs])
    stm = stm.merge(meta, on="date").sort_values("date").reset_index(drop=True)
    stm["rate_change"] = stm["rate"].diff()
    stm["next_change"] = stm["rate"].shift(-1) - stm["rate"]
    stm["decision"] = np.select([stm["rate_change"] > 0, stm["rate_change"] < 0], ["Raise", "Cut"], "Hold")
    stm.loc[0, "decision"] = "Hold"

    OUT.mkdir(parents=True, exist_ok=True)
    sent.to_csv(OUT / "sentences.csv", index=False)
    stm.to_csv(OUT / "statements.csv", index=False)

    # Does today's tone lead the next decision?
    valid = stm.dropna(subset=["next_change"])
    corr_next = float(valid["tone"].corr(valid["next_change"]))
    by_next = valid.assign(next=np.sign(valid["next_change"])).groupby("next")["tone"].mean()
    latest = stm.iloc[-1]
    summary = {
        "statements": int(len(stm)),
        "sentences": int(len(sent)),
        "first": stm["date"].iloc[0], "last": latest["date"],
        "latest_tone": round(float(latest["tone"]), 1),
        "latest_rate": float(latest["rate"]),
        "most_hawkish": stm.loc[stm["tone"].idxmax(), ["date", "tone"]].to_dict(),
        "most_dovish": stm.loc[stm["tone"].idxmin(), ["date", "tone"]].to_dict(),
        "corr_tone_next_change": round(corr_next, 2),
        "avg_tone_before_raise": round(float(by_next.get(1.0, np.nan)), 1),
        "avg_tone_before_hold": round(float(by_next.get(0.0, np.nan)), 1),
        "avg_tone_before_cut": round(float(by_next.get(-1.0, np.nan)), 1),
        "model": MODEL,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
