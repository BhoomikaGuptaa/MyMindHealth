# src/model.py
from transformers import pipeline
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv, re

# --------- Lazy global pipelines (avoid reloading every click) ----------
_emotion = None
_sentiment = None

def get_pipes():
    global _emotion, _sentiment
    if _emotion is None:
        # 7-emotion model
        _emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    if _sentiment is None:
        # robust tweet/short-text sentiment model
        _sentiment = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
    return _emotion, _sentiment

# --------- Heuristics & helpers ----------
NEGATIVE_HINTS = [
    r"\b(not\s*feeling\s*good|feeling\s*bad|messing\s*up|i\s*suck|i\s*failed|hopeless|blue|down|low)\b",
    r"\b(anxious|anxiety|stressed|overwhelmed|burnt\s*out|tired|exhausted|lonely|depressed)\b",
    r"\b(worried|scared|fear|angry|upset|frustrated|guilty|ashamed|worthless)\b"
]

PAIN_HINTS = [
    r"\b(stomach\s*(ache|hurts|pain)|headache|migraine|nausea|nauseous|vomit|throw(ing)? up|dizzy|dizziness|cramp(s)?|period cramps|back pain|sore throat|fever|cold|flu|cough)\b"
]

def _norm_sent_label(lbl: str) -> str:
    l = lbl.lower()
    mapping = {
        "negative":"negative","label_0":"negative","1 star":"negative","0":"negative",
        "neutral":"neutral","label_1":"neutral","2 stars":"neutral","1":"neutral",
        "positive":"positive","label_2":"positive","3 stars":"positive","2":"positive"
    }
    return mapping.get(l, l)

def _has_negative_keywords(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in NEGATIVE_HINTS)

def _has_pain(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in PAIN_HINTS)

def _top(scores: List[Dict[str, float]]) -> Tuple[str, float, List[Dict[str, float]]]:
    s = sorted(scores, key=lambda d: d["score"], reverse=True)
    return s[0]["label"], s[0]["score"], s

def _override_from_csv(text: str) -> Optional[str]:
    """Allow user-defined pattern -> emotion mappings without training."""
    p = Path("data/custom_overrides.csv")
    if not p.exists():
        return None
    t = text.lower()
    try:
        with p.open(newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                pat = (row.get("pattern") or "").strip().lower()
                emo = (row.get("emotion") or "").strip().lower()
                if not pat or not emo:
                    continue
                if re.search(pat, t):
                    return emo
    except Exception:
        return None
    return None

# --------- Main entry: analyze ----------
def analyze(text: str) -> Dict:
    emotion_pipe, sent_pipe = get_pipes()

    # emotion
    emo_scores = emotion_pipe(text)[0]
    emo_label, emo_conf, emo_all = _top(emo_scores)

    # sentiment (normalized labels)
    raw_sent = sent_pipe(text)[0]
    sent_scores = [{"label": _norm_sent_label(d["label"]), "score": d["score"]} for d in raw_sent]
    sent_label, sent_conf, sent_all = _top(sent_scores)

    negative_hint = _has_negative_keywords(text) or sent_label == "negative"

    # margin-aware adjustment: if negativity present and joy/neutral barely win, choose next best non-positive
    if negative_hint:
        sorted_emo = sorted(emo_all, key=lambda d: d["score"], reverse=True)
        top1 = sorted_emo[0]
        top2 = sorted_emo[1] if len(sorted_emo) > 1 else sorted_emo[0]
        margin = top1["score"] - top2["score"]
        if top1["label"].lower() in {"joy", "neutral"} and margin < 0.20:
            candidates = [d for d in sorted_emo if d["label"].lower() not in {"joy", "neutral"}]
            if candidates:
                top1 = candidates[0]
                emo_label, emo_conf = top1["label"], top1["score"]
                # lightly penalize joy/neutral in the display list
                adj = []
                for d in sorted_emo:
                    score = d["score"] - (0.15 if d["label"].lower() in {"joy","neutral"} else 0.0)
                    adj.append({"label": d["label"], "score": score})
                emo_all = sorted(adj, key=lambda d: d["score"], reverse=True)

    # pain-specific nudge: steer to fear/neutral if within small margin
    if _has_pain(text):
        sorted_emo = sorted(emo_all, key=lambda d: d["score"], reverse=True)
        top1 = sorted_emo[0]
        for preferred in ["fear", "neutral"]:
            cand = next((d for d in sorted_emo if d["label"].lower() == preferred), None)
            if cand and (top1["score"] - cand["score"]) < 0.15:
                emo_label, emo_conf = cand["label"], cand["score"]
                break

    # user overrides win last (explicit mapping)
    override = _override_from_csv(text)
    if override:
        emo_label = override

    return {
        "emotion_top": {"label": emo_label, "confidence": emo_conf},
        "emotion_all": emo_all,
        "sentiment_top": {"label": sent_label, "confidence": sent_conf},
        "sentiment_all": sent_all,
        "negative_hint": negative_hint
    }
