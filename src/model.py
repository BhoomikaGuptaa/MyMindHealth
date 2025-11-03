from transformers import pipeline

_classifier = None

def get_emotion_pipeline():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    return _classifier

def classify(text: str):
    clf = get_emotion_pipeline()
    scores = clf(text)[0]  # list of {label, score}
    scores = sorted(scores, key=lambda d: d["score"], reverse=True)
    top = scores[0]
    return top["label"], top["score"], scores
