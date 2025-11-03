import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from model import classify

DATA_PATH = Path("data/journal.csv")

st.set_page_config(page_title="MindMend", page_icon="🧠")
st.title("🧠 MindMend — Emotion Insights (Non-clinical)")
st.caption("Info only. If you’re in crisis in the US, call/text 988.")

user_text = st.text_area("How are you feeling today?", height=140, placeholder="Type anything that's on your mind...")

if st.button("Analyze"):
    if user_text.strip():
        label, score, scores = classify(user_text.strip())
        st.success(f"Top emotion: **{label}** (confidence {score:.2f})")

        with st.expander("All scores"):
            st.write(pd.DataFrame(scores))

        # Save to a simple journal CSV
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        row = {"timestamp": ts, "text": user_text.strip(), "top_emotion": label, "confidence": score}
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(DATA_PATH, index=False)
        st.toast("Saved to data/journal.csv")
    else:
        st.warning("Please enter a message first.")

st.divider()
st.subheader("📈 Your emotion trend")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    counts = df["top_emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
    st.bar_chart(counts.set_index("emotion"))
else:
    st.info("No entries yet. Analyze a message to start your journal.")
