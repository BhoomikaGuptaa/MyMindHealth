# src/app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from model import analyze

DATA_PATH = Path("data/journal.csv")

# ---------------- Theming & CSS ----------------
st.set_page_config(page_title="MyMindHealth", page_icon="🧠")

st.markdown("""
<style>
/* gradient background */
.stApp {
  background: linear-gradient(135deg, #0b1020 0%, #1b2a4a 45%, #2b0b45 100%);
}
/* container padding */
.block-container { padding-top: 1.5rem; }
/* buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.6rem 1rem;
  font-weight: 600;
}
/* text area */
textarea { border-radius: 12px !important; }
/* expander header */
.streamlit-expanderHeader { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🧠✨ MyMindHealth")
st.caption("Info only • Not medical advice • US crisis: call 911")
st.markdown("**Emotions:** 😊 joy • 😐 neutral • 😢 sadness • 😠 anger • 😨 fear • 🤢 disgust • 😲 surprise")

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Settings")
    reset = st.button("🗑️ Reset journal (delete data/journal.csv)")
    show_last_n = st.number_input("Show last N entries in chart", min_value=10, max_value=1000, value=100, step=10)

if reset:
    try:
        if DATA_PATH.exists():
            DATA_PATH.unlink()
            st.toast("Journal cleared.")
    except Exception as e:
        st.warning(f"Could not delete journal: {e}")

# ---------------- Main input ----------------
user_text = st.text_area("How are you feeling today?", height=140, placeholder="Type anything on your mind…")

col1, col2 = st.columns([1,1], vertical_alignment="bottom")
with col1:
    analyze_btn = st.button("Analyze")
with col2:
    tips_on = st.toggle("Show tips & quote", value=True)

# ---------------- Tips & Quotes ----------------
TIPS = {
    "sadness": [
        "Micro-action: drink a glass of water and step outside for 2 minutes.",
        "Text a friend one sentence about how you feel.",
        "Write 3 things that went okay today."
    ],
    "fear": [
        "Box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s (x4).",
        "Name the worry in one line; write one thing you can control.",
        "Reduce stimulation: 2 minutes eyes closed, slow breaths."
    ],
    "anger": [
        "Pause: 10 slow breaths, then re-evaluate.",
        "Write what you want to say; wait 20 min; rewrite kindly.",
        "Move your body for 2–5 minutes."
    ],
    "disgust": [
        "Notice & name it; ask: what value is being violated?",
        "Tidy a tiny area for 3 minutes.",
        "Rinse face with cool water to reset senses."
    ],
    "neutral": [
        "2-minute stretch; relax jaw/shoulders.",
        "Write one small intention for the next hour.",
        "Hydrate or have a healthy snack."
    ],
    "joy": [
        "Savor it: describe the moment with 3 adjectives.",
        "Share it with someone; celebrate a tiny win.",
        "Capture it with a quick photo or one-line journal."
    ],
    "surprise": [
        "If good: jot why it matters; if tough: name a next step.",
        "Grounding: 5 things you see, 4 feel, 3 hear.",
        "Reality-check: message someone you trust."
    ]
}
QUOTES = {
    "sadness": [
        "“Sometimes the bravest thing you can do is ask for help.”",
        "“If you’re going through hell, keep going.” — Churchill"
    ],
    "fear": [
        "“Feelings are just visitors; let them come and go.”",
        "“Do the thing you fear and the death of fear is certain.” — Emerson"
    ],
    "anger": [
        "“You don’t have to attend every argument you’re invited to.”",
        "“Speak when you are angry and you’ll make the best speech you’ll ever regret.” — Bierce"
    ],
    "neutral": [
        "“Small steps still move you forward.”",
        "“Every day may not be good, but there’s something good in every day.”"
    ],
    "joy": [
        "“Joy is not in things; it is in us.” — Wagner",
        "“Savor the little things; someday they’re the big things.”"
    ],
    "surprise": [
        "“The art of life is to adjust to the unexpected.”",
        "“Life is full of surprises, but never when you need one.” — C&H"
    ]
}

def pick_tips(emotion: str):
    return TIPS.get(emotion.lower(), TIPS["neutral"])

def pick_quote(emotion: str):
    from random import choice
    return choice(QUOTES.get(emotion.lower(), QUOTES["neutral"]))

# ---------------- Analyze & save ----------------
if analyze_btn:
    if user_text.strip():
        result = analyze(user_text.strip())
        emo = result["emotion_top"]["label"]
        emo_conf = result["emotion_top"]["confidence"]
        sent = result["sentiment_top"]["label"]
        sent_conf = result["sentiment_top"]["confidence"]

        st.success(f"Top emotion: **{emo}** ({emo_conf:.2f}) • Sentiment: **{sent}** ({sent_conf:.2f})")
        with st.expander("All emotion scores"):
            st.write(pd.DataFrame(result["emotion_all"]))
        with st.expander("All sentiment scores"):
            st.write(pd.DataFrame(result["sentiment_all"]))

        # Save to journal (auto-create folder)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        row = {
            "timestamp": ts,
            "text": user_text.strip(),
            "emotion": emo,
            "emotion_conf": emo_conf,
            "sentiment": sent,
            "sentiment_conf": sent_conf
        }
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(DATA_PATH, index=False)
        st.toast("Saved to data/journal.csv")

        if tips_on:
            st.subheader("🧩 Supportive suggestions")
            for tip in pick_tips(emo):
                st.write("• " + tip)
            st.subheader("📝 Quote")
            st.info(pick_quote(emo))
    else:
        st.warning("Please enter a message first.")

# ---------------- Trend chart w/ migration & recent filter ----------------
st.divider()
st.subheader("📈 Your emotion trend")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    # migrate old files that used 'top_emotion'
    if "emotion" not in df.columns and "top_emotion" in df.columns:
        df = df.rename(columns={"top_emotion": "emotion"})
        df.to_csv(DATA_PATH, index=False)

    if "emotion" in df.columns and not df.empty:
        df_recent = df.tail(int(show_last_n))
        counts = df_recent["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
        st.bar_chart(counts.set_index("emotion"))
    else:
        st.info("No emotion data yet. Analyze a message to start your journal.")
else:
    st.info("No entries yet. Analyze a message to start your journal.")
