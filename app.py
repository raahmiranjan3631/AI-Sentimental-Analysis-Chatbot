import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# 1️⃣ Load Hugging Face Pipelines (CPU-safe, no meta tensor)
# -------------------------------
@st.cache_resource
def load_models():
    # Sentiment Analysis Model
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=-1  # CPU only
    )

    # Zero-Shot Intent Classification Model
    intent_model_name = "facebook/bart-large-mnli"
    intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name)
    intent_pipeline = pipeline(
        "zero-shot-classification",
        model=intent_model,
        tokenizer=intent_tokenizer,
        device=-1  # CPU only
    )

    return sentiment_pipeline, intent_pipeline

sentiment_analyzer, intent_classifier = load_models()

# Candidate intents
candidate_labels = ["complaint", "query", "feedback", "request", "greeting"]

# -------------------------------
# 2️⃣ Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# -------------------------------
# 3️⃣ Helper Functions
# -------------------------------
def analyze_sentiment(user_input):
    result = sentiment_analyzer(user_input)[0]
    label = result["label"].lower()
    score = result["score"]
    if "positive" in label:
        sentiment = "positive"
    elif "negative" in label:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return sentiment, score

def detect_intent(user_input):
    result = intent_classifier(user_input, candidate_labels)
    intent = result["labels"][0]
    confidence = result["scores"][0]
    return intent, confidence

# -------------------------------
# 4️⃣ Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["💬 Live Chat", "📊 Analytics Dashboard", "📜 Chat History"])

# -------------------------------
# 5️⃣ Live Chat
# -------------------------------
if page == "💬 Live Chat":
    st.title("🤖 Customer Support Chatbot with NLP & Intent AI")
    user_input = st.text_input("You:", "")
    
    if st.button("Send") and user_input:
        sentiment, sent_score = analyze_sentiment(user_input)
        intent, intent_score = detect_intent(user_input)

        # Save user message
        st.session_state["messages"].append({
            "user": user_input,
            "sentiment": sentiment,
            "sentiment_score": sent_score,
            "intent": intent,
            "intent_score": intent_score
        })

        # Bot response logic
        if intent == "complaint" and sentiment == "negative":
            bot_response = "I understand your frustration 😟. Let me escalate this to our support team."
        elif intent == "query":
            bot_response = "Thanks for your question! I'll provide the best possible answer."
        elif intent == "feedback":
            bot_response = "We really appreciate your feedback 💡. It helps us improve."
        elif intent == "greeting":
            bot_response = "Hello 👋! How can I assist you today?"
        else:
            bot_response = "Got it ✅. I’ll make sure this is noted."

        st.session_state["messages"].append({"bot": bot_response})

    # Display chat history
    for msg in st.session_state["messages"]:
        if "user" in msg:
            st.markdown(
                f"🧑 **You:** {msg['user']} "
                f"*(sentiment: {msg['sentiment']} | intent: {msg['intent']})*"
            )
        else:
            st.markdown(f"🤖 **Bot:** {msg['bot']}")

# -------------------------------
# 6️⃣ Analytics Dashboard
# -------------------------------
elif page == "📊 Analytics Dashboard":
    st.title("📊 Sentiment & Intent Analytics")

    if st.session_state["messages"]:
        user_msgs = [m for m in st.session_state["messages"] if "user" in m]
        df = pd.DataFrame(user_msgs)

        # Sentiment stats
        st.subheader("Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

        # Intent stats
        st.subheader("Intent Distribution")
        st.bar_chart(df["intent"].value_counts())
    else:
        st.info("No chat data yet. Start chatting!")

# -------------------------------
# 7️⃣ Chat History
# -------------------------------
elif page == "📜 Chat History":
    st.title("📜 Chat History")
    if st.session_state["messages"]:
        for msg in st.session_state["messages"]:
            if "user" in msg:
                st.markdown(
                    f"🧑 **You:** {msg['user']} "
                    f"*(sentiment: {msg['sentiment']} | intent: {msg['intent']})*"
                )
            else:
                st.markdown(f"🤖 **Bot:** {msg['bot']}")
    else:
        st.info("No chat history yet.")
