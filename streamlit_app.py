
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import base64


#background
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #1e1e2f; /* Dark navy blue background */
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }

    .stApp {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Load model and tokenizer


def load_model():
    model_name = "unitary/toxic-bert"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model. The library will handle placing it on the CPU by default.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()
    return tokenizer, model
tokenizer, model = load_model()


ALL_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Toxic labels used by the model
LABELS = ["toxic", "threat"]
SELECTED_INDICES = [ALL_LABELS.index(label) for label in LABELS]
# App UI
st.title("Toxic Comment Classifier")
st.logo('logo.jpg', size='large')


user_input = st.text_area("Enter a comment:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            scores = F.sigmoid(outputs["logits"])[0].cpu().numpy()
            selected_scores = scores[SELECTED_INDICES]
        isDanger = False
        for score in scores:
            if score >= 0.5:
                st.badge("Danger", icon="🚨", color="red")
                set_bg_from_local("toxicBackground.png")
                isDanger=True
                break
        if not isDanger:
            st.badge("OK", icon="👌", color="green")


        df_scores = pd.DataFrame({
            'Toxicity Type': LABELS,
            'Score': selected_scores
        })

        # Display results
        st.subheader("📊 Toxicity Scores:")
        st.bar_chart(df_scores.set_index("Toxicity Type"))



        # Show raw scores as text too
        st.subheader("🔢 Raw Scores:")
        for label, score in zip(LABELS, selected_scores):
            st.write(f"**{label}**: {score:.2f}")