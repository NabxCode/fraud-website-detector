import os
import gradio as gr
import joblib
import pandas as pd
from typing import Dict, Tuple
from src.feature_extraction import URLFeatureExtractor

# Load model and extractor
model = joblib.load("models/phishing_detector.pkl")
extractor = URLFeatureExtractor()


def analyze_url(url: str) -> Tuple[str, str, Dict]:
    if not url or not url.strip():
        return "Please enter a URL", "", {}

    try:
        features = extractor.extract_all_features(url)

        # Keep only numeric features
        numeric_features = {
            k: v for k, v in features.items() if isinstance(v, (int, float))
        }

        # Convert to DataFrame (FIX)
        feature_df = pd.DataFrame([numeric_features])

        # Predictions
        proba = model.predict_proba(feature_df)[0]
        pred = model.predict(feature_df)[0]

        # Result formatting
        if pred == 1:
            result_text = f"🔴 PHISHING DETECTED\n\n"
            result_text += f"Confidence: {max(proba):.1%}\n"
            result_text += f"Phishing Probability: {proba[1]:.1%}\n"
        else:
            result_text = f"🟢 LEGITIMATE WEBSITE\n\n"
            result_text += f"Confidence: {max(proba):.1%}\n"
            result_text += f"Phishing Probability: {proba[1]:.1%}\n"

        # Warnings
        warnings = []

        if not features.get("has_https", 0):
            warnings.append("• No HTTPS")

        if features.get("has_ip_address", 0):
            warnings.append("• Contains IP address")

        if features.get("typosquatting_score", 0):
            warnings.append("• Possible brand impersonation")

        if (
            features.get("domain_age_days", -1) >= 0
            and features.get("domain_age_days", 999) < 30
        ):
            warnings.append("• Domain is very new")

        warning_text = "\n".join(warnings) if warnings else "No major warnings"

        return result_text, warning_text, features

    except Exception as e:
        return f"❌ Error: {str(e)}", "", {}


# UI
with gr.Blocks(title="AI Phishing Detector") as app:

    gr.Markdown("""
    # 🔐 AI Phishing Detector  
    ### Real-time Fraud URL Analysis System
    """)

    url_input = gr.Textbox(
        label="Enter URL",
        placeholder="https://example.com"
    )

    # Sample URLs (for demo)
    gr.Examples(
        examples=[
            "https://google.com",
            "https://facebook.com",
            "http://192.168.1.1/login",
            "http://paypal-login-secure.com",
            "http://free-gift-card-amazon.xyz"
        ],
        inputs=url_input
    )

    analyze_btn = gr.Button("Analyze URL")

    result_output = gr.Textbox(label="Result", lines=6)
    warning_output = gr.Textbox(label="Warnings", lines=6)
    feature_output = gr.JSON(label="Extracted Features")

    analyze_btn.click(
        fn=analyze_url,
        inputs=[url_input],
        outputs=[result_output, warning_output, feature_output]
    )


# Launch for Render
port = int(os.environ.get("PORT", 7860))

app.launch(
    server_name="0.0.0.0",
    server_port=port
)