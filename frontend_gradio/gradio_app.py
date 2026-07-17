"""
Gradio interface for the Fraud Website Detection System.
Layout/wiring only — static data lives in gradio_data.py, API calls and
HTML-rendering helpers live in gradio_api.py, styling lives in style.css.
"""

import os
import gradio as gr

from .gradio_data import EXAMPLES
from .gradio_api import (
    check_api_health, analyze_url, batch_analyze, extract_features_display,
    build_importance_bars_html, build_model_insights_html,
    build_feature_accordion_html, example_loaded_toast,
)

# ------------------------------------------------------------------
# CSS is loaded from an external file to keep this module readable.
# ------------------------------------------------------------------
_CSS_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'style.css')
with open(_CSS_PATH, 'r', encoding='utf-8') as _f:
    CSS = _f.read()


def create_gradio_interface() -> gr.Blocks:
    with gr.Blocks(title="Fraud Website Detector", css=CSS) as app:

        # ── Header ────────────────────────────────────────────────────
        gr.Markdown("""
# 🔍 Fraud Website Detection System
AI-powered phishing URL analyser — XGBoost model + 5-layer rule engine
""", elem_id="header-md")

        # ── API status ────────────────────────────────────────────────
        with gr.Row(elem_id="api-row"):
            api_status = gr.Textbox(
                value=check_api_health(),
                interactive=False,
                show_label=False,
                elem_id="api_status_box"
            )
            refresh_btn = gr.Button("🔄 Refresh Status", elem_id="refresh_btn")
        refresh_btn.click(
            fn=lambda: gr.update(value="⏳ Checking…", interactive=False),
            inputs=None, outputs=refresh_btn, queue=False
        ).then(
            fn=check_api_health, outputs=api_status
        ).then(
            fn=lambda: gr.update(value="🔄 Refresh Status", interactive=True),
            inputs=None, outputs=refresh_btn, queue=False
        )

        gr.HTML('<div class="section-divider"></div>')

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════
            # TAB 1 — Single URL
            # ══════════════════════════════════════════════════════════
            with gr.Tab("🔍 Single URL"):
                gr.HTML('<p class="section-heading">Analyse a URL</p>')

                url_input = gr.Textbox(
                    placeholder="https://example.com",
                    show_label=False,
                    lines=1,
                    elem_id="url_input"
                )
                example_status = gr.HTML("", elem_id="example_status")

                analyze_btn = gr.Button(
                    "🔍 Analyse URL", variant="primary", elem_id="analyze_btn")

                with gr.Row(elem_id="output-row"):
                    result_output = gr.Textbox(
                        label="Result",
                        lines=6,
                        interactive=False,
                        elem_id="result_box"
                    )
                    warnings_output = gr.Textbox(
                        label="Warning Indicators",
                        lines=6,
                        interactive=False,
                        elem_id="warning_box"
                    )

                analyze_btn.click(
                    fn=lambda: gr.update(
                        value="⏳ Analysing…", interactive=False),
                    inputs=None, outputs=analyze_btn, queue=False
                ).then(
                    fn=analyze_url, inputs=[url_input], outputs=[
                        result_output, warnings_output]
                ).then(
                    fn=lambda: gr.update(
                        value="🔍 Analyse URL", interactive=True),
                    inputs=None, outputs=analyze_btn, queue=False
                )

                gr.HTML('<div class="section-divider"></div>')

                gr.HTML('<p class="section-heading">📋 Try These Examples</p>')
                gr.HTML(
                    '<p style="color:#96a0b8;font-size:0.88rem;margin-bottom:12px">Click any URL to load it into the analyser above.</p>')

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=url_input,
                    outputs=example_status,
                    fn=example_loaded_toast,
                    label=None,
                    examples_per_page=17,
                )

                gr.HTML('<div class="section-divider"></div>')

                gr.HTML(
                    '<p class="section-heading">🔬 What Does the Model Check?</p>')
                gr.HTML('<p style="color:#96a0b8;font-size:0.88rem;margin-bottom:16px">The model analyses 28 signals from every URL. Click any feature to read a plain-English explanation.</p>')
                gr.HTML(build_feature_accordion_html())

            # ══════════════════════════════════════════════════════════
            # TAB 2 — Batch Analysis
            # ══════════════════════════════════════════════════════════
            with gr.Tab("📦 Batch Analysis"):
                gr.HTML('<p class="section-heading">Analyse Multiple URLs</p>')
                gr.HTML('<p style="color:#96a0b8;font-size:0.88rem;margin-bottom:12px">Paste one URL per line, then run the batch. Uses the same override chain as single-URL analysis.</p>')

                batch_input = gr.Textbox(
                    placeholder="http://paypa1-secure.com/verify\nhttps://www.paypal.com\n...",
                    show_label=False,
                    lines=8,
                    elem_id="batch_input"
                )
                batch_btn = gr.Button(
                    "📦 Run Batch Analysis", elem_id="batch_btn")

                batch_output = gr.Dataframe(
                    headers=["URL", "Verdict", "Confidence",
                             "Phishing Probability"],
                    interactive=False,
                    wrap=True,
                    elem_id="batch_output"
                )
                batch_btn.click(
                    fn=lambda: gr.update(
                        value="⏳ Running batch…", interactive=False),
                    inputs=None, outputs=batch_btn, queue=False
                ).then(
                    fn=batch_analyze, inputs=[
                        batch_input], outputs=[batch_output]
                ).then(
                    fn=lambda: gr.update(
                        value="📦 Run Batch Analysis", interactive=True),
                    inputs=None, outputs=batch_btn, queue=False
                )

            # ══════════════════════════════════════════════════════════
            # TAB 3 — Feature Extraction
            # ══════════════════════════════════════════════════════════
            with gr.Tab("🧬 Feature Extraction"):
                gr.HTML(
                    '<p class="section-heading">Inspect a URL\'s Raw Features</p>')
                gr.HTML('<p style="color:#96a0b8;font-size:0.88rem;margin-bottom:12px">Enter a URL to see all 28 engineered features the model actually sees, grouped by category.</p>')

                feat_url_input = gr.Textbox(
                    placeholder="https://example.com",
                    show_label=False,
                    lines=1,
                    elem_id="feat_url_input"
                )
                feat_btn = gr.Button(
                    "🧬 Extract Features", variant="primary", elem_id="feat_extract_btn")

                feat_output = gr.HTML("")
                feat_btn.click(
                    fn=lambda: gr.update(
                        value="⏳ Extracting…", interactive=False),
                    inputs=None, outputs=feat_btn, queue=False
                ).then(
                    fn=extract_features_display, inputs=[
                        feat_url_input], outputs=[feat_output]
                ).then(
                    fn=lambda: gr.update(
                        value="🧬 Extract Features", interactive=True),
                    inputs=None, outputs=feat_btn, queue=False
                )

            # ══════════════════════════════════════════════════════════
            # TAB 4 — Model Insights
            # ══════════════════════════════════════════════════════════
            with gr.Tab("📊 Model Insights"):
                gr.HTML('<p class="section-heading">Model Performance</p>')
                gr.HTML(build_model_insights_html())

                with gr.Row(elem_id="importance-heading-row"):
                    gr.HTML(
                        '<p class="section-heading" style="margin-top:24px">Feature Importance (XGBoost)</p>')
                    importance_refresh_btn = gr.Button(
                        "🔄 Refresh", elem_id="importance_refresh_btn")
                gr.HTML('<p style="color:#96a0b8;font-size:0.88rem;margin-bottom:4px">How much weight each of the 28 features carries in the model\'s decision, ranked highest to lowest — pulled live from the trained model.</p>')

                importance_chart = gr.HTML("")
                importance_refresh_btn.click(
                    fn=lambda: gr.update(value="⏳", interactive=False),
                    inputs=None, outputs=importance_refresh_btn, queue=False
                ).then(
                    fn=build_importance_bars_html, outputs=[importance_chart]
                ).then(
                    fn=lambda: gr.update(value="🔄 Refresh", interactive=True),
                    inputs=None, outputs=importance_refresh_btn, queue=False
                )

        gr.HTML('<div class="section-divider"></div>')
        gr.HTML(
            '<div style="text-align:center;padding:12px 0 4px;color:#96a0b8;font-size:0.85rem">'
            'Built by <a href="https://github.com/NabxCode" target="_blank" '
            'style="color:#22d3ee;text-decoration:none">@NabxCode</a> · '
            'Fraud Website Detection System</div>'
        )

    return app


if __name__ == "__main__":
    server_port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    app = create_gradio_interface()

    print("=" * 60)
    print("🚀 Starting Gradio Interface")
    print("=" * 60)
    print(f"Flask API:   python -m backend_flask.app")
    print(f"Gradio UI:   http://localhost:{server_port}")
    print("=" * 60)

    app.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        theme=gr.themes.Soft(),
        show_error=True,
        quiet=False
    )
