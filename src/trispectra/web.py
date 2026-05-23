from __future__ import annotations

import gradio as gr

from .inference import TriSpectraPredictor

APP_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, #f4d8bf 0%, transparent 28%),
    radial-gradient(circle at bottom right, #c6d6cf 0%, transparent 32%),
    linear-gradient(180deg, #f7f3eb 0%, #eee6da 100%);
}
.app-shell {
  max-width: 1100px;
  margin: 0 auto;
}
.hero {
  padding: 1.2rem 1.4rem;
  border: 1px solid rgba(58, 56, 51, 0.12);
  border-radius: 24px;
  background: rgba(255, 252, 246, 0.82);
  backdrop-filter: blur(8px);
}
.hero h1 {
  margin: 0;
  font-size: 2.4rem;
  letter-spacing: -0.04em;
}
.hero p {
  margin: 0.6rem 0 0 0;
  color: #514c43;
  font-size: 1rem;
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="slate",
)


def build_app() -> gr.Blocks:
    predictor = TriSpectraPredictor()

    def run_prediction(audio_path: str | None):
        if not audio_path:
            raise gr.Error("Upload or record an audio clip before running inference.")
        return predictor.predict_for_gradio(audio_path)

    with gr.Blocks(title="TriSpectra Audio Deepfake Detector") as app:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="hero">
                  <h1>TriSpectra Audio Deepfake Detector</h1>
                  <p>
                    Upload an audio file and the app will score 1-second chunks with the
                    published TriSpectra checkpoint from Hugging Face.
                  </p>
                </section>
                """
            )

            with gr.Row():
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio Input",
                )
                label_output = gr.Label(label="Prediction", num_top_classes=2)

            details_output = gr.JSON(label="Inference Details")
            submit = gr.Button("Analyze Audio", variant="primary")

            submit.click(
                fn=run_prediction,
                inputs=audio_input,
                outputs=[label_output, details_output],
            )

            gr.Markdown(
                "Supported formats depend on your local audio backend, but WAV, MP3, FLAC, and M4A are the intended targets."
            )

    return app
