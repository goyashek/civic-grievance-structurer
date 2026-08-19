"""Small local Gradio review app for the shared CivicStruct inference path."""

from __future__ import annotations

import argparse
from functools import partial
from threading import Lock
from typing import Any, Callable

from .inference import CivicStructInference, InferenceResult, MAX_COMPLAINT_CHARS


LIMITATION_NOTE = (
    "This is a local review demo. The examples are fictional, the model can return a visible "
    "failure, and any accepted JSON still needs human review before consequential use."
)
EXAMPLES = [
    "Bus 724 did not arrive near Nehru Place yesterday evening. I waited for almost an hour.",
    "There has been no water supply in Ward 4 since Monday morning. Please check the line.",
    "A streetlight beside the community clinic has been sparking at night for two days.",
]

_inference: CivicStructInference | None = None
_inference_lock = Lock()


def get_inference() -> CivicStructInference:
    """Load the frozen model only after the user submits a complaint."""

    global _inference
    if _inference is None:
        with _inference_lock:
            if _inference is None:
                _inference = CivicStructInference()
    return _inference


def _failure(
    error: dict[str, str], raw_response: str = "", latency: float = 0.0
) -> tuple[Any, str, str, str]:
    error_type = error.get("type", "inference_error")
    message = error.get("message", "the model did not return an accepted result")
    status = (
        "### Schema validation failed\n\n"
        f"`{error_type}`: {message}\n\n"
        f"Latency: `{latency:.2f}s`. No structured output was accepted."
    )
    return None, "", status, raw_response


def review_complaint(
    complaint: str, inference: CivicStructInference | None = None
) -> tuple[dict[str, Any] | None, str, str, str]:
    """Return structured JSON, summary, validation status, and raw response."""

    if not isinstance(complaint, str) or not complaint.strip():
        return _failure({"type": "invalid_input", "message": "complaint must contain text"})
    if len(complaint) > MAX_COMPLAINT_CHARS:
        return _failure(
            {
                "type": "invalid_input",
                "message": f"complaint must be at most {MAX_COMPLAINT_CHARS} characters",
            }
        )
    try:
        result: InferenceResult = (inference or get_inference()).infer(complaint)
    except Exception as exc:
        return _failure({"type": "load_error", "message": str(exc)})
    if not result.ok:
        return _failure(
            result.error or {"type": "inference_error", "message": "unknown failure"},
            result.raw_response or "",
            result.latency_seconds,
        )
    prediction = result.prediction or {}
    status = (
        "### Schema validation passed\n\n"
        f"Strict validation accepted the model output. Latency: `{result.latency_seconds:.2f}s`."
    )
    return (
        prediction,
        str(prediction.get("formal_summary", "")),
        status,
        result.raw_response or "",
    )


def build_demo(
    inference: CivicStructInference | None = None,
    handler: Callable[[str], tuple[dict[str, Any] | None, str, str, str]] | None = None,
):
    """Build the Gradio UI without loading the model."""

    import gradio as gr

    handler = handler or (
        review_complaint
        if inference is None
        else partial(review_complaint, inference=inference)
    )
    with gr.Blocks(title="CivicStruct review demo") as demo:
        gr.Markdown(
            "# CivicStruct review demo\n"
            "Turn one public-service complaint into a strictly validated JSON record."
        )
        gr.Markdown(LIMITATION_NOTE)
        with gr.Row():
            with gr.Column():
                complaint = gr.Textbox(
                    label="Complaint",
                    placeholder="Describe one public-service problem...",
                    lines=8,
                )
                run = gr.Button("Structure complaint", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=complaint, label="Fictional examples")
            with gr.Column():
                structured = gr.JSON(label="Structured JSON")
                summary = gr.Textbox(label="Formal summary", lines=3, interactive=False)
                status = gr.Markdown()
                raw = gr.Textbox(label="Raw model response", lines=6, interactive=False)
        run.click(handler, inputs=complaint, outputs=[structured, summary, status, raw])
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local CivicStruct Gradio review demo.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    build_demo().launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
