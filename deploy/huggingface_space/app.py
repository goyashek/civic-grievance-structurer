import spaces

from src.demo import _failure, build_demo, review_complaint
from src.inference import CivicStructInference


inference = None


@spaces.GPU(duration=120)
def hosted_review(complaint: str):
    global inference
    try:
        if inference is None:
            inference = CivicStructInference()
        return review_complaint(complaint, inference=inference)
    except Exception as exc:
        return _failure({"type": "load_error", "message": str(exc)})


demo = build_demo(handler=hosted_review)
demo.queue().launch()
