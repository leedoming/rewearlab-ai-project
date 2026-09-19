"""Model loading for the retrieval core.

Only loads models. No Streamlit, no detection/embedding/search logic.
Heavy dependencies (torch, transformers, open_clip) are imported lazily
inside the functions so importing this module never requires them, and
Streamlit's ``@st.cache_resource`` (or any other caching) is applied by
the UI layer around these functions, not in here (see IMPLEMENTATION_SPEC.md
section 6, "SHOULD").
"""

from .config import DETECTION_MODEL, EMBEDDING_MODEL


def _resolve_device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_detection_model(model_name=DETECTION_MODEL, device=None):
    """Load the fashion object-detection model + processor.

    Returns (image_processor, model, device), matching the tuple shape
    used by the existing Streamlit apps.
    """
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    resolved_device = _resolve_device(device)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    # .eval() matters, not just style: without it the model stays in the
    # train()-mode PyTorch defaults to after construction, so dropout (and
    # any batchnorm running-stats update) would stay active during
    # inference if this model had either. Added defensively alongside the
    # load_embedding_model fix below (found while investigating real
    # run-to-run non-determinism during Milestone 10's hubness pilot --
    # docs/evidence/milestone-10-hubness.md); this predates the Milestone 1
    # refactor (neither the original main-app/app.py nor
    # search-app/musinsa_detect.py ever called .eval() either).
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(resolved_device).eval()
    return image_processor, model, resolved_device


def load_embedding_model(model_name=EMBEDDING_MODEL, device=None):
    """Load the CLIP-family embedding model + its preprocessing transform.

    Returns (model, preprocess_val, device), matching the tuple shape used
    by the existing Streamlit apps.
    """
    import open_clip

    resolved_device = _resolve_device(device)
    # open_clip.create_model_and_transforms returns
    # (model, preprocess_TRAIN, preprocess_VAL) -- in that order, per its
    # own source. The pre-refactor code (and this function, until this fix)
    # unpacked the second value into a variable named `preprocess_val` and
    # discarded the real one, so every embedding call -- catalog ingestion
    # and query embedding alike -- was actually run through the TRAINING
    # transform (`RandomResizedCrop(scale=(0.9, 1.0), ...)`), not the
    # deterministic validation one. This is the real cause of the run-to-run
    # embedding non-determinism found while validating a hub-mitigation fix
    # in Milestone 10 (docs/evidence/milestone-10-hubness.md) -- a model
    # forward pass on an identical input tensor was confirmed bit-identical;
    # only the "preprocess" step varied between calls on the same image.
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    # See load_detection_model's comment: .eval() is required for
    # deterministic, dropout-free inference, and was missing here too.
    model = model.to(resolved_device).eval()
    return model, preprocess_val, resolved_device
