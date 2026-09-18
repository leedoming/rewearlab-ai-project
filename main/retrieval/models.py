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
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(resolved_device)
    return image_processor, model, resolved_device


def load_embedding_model(model_name=EMBEDDING_MODEL, device=None):
    """Load the CLIP-family embedding model + its preprocessing transform.

    Returns (model, preprocess_val, device), matching the tuple shape used
    by the existing Streamlit apps.
    """
    import open_clip

    resolved_device = _resolve_device(device)
    model, preprocess_val, _ = open_clip.create_model_and_transforms(model_name)
    model = model.to(resolved_device)
    return model, preprocess_val, resolved_device
