"""Image embedding.

Performs only: PIL Image -> model preprocessing -> embedding ->
normalization. No Streamlit dependency, no model loading (see
`retrieval.models.load_embedding_model`).
"""


def embed_image(image, model, preprocess, device, letterbox=False):
    """Embed a single PIL Image using an already-loaded CLIP-family model.

    Args:
        image: PIL Image.
        model: object with `.encode_image(tensor)` (e.g. an open_clip model).
        preprocess: callable PIL Image -> tensor (the model's val transform).
        device: torch device the model lives on.
        letterbox: if True, pad `image` to a square with
            `retrieval.preprocessing.letterbox_to_square` before handing it
            to `preprocess`, so the model's own non-aspect-preserving
            resize only scales the image rather than stretching it. Off by
            default to keep existing callers' behavior unchanged; see
            docs/evidence/milestone-10-letterbox.md for why this exists.

    Returns:
        A 1-D numpy array: the L2-normalized image embedding.
    """
    import torch

    if letterbox:
        from .preprocessing import letterbox_to_square

        image = letterbox_to_square(image)

    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]
