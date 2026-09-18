"""Image embedding.

Performs only: PIL Image -> model preprocessing -> embedding ->
normalization. No Streamlit dependency, no model loading (see
`retrieval.models.load_embedding_model`).
"""


def embed_image(image, model, preprocess, device):
    """Embed a single PIL Image using an already-loaded CLIP-family model.

    Args:
        image: PIL Image.
        model: object with `.encode_image(tensor)` (e.g. an open_clip model).
        preprocess: callable PIL Image -> tensor (the model's val transform).
        device: torch device the model lives on.

    Returns:
        A 1-D numpy array: the L2-normalized image embedding.
    """
    import torch

    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]
