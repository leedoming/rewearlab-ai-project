from PIL import Image

from retrieval.embedding import embed_image


class _FakeTensor:
    def __init__(self, size):
        self.size = size

    def unsqueeze(self, _dim):
        return self

    def to(self, _device):
        return self


class _FakeFeatures:
    def __init__(self, value):
        self.value = value

    def norm(self, dim=-1, keepdim=True):
        return _FakeFeatures(1.0)

    def __truediv__(self, _other):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.array([self.value])


class _FakeModel:
    def encode_image(self, tensor):
        return _FakeFeatures(tensor.size)


def _fake_preprocess(image):
    return _FakeTensor(image.size)


def test_embed_image_without_letterbox_preprocesses_the_image_unchanged():
    image = Image.new("RGB", (50, 200))
    result = embed_image(image, _FakeModel(), _fake_preprocess, device="cpu")
    assert tuple(result) == (50, 200)


def test_embed_image_with_letterbox_pads_before_preprocessing():
    image = Image.new("RGB", (50, 200))
    result = embed_image(image, _FakeModel(), _fake_preprocess, device="cpu", letterbox=True)
    assert tuple(result) == (200, 200)
