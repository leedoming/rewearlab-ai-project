"""Tests for detect_fashion_items' argument-resolution contract.

Uses fakes for image_processor/model — never loads a real detection model
(no torch/transformers required for these cases).
"""

import pytest

from retrieval.detection import _resolve_detection_components


class FakeParameter:
    def __init__(self, device):
        self.device = device


class FakeModel:
    """Duck-types torch.nn.Module.parameters() enough to resolve a device."""

    def __init__(self, device="fake-cuda:0"):
        self._device = device

    def parameters(self):
        yield FakeParameter(self._device)


def test_both_omitted_loads_via_load_detection_model(monkeypatch):
    import retrieval.detection as detection_module

    sentinel = ("loaded-processor", "loaded-model", "loaded-device")
    monkeypatch.setattr(detection_module, "load_detection_model", lambda model_name, device: sentinel)

    result = _resolve_detection_components(None, None, None, "some-model-name")

    assert result == sentinel


def test_both_supplied_with_device_omitted_infers_device_from_model():
    processor = object()
    model = FakeModel(device="fake-cuda:0")

    resolved_processor, resolved_model, resolved_device = _resolve_detection_components(
        processor, model, None, "unused-model-name"
    )

    assert resolved_processor is processor
    assert resolved_model is model
    assert resolved_device == "fake-cuda:0"


def test_both_supplied_with_device_uses_it_as_is():
    processor = object()
    model = FakeModel(device="fake-cuda:0")

    resolved_processor, resolved_model, resolved_device = _resolve_detection_components(
        processor, model, "explicit-cpu", "unused-model-name"
    )

    assert resolved_device == "explicit-cpu"


def test_only_image_processor_supplied_raises():
    with pytest.raises(ValueError):
        _resolve_detection_components(object(), None, None, "unused-model-name")


def test_only_model_supplied_raises():
    with pytest.raises(ValueError):
        _resolve_detection_components(None, FakeModel(), None, "unused-model-name")


def test_only_model_supplied_with_device_still_raises():
    # Even with device given, model without image_processor is inconsistent.
    with pytest.raises(ValueError):
        _resolve_detection_components(None, FakeModel(), "explicit-cpu", "unused-model-name")
