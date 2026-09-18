import pytest
from PIL import Image

from retrieval.preprocessing import crop_image


def make_image(size=(100, 100)):
    return Image.new("RGB", size)


def test_crop_within_bounds():
    image = make_image()
    cropped = crop_image(image, [10, 20, 60, 70])
    assert cropped.size == (50, 50)


def test_crop_clamps_negative_coordinates():
    image = make_image()
    cropped = crop_image(image, [-20, -20, 30, 30])
    assert cropped.size == (30, 30)


def test_crop_clamps_coordinates_beyond_image_bounds():
    image = make_image(size=(100, 100))
    cropped = crop_image(image, [50, 50, 500, 500])
    assert cropped.size == (50, 50)


def test_crop_fully_out_of_bounds_bbox_raises():
    # Pre-existing behavior inherited from the original crop_image()
    # duplicated across musinsa_to_chromadb.py / musinsa_detect.py / app.py:
    # clamping x1/y1 up and x2/y2 down independently can leave x2 < x1 when
    # the bbox is entirely outside the image, and PIL raises. Real detector
    # output never produces this (boxes are always within the source image),
    # so this milestone preserves rather than changes the behavior.
    image = make_image(size=(100, 100))
    with pytest.raises(ValueError):
        crop_image(image, [200, 200, 300, 300])


def test_crop_with_padding_expands_and_still_clamps():
    image = make_image(size=(100, 100))
    # bbox near the edge; 50% padding should push past the boundary and clamp.
    cropped = crop_image(image, [0, 0, 20, 20], padding_ratio=0.5)
    # padding adds 10px each side -> [-10, -10, 30, 30] -> clamped to [0, 0, 30, 30]
    assert cropped.size == (30, 30)


def test_crop_with_zero_padding_matches_no_padding_arg():
    image = make_image()
    assert crop_image(image, [10, 10, 40, 40]).size == crop_image(image, [10, 10, 40, 40], padding_ratio=0.0).size
