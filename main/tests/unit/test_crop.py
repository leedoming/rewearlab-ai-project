import pytest
from PIL import Image

from retrieval.preprocessing import crop_image, letterbox_to_square


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


def test_crop_rejects_negative_padding():
    with pytest.raises(ValueError, match="padding_ratio"):
        crop_image(make_image(), [10, 10, 40, 40], padding_ratio=-0.5)


def test_letterbox_pads_tall_image_to_square_without_scaling_content():
    image = make_image(size=(50, 200))
    padded = letterbox_to_square(image)
    assert padded.size == (200, 200)


def test_letterbox_pads_wide_image_to_square():
    image = make_image(size=(200, 50))
    padded = letterbox_to_square(image)
    assert padded.size == (200, 200)


def test_letterbox_is_a_noop_size_wise_on_already_square_image():
    image = make_image(size=(80, 80))
    padded = letterbox_to_square(image)
    assert padded.size == (80, 80)


def test_letterbox_centers_original_content_and_fills_the_rest():
    # A tall, distinctively-colored image on a square gray canvas: the
    # original pixels must land centered, and the padding must use the
    # given fill color, not leave the new canvas's default black.
    image = Image.new("RGB", (10, 20), (255, 0, 0))
    padded = letterbox_to_square(image, fill=(128, 128, 128))
    assert padded.size == (20, 20)
    assert padded.getpixel((0, 0)) == (128, 128, 128)  # padding column
    assert padded.getpixel((5, 10)) == (255, 0, 0)  # original content, centered
