import pytest
from argos.geometry import ImagePoint, PixelBox, normalized_to_pixel, pixel_to_normalized
from argos.models import RangeEstimate, TargetState


def test_box_center_and_target_normalized_coordinates():
    target = TargetState.from_box(7, PixelBox(20, 10, 40, 20), 200, 100)
    assert (target.pixel_x, target.pixel_y) == (40, 20)
    assert (target.normalized_x, target.normalized_y) == (0.2, 0.2)
    assert target.track_id == 7
    assert target.identity_label == "Unknown"
    assert target.range_estimate is None


def test_coordinate_conversion_round_trip():
    point = ImagePoint(125, 60)
    assert normalized_to_pixel(pixel_to_normalized(point, 500, 240), 500, 240) == point


@pytest.mark.parametrize("size", [(0, 1), (1, 0), (-1, 2)])
def test_invalid_dimensions_fail(size):
    with pytest.raises(ValueError):
        pixel_to_normalized(ImagePoint(1, 1), *size)


def test_range_estimate_is_explicit_and_neutral():
    result = RangeEstimate(2.4, 0.8, "stereo")
    assert result.meters == 2.4 and result.method == "stereo"
