import pytest
from spatialist.vector import bbox

from cesard.metadata.extract import geometry_from_vec


@pytest.mark.parametrize(
    'coordinates, crs, expected_bbox, expected_center, expected_native, expected_geometry_type',
    [
        pytest.param(
            {
                'xmin': 10,
                'xmax': 11,
                'ymin': 50,
                'ymax': 51,
            },
            4326,
            [10, 50, 11, 51],
            '50.5 10.5',
            None,
            'Polygon',
            id='geographic',
        ),
        pytest.param(
            {
                'xmin': 600000,
                'xmax': 709800,
                'ymin': 5790240,
                'ymax': 5900040,
            },
            32660,
            pytest.approx([178, 52, -179, 53], abs=1),
            '52.73138719106183 179.3033751358137',
            [600000, 5790240, 709800, 5900040],
            'MultiPolygon',
            id='utm',
        ),
        pytest.param(
            {
                'xmin': 178,
                'xmax': -178,
                'ymin': 50,
                'ymax': 51,
            },
            4326,
            [178, 50, -178, 51],
            '50.5 180.0',
            None,
            'MultiPolygon',
            id='antimeridian',
        ),
    ],
)
def test_geometry_from_vec(
        coordinates: dict[str, int | float],
        crs: int,
        expected_bbox: list[int | float],
        expected_center: str,
        expected_native: list[int | float] | None,
        expected_geometry_type: str,
) -> None:
    with bbox(coordinates=coordinates, crs=crs) as vector:
        result = geometry_from_vec(vector)
    
    assert result['bbox'] == expected_bbox
    assert result['center'] == pytest.approx(expected_center)
    assert result['geometry']['type'] == expected_geometry_type
    assert isinstance(result['envelope'], str)
    
    if expected_native is None:
        assert 'bbox_native' not in result
    else:
        assert result['bbox_native'] == expected_native
