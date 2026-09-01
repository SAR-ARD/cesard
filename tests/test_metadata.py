import pytest
from spatialist.vector import bbox

from cesard.metadata.extract import geometry_from_vec


@pytest.mark.parametrize(
    'coordinates, crs, expected_bbox, expected_bbox_native, expected_center, expected_envelope, expected_geometry_type',
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
            None,
            '50.5 10.5',
            ['50.0 10.0 50.0 11.0 51.0 11.0 51.0 10.0 50.0 10.0'],
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
            [600000, 5790240, 709800, 5900040],
            '52.73138719106183 179.3033751358137',
            [
                '52.25345680616433 178.46498036715545 '
                '52.224411325476815 180.0 '
                '53.21185796769914 180.0 '
                '53.24020834899015 178.49846042833093 '
                '52.25345680616433 178.46498036715545',
                '53.20820137227573 -179.85823009552806 '
                '53.21185796764875 -180.0 '
                '52.22441132546851 -180.0 '
                '52.22441132542593 -180.0 '
                '52.2225660331335 -179.92832151296628 '
                '53.20820137227573 -179.85823009552806'],
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
            None,
            '50.5 180.0',
            ['50.0 178.0 50.0 180.0 51.0 180.0 51.0 178.0 50.0 178.0',
             '50.0 -180.0 50.0 -178.0 51.0 -178.0 51.0 -180.0 50.0 -180.0'],
            'MultiPolygon',
            id='antimeridian',
        ),
    ],
)
def test_geometry_from_vec(
        coordinates: dict[str, int | float],
        crs: int,
        expected_bbox: list[int | float],
        expected_bbox_native: list[int | float] | None,
        expected_center: str,
        expected_envelope: str,
        expected_geometry_type: str,
) -> None:
    with bbox(coordinates=coordinates, crs=crs) as vector:
        result = geometry_from_vec(vector)
    
    assert result['bbox'] == expected_bbox
    assert result['center'] == pytest.approx(expected_center)
    assert result['geometry']['type'] == expected_geometry_type
    assert result['envelope'] == expected_envelope
    
    if expected_bbox_native is None:
        assert 'bbox_native' not in result
    else:
        assert result['bbox_native'] == expected_bbox_native
