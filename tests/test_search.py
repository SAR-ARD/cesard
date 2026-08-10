import pytest
from cesard.search import ASFArchive
from spatialist.vector import bbox


@pytest.mark.parametrize(
    'extent',
    [
        {'xmin': 10, 'xmax': 10.5, 'ymin': 50, 'ymax': 50.5},
        {'xmin': 177.5, 'xmax': -174.5, 'ymin': 50, 'ymax': 53},
    ],
    ids=[
        'regular',
        'antimeridian',
    ],
)
def test_asf_antimeridian(extent):
    with bbox(extent, 4326) as box:
        with ASFArchive() as db:
            scenes = db.select(sensor='S1A', product='GRD', vectorobject=box,
                               mindate='2025-01-01', maxdate='2025-01-31',
                               return_value='ASF')
    assert len(scenes) == 14
