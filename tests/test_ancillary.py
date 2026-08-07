import pytest
from spatialist.vector import bbox
from cesard.ancillary import combine_polygons
from osgeo import ogr


@pytest.mark.parametrize(
    'input_extents, expected_extent',
    [
        (
                (
                        {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51},
                        {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51},
                ),
                {'xmin': 10, 'xmax': 12, 'ymin': 50, 'ymax': 51},
        ),
    ],
)
def test_combine_polygons_preserves_polygons(
        input_extents,
        expected_extent,
):
    vectors = [bbox(extent, 4326) for extent in input_extents]
    
    try:
        with combine_polygons(vectors) as combined:
            assert combined.extent == expected_extent
            assert combined.nfeatures == 2
            assert combined.geomType == ogr.wkbPolygon
    finally:
        for vector in vectors:
            vector.close()


def test_combine_polygons_multipolygon_roundtrip():
    ext1 = {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51}
    ext2 = {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51}
    ext3 = {'xmin': 21, 'xmax': 22, 'ymin': 50, 'ymax': 51}
    
    with bbox(ext1, 4326) as vec1:
        with bbox(ext2, 4326) as vec2:
            # Two Polygon -> one MultiPolygon.
            with combine_polygons(
                    [vec1, vec2],
                    multipolygon=True,
            ) as multipolygon:
                assert multipolygon.nfeatures == 1
                assert multipolygon.geomType == ogr.wkbMultiPolygon
                
                # One MultiPolygon -> two Polygon.
                with combine_polygons(
                        multipolygon,
                        explode=True,
                ) as exploded:
                    assert exploded.nfeatures == 2
                    assert exploded.geomType == ogr.wkbPolygon
                
                # One MultiPolygon -> one MultiPolygon.
                with combine_polygons(
                        multipolygon,
                        multipolygon=True,
                ) as multipolygon_again:
                    assert multipolygon_again.nfeatures == 1
                    assert multipolygon_again.geomType == ogr.wkbMultiPolygon
                
                # MultiPolygon + Polygon -> three Polygon.
                with bbox(ext3, 4326) as vec3:
                    with combine_polygons(
                            [multipolygon, vec3],
                            explode=True,
                    ) as combined:
                        assert combined.nfeatures == 3
                        assert combined.geomType == ogr.wkbPolygon


@pytest.mark.parametrize(
    'explode, multipolygon, expected_features, expected_geom_type',
    [
        (True, False, 2, ogr.wkbPolygon),
        (False, True, 1, ogr.wkbMultiPolygon),
    ],
    ids=[
        'split-into-polygons',
        'retain-as-multipolygon',
    ],
)
def test_combine_polygons_antimeridian(
        explode,
        multipolygon,
        expected_features,
        expected_geom_type,
):
    antimeridian_extent = {
        'xmin': 179,
        'xmax': -179,
        'ymin': 50,
        'ymax': 51,
    }
    
    with bbox(antimeridian_extent, 4326) as vector:
        with combine_polygons(
                vector,
                explode=explode,
                multipolygon=multipolygon,
        ) as combined:
            assert combined.nfeatures == expected_features
            assert combined.geomType == expected_geom_type


def test_combine_polygons_mixed_antimeridian():
    ordinary_extent = {
        'xmin': 10,
        'xmax': 11,
        'ymin': 50,
        'ymax': 51,
    }
    antimeridian_extent = {
        'xmin': 179,
        'xmax': -179,
        'ymin': 50,
        'ymax': 51,
    }
    
    with bbox(ordinary_extent, 4326) as ordinary:
        with bbox(antimeridian_extent, 4326) as antimeridian:
            # One ordinary polygon plus two polygons created by splitting
            # the antimeridian-crossing polygon.
            with combine_polygons(
                    [ordinary, antimeridian],
                    explode=True,
            ) as combined:
                assert combined.nfeatures == 3
                assert combined.geomType == ogr.wkbPolygon
