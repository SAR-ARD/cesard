from spatialist.vector import bbox
from cesard.ancillary import combine_polygons
from osgeo import ogr

def test_combine_polygons():
    ext1 = {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51}
    ext2 = {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51}
    ext3 = {'xmin': 21, 'xmax': 22, 'ymin': 50, 'ymax': 51}
    ext12 = {'xmin': 10, 'xmax': 12, 'ymin': 50, 'ymax': 51}
    with bbox(ext1, 4326) as vec1:
        with bbox(ext2, 4326) as vec2:
            # 2x Polygon -> 2x Polygon
            with combine_polygons([vec1, vec2]) as vec3:
                assert vec3.extent == ext12
                assert vec3.nfeatures == 2
                assert vec3.geomType == ogr.wkbPolygon
            # 2x Polygon -> 1x MultiPolygon
            with combine_polygons([vec1, vec2], multipolygon=True) as vec3:
                assert vec3.extent == ext12
                assert vec3.nfeatures == 1
                assert vec3.geomType == ogr.wkbMultiPolygon
                # 1x MultiPolygon -> 2x Polygon
                with combine_polygons([vec3], explode=True) as vec4:
                    assert vec4.nfeatures == 2
                    assert vec4.geomType == ogr.wkbPolygon
                # 1x MultiPolygon -> 1x MultiPolygon
                with combine_polygons([vec3], multipolygon=True) as vec4:
                    assert vec4.nfeatures == 1
                    assert vec4.geomType == ogr.wkbMultiPolygon
                # 1x Polygon + 1x MultiPolygon -> 3x Polygon
                with bbox(ext3, 4326) as vec5:
                    with combine_polygons([vec3, vec5], explode=True, multipolygon=False) as vec6:
                        assert vec6.nfeatures == 3
                        assert vec6.geomType == ogr.wkbPolygon
