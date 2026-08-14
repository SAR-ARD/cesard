import os
from cesard.dem import to_mgrs
from spatialist import Raster


def test_to_mgrs(tmpdir):
    out = os.path.join(str(tmpdir), 'dem.tif')
    overviews = [2, 4, 9, 18, 36]
    create_options = [
        'BLOCKSIZE=512',
        'OVERVIEW_RESAMPLING=AVERAGE'
    ]
    to_mgrs(
        tile='32TPS',
        dst=out,
        dem_type='Copernicus 30m Global DEM',
        overviews=overviews,
        tr=(60, 60),
        create_options=create_options,
        threads=4
    )
    with Raster(out) as ras:
        assert ras.raster.RasterCount == 1
        band = ras.raster.GetRasterBand(1)
        assert band.GetBlockSize() == [512, 512]
        
        overviews_test = []
        overviews_test_resampling = set()
        for j in range(band.GetOverviewCount()):
            ovr = band.GetOverview(j)
            x_factor = round(ras.raster.RasterXSize / ovr.XSize)
            y_factor = round(ras.raster.RasterYSize / ovr.YSize)
            assert x_factor == y_factor
            overviews_test.append(x_factor)
            overviews_test_resampling.add(ovr.GetMetadataItem('RESAMPLING'))
        assert sorted(overviews_test) == overviews
        assert len(overviews_test_resampling) == 1
        assert overviews_test_resampling.pop() == 'AVERAGE'
        
        band = None
