import os
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import geopandas as gpd
from lxml import etree
from time import gmtime, strftime
from copy import deepcopy
from scipy.interpolate import RBFInterpolator
from osgeo import gdal
from spatialist.vector import bbox
from spatialist.raster import Raster, Dtype
from spatialist.auxil import gdalbuildvrt
from pyroSAR.drivers import ID

from cesard.ancillary import combine_polygons
import logging

log = logging.getLogger('cesard')


def create_vrt(
        src: str | list[str],
        dst: str,
        fun: str,
        relpaths: bool = False,
        scale: int | None = None,
        offset: float | None = None,
        dtype: str | None = None,
        args: dict[str, int | float | str] | None = None,
        options: dict | None = None,
        overviews: list[int] | None = None,
        overview_resampling: str | None = None
) -> None:
    """
    Create a GDAL VRT file executing an on-the-fly pixel function.

    Parameters
    ----------
    src:
        The input dataset(s).
    dst:
        The output dataset.
    fun:
        A `PixelFunctionType` that should be applied on the fly when opening the VRT file. The function is applied to a
        band that derives its pixel information from the source bands. A list of possible options can be found here:
        https://gdal.org/drivers/raster/vrt.html#default-pixel-functions.
        Furthermore, the option 'decibel' can be specified, which will implement a custom pixel function that uses
        Python code for decibel conversion (10*log10).
    relpaths:
        Should all `SourceFilename` XML elements with attribute `@relativeToVRT="0"` be updated to be paths relative to
        the output VRT file? Default is False.
    scale:
         The scale that should be applied when computing “real” pixel values from scaled pixel values on a raster band.
         Will be ignored if `fun='decibel'`.
    offset:
        The offset that should be applied when computing “real” pixel values from scaled pixel values on a raster band.
        Will be ignored if `fun='decibel'`.
    dtype:
        the data type of the written VRT file; default None: same data type as source data.
        data type notations of GDAL (e.g. `Float32`) and numpy (e.g. `int8`) are supported.
    args:
        arguments for `fun` passed as `PixelFunctionArguments`. Requires GDAL>=3.5 to be read.
    options:
        Additional parameters passed to :func:`osgeo.gdal.BuildVRT`.
    overviews:
        Internal overview levels to be created for each raster file.
    overview_resampling:
        Resampling method for overview levels.

    Examples
    --------
    linear gamma0 backscatter as input:

    >>> src = 's1a-iw-nrb-20220601t052704-043465-0530a1-32tpt-vh-g-lin.tif'

    decibel scaling I:
    use `log10` pixel function and additional `Scale` parameter.
    Known to display well in QGIS, but `Scale` is ignored when reading array in Python.

    >>> dst = src.replace('-lin.tif', '-log1.vrt')
    >>> create_vrt(src=src, dst=dst, fun='log10', scale=10)

    decibel scaling II:
    use custom Python pixel function. Requires additional environment variable GDAL_VRT_ENABLE_PYTHON set to YES.

    >>> dst = src.replace('-lin.tif', '-log2.vrt')
    >>> create_vrt(src=src, dst=dst, fun='decibel')

    decibel scaling III:
    use `dB` pixel function with additional `PixelFunctionArguments`. Works best but requires GDAL>=3.5.

    >>> dst = src.replace('-lin.tif', '-log3.vrt')
    >>> create_vrt(src=src, dst=dst, fun='dB', args={'fact': 10})
    """
    options = {} if options is None else options
    gdalbuildvrt(src=src, dst=dst, **options)
    tree = etree.parse(dst)
    root = tree.getroot()
    band = tree.find('VRTRasterBand')
    band.attrib['subClass'] = 'VRTDerivedRasterBand'
    
    if dtype is not None:
        band.attrib['dataType'] = Dtype(dtype).gdalstr
    
    if fun == 'decibel':
        pxfun_language = etree.SubElement(band, 'PixelFunctionLanguage')
        pxfun_language.text = 'Python'
        pxfun_type = etree.SubElement(band, 'PixelFunctionType')
        pxfun_type.text = fun
        pxfun_code = etree.SubElement(band, 'PixelFunctionCode')
        pxfun_code.text = etree.CDATA("""
    import numpy as np
    def decibel(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
        np.multiply(np.log10(in_ar[0], where=in_ar[0]>0.0, out=out_ar, dtype='float32'), 10.0, out=out_ar, dtype='float32')
        """)
    else:
        pixfun_type = etree.SubElement(band, 'PixelFunctionType')
        pixfun_type.text = fun
        if args is not None:
            arg = etree.SubElement(band, 'PixelFunctionArguments')
            for key, value in args.items():
                arg.attrib[key] = str(value)
        if scale is not None:
            sc = etree.SubElement(band, 'Scale')
            sc.text = str(scale)
        if offset is not None:
            off = etree.SubElement(band, 'Offset')
            off.text = str(offset)
    
    if any([overviews, overview_resampling]) is not None:
        ovr = tree.find('OverviewList')
        if ovr is None:
            ovr = etree.SubElement(root, 'OverviewList')
        if overview_resampling is not None:
            ovr.attrib['resampling'] = overview_resampling.lower()
        if overviews is not None:
            ov = str(overviews)
            for x in ['[', ']', ',']:
                ov = ov.replace(x, '')
            ovr.text = ov
    
    if relpaths:
        srcfiles = tree.xpath('//SourceFilename[@relativeToVRT="0"]')
        for srcfile in srcfiles:
            repl = os.path.relpath(srcfile.text, start=os.path.dirname(dst))
            repl = repl.replace('\\', '/')
            srcfile.text = repl
            srcfile.attrib['relativeToVRT'] = '1'
    
    etree.indent(root)
    tree.write(dst, pretty_print=True, xml_declaration=False, encoding='utf-8')


def create_rgb_vrt(
        outname: str,
        infiles: list[str],
        overviews: list[int],
        overview_resampling: str
) -> None:
    """
    Creation of the color composite GDAL VRT file.

    Parameters
    ----------
    outname:
        Full path to the output VRT file.
    infiles:
        A list of paths pointing to the linear scaled measurement backscatter files.
    overviews:
        Internal overview levels to be defined for the created VRT file.
    overview_resampling:
        Resampling method applied to overview pyramids.
    """
    
    # make sure order is right and co-polarization (VV or HH) is first
    pols = [re.search('[hv]{2}', os.path.basename(f)).group() for f in infiles]
    if pols[1] in ['vv', 'hh']:
        infiles.reverse()
        pols.reverse()
    
    # format overview levels
    ov = str(overviews)
    for x in ['[', ']', ',']:
        ov = ov.replace(x, '')
    
    # create VRT file and change its content
    gdalbuildvrt(src=infiles, dst=outname, separate=True)
    
    tree = etree.parse(outname)
    root = tree.getroot()
    srs = tree.find('SRS').text
    geotrans = tree.find('GeoTransform').text
    bands = tree.findall('VRTRasterBand')
    vrt_nodata = bands[0].find('NoDataValue').text
    complex_src = [band.find('ComplexSource') for band in bands]
    for cs in complex_src:
        cs.remove(cs.find('NODATA'))
    
    new_band = etree.SubElement(root, 'VRTRasterBand',
                                attrib={'dataType': 'Float32', 'band': '3',
                                        'subClass': 'VRTDerivedRasterBand'})
    new_band_na = etree.SubElement(new_band, 'NoDataValue')
    new_band_na.text = 'nan'
    pxfun_type = etree.SubElement(new_band, 'PixelFunctionType')
    pxfun_type.text = 'mul'
    for cs in complex_src:
        new_band.append(deepcopy(cs))
    
    src = new_band.findall('ComplexSource')[1]
    fname = src.find('SourceFilename')
    fname_old = fname.text
    src_attr = src.find('SourceProperties').attrib
    fname.text = etree.CDATA("""
    <VRTDataset rasterXSize="{rasterxsize}" rasterYSize="{rasterysize}">
        <SRS dataAxisToSRSAxisMapping="1,2">{srs}</SRS>
        <GeoTransform>{geotrans}</GeoTransform>
        <VRTRasterBand dataType="{dtype}" band="1" subClass="VRTDerivedRasterBand">
            <NoDataValue>{vrt_nodata}</NoDataValue>
            <PixelFunctionType>{px_fun}</PixelFunctionType>
            <ComplexSource>
              <SourceFilename relativeToVRT="1">{fname}</SourceFilename>
              <SourceBand>1</SourceBand>
              <SourceProperties RasterXSize="{rasterxsize}" RasterYSize="{rasterysize}" DataType="{dtype}" BlockXSize="{blockxsize}" BlockYSize="{blockysize}"/>
              <SrcRect xOff="0" yOff="0" xSize="{rasterxsize}" ySize="{rasterysize}"/>
              <DstRect xOff="0" yOff="0" xSize="{rasterxsize}" ySize="{rasterysize}"/>
            </ComplexSource>
        </VRTRasterBand>
        <OverviewList resampling="{ov_resampling}">{ov}</OverviewList>
    </VRTDataset>
    """.format(rasterxsize=src_attr['RasterXSize'], rasterysize=src_attr['RasterYSize'], srs=srs, geotrans=geotrans,
               dtype=src_attr['DataType'], px_fun='inv', fname=fname_old, vrt_nodata=vrt_nodata,
               blockxsize=src_attr['BlockXSize'], blockysize=src_attr['BlockYSize'],
               ov_resampling=overview_resampling.lower(), ov=ov))
    
    bands = tree.findall('VRTRasterBand')
    for band, col in zip(bands, ['Red', 'Green', 'Blue']):
        color = etree.Element('ColorInterp')
        color.text = col
        band.insert(0, color)
    
    ovr = etree.SubElement(root, 'OverviewList', attrib={'resampling': overview_resampling.lower()})
    ovr.text = ov
    
    etree.indent(root)
    tree.write(outname, pretty_print=True, xml_declaration=False, encoding='utf-8')


def calc_product_start_stop(
        src_ids: list[ID],
        extent: dict[str, int | float],
        epsg: int
) -> tuple[datetime, datetime]:
    """
    Calculates the start and stop times of the ARD product.
    The geolocation grid points including their azimuth time information are
    extracted first from the metadata of each source product.
    These grid points are then used to interpolate the azimuth time for the
    coordinates of the MGRS tile extent. The lowest and highest interpolated
    value are returned as product acquisition start and stop times of the
    ARD product.

    Parameters
    ----------
    src_ids:
        List of :class:`~pyroSAR.drivers.ID` objects of all source products
        that overlap with the current MGRS tile.
    extent:
        Spatial extent of the MGRS tile, derived from a
        :class:`~spatialist.vector.Vector` object.
    epsg:
        The coordinate reference system of the extent as an EPSG code.

    Returns
    -------
        Start and stop time of the ARD product in UTC.
    
    See Also
    --------
    pyroSAR.drivers.SAFE.geo_grid
    scipy.interpolate.RBFInterpolator
    """
    with bbox(extent, epsg) as tile_geom:
        tile_geom.reproject(4326)
        scene_geoms = [x.geometry() for x in src_ids]
        with combine_polygons(scene_geoms) as scene_geom:
            intersection = gpd.overlay(df1=tile_geom.to_geopandas(),
                                       df2=scene_geom.to_geopandas(),
                                       how='intersection')
            tile_geom_pts = intersection.get_coordinates().to_numpy()
        scene_geoms = None
    
    # combine geo grid of all scenes into one
    gdfs = []
    for src_id in src_ids:
        with src_id.geo_grid() as vec:
            gdfs.append(vec.to_geopandas())
    gdf = pd.concat(gdfs, ignore_index=True)
    
    # remove duplicate points
    gdf["xy"] = gdf.geometry.apply(lambda p: (p.x, p.y))
    gdf = gdf.drop_duplicates(subset="xy").copy()
    gdf.drop(columns="xy", inplace=True)
    
    # get grid point coordinates and numerical time stamps for interpolation
    gdf['timestamp'] = gdf['azimuthTime'].astype(np.int64) / 10 ** 9
    gridpts = gdf.get_coordinates().to_numpy()
    az_time = gdf['timestamp'].values
    
    # perform interpolation
    rbf = RBFInterpolator(y=gridpts, d=az_time)
    interpolated = rbf(tile_geom_pts)
    
    # check interpolation validity
    if np.isnan(interpolated).any():
        raise RuntimeError('The interpolated array contains NaN values.')
    
    # Make sure the interpolated values do not exceed the actual values.
    # This might happen when the source product geometries are slightly
    # larger than the geo grid extent.
    out = [max(min(interpolated), min(gdf['timestamp'])),
           min(max(interpolated), max(gdf['timestamp']))]
    
    # double-check that values are plausible
    if out[0] < min(gdf['timestamp']) or out[1] > max(gdf['timestamp']):
        raise RuntimeError('The interpolated values exceed the input range.')
    if out[0] >= out[1]:
        raise RuntimeError('The determined acquisition start is larger '
                           'than or equal to the acquisition end.')
    
    return (datetime.fromtimestamp(out[0], tz=timezone.utc),
            datetime.fromtimestamp(out[1], tz=timezone.utc))


def create_data_mask(
        outname: str,
        datasets: list[dict],
        extent: dict[str, int | float],
        epsg: int,
        driver: str,
        creation_opt: list[str],
        overviews: list[int],
        overview_resampling: str,
        dst_nodata: int | str,
        product_type: str,
        lsm_encoding: dict[str, int],
        wbm: str | None = None
) -> None:
    """
    Creation of the Data Mask image.
    
    Parameters
    ----------
    outname:
        Full path to the output data mask file.
    datasets:
        List of processed output files that match the source scenes and overlap
        with the current MGRS tile. An error will be thrown if not all datasets
        contain a key `datamask`. The function will return without an error if
        not all datasets contain a key `dm`.
    extent:
        Spatial extent of the MGRS tile, derived from a
        :class:`~spatialist.vector.Vector` object.
    epsg:
        The coordinate reference system as an EPSG code.
    driver:
        GDAL driver to use for raster file creation.
    creation_opt:
        GDAL creation options to use for raster file creation. Should match
        specified GDAL driver.
    overviews:
        Internal overview levels to be created for each raster file.
    overview_resampling:
        Resampling method for overview levels.
    dst_nodata:
        Nodata value to write to the output raster.
    product_type:
        The type of ARD product that is being created. Either 'NRB' or 'ORB'.
    lsm_encoding:
        a dictionary containing the layover shadow mask encoding.
    wbm:
        Path to a water body mask file with the dimensions of an MGRS tile.
        Optional if `product_type='NRB', mandatory if `product_type='ORB'`.
    """
    measurement_keys = [x for x in datasets[0].keys() if re.search('[gs]-lin', x)]
    measurement = [scene[measurement_keys[0]] for scene in datasets]
    datamask = [scene['datamask'] for scene in datasets]
    ls = []
    for scene in datasets:
        if 'dm' in scene:
            ls.append(scene['dm'])
        else:
            return  # do not create a data mask if not all scenes have a layover-shadow mask
    
    dm_bands = ['not layover, nor shadow',
                'layover',
                'shadow',
                'ocean',
                'lakes',
                'rivers']
    
    if product_type == 'ORB':
        if wbm is None:
            raise RuntimeError('Water body mask is required for ORB products')
    
    tile_bounds = [extent['xmin'], extent['ymin'], extent['xmax'], extent['ymax']]
    
    vrt_ls = '/vsimem/' + os.path.dirname(outname) + 'ls.vrt'
    vrt_valid = '/vsimem/' + os.path.dirname(outname) + 'valid.vrt'
    vrt_measurement = '/vsimem/' + os.path.dirname(outname) + 'measurement.vrt'
    gdalbuildvrt(src=ls, dst=vrt_ls, outputBounds=tile_bounds, void=False)
    gdalbuildvrt(src=datamask, dst=vrt_valid, outputBounds=tile_bounds, void=False)
    gdalbuildvrt(src=measurement, dst=vrt_measurement, outputBounds=tile_bounds, void=False)
    
    with Raster(vrt_ls) as ras_ls:
        with bbox(extent, crs=epsg) as tile_vec:
            ras_ls_res = ras_ls.res
            rows = ras_ls.rows
            cols = ras_ls.cols
            geotrans = ras_ls.raster.GetGeoTransform()
            proj = ras_ls.raster.GetProjection()
            arr_dm = ras_ls.array()
            
            # Get Water Body Mask
            if wbm is not None:
                with Raster(wbm) as ras_wbm:
                    ras_wbm_cols = ras_wbm.cols
                    cols_ratio = ras_wbm_cols / cols
                    if cols_ratio > 1:
                        # create low resolution VRT
                        wbm_lowres = wbm.replace('.tif', f'_{ras_ls_res[0]}m.vrt')
                        if not os.path.isfile(wbm_lowres):
                            options = {'xRes': ras_ls_res[0], 'yRes': ras_ls_res[1],
                                       'resampleAlg': 'mode'}
                            gdalbuildvrt(src=wbm, dst=wbm_lowres, **options)
                        with Raster(wbm_lowres) as ras_wbm_lowres:
                            arr_wbm = ras_wbm_lowres.array()
                    else:
                        arr_wbm = ras_wbm.array()
            else:
                del dm_bands[3:]
            
            c = lsm_encoding['not layover, not shadow']
            l = lsm_encoding['layover']
            s = lsm_encoding['shadow']
            ls = lsm_encoding['layover in shadow']
            n = lsm_encoding['nodata']
            
            # Extend the shadow class of the data mask with nodata values
            # from backscatter data and create final array
            with Raster(vrt_valid)[tile_vec] as ras_valid:
                with Raster(vrt_measurement)[tile_vec] as ras_measurement:
                    arr_valid = ras_valid.array()
                    arr_measurement = ras_measurement.array()
                    
                    arr_dm[~np.isfinite(arr_dm)] = n
                    arr_dm = np.where(((arr_valid == 1) & (np.isnan(arr_measurement))),
                                      s, arr_dm)
                    arr_dm[arr_valid != 1] = n
                    del arr_measurement
                    del arr_valid
        
        outname_tmp = '/vsimem/' + os.path.basename(outname) + '.vrt'
        gdriver = gdal.GetDriverByName('GTiff')
        ds_tmp = gdriver.Create(outname_tmp, rows, cols, len(dm_bands), gdal.GDT_Byte,
                                options=['ALPHA=UNSPECIFIED', 'PHOTOMETRIC=MINISWHITE'])
        gdriver = None
        ds_tmp.SetGeoTransform(geotrans)
        ds_tmp.SetProjection(proj)
        
        for i, name in enumerate(dm_bands):
            band = ds_tmp.GetRasterBand(i + 1)
            
            # not layover, nor shadow
            if i == 0:
                arr = arr_dm == c
            # layover | layover in shadow
            elif i == 1:
                arr = (arr_dm == l) | (arr_dm == ls)
            # shadow | layover in shadow
            elif i == 2:
                arr = (arr_dm == s) | (arr_dm == ls)
            # ocean
            elif i == 3:
                arr = arr_wbm == 1
            # lakes
            elif i == 4:
                arr = arr_wbm == 2
            # rivers
            elif i == 5:
                arr = arr_wbm == 3
            else:
                raise ValueError(f'unknown array value: {i}')
            
            arr = arr.astype('uint8')
            arr[arr_dm == n] = dst_nodata
            band.WriteArray(arr)
            band.SetNoDataValue(dst_nodata)
            band.SetDescription(name)
            band.FlushCache()
            band = None
            del arr
        
        ds_tmp.SetMetadataItem('TIFFTAG_DATETIME', strftime('%Y:%m:%d %H:%M:%S', gmtime()))
        ds_tmp.BuildOverviews(overview_resampling, overviews)
        outDataset_cog = gdal.GetDriverByName(driver).CreateCopy(outname, ds_tmp, strict=1, options=creation_opt)
        outDataset_cog = None
        ds_tmp = None
        tile_vec = None


def create_acq_id_image(
        outname: str,
        ref_tif: str,
        datasets: list[dict],
        src_ids: list[ID],
        extent: dict[str, int | float],
        epsg: int,
        driver: str,
        creation_opt: list[str],
        overviews: list[int],
        dst_nodata: int | str
) -> None:
    """
    Creation of the Acquisition ID image.

    Parameters
    ----------
    outname:
        Full path to the output data mask file.
    ref_tif:
        Full path to any GeoTIFF file of the ARD product.
    datasets:
        List of processed output files that match the source SLC scenes and overlap with the current MGRS tile.
    src_ids:
        List of :class:`~pyroSAR.drivers.ID` objects of all source SLC scenes that overlap with the current MGRS tile.
    extent:
        Spatial extent of the MGRS tile, derived from a :class:`~spatialist.vector.Vector` object.
    epsg:
        The CRS used for the ARD product; provided as an EPSG code.
    driver:
        GDAL driver to use for raster file creation.
    creation_opt:
        GDAL creation options to use for raster file creation. Should match specified GDAL driver.
    overviews:
        Internal overview levels to be created for each raster file.
    dst_nodata:
        Nodata value to write to the output raster.
    """
    src_scenes = [sid.scene for sid in src_ids]
    
    tile_bounds = [extent['xmin'], extent['ymin'], extent['xmax'], extent['ymax']]
    
    arr_list = []
    for dataset in datasets:
        vrt_valid = '/vsimem/' + os.path.dirname(outname) + 'mosaic.vrt'
        gdalbuildvrt(src=dataset['datamask'], dst=vrt_valid, outputBounds=tile_bounds, void=False)
        with bbox(extent, crs=epsg) as tile_vec:
            with Raster(vrt_valid)[tile_vec] as vrt_ras:
                vrt_arr = vrt_ras.array()
                arr_list.append(vrt_arr)
                del vrt_arr
            tile_vec = None
    
    src_scenes_clean = [os.path.basename(src).replace('.zip', '').replace('.SAFE', '') for src in src_scenes]
    tag = '{{"{src1}": 1}}'.format(src1=src_scenes_clean[0])
    out_arr = np.full(arr_list[0].shape, dst_nodata)
    out_arr[arr_list[0] == 1] = 1
    if len(arr_list) == 2:
        out_arr[arr_list[1] == 1] = 2
        tag = '{{"{src1}": 1, "{src2}": 2}}'.format(src1=src_scenes_clean[0], src2=src_scenes_clean[1])
    
    creation_opt.append('TIFFTAG_IMAGEDESCRIPTION={}'.format(tag))
    with Raster(ref_tif) as ref_ras:
        ref_ras.write(outname, format=driver, array=out_arr.astype('uint8'), nodata=dst_nodata, overwrite=True,
                      overviews=overviews, options=creation_opt)
