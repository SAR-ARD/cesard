import re
import json
import numpy as np
from datetime import datetime
from spatialist import Raster
from spatialist.auxil import crsConvert
from spatialist.vector import Vector
from osgeo import gdal, ogr
from typing import Any

gdal.UseExceptions()


def vec_from_srccoords(
        coord_list: list[list[tuple[float, float]]],
        crs: int | str,
        layername: str = 'polygon'
) -> Vector:
    """
    Creates a single :class:`~spatialist.vector.Vector` object from a list
    of footprint coordinates of source scenes.
    
    Parameters
    ----------
    coord_list:
        List containing for each source scene a list of coordinate pairs as
        retrieved from the metadata stored in an :class:`~pyroSAR.drivers.ID`
        object.
    crs:
        the coordinate reference system of the provided coordinates.
    layername:
        the layer name of the output vector object
    
    Returns
    -------
        the vector object
    """
    srs = crsConvert(crs, 'osr')
    pts = ogr.Geometry(ogr.wkbMultiPoint)
    for footprint in coord_list:
        for lon, lat in footprint:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lon, lat)
            pts.AddGeometry(point)
    geom = pts.ConvexHull()
    vec = Vector(driver='Memory')
    vec.addlayer(layername, srs, geom.GetGeometryType())
    vec.addfeature(geom)
    point = None
    pts = None
    geom = None
    return vec


def geometry_from_vec(
        vectorobject: Vector
) -> dict[str, Any]:
    """
    Get geometry information for usage in STAC and XML metadata from a :class:`spatialist.vector.Vector` object.
    
    Parameters
    ----------
    vectorobject:
        The vector object to extract geometry information from.
    
    Returns
    -------
        A dictionary containing the geometry information extracted from the vector object.
    """
    out = {}
    vec = vectorobject
    
    # For STAC metadata
    if vec.getProjection(type='epsg') != 4326:
        ext = vec.extent
        out['bbox_native'] = [ext['xmin'], ext['ymin'], ext['xmax'], ext['ymax']]
        vec.reproject(4326)
    feat = vec.getfeatures()[0]
    geom = feat.GetGeometryRef()
    out['geometry'] = json.loads(geom.ExportToJson())
    ext = vec.extent
    out['bbox'] = [ext['xmin'], ext['ymin'], ext['xmax'], ext['ymax']]
    
    # For XML metadata
    c_x = (ext['xmax'] + ext['xmin']) / 2
    c_y = (ext['ymax'] + ext['ymin']) / 2
    out['center'] = '{} {}'.format(c_y, c_x)
    wkt = geom.ExportToWkt().removeprefix('POLYGON ((').removesuffix('))')
    wkt_list = ['{} {}'.format(x[1], x[0]) for x in [y.split(' ') for y in wkt.split(',')]]
    out['envelope'] = ' '.join(wkt_list)
    
    return out


def calc_enl(
        tif: str,
        block_size: int = 30,
        return_arr: bool = False,
        decimals: int = 2
) -> float | np.ndarray | None:
    """
    Calculate the Equivalent Number of Looks (ENL) for a linear-scaled backscatter
    measurement GeoTIFF file. The calculation is performed block-wise for the
    entire image and by default the median ENL value is returned.
    
    Parameters
    ----------
    tif:
        The path to a linear-scaled backscatter measurement GeoTIFF file.
    block_size:
        The block size to use for the calculation. Remainder pixels are discarded,
         if the array dimensions are not evenly divisible by the block size.
         Default is 30, which calculates ENL for 30x30 pixel blocks.
    return_arr:
        If True, the calculated ENL array is returned. Default is False.
    decimals:
        Number of decimal places to round the calculated ENL value to. Default is 2.
    
    Raises
    ------
    RuntimeError
        if the input array contains only NaN values
    
    Returns
    -------
        If `return_enl_arr=True`, an array of ENL values is returned. Otherwise,
        the median ENL value is returned. If the ENL array contains only NaN and
        `return_enl_arr=False`, the return value is `None`.
    
    References
    ----------
    :cite:`anfinsen.etal_2009`
    """
    with Raster(tif) as ras:
        arr = ras.array()
    arr[np.isinf(arr)] = np.nan
    
    if len(arr[~np.isnan(arr)]) == 0:
        raise RuntimeError('cannot compute ENL for an empty array')
    
    num_blocks_rows = arr.shape[0] // block_size
    num_blocks_cols = arr.shape[1] // block_size
    if num_blocks_rows == 0 or num_blocks_cols == 0:
        raise ValueError("Block size is too large for the input data dimensions.")
    blocks = arr[:num_blocks_rows * block_size,
    :num_blocks_cols * block_size].reshape(num_blocks_rows, block_size,
                                           num_blocks_cols, block_size)
    
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Mean of empty slice")
        _mean = np.nanmean(blocks, axis=(1, 3))
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
        _std = np.nanstd(blocks, axis=(1, 3))
    enl = np.divide(_mean ** 2, _std ** 2,
                    out=np.full_like(_mean, fill_value=np.nan), where=_std != 0)
    out_arr = np.zeros((num_blocks_rows, num_blocks_cols))
    out_arr[:num_blocks_rows, :num_blocks_cols] = enl
    if not return_arr:
        if len(enl[~np.isnan(enl)]) == 0:
            return None
        out_median = np.nanmedian(out_arr).item()
        return round(out_median, decimals)
    else:
        return out_arr


def calc_performance_estimates(
        files: list[str],
        decimals: int = 2
):
    """
    Calculates the performance estimates specified in CARD4L NRB 1.6.9 for all noise power images if available.
    
    Parameters
    ----------
    files:
        List of paths pointing to the noise power images the estimates should be calculated for.
    decimals:
        Number of decimal places to round the calculated values to. Default is 2.
    
    Returns
    -------
    out: dict
        Dictionary containing the calculated estimates for each available polarization.
    """
    out = {}
    for f in files:
        pol = re.search('np-([vh]{2})', f).group(1).upper()
        with Raster(f) as ras:
            arr = ras.array()
            # The following need to be of type float, not numpy.float32 in order to be JSON serializable
            _min = float(np.nanmin(arr))
            _max = float(np.nanmax(arr))
            _mean = float(np.nanmean(arr))
            del arr
        out[pol] = {'minimum': round(_min, decimals),
                    'maximum': round(_max, decimals),
                    'mean': round(_mean, decimals)}
    return out


def evaluate_types(
        meta: Any
) -> None:
    """
    Evaluate the types of a metadata dictionary as extracted by e.g.
    :func:`s1ard.metadata.extract.meta_dict`.
    Currently allowed:
    
    - :py:obj:`bool`
    - :py:obj:`int`
    - :py:obj:`float`
    - :py:obj:`str`
    - :py:obj:`None`
    - :py:obj:`datetime.datetime`
    
    Parameters
    ----------
    meta
        the metadata dictionary to be evaluated.

    Raises
    ------
    TypeError

    """
    collector = dict()
    
    def get_typing(item=meta, collector=None, key=''):
        fill = '' if key == '' else '.'
        if isinstance(item, dict):
            return {k: get_typing(v, key=f'{key}{fill}{k}', collector=collector)
                    for k, v in item.items()}
        elif isinstance(item, list):
            return [get_typing(v, key=f'{key}{fill}{k}', collector=collector)
                    for k, v in enumerate(item)]
        else:
            t = type(item)
            # isinstance(np.float64(3.), float) == True
            allowed = [bool, int, float, str, type(None), None, datetime]
            if t not in allowed and collector is not None:
                collector[key] = t
            return t
    
    get_typing(item=meta, collector=collector)
    if len(collector) > 0:
        display = '\n'.join([f'{k}: {v}' for k, v in collector.items()])
        raise TypeError(f'the passed metadata dictionary contains unsupported types:\n'
                        f'{display}')


def get_header_size(tif: str) -> int:
    """
    Gets the header size of a GeoTIFF file in bytes.
    The code used in this function and its helper function `_get_block_offset` were extracted from the following
    source:
    
    https://github.com/OSGeo/gdal/blob/master/swig/python/gdal-utils/osgeo_utils/samples/validate_cloud_optimized_geotiff.py
    
    Copyright (c) 2017, Even Rouault
    
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.
    
    Parameters
    ----------
    tif:
        A path to a GeoTIFF file of the currently processed ARD product.

    Returns
    -------
    header_size:
        The size of all IFD headers of the GeoTIFF file in bytes.
    """
    
    def _get_block_offset(band):
        blockxsize, blockysize = band.GetBlockSize()
        for y in range(int((band.YSize + blockysize - 1) / blockysize)):
            for x in range(int((band.XSize + blockxsize - 1) / blockxsize)):
                block_offset = band.GetMetadataItem('BLOCK_OFFSET_%d_%d' % (x, y), 'TIFF')
                if block_offset:
                    return int(block_offset)
        return 0
    
    details = {}
    ds = gdal.Open(tif)
    main_band = ds.GetRasterBand(1)
    ovr_count = main_band.GetOverviewCount()
    
    block_offset = _get_block_offset(band=main_band)
    details['data_offsets'] = {}
    details['data_offsets']['main'] = block_offset
    for i in range(ovr_count):
        ovr_band = ds.GetRasterBand(1).GetOverview(i)
        block_offset = _get_block_offset(band=ovr_band)
        details['data_offsets']['overview_%d' % i] = block_offset
    
    headers_size = min(details['data_offsets'][k] for k in details['data_offsets'])
    if headers_size == 0:
        headers_size = gdal.VSIStatL(tif).size
    return headers_size
