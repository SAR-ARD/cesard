import os
import re
import json
from lxml import etree
import numpy as np
from spatialist import Raster
from spatialist.auxil import crsConvert
from spatialist.vector import Vector
from osgeo import gdal, ogr

gdal.UseExceptions()


def vec_from_srccoords(coord_list, crs, layername='polygon'):
    """
    Creates a single :class:`~spatialist.vector.Vector` object from a list
    of footprint coordinates of source scenes.
    
    Parameters
    ----------
    coord_list: list[list[tuple[float]]]
        List containing for each source scene a list of coordinate pairs as
        retrieved from the metadata stored in an :class:`~pyroSAR.drivers.ID`
        object.
    crs: int or str
        the coordinate reference system of the provided coordinates.
    layername: str
        the layer name of the output vector object
    
    Returns
    -------
    spatialist.vector.Vector
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


def get_src_meta(sid):
    """
    Retrieve the manifest and annotation XML data of a scene as a dictionary using an :class:`pyroSAR.drivers.ID`
    object.
    
    Parameters
    ----------
    sid:  pyroSAR.drivers.ID
        A pyroSAR :class:`~pyroSAR.drivers.ID` object generated with e.g. :func:`pyroSAR.drivers.identify`.
    
    Returns
    -------
    dict
        A dictionary containing the parsed `etree.ElementTree` objects for the manifest and annotation XML files.
    """
    files = sid.findfiles(r'^s1[abcd].*-[vh]{2}-.*\.xml$')
    pols = list(set([re.search('[vh]{2}', os.path.basename(a)).group() for a in files]))
    annotation_files = list(filter(re.compile(pols[0]).search, files))
    
    a_files_base = [os.path.basename(a) for a in annotation_files]
    swaths = [re.search('-(iw[1-3]*|ew[1-5]*|s[1-6])', a).group(1) for a in a_files_base]
    
    annotation_dict = {}
    for s, a in zip(swaths, annotation_files):
        annotation_dict[s.upper()] = etree.fromstring(sid.getFileObj(a).getvalue())
    
    with sid.getFileObj(sid.findfiles('manifest.safe')[0]) as input_man:
        manifest = etree.fromstring(input_man.getvalue())
    
    return {'manifest': manifest,
            'annotation': annotation_dict}


def geometry_from_vec(vectorobject):
    """
    Get geometry information for usage in STAC and XML metadata from a :class:`spatialist.vector.Vector` object.
    
    Parameters
    ----------
    vectorobject: spatialist.vector.Vector
        The vector object to extract geometry information from.
    
    Returns
    -------
    out: dict
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


def find_in_annotation(annotation_dict, pattern, single=False, out_type='str'):
    """
    Search for a pattern in all XML annotation files provided and return a dictionary of results.
    
    Parameters
    ----------
    annotation_dict: dict
        A dict of annotation files in the form: {'swath ID': `lxml.etree._Element` object}
    pattern: str
        The pattern to search for in each annotation file.
    single: bool
        If True, the results found in each annotation file are expected to be the same and therefore only a single
        value will be returned instead of a dict. If the results differ, an error is raised. Default is False.
    out_type: str
        Output type to convert the results to. Can be one of the following:
        
        - 'str' (default)
        - 'float'
        - 'int'
    
    Returns
    -------
    out: dict
        A dictionary of the results containing a list for each of the annotation files. E.g.,
        {'swath ID': list[str or float or int]}
    """
    out = {}
    for s, a in annotation_dict.items():
        swaths = [x.text for x in a.findall('.//swathProcParams/swath')]
        items = a.findall(pattern)
        
        parent = items[0].getparent().tag
        if parent in ['azimuthProcessing', 'rangeProcessing']:
            for i, val in enumerate(items):
                out[swaths[i]] = val.text
        else:
            out[s] = [x.text for x in items]
            if len(out[s]) == 1:
                out[s] = out[s][0]
    
    def _convert(obj, type):
        if isinstance(obj, list):
            return [_convert(x, type) for x in obj]
        elif isinstance(obj, str):
            if type == 'float':
                return float(obj)
            if type == 'int':
                return int(obj)
    
    if out_type != 'str':
        for k, v in list(out.items()):
            out[k] = _convert(v, out_type)
    
    err_msg = 'Search result for pattern "{}" expected to be the same in all annotation files.'
    if single:
        val = list(out.values())[0]
        for k in out:
            if out[k] != val:
                raise RuntimeError(err_msg.format(pattern))
        if out_type != 'str':
            return _convert(val, out_type)
        else:
            return val
    else:
        return out


def calc_enl(tif, block_size=30, return_arr=False, decimals=2):
    """
    Calculate the Equivalent Number of Looks (ENL) for a linear-scaled backscatter
    measurement GeoTIFF file. The calculation is performed block-wise for the
    entire image and by default the median ENL value is returned.
    
    Parameters
    ----------
    tif: str
        The path to a linear-scaled backscatter measurement GeoTIFF file.
    block_size: int, optional
        The block size to use for the calculation. Remainder pixels are discarded,
         if the array dimensions are not evenly divisible by the block size.
         Default is 30, which calculates ENL for 30x30 pixel blocks.
    return_arr: bool, optional
        If True, the calculated ENL array is returned. Default is False.
    decimals: int, optional
        Number of decimal places to round the calculated ENL value to. Default is 2.
    
    Raises
    ------
    RuntimeError
        if the input array contains only NaN values
    
    Returns
    -------
    float or None or numpy.ndarray
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
        out_median = np.nanmedian(out_arr)
        return np.round(out_median, decimals)
    else:
        return out_arr


def calc_performance_estimates(files, decimals=2):
    """
    Calculates the performance estimates specified in CARD4L NRB 1.6.9 for all noise power images if available.
    
    Parameters
    ----------
    files: list[str]
        List of paths pointing to the noise power images the estimates should be calculated for.
    decimals: int, optional
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


def get_header_size(tif):
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
    tif: str
        A path to a GeoTIFF file of the currently processed ARD product.

    Returns
    -------
    header_size: int
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
