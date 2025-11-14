import os
import re
import inspect
from dateutil.parser import parse as dateparse
from datetime import datetime, timedelta
from spatialist.vector import Vector, crsConvert, wkt2vector
import asf_search as asf
from pyroSAR.drivers import ID
from cesard.ancillary import date_to_utc, combine_polygons
from cesard.tile_extraction import aoi_from_tile, tile_from_aoi
from typing import Any, Literal
import logging

log = logging.getLogger('cesard')

SENSOR = Literal["S1A", "S1B", "S1C", "S1D"]
PRODUCT = Literal["GRD", "SLC"]
ACQUISITION_MODE = Literal["IW", "EW", "SM"]


class ASF(ID):
    """
    Simple SAR metadata handler for scenes in the ASF archive. The interface
    is consistent with the driver classes in :mod:`pyroSAR.drivers` but does
    not implement the full functionality due to limited content of the CMR
    metadata catalog. Registered attributes:
    
    - acquisition_mode
    - coordinates
    - frameNumber
    - orbit
    - orbitNumber_abs
    - orbitNumber_rel
    - polarizations
    - product
    - projection
    - sensor
    - start
    - stop
    """
    
    def __init__(
            self,
            meta: dict[str, Any]
    ):
        self.scene = meta['properties']['url']
        self._meta = meta
        self.meta = self.scanMetadata()
        super(ASF, self).__init__(self.meta)
    
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ASF):
            return NotImplemented
        return self.outname_base() < other.outname_base()
    
    def scanMetadata(self) -> dict[str, int | float | str | datetime | list[tuple[float, float]]]:
        meta = dict()
        meta['acquisition_mode'] = self._meta['properties']['beamModeType']
        meta['coordinates'] = [tuple(x) for x in self._meta['geometry']['coordinates'][0]]
        fname = os.path.splitext(self._meta['properties']['fileName'])[0]
        meta['frameNumber'] = fname[-11:-5]
        meta['orbit'] = self._meta['properties']['flightDirection'][0]
        meta['orbitNumber_abs'] = self._meta['properties']['orbit']
        meta['orbitNumber_rel'] = self._meta['properties']['pathNumber']
        meta['polarizations'] = self._meta['properties']['polarization'].split('+')
        product = self._meta['properties']['processingLevel']
        meta['product'] = re.search('GRD|SLC|SM|OCN', product).group()
        meta['projection'] = crsConvert(4326, 'wkt')
        meta['sensor'] = self._meta['properties']['platform'].replace('entinel-', '')
        start = self._meta['properties']['startTime']
        stop = self._meta['properties']['stopTime']
        pattern = '%Y%m%dT%H%M%S'
        meta['start'] = dateparse(start).strftime(pattern)
        meta['stop'] = dateparse(stop).strftime(pattern)
        meta['spacing'] = None
        meta['samples'] = None
        meta['lines'] = None
        meta['cycleNumber'] = None
        meta['sliceNumber'] = None
        meta['totalSlices'] = None
        return meta


class ASFArchive(object):
    """
    Search for scenes in the Alaska Satellite Facility (ASF) catalog.
    """
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return
    
    @staticmethod
    def select(
            sensor: SENSOR | None = None,
            product: PRODUCT | None = None,
            acquisition_mode: ACQUISITION_MODE | None = None,
            mindate: str | datetime | None = None,
            maxdate: str | datetime | None = None,
            vectorobject: Vector | None = None,
            date_strict: bool = True,
            return_value: str | list[str] = 'scene'
    ) -> list[str | tuple[str] | ASF]:
        """
        Select scenes from the ASF catalog. This is a simple wrapper around the function
        :func:`~cesard.search.asf_select` to be consistent with the interfaces of the
        other search classes.

        Parameters
        ----------
        sensor:
            the satellite
        product:
            the product type
        acquisition_mode:
            the satellite acquisition mode
        mindate:
            the minimum acquisition date; timezone-unaware dates are interpreted as UTC.
        maxdate:
            the maximum acquisition date; timezone-unaware dates are interpreted as UTC.
        vectorobject:
            a geometry with which the scenes need to overlap. The object may only contain one feature.
        date_strict:
            treat dates as strict limits or also allow flexible limits to incorporate scenes
            whose acquisition period overlaps with the defined limit?
            
            - strict: start >= mindate & stop <= maxdate
            - not strict: stop >= mindate & start <= maxdate
        return_value:
            the metadata return value; see :func:`~cesard.search.asf_select` for details
        
        Returns
        -------
            the scene metadata attributes as specified with `return_value`;
            see :func:`~cesard.search.asf_select` for details
        
        See Also
        --------
        asf_select
        """
        return asf_select(sensor, product, acquisition_mode, mindate, maxdate, vectorobject,
                          return_value=return_value, date_strict=date_strict)


def asf_select(
        sensor: SENSOR | None = None,
        product: PRODUCT | None = None,
        acquisition_mode: ACQUISITION_MODE | None = None,
        mindate: str | datetime | None = None,
        maxdate: str | datetime | None = None,
        vectorobject: Vector | None = None,
        date_strict: bool = True,
        return_value: str | list[str] = 'scene'
) -> list[str | tuple[str] | ASF]:
    """
    Search scenes in the Alaska Satellite Facility (ASF) data catalog.
    This is a simple interface to the
    `asf_search <https://github.com/asfadmin/Discovery-asf_search>`_ package.
    
    Parameters
    ----------
    sensor:
        the satellite
    product:
        the product type
    acquisition_mode:
        the satellite acquisition mode
    mindate:
        the minimum acquisition date; timezone-unaware dates are interpreted as UTC.
    maxdate:
        the maximum acquisition date; timezone-unaware dates are interpreted as UTC.
    vectorobject:
        a geometry with which the scenes need to overlap. The object may only contain one feature.
    date_strict:
        treat dates as strict limits or also allow flexible limits to incorporate scenes
        whose acquisition period overlaps with the defined limit?
        
        - strict: start >= mindate & stop <= maxdate
        - not strict: stop >= mindate & start <= maxdate
    
    return_value:
        the query return value(s). Options:
        
        - acquisition_mode: the sensor's acquisition mode
        - frameNumber: the frame or datatake number
        - geometry_wkb: the scene's footprint geometry formatted as WKB
        - geometry_wkt: the scene's footprint geometry formatted as WKT
        - mindate: the acquisition start datetime in UTC formatted as YYYYmmddTHHMMSS
        - maxdate: the acquisition end datetime in UTC formatted as YYYYmmddTHHMMSS
        - product: the product type
        - scene: the scene's storage location path (default)
        - sensor: the satellite platform

    Returns
    -------
        the scene metadata attributes as specified with `return_value`; the return type
        is a list of strings, tuples, or :class:`~cesard.search.ASF` objects depending on
        whether `return_type` is of type string, list or :class:`~cesard.search.ASF`.
    
    """
    if isinstance(return_value, str):
        return_values = [return_value]
    else:
        return_values = return_value
    
    if product == 'GRD':
        processing_level = ['GRD_HD', 'GRD_MD', 'GRD_MS', 'GRD_HS', 'GRD_FD']
    else:
        processing_level = product
    if acquisition_mode == 'SM':
        beam_mode = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    else:
        beam_mode = acquisition_mode
    if vectorobject is not None:
        if vectorobject.nfeatures > 1:
            raise RuntimeError("'vectorobject' contains more than one feature.")
        with vectorobject.clone() as geom:
            geom.reproject(4326)
            geometry = geom.convert2wkt(set3D=False)[0]
    else:
        geometry = None
    
    start = date_to_utc(mindate, as_datetime=True)
    stop = date_to_utc(maxdate, as_datetime=True)
    
    lookup_platform = {'S1A': 'Sentinel-1A',
                       'S1B': 'Sentinel-1B',
                       'S1C': 'Sentinel-1C',
                       'S1D': 'Sentinel-1D'}
    platform = lookup_platform[sensor] if sensor is not None else None
    
    result = asf.search(platform=platform,
                        processingLevel=processing_level,
                        beamMode=beam_mode,
                        start=start,
                        end=stop,
                        intersectsWith=geometry).geojson()
    features = result['features']
    
    def date_extract(item, key):
        return date_to_utc(date=item['properties'][key], as_datetime=True)
    
    if date_strict:
        features = [x for x in features
                    if start <= date_extract(x, 'startTime')
                    and date_extract(x, 'stopTime') <= stop]
    
    features = sorted([ASF(x) for x in features])
    
    out = []
    for item in features:
        values = []
        for key in return_values:
            if key == 'ASF':
                values.append(item)
            elif key == 'mindate':
                values.append(getattr(item, 'start'))
            elif key == 'maxdate':
                values.append(getattr(item, 'stop'))
            elif key == 'geometry_wkb':
                with item.geometry() as vec:
                    value = vec.to_geopandas().to_wkb()['geometry'][0]
                    values.append(value)
            elif key == 'geometry_wkt':
                with item.geometry() as vec:
                    value = vec.to_geopandas().to_wkt()['geometry'][0]
                    values.append(value)
            elif hasattr(item, key):
                values.append(getattr(item, key))
            else:
                raise ValueError(f'invalid return value: {key}')
        if len(return_values) == 1:
            out.append(values[0])
        else:
            out.append(tuple(values))
    return out


def scene_select(
        archive: Any,
        aoi_tiles: list[str] | None = None,
        aoi_geometry: str | None = None,
        return_value: str | list[str] = 'scene',
        **kwargs: Any
) -> tuple[list[str | tuple[str]], list[str]]:
    """
    Central scene search utility. Selects scenes from a database and returns their file names
    together with the MGRS tile names for which to process ARD products.
    The list of MGRS tile names is either identical to the list provided with `aoi_tiles`,
    the list of all tiles overlapping with `aoi_geometry` or `vectorobject` (via `kwargs`),
    or the list of all tiles overlapping with an initial scene search result if no geometry
    has been defined via `aoi_tiles` or `aoi_geometry`. In the latter (most complex) case,
    the search procedure is as follows:
    
     - perform a first search matching all other search parameters
     - derive all MGRS tile geometries overlapping with the selection
     - derive the minimum and maximum acquisition times of the selection as search parameters
       `mindate` and `maxdate`
     - extend the `mindate` and `maxdate` search parameters by one minute
     - perform a second search with the extended time range and the derived MGRS tile geometries
     - filter the search result to scenes overlapping with the initial time range (if defined
       via `mindate` or `maxdate`)
    
    As consequence, if one defines the search parameters to only return one scene, the neighboring
    acquisitions will also be returned. This is because the scene overlaps with a set of MGRS
    tiles of which many or all will also overlap with these neighboring acquisitions. To ensure
    full coverage of all MGRS tiles, the neighbors of the scene in focus have to be processed too.
    
    This function has three ways to define search geometries. In order of priority overriding others:
    `aoi_tiles` > `aoi_geometry` > `vectorobject` (via `kwargs`). In the latter two cases, the search
    geometry is extended to the common footprint of all MGRS tiles overlapping with the initial geometry
    to ensure full coverage of all tiles.
    
    Parameters
    ----------
    archive:
        an open scene archive connection. pyroSAR.drivers.Archive or s1ard.search.STACArchive
        or s1ard.search.STACParquetArchive or ASFArchive.
    aoi_tiles:
        a list of MGRS tile names for spatial search
    aoi_geometry:
        the name of a vector geometry file for spatial search
    return_value:
        the query return value(s). Default 'scene': return the scene's storage location path.
        See the documentation of `archive.select` for options.
    kwargs:
        further search arguments passed to the `select` method of `archive`.
        The `date_strict` argument has no effect. Whether an ARD product is strictly in the defined
        time range cannot be determined by this function, and it thus has to add a time buffer.
        When `date_strict=True`, more scenes will be filtered out in the last step described above.

    Returns
    -------
        a tuple containing
    
        - the list of return values; single value:string, multiple values: tuple
        - the list of MGRS tiles
    
    """
    args = kwargs.copy()
    if 'mindate' in args.keys():
        args['mindate'] = date_to_utc(args['mindate'], as_datetime=True)
        mindate_init = args['mindate']
    else:
        mindate_init = None
    if 'maxdate' in args.keys():
        args['maxdate'] = date_to_utc(args['maxdate'], as_datetime=True)
        maxdate_init = args['maxdate']
    else:
        maxdate_init = None
    for key in ['acquisition_mode']:
        if key not in args.keys():
            args[key] = None
    
    if args['acquisition_mode'] == 'SM':
        args['acquisition_mode'] = ('S1', 'S2', 'S3', 'S4', 'S5', 'S6')
    
    signature = inspect.signature(archive.select)
    if 'return_value' not in signature.parameters:
        raise RuntimeError("the 'select' method of 'archive' does not take "
                           "a 'return_value' parameter")
    
    return_values = return_value if isinstance(return_value, list) else [return_value]
    if isinstance(archive, ASFArchive):
        args['return_value'] = 'ASF'
    else:
        args['return_value'] = return_value
    
    vec = None
    if aoi_tiles is not None:
        log.debug("reading geometries of 'aoi_tiles'")
        vec = aoi_from_tile(tile=aoi_tiles)
    elif aoi_geometry is not None:
        log.debug("extracting tiles overlapping with 'aoi_geometry'")
        with Vector(aoi_geometry) as geom:
            vec = tile_from_aoi(vector=geom,
                                return_geometries=True)
    elif 'vectorobject' in args.keys() and args['vectorobject'] is not None:
        log.debug("extracting tiles overlapping with 'vectorobject'")
        vec = tile_from_aoi(vector=args['vectorobject'],
                            return_geometries=True)
    if vec is not None:
        if not isinstance(vec, list):
            vec = [vec]
        
        if aoi_tiles is None:
            aoi_tiles = [x.mgrs for x in vec]
        log.debug(f"got {len(aoi_tiles)} tiles")
    
    # derive geometries and tiles from scene footprints
    if vec is None:
        log.debug("performing initial scene search without geometry constraint")
        args['return_value'] = ['mindate', 'maxdate', 'geometry_wkt']
        selection_tmp = archive.select(**args)
        log.debug(f'got {len(selection_tmp)} scenes')
        mindates, maxdates, geometries_init = zip(*selection_tmp)
        # The geometry of scenes crossing the antimeridian is stored as multipolygon.
        # Since the processor is currently not able to process these scenes, they are
        # removed in this step.
        geometries = [x for x in geometries_init if x.startswith('POLYGON')]
        if len(geometries) < len(geometries_init):
            log.debug(f'removed {len(geometries_init) - len(geometries)} '
                      f'scenes crossing the antimeridian')
        del selection_tmp
        
        log.debug(f"loading geometries")
        scenes_geom = [wkt2vector(x, srs=4326) for x in geometries]
        # select all tiles overlapping with the scenes for further processing
        log.debug("extracting all tiles overlapping with initial scene selection")
        vec = tile_from_aoi(vector=scenes_geom,
                            return_geometries=True)
        if not isinstance(vec, list):
            vec = [vec]
        aoi_tiles = [x.mgrs for x in vec]
        log.debug(f"got {len(aoi_tiles)} tiles")
        del scenes_geom
        
        args['mindate'] = min([date_to_utc(x, as_datetime=True) for x in mindates])
        args['maxdate'] = max([date_to_utc(x, as_datetime=True) for x in maxdates])
        del mindates, maxdates, geometries
    
    # extend the time range to fully cover all tiles
    # (one additional scene needed before and after each data take group)
    if 'mindate' in args.keys():
        args['mindate'] -= timedelta(minutes=1)
    if 'maxdate' in args.keys():
        args['maxdate'] += timedelta(minutes=1)
    
    args['return_value'] = return_values.copy()
    for key in ['mindate', 'maxdate']:
        if key not in args['return_value']:
            args['return_value'].append(key)
    
    log.debug("performing main scene search")
    with combine_polygons(vec, multipolygon=True) as combined:
        args['vectorobject'] = combined
        selection = archive.select(**args)
    del vec
    
    # reduce the selection to the time range defined by the user
    if mindate_init is not None or maxdate_init is not None:
        i = 0
        while i < len(selection):
            values = dict(zip(args['return_value'], selection[i]))
            start = date_to_utc(values['mindate'], as_datetime=True)
            stop = date_to_utc(values['maxdate'], as_datetime=True)
            delete = False
            if mindate_init is not None and stop < mindate_init:
                delete = True
            if maxdate_init is not None and start > maxdate_init:
                delete = True
            if delete:
                del selection[i]
            else:
                i += 1
    
    # sort the return values by the scene's basename
    rv_scene_key = args['return_value'].index('scene')
    selection = sorted(selection, key=lambda k: os.path.basename(k[rv_scene_key]))
    
    # reduce the return values to those defined by the user
    indices = [i for i, key in enumerate(args['return_value']) if key in return_values]
    if len(indices) == 1:
        selection = [scene[indices[0]] for scene in selection]
    else:
        selection = [tuple(scene[i] for i in indices) for scene in selection]
    log.debug(f"got {len(selection)} scenes")
    return selection, aoi_tiles
