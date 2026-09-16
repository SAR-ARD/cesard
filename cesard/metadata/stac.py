import logging
import os
import re
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Mapping

import pystac
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.file import ByteOrder, FileExtension
from pystac.extensions.mgrs import MgrsExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
from pystac.extensions.sar import (
    FrequencyBand,
    ObservationDirection,
    Polarization,
    SarExtension,
)
from pystac.extensions.sat import OrbitState, SatExtension
from pystac.extensions.view import ViewExtension
from spatialist import Raster
from spatialist.ancillary import finder

from cesard.ancillary import compute_hash
from cesard.metadata.extract import get_header_size
from cesard.metadata.mapping import ASSET_MAP
from cesard.metadata.model import (
    ARDMetadata,
    NOT_IMPLEMENTED_NUMBER,
)

log = logging.getLogger('cesard')

MetadataInput = ARDMetadata | Mapping[str, Any]


def _as_model(meta: MetadataInput) -> ARDMetadata:
    """Return *meta* as validated :class:`ARDMetadata`.

    Legacy ``common/prod/source`` dictionaries remain supported during the
    migration period and are converted using :meth:`ARDMetadata.from_legacy`.
    """
    if isinstance(meta, ARDMetadata):
        return meta
    return ARDMetadata.from_legacy(meta)


def _mean(values: Mapping[str, float]) -> float:
    """Return the arithmetic mean of per-swath values."""
    return mean(values.values())


def _bandwidth_ghz(
        values: Mapping[str, float | int | None]
) -> dict[str, float] | None:
    """Convert per-swath look bandwidth from Hz to GHz.

    CARD4L STAC look-bandwidth properties require non-negative numeric values.
    If a producer reports ``None`` or the ``-99999`` implementation sentinel,
    the optional STAC property is omitted instead of serializing an invalid
    value or destroying the sentinel by unit conversion.
    """
    if any(value is None or value == NOT_IMPLEMENTED_NUMBER
           for value in values.values()):
        return None
    return {key: float(value) / 1e9 for key, value in values.items()}


def parse(
        meta: MetadataInput,
        target: str,
        assets: list[str],
        exist_ok: bool = False
) -> None:
    """Create source- and product-level STAC JSON metadata.

    Parameters
    ----------
    meta:
        Validated :class:`~cesard.metadata.model.ARDMetadata` or a legacy
        ``common/prod/source`` metadata dictionary. Legacy dictionaries are
        converted and validated with :meth:`ARDMetadata.from_legacy`.
    target:
        Path pointing to the root directory of a product scene.
    assets:
        Paths to all GeoTIFF and VRT assets of the ARD product.
    exist_ok:
        Do not create files if they already exist?
    """
    meta = _as_model(meta)
    source_json(meta=meta, target=target, exist_ok=exist_ok)
    product_json(meta=meta, target=target, assets=assets, exist_ok=exist_ok)


def source_json(
        meta: MetadataInput,
        target: str,
        exist_ok: bool = False
) -> None:
    """Create source-level STAC JSON metadata."""
    meta = _as_model(meta)
    common = meta.common
    product = meta.product
    metadir = os.path.join(target, 'source')
    
    for uid, source in meta.sources.items():
        scene = os.path.splitext(os.path.basename(source.filename))[0]
        outname = os.path.join(metadir, f'{scene}.json')
        if os.path.isfile(outname) and exist_ok:
            continue
        
        log.info(f'creating {os.path.relpath(outname, target)}')
        start = source.time_start
        stop = source.time_stop
        date = start + (stop - start) / 2
        
        item = pystac.Item(
            id=scene,
            geometry=source.geometry.geometry,
            bbox=list(source.geometry.bbox),
            datetime=date,
            properties={},
        )
        
        # Common metadata
        item.common_metadata.start_datetime = start
        item.common_metadata.end_datetime = stop
        item.common_metadata.created = source.processing.date
        item.common_metadata.instruments = [common.instrument_short_name.lower()]
        item.common_metadata.constellation = common.constellation
        item.common_metadata.platform = common.platform_full_name
        
        # SAT extension
        sat_ext = SatExtension.ext(item, add_if_missing=True)
        sat_ext.apply(
            orbit_state=OrbitState[common.orbit_direction.upper()],
            relative_orbit=common.orbit_number_relative,
            absolute_orbit=common.orbit_number_absolute,
            anx_datetime=source.orbit.ascending_node_date,
        )
        
        # SAR extension: STAC requires scalar values, while the canonical
        # interface retains the per-swath values needed by CARD4L/XML.
        sar_ext = SarExtension.ext(item, add_if_missing=True)
        sar_ext.apply(
            instrument_mode=common.operational_mode,
            frequency_band=FrequencyBand[common.radar_band.upper()],
            polarizations=[Polarization[pol] for pol in common.polarizations],
            product_type=source.product_type,
            center_frequency=common.radar_center_frequency / 1e9,
            resolution_range=_mean(source.range.resolution),
            resolution_azimuth=_mean(source.azimuth.resolution),
            pixel_spacing_range=_mean(source.range.pixel_spacing),
            pixel_spacing_azimuth=_mean(source.azimuth.pixel_spacing),
            looks_range=_mean(source.range.number_of_looks),
            looks_azimuth=_mean(source.azimuth.number_of_looks),
            looks_equivalent_number=source.performance.equivalent_number_of_looks,
            observation_direction=ObservationDirection[common.antenna_look_direction],
        )
        
        # View extension
        view_ext = ViewExtension.ext(item, add_if_missing=True)
        view_ext.apply(
            incidence_angle=source.incidence_angle.mid_swath,
            azimuth=source.instrument_azimuth_angle,
        )
        
        # Processing extension
        item.stac_extensions.append(
            'https://stac-extensions.github.io/processing/v1.1.0/schema.json'
        )
        if source.processing.facility is not None:
            item.properties['processing:facility'] = source.processing.facility
        item.properties['processing:software'] = source.processing.software
        item.properties['processing:level'] = common.processing_level
        
        # CARD4L extension
        item.stac_extensions.append(
            'https://stac-extensions.github.io/card4l/v0.1.0/sar/source.json'
        )
        item.properties['card4l:specification'] = product.card4l.specification
        item.properties['card4l:specification_version'] = product.card4l.version
        item.properties['card4l:beam_id'] = common.swath_identifier
        item.properties['card4l:orbit_mean_altitude'] = common.orbit_mean_altitude
        
        proc_param = {}
        if source.lut_applied is not None:
            proc_param['lut_applied'] = source.lut_applied
        
        range_bandwidth = _bandwidth_ghz(source.range.look_bandwidth)
        if range_bandwidth is not None:
            proc_param['range_look_bandwidth'] = range_bandwidth
        
        azimuth_bandwidth = _bandwidth_ghz(source.azimuth.look_bandwidth)
        if azimuth_bandwidth is not None:
            proc_param['azimuth_look_bandwidth'] = azimuth_bandwidth
        
        if proc_param:
            item.properties['card4l:source_processing_parameters'] = proc_param
        
        properties = {
            'card4l:incidence_angle_far_range': source.incidence_angle.maximum,
            'card4l:incidence_angle_near_range': source.incidence_angle.minimum,
            'card4l:integrated_sidelobe_ratio': source.performance.integrated_side_lobe_ratio,
            'card4l:ionosphere_indicator': source.ionosphere_indicator,
            'card4l:mean_faraday_rotation_angle': source.faraday_mean_rotation_angle,
            'card4l:noise_equivalent_intensity': {
                pol: estimate.model_dump()
                for pol, estimate in source.performance.estimates.items()
            },
            'card4l:noise_equivalent_intensity_type': source.performance.noise_equivalent_intensity_type,
            'card4l:orbit_data_source': source.orbit.data_source,
            'card4l:peak_sidelobe_ratio': source.performance.peak_side_lobe_ratio,
            'card4l:resolution_range': source.range.resolution,
            'card4l:resolution_azimuth': source.azimuth.resolution,
            'card4l:source_geometry': source.data_geometry,
        }
        for key, value in properties.items():
            if value is not None:
                item.properties[key] = value
        
        # Links
        links = [
            {
                'rel': 'card4l-document',
                'target': product.card4l.document,
                'media_type': 'application/pdf',
                'title': (
                    'CARD4L Product Family Specification: '
                    f'{product.name} (v{product.card4l.version})'
                ),
            },
            {
                'rel': 'about',
                'target': source.doi,
                'media_type': None,
                'title': 'Product definition reference.',
            },
            {
                'rel': 'access',
                'target': source.access,
                'media_type': None,
                'title': 'Product data access.',
            },
            {
                'rel': 'satellite',
                'target': common.platform_reference,
                'media_type': None,
                'title': 'CEOS Missions, Instruments and Measurements Database record',
            },
            {
                'rel': 'state-vectors',
                'target': source.orbit.state_vector,
                'media_type': None,
                'title': 'Orbit data file containing state vectors.',
            },
            {
                'rel': 'sensor-calibration',
                'target': source.sensor_calibration,
                'media_type': None,
                'title': 'Reference describing sensor calibration parameters.',
            },
            {
                'rel': 'pol-cal-matrices',
                'target': source.polarimetric_calibration_matrices,
                'media_type': None,
                'title': 'Reference to the complex-valued polarimetric distortion matrices.',
            },
            {
                'rel': 'referenced-faraday-rotation',
                'target': source.faraday_rotation_reference,
                'media_type': None,
                'title': (
                    'Reference describing the method used to derive the estimate '
                    'for the mean Faraday rotation angle.'
                ),
            },
        ]
        for link in links:
            if link['target'] is not None:
                item.add_link(link=pystac.Link(**link))
        
        # Assets
        relpath = os.path.relpath(outname.replace('.json', '.xml'), metadir)
        xml_relpath = './' + relpath.replace('\\', '/')
        item.add_asset(
            key='card4l',
            asset=pystac.Asset(
                href=xml_relpath,
                title='Metadata in XML format.',
                media_type=pystac.MediaType.XML,
                roles=['metadata', 'card4l'],
            ),
        )
        _asset_add_orig_src(metadir=metadir, uid=uid, item=item)
        
        # Source validation stays disabled for now, as in the legacy writer.
        # Some legacy source metadata intentionally represents unavailable
        # mandatory CARD4L values and needs a dedicated policy before enabling it.
        # item.validate()
        item.save_object(dest_href=outname)


def _asset_add_orig_src(
        metadir: str,
        uid: str,
        item: pystac.Item
) -> None:
    """
    Helper function to add the original source metadata files as assets to a STAC item.
    
    Parameters
    ----------
    metadir:
        Source directory of the current ARD product.
    uid:
        Unique identifier of a source scene.
    item:
        The pystac.Item to add the assets to.
    
    """
    pattern = r'^(.+?)-\d{8}t\d{6}-\d{8}t\d{6}-\w+-\w+-\d{3}\.xml$'
    prefixes = {'calibration': 'Calibration metadata',
                'noise': 'Estimated thermal noise look-up tables',
                'rfi': 'Radio Frequency Interference metadata'}
    
    root_dir = os.path.join(metadir, uid)
    if not os.path.isdir(root_dir):
        return
    file_list = finder(target=root_dir, matchlist=['*.safe', '*.xml'], foldermode=0)
    if len(file_list) > 0:
        for file in file_list:
            basename = os.path.basename(file)
            href = './' + os.path.relpath(file, metadir).replace('\\', '/')
            if basename == 'manifest.safe':
                key = 'manifest'
                title = 'Mandatory product metadata'
            else:
                try:
                    key = re.match(pattern, basename).group(1)
                except AttributeError:
                    raise RuntimeError(
                        'Unexpected file in original source metadata directory: ' + os.path.join(root_dir, file))
                title = prefixes.get(key.split('-')[0], 'Measurement metadata')
                if title == 'Measurement metadata':
                    title = title + ' ({},{})'.format(key.split('-')[1].upper(), key.split('-')[3].upper())
                else:
                    title = title + ' ({},{})'.format(key.split('-')[2].upper(), key.split('-')[4].upper())
            
            item.add_asset(key=key,
                           asset=pystac.Asset(href=href,
                                              title=title,
                                              media_type=pystac.MediaType.XML,
                                              roles=['metadata']))


def product_json(
        meta: MetadataInput,
        target: str,
        assets: list[str],
        exist_ok: bool = False
) -> None:
    """Create product-level STAC JSON metadata."""
    meta = _as_model(meta)
    common = meta.common
    product = meta.product
    
    scene_id = os.path.basename(target)
    outname = os.path.join(target, f'{scene_id}.json')
    if os.path.isfile(outname) and exist_ok:
        return
    
    log.info(f'creating {os.path.relpath(outname, target)}')
    start = product.time_start
    stop = product.time_stop
    date = start + (stop - start) / 2
    mgrs = product.grid.mgrs_id
    
    item = pystac.Item(
        id=scene_id,
        geometry=product.geometry.geometry,
        bbox=list(product.geometry.bbox),
        datetime=date,
        properties={},
    )
    
    # Common metadata
    item.common_metadata.license = product.license
    item.common_metadata.start_datetime = start
    item.common_metadata.end_datetime = stop
    item.common_metadata.created = product.time_created
    item.common_metadata.instruments = [common.instrument_short_name.lower()]
    item.common_metadata.constellation = common.constellation
    item.common_metadata.platform = common.platform_full_name
    item.common_metadata.gsd = product.grid.pixel_spacing_column
    
    # Extensions
    sar_ext = SarExtension.ext(item, add_if_missing=True)
    sat_ext = SatExtension.ext(item, add_if_missing=True)
    proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
    mgrs_ext = MgrsExtension.ext(item, add_if_missing=True)
    ## add extensions that are not directly supported by pystac
    item.stac_extensions.append(
        'https://stac-extensions.github.io/processing/v1.1.0/schema.json'
    )
    item.stac_extensions.append(
        'https://stac-extensions.github.io/card4l/v0.1.0/sar/product.json'
    )
    ###############################################################################################
    # sat extension
    sat_ext.apply(
        orbit_state=OrbitState[common.orbit_direction.upper()],
        relative_orbit=common.orbit_number_relative,
        absolute_orbit=common.orbit_number_absolute,
    )
    ###############################################################################################
    # sar extension
    sar_ext.apply(
        instrument_mode=common.operational_mode,
        frequency_band=FrequencyBand[common.radar_band.upper()],
        polarizations=[Polarization[pol] for pol in common.polarizations],
        product_type=product.product_type,
        looks_range=product.backscatter.range_number_of_looks,
        looks_azimuth=product.backscatter.azimuth_number_of_looks,
        looks_equivalent_number=product.backscatter.equivalent_number_of_looks,
    )
    ###############################################################################################
    # projection extension
    native_bbox = product.geometry.bbox_native
    proj_ext.apply(
        epsg=product.grid.epsg,
        wkt2=product.grid.wkt,
        bbox=list(native_bbox) if native_bbox is not None else None,
        # Projection Extension uses Y, X = rows, columns order.
        shape=[product.grid.rows, product.grid.columns],
        transform=list(product.grid.transform),
    )
    ###############################################################################################
    # mgrs extension
    mgrs_ext.apply(
        latitude_band=mgrs[2:3],
        grid_square=mgrs[3:],
        utm_zone=int(mgrs[:2]),
    )
    ###############################################################################################
    # processing extension
    if product.processing.facility is not None:
        item.properties['processing:facility'] = product.processing.facility
    item.properties['processing:software'] = product.processing.software
    item.properties['processing:level'] = common.processing_level
    ###############################################################################################
    # card4l extension
    item.properties['card4l:specification'] = product.card4l.specification
    item.properties['card4l:specification_version'] = product.card4l.version
    item.properties['card4l:beam_id'] = common.swath_identifier
    item.properties['card4l:measurement_type'] = product.backscatter.measurement
    item.properties['card4l:measurement_convention'] = product.backscatter.convention
    item.properties['card4l:pixel_coordinate_convention'] = product.grid.pixel_coordinate_convention
    item.properties['card4l:speckle_filtering'] = product.speckle_filter_applied
    item.properties['card4l:noise_removal_applied'] = product.noise_removal.applied
    item.properties['card4l:conversion_eq'] = product.backscatter.conversion_equation
    item.properties['card4l:relative_radiometric_accuracy'] = product.radiometric_accuracy.relative
    item.properties['card4l:absolute_radiometric_accuracy'] = product.radiometric_accuracy.absolute
    item.properties['card4l:resampling_method'] = product.geometric_correction.resampling_method
    item.properties['card4l:dem_resampling_method'] = product.dem.resampling_method
    item.properties['card4l:egm_resampling_method'] = product.dem.egm_resampling_method
    item.properties['card4l:gridding_convention'] = 'Sentinel-2 MGRS'
    
    accuracy = product.geometric_correction.accuracy
    item.properties['card4l:geometric_accuracy_type'] = accuracy.type
    item.properties['card4l:northern_geometric_accuracy'] = {
        'bias': accuracy.northern.bias,
        'stddev': accuracy.northern.standard_deviation,
    }
    item.properties['card4l:eastern_geometric_accuracy'] = {
        'bias': accuracy.eastern.bias,
        'stddev': accuracy.eastern.standard_deviation,
    }
    item.properties['card4l:geometric_accuracy_radial_rmse'] = accuracy.radial_rmse
    ###############################################################################################
    # links
    links = [
        {
            'rel': 'card4l-document',
            'target': product.card4l.document.replace('.pdf', '.docx'),
            'media_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'title': (
                'CARD4L Product Family Specification: '
                f'{product.name} (v{product.card4l.version})'
            ),
        },
        {
            'rel': 'card4l-document',
            'target': product.card4l.document,
            'media_type': 'application/pdf',
            'title': (
                'CARD4L Product Family Specification: '
                f'{product.name} (v{product.card4l.version})'
            ),
        },
        {
            'rel': 'about',
            'target': product.doi,
            'title': 'Product definition reference.',
            'media_type': None,
        },
        {
            'rel': 'access',
            'target': product.access,
            'title': 'Product data access',
            'media_type': None,
        },
        {
            'rel': 'related',
            'target': product.ancillary_data_kml,
            'title': (
                'Sentinel-2 Military Grid Reference System (MGRS) tiling '
                'grid file used as auxiliary data during processing'
            ),
            'media_type': None,
        },
        {
            'rel': 'noise-removal',
            'target': product.noise_removal.algorithm,
            'title': 'Reference to the noise removal algorithm details',
            'media_type': None,
        },
        {
            'rel': 'radiometric-terrain-correction',
            'target': product.rtc_algorithm,
            'title': 'Reference to the Radiometric Terrain Correction algorithm details',
            'media_type': None,
        },
        {
            'rel': 'radiometric-accuracy',
            'target': product.radiometric_accuracy.reference,
            'title': 'Reference describing the radiometric uncertainty of the product',
            'media_type': None,
        },
        {
            'rel': 'geometric-correction',
            'target': product.geometric_correction.algorithm,
            'title': 'Reference to the Geometric Correction algorithm details',
            'media_type': None,
        },
        # CARD4L currently expects both link relations although the DEM type
        # itself is available in the canonical metadata model.
        {
            'rel': 'elevation-model',
            'target': product.dem.reference,
            'title': (
                'Digital Elevation Model used as auxiliary data during processing: '
                f'{product.dem.name}'
            ),
            'media_type': None,
        },
        {
            'rel': 'surface-model',
            'target': product.dem.reference,
            'title': (
                'Digital Elevation Model used as auxiliary data during processing: '
                f'{product.dem.name}'
            ),
            'media_type': None,
        },
        {
            'rel': 'earth-gravitational-model',
            'target': product.dem.egm_reference,
            'title': 'Reference to the Earth Gravitational Model (EGM) used for geometric correction',
            'media_type': None,
        },
        {
            'rel': 'geometric-accuracy',
            'target': accuracy.reference,
            'title': 'Reference documenting the estimate of absolute localization error',
            'media_type': None,
        },
        {
            'rel': 'gridding-convention',
            'target': product.gridding_convention_url,
            'title': 'Reference describing the gridding convention used',
            'media_type': None,
        },
    ]
    
    if product.wind_normalization is not None:
        links.append(
            {
                'rel': 'wind-norm-reference',
                'target': product.wind_normalization.reference_model,
                'title': 'Reference to the model used to create the wind normalisation layer',
                'media_type': None,
            }
        )
    
    for source in meta.sources.values():
        source_name = os.path.basename(source.filename).split('.')[0]
        source_target = os.path.join('./source', f'{source_name}.json').replace('\\', '/')
        links.append(
            {
                'rel': 'derived_from',
                'target': source_target,
                'media_type': 'application/json',
                'title': 'Source metadata formatted in STAC compliant JSON format',
            }
        )
    
    for link in links:
        if link['target'] is not None:
            item.add_link(link=pystac.Link(**link))
    ###############################################################################################
    # assets
    assets = assets.copy()
    xml = outname.replace('.json', '.xml')
    if os.path.isfile(xml):
        assets.append(xml)
    
    assets_dict = {
        'measurement': {},
        'annotation': {},
        'metadata': {},
    }
    for asset in assets:
        relpath = './' + os.path.relpath(asset, target).replace('\\', '/')
        
        size = os.path.getsize(asset)
        checksum = compute_hash(asset)
        created = None
        header_size = None
        media_type = pystac.MediaType.COG  # COG, VRT
        byte_order = None
        nodata = None
        if asset.endswith('.tif'):
            with Raster(asset) as ras:
                nodata = ras.nodata
            created = datetime.fromtimestamp(
                os.path.getctime(asset), tz=timezone.utc
            ).isoformat()
            header_size = get_header_size(tif=asset)
            byte_order = ByteOrder.LITTLE_ENDIAN
        
        if 'measurement' in asset:
            key, title = _asset_get_key_title(meta=meta, asset=asset)
            stac_asset = pystac.Asset(
                href=relpath,
                title=title,
                media_type=media_type,
                roles=['backscatter', 'data'],
                extra_fields=None,
            )
            if asset.endswith('.tif'):
                stac_asset.extra_fields = {
                    'created': created,
                    'card4l:border_pixels': product.grid.number_of_border_pixels,
                }
                _asset_handle_raster_ext(stac_asset=stac_asset, nodata=nodata)
            file_ext = FileExtension.ext(stac_asset)
            file_ext.apply(
                byte_order=byte_order,
                size=size,
                header_size=header_size,
                checksum=checksum,
            )
            assets_dict['measurement'][key] = stac_asset
        
        elif 'annotation' in asset:
            key, title = _asset_get_key_title(meta=meta, asset=asset)
            if key == 'np':
                pol = re.search('-[vh]{2}', relpath).group().removeprefix('-')
                asset_key = f'noise-power-{pol}'
                title = f'{title} {pol.upper()}'
            else:
                asset_key = ASSET_MAP[key]['role']
            
            unit = ASSET_MAP[key]['unit'] or 'unitless'
            stac_asset = pystac.Asset(
                href=relpath,
                title=title,
                media_type=media_type,
                roles=[ASSET_MAP[key]['role'], 'metadata'],
                extra_fields=None,
            )
            if key == 'ei' and product.ellipsoidal_height is not None:
                stac_asset.extra_fields = {
                    'card4l:ellipsoidal_height': product.ellipsoidal_height
                }
            file_ext = FileExtension.ext(stac_asset)
            file_ext.apply(
                byte_order=byte_order,
                size=size,
                header_size=header_size,
                checksum=checksum,
            )
            _asset_handle_raster_ext(
                stac_asset=stac_asset,
                nodata=nodata,
                key=key,
                meta=meta,
                asset=asset,
                unit=unit,
            )
            assets_dict['annotation'][asset_key] = stac_asset
        
        else:
            stac_asset = pystac.Asset(
                href=relpath,
                title='Metadata in XML format.',
                media_type=pystac.MediaType.XML,
                roles=['metadata', 'card4l'],
            )
            file_ext = FileExtension.ext(stac_asset)
            file_ext.apply(
                byte_order=byte_order,
                size=size,
                header_size=header_size,
                checksum=checksum,
            )
            assets_dict['metadata']['card4l'] = stac_asset
    
    for category in ['measurement', 'annotation', 'metadata']:
        for key in sorted(assets_dict[category]):
            item.add_asset(key=key, asset=assets_dict[category][key])
    
    ## add schema URIs for extensions used in asset metadata
    if any(x in item.get_assets() for x in ['acquisition-id', 'data-mask']):
        ClassificationExtension.add_to(item)
    FileExtension.add_to(item)
    RasterExtension.add_to(item)
    ###############################################################################################
    # downgrade extensions as required by CARD4L v0.1.0.
    item.stac_extensions = [
        'https://stac-extensions.github.io/file/v2.0.0/schema.json'
        if ext.startswith('https://stac-extensions.github.io/file/')
        else 'https://stac-extensions.github.io/projection/v1.0.0/schema.json'
        if ext.startswith('https://stac-extensions.github.io/projection/')
        else ext
        for ext in item.stac_extensions
    ]
    ###############################################################################################
    # validate and save
    item.validate()
    item.save_object(dest_href=outname)


def _asset_get_key_title(
        meta: ARDMetadata,
        asset: str
) -> tuple[str, str]:
    """Return the STAC asset key and title for *asset*."""
    key = None
    title = None
    if 'measurement' in asset:
        title_dict = {
            'g': 'gamma nought',
            's': 'sigma nought',
            'lin': 'linear',
            'log': 'logarithmic',
        }
        pattern = (
            '(?P<key>(?P<pol>[vhc]{2})-'
            '(?P<nought>[gs])-'
            '(?P<scaling>lin|log)'
            '(?P<windnorm>-wn|))'
        )
        info = re.search(pattern, asset).groupdict()
        key = info['key']
        
        if re.search('cc-[gs]-lin', key):
            # Copy because canonical polarizations are immutable and the helper
            # must never mutate the metadata object.
            pols = list(meta.common.polarizations)
            co = pols.pop(0) if pols[0][0] == pols[0][1] else pols.pop(1)
            cross = pols[0]
            title = (
                'RGB color composite ('
                '{co}-{nought}-lin, '
                '{cross}-{nought}-lin, '
                '{co}-{nought}-lin/{cross}-{nought}-lin)'
            ).format(
                co=co.lower(),
                cross=cross.lower(),
                nought=info['nought'],
            )
        else:
            if info['windnorm'] == '-wn':
                skeleton = (
                    '{pol} {nought} {subtype} wind normalisation ratio, '
                    '{scale} scaling'
                )
            else:
                skeleton = (
                    '{pol} {nought} {subtype} backscatter, {scale} scaling'
                )
            title = skeleton.format(
                pol=info['pol'].upper(),
                nought=title_dict[info['nought']],
                subtype='RTC',
                scale=title_dict[info['scaling']],
            )
    
    elif 'annotation' in asset:
        pattern = '(dm|ei|em|lc|ld|li|gs|id|np-[vh]{2}|sg|wm).tif'
        key = re.search(pattern, asset).groups()[0][:2]
        title = ASSET_MAP[key]['title']
    
    return key, title


def _asset_handle_raster_ext(
        stac_asset: pystac.Asset,
        nodata: float | None,
        key: str | None = None,
        meta: ARDMetadata | None = None,
        asset: str | None = None,
        unit: str | None = None,
) -> None:
    """Apply the STAC Raster Extension to an asset."""
    raster_ext = RasterExtension.ext(stac_asset)
    
    if 'measurement' in stac_asset.href:
        raster_ext.bands = [
            RasterBand.create(
                nodata=nodata,
                data_type=DataType.FLOAT32,
                unit='natural',
            )
        ]
    
    elif 'annotation' in stac_asset.href:
        if key is None or meta is None or asset is None:
            raise ValueError(
                '`key`, `meta` and `asset` parameters need to be defined '
                f'to handle RasterExtension for {stac_asset.href}'
            )
        if unit is None:
            unit = ASSET_MAP[key]['unit'] or 'unitless'
        
        if key == 'id':
            source_names = [
                os.path.basename(source.filename)
                .replace('.SAFE', '')
                .replace('.zip', '')
                for source in meta.sources.values()
            ]
            band = RasterBand.create(
                nodata=nodata,
                data_type=DataType.UINT8,
                unit=unit,
            )
            class_ext = ClassificationExtension.ext(band)
            class_ext.classes = [
                Classification.create(value=i + 1, description=name)
                for i, name in enumerate(source_names)
            ]
            raster_ext.bands = [band]
        
        elif key == 'dm':
            with Raster(asset) as dm_ras:
                band_descr = [
                    dm_ras.raster.GetRasterBand(band).GetDescription()
                    for band in range(1, dm_ras.bands + 1)
                ]
            samples = [
                value for value in band_descr
                if value in ASSET_MAP[key]['allowed']
            ]
            bands = []
            for sample in samples:
                band = RasterBand.create(
                    nodata=nodata,
                    data_type=DataType.UINT8,
                    unit=unit,
                )
                class_ext = ClassificationExtension.ext(
                    band, add_if_missing=True
                )
                class_ext.classes = [
                    Classification.create(value=1, description=sample)
                ]
                bands.append(band)
            raster_ext.bands = bands
        
        else:
            raster_ext.bands = [
                RasterBand.create(
                    nodata=nodata,
                    data_type=DataType.FLOAT32,
                    unit=unit,
                )
            ]
    
    if key == 'em':
        raster_ext.bands[0].spatial_resolution = int(meta.product.dem.gsd.value)
