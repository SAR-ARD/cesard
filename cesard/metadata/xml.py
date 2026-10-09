import os
import re
from copy import deepcopy
from dataclasses import dataclass
from lxml import etree
from datetime import datetime, timezone
from spatialist import Raster
from spatialist.ancillary import finder
from statistics import mean
from cesard.metadata.mapping import ASSET_MAP, NS_MAP
from cesard.metadata.model import ARDMetadata, ProcessingMetadata
from cesard.metadata.extract import get_header_size
from typing import Mapping, Literal
import logging

log = logging.getLogger('cesard')


@dataclass(frozen=True, slots=True)
class XmlField:
    """Description of one ordered XML text element."""
    
    parent: etree._Element
    name: str
    value: object | None = None
    attributes: Mapping[str, object] | None = None
    omit_if_none: bool = False


def _append_xml_field(
        field: XmlField,
        nsmap: Mapping[str, str],
        ard_ns: str,
) -> etree._Element | None:
    """Append one XML element according to its absence policy."""
    if field.value is None and field.omit_if_none:
        return None
    
    attributes = {
        name: str(value)
        for name, value in (field.attributes or {}).items()
        if value is not None
    }
    element = etree.SubElement(
        field.parent,
        _nsc(field.name, nsmap, ard_ns=ard_ns),
        attrib=attributes,
    )
    if field.value is not None:
        element.text = str(field.value)
    return element


def _processor_version(processing: ProcessingMetadata) -> str | None:
    """Return the version of the primary processor, if available."""
    if processing.processor is None:
        return None
    return processing.software.get(processing.processor)


def parse(
        meta: ARDMetadata,
        target: str,
        assets: list[str],
        nsmap_ard: dict[str, dict[Literal['NRB', 'ORB'], dict[Literal['key', 'source', 'product'], str]]],
        exist_ok: bool = False
) -> None:
    """
    Wrapper for :func:`~cesard.metadata.xml.source_xml`
    and :func:`~cesard.metadata.xml.product_xml`.

    Parameters
    ----------
    meta
        A validated :class:`~cesard.metadata.model.ARDMetadata` object.
    target
        A path pointing to the root directory of a product scene.
    assets
        List of paths to all GeoTIFF and VRT assets of the currently
        processed ARD product.
    nsmap_ard
        A dictionary containing the namespace mappings for the ARD product.
    exist_ok
        Do not create files if they already exist?
    """
    
    nsmap = deepcopy(NS_MAP)
    product = meta.product.product_type
    constellation = meta.common.constellation
    nsmap_product = nsmap_ard[constellation][product]
    key = nsmap_product['key']
    
    nsmap.update({
        key: nsmap_product['source']
    })
    source_xml(
        meta=meta, target=target,
        nsmap=nsmap, ard_ns=key,
        exist_ok=exist_ok
    )
    
    nsmap.update({
        key: nsmap_product['product']
    })
    product_xml(
        meta=meta, target=target, assets=assets,
        nsmap=nsmap, ard_ns=key,
        exist_ok=exist_ok
    )


def source_xml(
        meta: ARDMetadata,
        target: str,
        nsmap: dict[str, str],
        ard_ns: str,
        exist_ok: bool = False
) -> None:
    """
    Function to generate source-level metadata for an ARD product in `OGC 10-157r4` compliant XML format.

    Parameters
    ----------
    meta:
        A validated :class:`~cesard.metadata.model.ARDMetadata` object.
    target:
        A path pointing to the root directory of a product scene.
    nsmap:
        Dictionary listing abbreviation (key) and URI (value) of all necessary XML namespaces.
    ard_ns:
        Abbreviation of the ARD namespace. E.g., `s1-nrb` for the NRB ARD product.
    exist_ok:
        Do not create files if they already exist?
    """
    metadir = os.path.join(target, 'source')
    for uid, source in meta.sources.items():
        scene = os.path.basename(source.filename).split('.')[0]
        outname = os.path.join(metadir, '{}.xml'.format(scene))
        if os.path.isfile(outname) and exist_ok:
            continue
        log.info(f'creating {os.path.relpath(outname, target)}')
        timeStart = source.time_start.isoformat()
        timeStop = source.time_stop.isoformat()
        
        root = etree.Element(_nsc('_:EarthObservation', nsmap, ard_ns=ard_ns), nsmap=nsmap,
                             attrib={_nsc('gml:id', nsmap): scene + '_1'})
        _om_time(root=root, nsmap=nsmap, scene_id=scene, time_start=timeStart, time_stop=timeStop)
        _om_procedure(root=root, nsmap=nsmap, ard_ns=ard_ns, scene_id=scene, meta=meta, uid=uid, prod=False)
        observedProperty = etree.SubElement(root, _nsc('om:observedProperty', nsmap),
                                            attrib={'nilReason': 'inapplicable'})
        _om_feature_of_interest(root=root, nsmap=nsmap, scene_id=scene,
                                extent=source.geometry.xml_envelopes,
                                center=source.geometry.xml_center)
        
        ################################################################################################################
        result = etree.SubElement(root, _nsc('om:result', nsmap))
        earthObservationResult = etree.SubElement(result, _nsc('eop:EarthObservationResult', nsmap),
                                                  attrib={_nsc('gml:id', nsmap): scene + '_9'})
        product = etree.SubElement(earthObservationResult, _nsc('eop:product', nsmap))
        productInformation = etree.SubElement(product, _nsc('_:ProductInformation', nsmap, ard_ns=ard_ns))
        fileName = etree.SubElement(productInformation, _nsc('eop:fileName', nsmap))
        serviceReference = etree.SubElement(fileName, _nsc('ows:ServiceReference', nsmap),
                                            attrib={_nsc('xlink:href', nsmap): scene})
        requestMessage = etree.SubElement(serviceReference, _nsc('ows:RequestMessage', nsmap))
        
        org_src_files_dir = os.path.join(metadir, uid)
        if os.path.isdir(org_src_files_dir):
            org_src_files = finder(target=org_src_files_dir, matchlist=['*.safe', '*.xml'], foldermode=0)
            if len(org_src_files) > 0:
                for file in org_src_files:
                    href = './' + os.path.relpath(file, metadir).replace('\\', '/')
                    product = etree.SubElement(earthObservationResult, _nsc('eop:product', nsmap))
                    productInformation = etree.SubElement(product, _nsc('_:ProductInformation', nsmap, ard_ns=ard_ns))
                    fileName = etree.SubElement(productInformation, _nsc('eop:fileName', nsmap))
                    serviceReference = etree.SubElement(fileName, _nsc('ows:ServiceReference', nsmap),
                                                        attrib={_nsc('xlink:href', nsmap): href})
                    requestMessage = etree.SubElement(serviceReference, _nsc('ows:RequestMessage', nsmap))
                    dataFormat = etree.SubElement(productInformation, _nsc('_:dataFormat', nsmap, ard_ns=ard_ns))
                    dataFormat.text = 'XML'
        ################################################################################################################
        metaDataProperty = etree.SubElement(root, _nsc('eop:metaDataProperty', nsmap))
        earthObservationMetaData = etree.SubElement(metaDataProperty, _nsc('_:EarthObservationMetaData', nsmap,
                                                                           ard_ns=ard_ns))
        
        identifier = etree.SubElement(earthObservationMetaData, _nsc('eop:identifier', nsmap))
        identifier.text = scene
        doi = etree.SubElement(earthObservationMetaData, _nsc('eop:doi', nsmap))
        doi.text = source.doi
        acquisitionType = etree.SubElement(earthObservationMetaData, _nsc('eop:acquisitionType', nsmap))
        acquisitionType.text = source.acquisition_type
        status = etree.SubElement(earthObservationMetaData, _nsc('eop:status', nsmap))
        status.text = source.status
        
        processing = etree.SubElement(
            earthObservationMetaData,
            _nsc('eop:processing', nsmap)
        )
        processingInformation = etree.SubElement(
            processing,
            _nsc('_:ProcessingInformation', nsmap, ard_ns=ard_ns)
        )
        
        fields = [
            XmlField(
                parent=processingInformation,
                name='eop:processingCenter',
                attributes={
                    'codeSpace': 'urn:esa:eop:Sentinel1:facility',
                },
                value=source.processing.facility,
            ),
            XmlField(
                parent=processingInformation,
                name='eop:processingDate',
                value=(
                    source.processing.date.isoformat()
                    if source.processing.date is not None
                    else None
                ),
            ),
            XmlField(
                parent=processingInformation,
                name='eop:processorName',
                value=source.processing.processor,
            ),
            XmlField(
                parent=processingInformation,
                name='eop:processorVersion',
                value=_processor_version(source.processing),
            ),
            XmlField(
                parent=processingInformation,
                name='eop:processingMode',
                value=source.processing.mode,
            ),
            XmlField(
                parent=processingInformation,
                name='_:orbitDataSource',
                value=(
                    source.orbit.data_source.upper()
                    if source.orbit.data_source is not None
                    else None
                ),
            ),
            XmlField(
                parent=processingInformation,
                name='_:orbitStateVector',
                attributes={'access': source.orbit.data_access},
                value=source.orbit.state_vector,
                omit_if_none=True,
            ),
            XmlField(
                parent=processingInformation,
                name='_:lutApplied',
                value=source.lut_applied,
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:productType',
                attributes={
                    'codeSpace': 'urn:esa:eop:Sentinel1:class',
                },
                value=source.product_type,
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:dataGeometry',
                value=source.data_geometry,
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:azimuthPixelSpacing',
                attributes={'uom': 'm'},
                value=mean(source.azimuth.pixel_spacing.values()),
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:rangePixelSpacing',
                attributes={'uom': 'm'},
                value=mean(source.range.pixel_spacing.values()),
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:meanFaradayRotationAngle',
                attributes={'uom': 'deg'},
                value=source.faraday_mean_rotation_angle,
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:referenceFaradayRotation',
                attributes={
                    _nsc('xlink:href', nsmap):
                        source.faraday_rotation_reference,
                },
            ),
            XmlField(
                parent=earthObservationMetaData,
                name='_:ionosphereIndicator',
                value=source.ionosphere_indicator,
            ),
        ]
        
        for field in fields:
            _append_xml_field(
                field=field,
                nsmap=nsmap,
                ard_ns=ard_ns,
            )
        
        processingLevel = etree.SubElement(
            processingInformation,
            _nsc('_:processingLevel', nsmap, ard_ns=ard_ns)
        )
        processingLevel.text = meta.common.processing_level
        
        for swath in source.swaths:
            fields = [
                XmlField(
                    parent=processingInformation,
                    name='_:azimuthLookBandwidth',
                    attributes={'uom': 'Hz', 'beam': swath},
                    value=source.azimuth.look_bandwidth[swath],
                ),
                XmlField(
                    parent=processingInformation,
                    name='_:rangeLookBandwidth',
                    attributes={'uom': 'Hz', 'beam': swath},
                    value=source.range.look_bandwidth[swath],
                ),
                XmlField(
                    parent=earthObservationMetaData,
                    name='_:azimuthNumberOfLooks',
                    attributes={'beam': swath},
                    value=source.azimuth.number_of_looks[swath],
                ),
                XmlField(
                    parent=earthObservationMetaData,
                    name='_:rangeNumberOfLooks',
                    attributes={'beam': swath},
                    value=source.range.number_of_looks[swath],
                ),
                XmlField(
                    parent=earthObservationMetaData,
                    name='_:azimuthResolution',
                    attributes={'uom': 'm', 'beam': swath},
                    value=source.azimuth.resolution[swath],
                ),
                XmlField(
                    parent=earthObservationMetaData,
                    name='_:rangeResolution',
                    attributes={'uom': 'm', 'beam': swath},
                    value=source.range.resolution[swath],
                ),
            ]
            
            for field in fields:
                _append_xml_field(
                    field=field,
                    nsmap=nsmap,
                    ard_ns=ard_ns,
                )
        
        performance = etree.SubElement(
            earthObservationMetaData,
            _nsc('_:performance', nsmap, ard_ns=ard_ns)
        )
        performanceIndicators = etree.SubElement(
            performance,
            _nsc('_:PerformanceIndicators', nsmap, ard_ns=ard_ns)
        )
        noiseEquivalentIntensityType = etree.SubElement(
            performanceIndicators,
            _nsc('_:noiseEquivalentIntensityType', nsmap, ard_ns=ard_ns),
            attrib={'uom': 'dB'}
        )
        noiseEquivalentIntensityType.text = str(source.performance.noise_equivalent_intensity_type)
        
        for pol in meta.common.polarizations:
            perf = source.performance.estimates[pol]
            for stat in ['minimum', 'mean', 'maximum']:
                _append_xml_field(
                    field=XmlField(
                        parent=performanceIndicators,
                        name='_:estimates',
                        attributes={'pol': pol, 'type': stat},
                        value=getattr(perf, stat),
                    ),
                    nsmap=nsmap,
                    ard_ns=ard_ns,
                )
        
        fields = [
            ('_:equivalentNumberOfLooks',
             source.performance.equivalent_number_of_looks),
            ('_:peakSideLobeRatio',
             source.performance.peak_side_lobe_ratio),
            ('_:integratedSideLobeRatio',
             source.performance.integrated_side_lobe_ratio),
        ]
        
        for field_dst, value in fields:
            element = etree.SubElement(performanceIndicators, _nsc(field_dst, nsmap, ard_ns=ard_ns))
            element.text = str(value)
        
        polCalMatrices = etree.SubElement(
            earthObservationMetaData,
            _nsc('_:polCalMatrices', nsmap, ard_ns=ard_ns),
            attrib={_nsc('xlink:href', nsmap): str(source.polarimetric_calibration_matrices)}
        )
        ################################################################################################################
        etree.indent(root)
        tree = etree.ElementTree(root)
        tree.write(outname, pretty_print=True, xml_declaration=True, encoding='utf-8')


def product_xml(
        meta: ARDMetadata,
        target: str,
        assets: list[str],
        nsmap: dict[str, str],
        ard_ns: str,
        exist_ok: bool = False
) -> None:
    """
    Function to generate product-level metadata for an ARD product in `OGC 10-157r4` compliant XML format.

    Parameters
    ----------
    meta:
        A validated :class:`~cesard.metadata.model.ARDMetadata` object.
    target:
        A path pointing to the root directory of a product scene.
    assets:
        List of paths to all GeoTIFF and VRT assets of the currently processed ARD product.
    nsmap:
        Dictionary listing abbreviation (key) and URI (value) of all necessary XML namespaces.
    ard_ns:
        Abbreviation of the ARD namespace. E.g., `s1-nrb` for the NRB ARD product.
    exist_ok:
        Do not create files if they already exist?
    """
    scene_id = os.path.basename(target)
    outname = os.path.join(target, '{}.xml'.format(scene_id))
    if os.path.isfile(outname) and exist_ok:
        return
    log.info(f'creating {os.path.relpath(outname, target)}')
    timeCreated = meta.product.time_created.isoformat()
    timeStart = meta.product.time_start.isoformat()
    timeStop = meta.product.time_stop.isoformat()
    
    root = etree.Element(_nsc('_:EarthObservation', nsmap, ard_ns=ard_ns), nsmap=nsmap,
                         attrib={_nsc('gml:id', nsmap): scene_id + '_1'})
    _om_time(root=root, nsmap=nsmap, scene_id=scene_id, time_start=timeStart, time_stop=timeStop)
    _om_procedure(root=root, nsmap=nsmap, ard_ns=ard_ns, scene_id=scene_id, meta=meta, prod=True)
    observedProperty = etree.SubElement(root, _nsc('om:observedProperty', nsmap),
                                        attrib={'nilReason': 'inapplicable'})
    _om_feature_of_interest(root=root, nsmap=nsmap, scene_id=scene_id,
                            extent=meta.product.geometry.xml_envelopes,
                            center=meta.product.geometry.xml_center)
    
    ####################################################################################################################
    result = etree.SubElement(root, _nsc('om:result', nsmap))
    earthObservationResult = etree.SubElement(result, _nsc('eop:EarthObservationResult', nsmap),
                                              attrib={_nsc('gml:id', nsmap): scene_id + '_9'})
    product = etree.SubElement(earthObservationResult, _nsc('eop:product', nsmap))
    productInformation = etree.SubElement(product, _nsc('_:ProductInformation', nsmap, ard_ns=ard_ns))
    fileName = etree.SubElement(productInformation, _nsc('eop:fileName', nsmap))
    serviceReference = etree.SubElement(fileName, _nsc('ows:ServiceReference', nsmap),
                                        attrib={_nsc('xlink:href', nsmap): scene_id})
    requestMessage = etree.SubElement(serviceReference, _nsc('ows:RequestMessage', nsmap))
    
    for asset in assets:
        relpath = './' + os.path.relpath(asset, target).replace('\\', '/')
        
        no_data = None
        header_size = None
        data_format = 'VRT'
        byte_order = None
        data_type = None
        z_error = None
        if asset.endswith('.tif'):
            with Raster(asset) as ras:
                no_data = str(ras.nodata)
            header_size = str(get_header_size(tif=asset))
            data_format = 'COG'
            byte_order = 'little-endian'
            data_type = 'FLOAT 32'
            prefix = '[0-9a-z]{5}-'
            match = re.search(prefix + f"({'|'.join(meta.product.compression.z_errors.keys())})",
                              os.path.basename(asset))
            if match is not None:
                k = match.group()
                k = k.removeprefix(re.search(prefix, k).group())
                z_error = str(meta.product.compression.z_errors[k])
        
        product = etree.SubElement(earthObservationResult, _nsc('eop:product', nsmap))
        productInformation = etree.SubElement(product, _nsc('_:ProductInformation', nsmap, ard_ns=ard_ns))
        fileName = etree.SubElement(productInformation, _nsc('eop:fileName', nsmap))
        serviceReference = etree.SubElement(fileName, _nsc('ows:ServiceReference', nsmap),
                                            attrib={_nsc('xlink:href', nsmap): relpath})
        requestMessage = etree.SubElement(serviceReference, _nsc('ows:RequestMessage', nsmap))
        
        size = etree.SubElement(productInformation, _nsc('eop:size', nsmap), attrib={'uom': 'bytes'})
        size.text = str(os.path.getsize(asset))
        
        lookup = [
            ('headerSize', header_size, {'uom': 'bytes'}),
            ('byteOrder', byte_order, None),
            ('dataFormat', data_format, None),
            ('noDataValue', no_data, None)
        ]
        for key, value, attrib in lookup:
            if value is not None:
                element = etree.SubElement(_parent=productInformation,
                                           _tag=_nsc(text=f'_:{key}', nsmap=nsmap, ard_ns=ard_ns),
                                           attrib=attrib)
                element.text = value
        
        if data_type is not None:
            dataType = etree.SubElement(productInformation, _nsc('_:dataType', nsmap, ard_ns=ard_ns))
            dataType.text = data_type.split()[0]
            bitsPerSample = etree.SubElement(productInformation, _nsc('_:bitsPerSample', nsmap, ard_ns=ard_ns))
            bitsPerSample.text = data_type.split()[1]
        if z_error is not None:
            compressionType = etree.SubElement(productInformation, _nsc('_:compressionType', nsmap, ard_ns=ard_ns))
            compressionType.text = meta.product.compression.type
            compressionzError = etree.SubElement(productInformation, _nsc('_:compressionZError', nsmap, ard_ns=ard_ns))
            compressionzError.text = z_error
        
        if 'annotation' in asset:
            pattern = '(dm|ei|em|lc|ld|li|gs|id|np-[vh]{2}|sg|wm).tif'
            key = re.search(pattern, asset).groups()[0][:2]
            
            sampleType = etree.SubElement(productInformation, _nsc('_:sampleType', nsmap, ard_ns=ard_ns),
                                          attrib={'uom': 'unitless' if ASSET_MAP[key]['unit'] is None else
                                          ASSET_MAP[key]['unit']})
            sampleType.text = ASSET_MAP[key]['type']
            
            if key in ['dm', 'id']:
                dataType.text = 'UINT'
                bitsPerSample.text = '8'
                
                if key == 'dm':
                    with Raster(asset) as dm_ras:
                        band_descr = [dm_ras.raster.GetRasterBand(band).GetDescription() for band in
                                      range(1, dm_ras.bands + 1)]
                    samples = [x for x in band_descr if x in ASSET_MAP[key]['allowed']]
                    for i, sample in enumerate(samples):
                        bitValue = etree.SubElement(productInformation, _nsc('_:bitValue', nsmap, ard_ns=ard_ns),
                                                    attrib={'band': str(i + 1),
                                                            'name': sample})
                        bitValue.text = '1'
                else:  # key == 'id'
                    src_list = list(meta.sources.keys())
                    src_target = [os.path.basename(meta.sources[src].filename).replace('.SAFE',
                                                                                       '').replace('.zip', '')
                                  for src in src_list]
                    for i, s in enumerate(src_target):
                        bitValue = etree.SubElement(productInformation, _nsc('_:bitValue', nsmap, ard_ns=ard_ns),
                                                    attrib={'band': '1', 'name': s})
                        bitValue.text = str(i + 1)
            
            if key == 'ei':
                ellipsoidalHeight = etree.SubElement(productInformation, _nsc('_:ellipsoidalHeight', nsmap,
                                                                              ard_ns=ard_ns),
                                                     attrib={'uom': 'm'})
                ellipsoidalHeight.text = (
                    None if meta.product.ellipsoidal_height is None
                    else str(meta.product.ellipsoidal_height)
                )
        
        if 'measurement' in asset and not asset.endswith('.vrt'):
            creationTime = etree.SubElement(productInformation, _nsc('_:creationTime', nsmap, ard_ns=ard_ns))
            creationTime.text = datetime.fromtimestamp(os.path.getctime(asset), tz=timezone.utc).isoformat()
            polarisation = etree.SubElement(productInformation, _nsc('_:polarisation', nsmap, ard_ns=ard_ns))
            polarisation.text = re.search('-[vh]{2}', relpath).group().removeprefix('-').upper()
            numBorderPixels = etree.SubElement(productInformation, _nsc('_:numBorderPixels', nsmap, ard_ns=ard_ns))
            numBorderPixels.text = str(meta.product.grid.number_of_border_pixels)
    
    ####################################################################################################################
    metaDataProperty = etree.SubElement(root, _nsc('eop:metaDataProperty', nsmap))
    earthObservationMetaData = etree.SubElement(metaDataProperty, _nsc('_:EarthObservationMetaData', nsmap,
                                                                       ard_ns=ard_ns))
    
    identifier = etree.SubElement(earthObservationMetaData, _nsc('eop:identifier', nsmap))
    identifier.text = scene_id
    if meta.product.doi is not None:
        doi = etree.SubElement(earthObservationMetaData, _nsc('eop:doi', nsmap))
        doi.text = meta.product.doi
    acquisitionType = etree.SubElement(earthObservationMetaData, _nsc('eop:acquisitionType', nsmap))
    acquisitionType.text = meta.product.acquisition_type
    status = etree.SubElement(earthObservationMetaData, _nsc('eop:status', nsmap))
    status.text = meta.product.status
    
    processing = etree.SubElement(earthObservationMetaData, _nsc('eop:processing', nsmap))
    processingInformation = etree.SubElement(processing, _nsc('_:ProcessingInformation', nsmap, ard_ns=ard_ns))
    if meta.product.processing.facility is not None:
        processingCenter = etree.SubElement(processingInformation, _nsc('eop:processingCenter', nsmap),
                                            attrib={'codeSpace': 'urn:esa:eop:Sentinel1:facility'})
        processingCenter.text = meta.product.processing.facility
    processingDate = etree.SubElement(processingInformation, _nsc('eop:processingDate', nsmap))
    processingDate.text = timeCreated
    processorName = etree.SubElement(processingInformation, _nsc('eop:processorName', nsmap))
    processorName.text = meta.product.processing.processor
    processorVersion = etree.SubElement(processingInformation, _nsc('eop:processorVersion', nsmap))
    processorVersion.text = _processor_version(meta.product.processing)
    processingMode = etree.SubElement(processingInformation, _nsc('eop:processingMode', nsmap),
                                      attrib={'codeSpace': 'urn:esa:eop:Sentinel1:class'})
    processingMode.text = meta.product.processing.mode
    processingLevel = etree.SubElement(processingInformation, _nsc('_:processingLevel', nsmap, ard_ns=ard_ns))
    processingLevel.text = meta.common.processing_level
    for src in list(meta.sources.keys()):
        src_path = '{}.xml'.format(os.path.basename(meta.sources[src].filename).split('.')[0])
        src_target = os.path.join('./source', src_path).replace('\\', '/')
        sourceProduct = etree.SubElement(processingInformation, _nsc('_:sourceProduct', nsmap, ard_ns=ard_ns),
                                         attrib={_nsc('xlink:href', nsmap): src_target})
    auxData1 = etree.SubElement(processingInformation, _nsc('_:auxiliaryDataSetFileName', nsmap, ard_ns=ard_ns),
                                attrib={_nsc('xlink:href', nsmap): meta.product.grid.definition_reference})
    speckleFilterApplied = etree.SubElement(processingInformation, _nsc('_:speckleFilterApplied', nsmap,
                                                                        ard_ns=ard_ns))
    speckleFilterApplied.text = str(meta.product.speckle_filter_applied).lower()
    noiseRemovalApplied = etree.SubElement(processingInformation, _nsc('_:noiseRemovalApplied', nsmap, ard_ns=ard_ns))
    noiseRemovalApplied.text = str(meta.product.noise_removal.applied).lower()
    value = meta.product.noise_removal.algorithm
    if meta.product.noise_removal.applied and value is not None:
        noiseRemovalAlgorithm = etree.SubElement(
            processingInformation,
            _nsc('_:noiseRemovalAlgorithm', nsmap, ard_ns=ard_ns),
            attrib={_nsc('xlink:href', nsmap): value},
        )
    if meta.product.rtc_algorithm is not None:
        rtcAlgorithm = etree.SubElement(processingInformation, _nsc('_:RTCAlgorithm', nsmap, ard_ns=ard_ns),
                                        attrib={_nsc('xlink:href', nsmap): meta.product.rtc_algorithm})
    if meta.product.wind_normalization is not None:
        windNormBackscatterMeasurement = etree.SubElement(processingInformation,
                                                          _nsc('_:windNormBackscatterMeasurement',
                                                               nsmap, ard_ns=ard_ns))
        windNormBackscatterMeasurement.text = meta.product.wind_normalization.backscatter_measurement
        windNormBackscatterConvention = etree.SubElement(processingInformation,
                                                         _nsc('_:windNormBackscatterConvention', nsmap, ard_ns=ard_ns))
        windNormBackscatterConvention.text = meta.product.wind_normalization.backscatter_convention
        windNormReferenceDirection = etree.SubElement(processingInformation,
                                                      _nsc('_:windNormReferenceDirection', nsmap, ard_ns=ard_ns),
                                                      attrib={'uom': 'deg'})
        windNormReferenceDirection.text = str(meta.product.wind_normalization.reference_direction)
        
        windNormReferenceModel = etree.SubElement(processingInformation, _nsc('_:windNormReferenceModel', nsmap,
                                                                              ard_ns=ard_ns),
                                                  attrib={_nsc('xlink:href', nsmap):
                                                              meta.product.wind_normalization.reference_model})
        windNormReferenceSpeed = etree.SubElement(processingInformation,
                                                  _nsc('_:windNormReferenceSpeed', nsmap, ard_ns=ard_ns),
                                                  attrib={'uom': 'm_s'})
        windNormReferenceSpeed.text = str(meta.product.wind_normalization.reference_speed)
        windNormReferenceType = etree.SubElement(processingInformation,
                                                 _nsc('_:windNormReferenceType', nsmap, ard_ns=ard_ns))
        windNormReferenceType.text = meta.product.wind_normalization.reference_type
    
    geoCorrAlgorithm = etree.SubElement(processingInformation, _nsc('_:geoCorrAlgorithm', nsmap, ard_ns=ard_ns),
                                        attrib={_nsc('xlink:href', nsmap): str(
                                            meta.product.geometric_correction.algorithm)})
    geoCorrResamplingMethod = etree.SubElement(processingInformation, _nsc('_:geoCorrResamplingAlgorithm', nsmap,
                                                                           ard_ns=ard_ns))
    geoCorrResamplingMethod.text = meta.product.geometric_correction.resampling_method.upper()
    demReference = etree.SubElement(processingInformation, _nsc('_:DEMReference', nsmap, ard_ns=ard_ns),
                                    attrib={'name': meta.product.dem.name,
                                            'dem': meta.product.dem.type,
                                            _nsc('xlink:href', nsmap): meta.product.dem.reference})
    demResamplingMethod = etree.SubElement(processingInformation, _nsc('_:DEMResamplingMethod', nsmap, ard_ns=ard_ns))
    demResamplingMethod.text = meta.product.dem.resampling_method.upper()
    demAccess = etree.SubElement(processingInformation, _nsc('_:DEMAccess', nsmap, ard_ns=ard_ns),
                                 attrib={_nsc('xlink:href', nsmap): meta.product.dem.access})
    demGSD = etree.SubElement(processingInformation, _nsc('_:DEMGroundSamplingDistance', nsmap, ard_ns=ard_ns),
                              attrib={'uom': meta.product.dem.gsd.unit})
    demGSD.text = str(meta.product.dem.gsd.value)
    
    if meta.product.dem.egm_resampling_method is not None:
        egmReference = etree.SubElement(
            processingInformation,
            _nsc('_:EGMReference', nsmap, ard_ns=ard_ns),
            attrib={_nsc('xlink:href', nsmap): meta.product.dem.egm_reference}
        )
        egmResamplingMethod = etree.SubElement(
            processingInformation,
            _nsc('_:EGMResamplingMethod', nsmap,
                 ard_ns=ard_ns)
        )
        egmResamplingMethod.text = meta.product.dem.egm_resampling_method.upper()
    
    productType = etree.SubElement(earthObservationMetaData, _nsc('_:productType', nsmap, ard_ns=ard_ns),
                                   attrib={'codeSpace': 'urn:esa:eop:Sentinel1:class'})
    productType.text = meta.product.product_type
    refDoc = etree.SubElement(earthObservationMetaData, _nsc('_:refDoc', nsmap, ard_ns=ard_ns),
                              attrib={'name': meta.product.name,
                                      'version': meta.product.card4l.version,
                                      _nsc('xlink:href', nsmap): meta.product.card4l.document})
    lookup = [
        ('azimuthNumberOfLooks', meta.product.backscatter.azimuth_number_of_looks, None),
        ('rangeNumberOfLooks', meta.product.backscatter.range_number_of_looks, None),
        ('equivalentNumberOfLooks', meta.product.backscatter.equivalent_number_of_looks, None),
        ('radiometricAccuracyRelative', meta.product.radiometric_accuracy.relative, {'uom': 'dB'}),
        ('radiometricAccuracyAbsolute', meta.product.radiometric_accuracy.absolute, {'uom': 'dB'}),
    ]
    for key, value, attrib in lookup:
        element = etree.SubElement(
            _parent=earthObservationMetaData,
            _tag=_nsc(text=f'_:{key}', nsmap=nsmap, ard_ns=ard_ns),
            attrib=attrib,
        )
        element.text = str(value)
    
    radacc_ref = str(meta.product.radiometric_accuracy.reference)
    radiometricAccuracyReference = etree.SubElement(earthObservationMetaData,
                                                    _nsc('_:radiometricAccuracyReference', nsmap, ard_ns=ard_ns),
                                                    attrib={_nsc('xlink:href', nsmap): radacc_ref})
    
    accuracy = meta.product.geometric_correction.accuracy
    lookup = [
        ('geoCorrAccuracyType', accuracy.type, None),
        ('geoCorrAccuracyNorthernSTDev', accuracy.northern.standard_deviation, {'uom': 'm'}),
        ('geoCorrAccuracyNorthernBias', accuracy.northern.bias, {'uom': 'm'}),
        ('geoCorrAccuracyEasternSTDev', accuracy.eastern.standard_deviation, {'uom': 'm'}),
        ('geoCorrAccuracyEasternBias', accuracy.eastern.bias, {'uom': 'm'}),
        ('geoCorrAccuracy_rRMSE', accuracy.radial_rmse, {'uom': 'm'}),
    ]
    for key, value, attrib in lookup:
        element = etree.SubElement(
            _parent=earthObservationMetaData,
            _tag=_nsc(f'_:{key}', nsmap, ard_ns=ard_ns),
            attrib=attrib,
        )
        element.text = str(value)
    
    geoacc_ref = meta.product.geometric_correction.accuracy.reference
    if geoacc_ref is not None:
        geoCorrAccuracyReference = etree.SubElement(earthObservationMetaData,
                                                    _tag=_nsc('_:geoCorrAccuracyReference', nsmap, ard_ns=ard_ns),
                                                    attrib={_nsc('xlink:href', nsmap): geoacc_ref})
    
    lookup = [
        ('numLines', meta.product.grid.rows, None),
        ('numPixelsPerLine', meta.product.grid.columns, None),
        ('columnSpacing', meta.product.grid.pixel_spacing_column, {'uom': 'm'}),
        ('rowSpacing', meta.product.grid.pixel_spacing_row, {'uom': 'm'}),
        ('pixelCoordinateConvention', meta.product.grid.pixel_coordinate_convention, None),
        ('backscatterMeasurement', meta.product.backscatter.measurement, None),
        ('backscatterConvention', meta.product.backscatter.convention, None),
        ('backscatterConversionEq', meta.product.backscatter.conversion_equation, {'uom': 'dB'}),
    ]
    for key, value, attrib in lookup:
        element = etree.SubElement(
            _parent=earthObservationMetaData,
            _tag=_nsc(text=f'_:{key}', nsmap=nsmap, ard_ns=ard_ns),
            attrib=attrib,
        )
        element.text = str(value)
    
    griddingConvention = etree.SubElement(_parent=earthObservationMetaData,
                                          _tag=_nsc('_:griddingConvention', nsmap, ard_ns=ard_ns),
                                          attrib={_nsc('xlink:href', nsmap):
                                                      meta.product.grid.convention_reference})
    
    lookup = [
        ('mgrsID', meta.product.grid.mgrs_id, None),
        ('crsEPSG', meta.product.grid.epsg, {'codeSpace': 'urn:esa:eop:crs'}),
        ('crsWKT', meta.product.grid.wkt, None),
    ]
    for key, value, attrib in lookup:
        element = etree.SubElement(
            _parent=earthObservationMetaData,
            _tag=_nsc(text=f'_:{key}', nsmap=nsmap, ard_ns=ard_ns),
            attrib=attrib,
        )
        element.text = str(value)
    ####################################################################################################################
    etree.indent(root)
    tree = etree.ElementTree(root)
    tree.write(outname, pretty_print=True, xml_declaration=True, encoding='utf-8')


def _nsc(
        text: str,
        nsmap: Mapping[str, str],
        ard_ns: str | None = None
) -> str:
    ns, key = text.split(':', maxsplit=1)
    if ard_ns is not None and ns == '_':
        ns = ard_ns
    return f'{{{nsmap[ns]}}}{key}'


def _om_time(
        root: etree.Element,
        nsmap: dict[str, str],
        scene_id: str,
        time_start: str,
        time_stop: str
) -> None:
    """
    Creates the `om:phenomenonTime` and `om:resultTime` XML elements.

    Parameters
    ----------
    root:
        Root XML element.
    nsmap:
        Dictionary listing abbreviation (key) and URI (value) of all necessary XML namespaces.
    scene_id:
        Scene basename.
    time_start:
        Start time of the scene acquisition.
    time_stop:
        Stop time of the acquisition.
    """
    phenomenonTime = etree.SubElement(root, _nsc('om:phenomenonTime', nsmap))
    timePeriod = etree.SubElement(phenomenonTime, _nsc('gml:TimePeriod', nsmap),
                                  attrib={_nsc('gml:id', nsmap): scene_id + '_2'})
    beginPosition = etree.SubElement(timePeriod, _nsc('gml:beginPosition', nsmap))
    beginPosition.text = time_start
    endPosition = etree.SubElement(timePeriod, _nsc('gml:endPosition', nsmap))
    endPosition.text = time_stop
    
    resultTime = etree.SubElement(root, _nsc('om:resultTime', nsmap))
    timeInstant = etree.SubElement(resultTime, _nsc('gml:TimeInstant', nsmap),
                                   attrib={_nsc('gml:id', nsmap): scene_id + '_3'})
    timePosition = etree.SubElement(timeInstant, _nsc('gml:timePosition', nsmap))
    timePosition.text = time_stop


def _om_procedure(
        root: etree.Element,
        nsmap: dict[str, str],
        ard_ns: str,
        scene_id: str,
        meta: ARDMetadata,
        uid: str | None = None,
        prod: bool = True
) -> None:
    """
    Creates the `om:procedure/eop:EarthObservationEquipment` XML elements and all relevant subelements for source and
    product metadata. Differences between source and product are controlled using the `prod=[True|False]` switch.

    Parameters
    ----------
    root:
        Root XML element.
    nsmap:
        Dictionary listing abbreviation (key) and URI (value) of all necessary XML namespaces.
    ard_ns:
        Abbreviation of the ARD namespace. E.g., `s1-nrb` for the NRB ARD product.
    scene_id: str
        Scene basename.
    meta:
        A validated :class:`~cesard.metadata.model.ARDMetadata` object.
    uid:
        Unique identifier of a source SLC scene.
    prod:
        Return XML subelements for further usage in :func:`~cesard.metadata.xml.product_xml` parsing function?
        Default is True. If False, the XML subelements for further usage in the :func:`~cesard.metadata.xml.source_xml`
        parsing function will be returned.
    """
    source = None
    if not prod:
        if uid is None:
            raise ValueError("'uid' must be provided for source metadata")
        source = meta.sources[uid]
    
    procedure = etree.SubElement(root, _nsc('om:procedure', nsmap))
    earthObservationEquipment = etree.SubElement(procedure, _nsc('eop:EarthObservationEquipment', nsmap),
                                                 attrib={_nsc('gml:id', nsmap): scene_id + '_4'})
    
    # eop:platform
    platform0 = etree.SubElement(earthObservationEquipment, _nsc('eop:platform', nsmap))
    if prod:
        platform1 = etree.SubElement(platform0, _nsc('eop:Platform', nsmap))
    else:
        platform1 = etree.SubElement(platform0, _nsc('_:Platform', nsmap, ard_ns=ard_ns))
    shortName = etree.SubElement(platform1, _nsc('eop:shortName', nsmap))
    shortName.text = meta.common.platform_short_name.upper()
    serialIdentifier = etree.SubElement(platform1, _nsc('eop:serialIdentifier', nsmap))
    serialIdentifier.text = meta.common.platform_identifier
    if source is not None:
        satReference = etree.SubElement(platform1, _nsc('_:satelliteReference', nsmap, ard_ns=ard_ns),
                                        attrib={_nsc('xlink:href', nsmap): meta.common.platform_reference})
    
    # eop:instrument
    instrument0 = etree.SubElement(earthObservationEquipment, _nsc('eop:instrument', nsmap))
    instrument1 = etree.SubElement(instrument0, _nsc('eop:Instrument', nsmap))
    shortName = etree.SubElement(instrument1, _nsc('eop:shortName', nsmap))
    shortName.text = meta.common.instrument_short_name
    
    # eop:sensor
    sensor0 = etree.SubElement(earthObservationEquipment, _nsc('eop:sensor', nsmap))
    sensor1 = etree.SubElement(sensor0, _nsc('_:Sensor', nsmap, ard_ns=ard_ns))
    sensorType = etree.SubElement(sensor1, _nsc('eop:sensorType', nsmap))
    sensorType.text = meta.common.sensor_type
    operationalMode = etree.SubElement(sensor1, _nsc('eop:operationalMode', nsmap),
                                       attrib={'codeSpace': 'urn:esa:eop:C-SAR:operationalMode'})
    operationalMode.text = meta.common.operational_mode
    swathIdentifier = etree.SubElement(sensor1, _nsc('eop:swathIdentifier', nsmap),
                                       attrib={'codeSpace': 'urn:esa:eop:C-SAR:swathIdentifier'})
    swathIdentifier.text = meta.common.swath_identifier
    radarBand = etree.SubElement(sensor1, _nsc('_:radarBand', nsmap, ard_ns=ard_ns))
    radarBand.text = meta.common.radar_band
    if source is not None:
        radarCenterFreq = etree.SubElement(sensor1, _nsc('_:radarCenterFrequency', nsmap, ard_ns=ard_ns),
                                           attrib={'uom': 'Hz'})
        radarCenterFreq.text = '{:.3e}'.format(meta.common.radar_center_frequency)
        value = source.sensor_calibration
        if value is not None:
            sensorCalibration = etree.SubElement(sensor1, _nsc('_:sensorCalibration',
                                                               nsmap, ard_ns=ard_ns),
                                                 attrib={_nsc('xlink:href', nsmap): value})
    
    # eop:acquisitionParameters
    acquisitionParameters = etree.SubElement(earthObservationEquipment, _nsc('eop:acquisitionParameters', nsmap))
    acquisition = etree.SubElement(acquisitionParameters, _nsc('_:Acquisition', nsmap, ard_ns=ard_ns))
    orbitNumber = etree.SubElement(acquisition, _nsc('eop:orbitNumber', nsmap))
    orbitNumber.text = str(meta.common.orbit_number_absolute)
    orbitDirection = etree.SubElement(acquisition, _nsc('eop:orbitDirection', nsmap))
    orbitDirection.text = meta.common.orbit_direction.upper()
    wrsLongitudeGrid = etree.SubElement(acquisition, _nsc('eop:wrsLongitudeGrid', nsmap),
                                        attrib={'codeSpace': 'urn:esa:eop:Sentinel1:relativeOrbits'})
    wrsLongitudeGrid.text = str(meta.common.wrs_longitude_grid)
    if source is not None:
        value = source.orbit.ascending_node_date
        ascendingNodeDate = etree.SubElement(acquisition, _nsc('eop:ascendingNodeDate', nsmap))
        if value is not None:
            ascendingNodeDate.text = value.isoformat()
        startTimeFromAscendingNode = etree.SubElement(acquisition, _nsc('eop:startTimeFromAscendingNode', nsmap),
                                                      attrib={'uom': 'ms'})
        startTimeFromAscendingNode.text = (
            None if source.orbit.start_time_from_ascending_node is None
            else str(source.orbit.start_time_from_ascending_node)
        )
        completionTimeFromAscendingNode = etree.SubElement(acquisition,
                                                           _nsc('eop:completionTimeFromAscendingNode', nsmap),
                                                           attrib={'uom': 'ms'})
        completionTimeFromAscendingNode.text = (
            None if source.orbit.completion_time_from_ascending_node is None
            else str(source.orbit.completion_time_from_ascending_node)
        )
        instrumentAzimuthAngle = etree.SubElement(acquisition, _nsc('eop:instrumentAzimuthAngle', nsmap),
                                                  attrib={'uom': 'deg'})
        instrumentAzimuthAngle.text = str(source.instrument_azimuth_angle)
    polarisationMode = etree.SubElement(acquisition, _nsc('sar:polarisationMode', nsmap))
    polarisationMode.text = meta.common.polarization_mode
    polarisationChannels = etree.SubElement(acquisition, _nsc('sar:polarisationChannels', nsmap))
    polarisationChannels.text = ', '.join(meta.common.polarizations)
    if prod:
        numberOfAcquisitions = etree.SubElement(acquisition, _nsc('_:numberOfAcquisitions', nsmap, ard_ns=ard_ns))
        numberOfAcquisitions.text = str(meta.product.number_of_acquisitions)
    else:
        antennaLookDirection = etree.SubElement(acquisition, _nsc('sar:antennaLookDirection', nsmap))
        antennaLookDirection.text = meta.common.antenna_look_direction
        minimumIncidenceAngle = etree.SubElement(acquisition, _nsc('sar:minimumIncidenceAngle', nsmap),
                                                 attrib={'uom': 'deg'})
        minimumIncidenceAngle.text = str(source.incidence_angle.minimum)
        maximumIncidenceAngle = etree.SubElement(acquisition, _nsc('sar:maximumIncidenceAngle', nsmap),
                                                 attrib={'uom': 'deg'})
        maximumIncidenceAngle.text = str(source.incidence_angle.maximum)
        orbitMeanAltitude = etree.SubElement(acquisition, _nsc('_:orbitMeanAltitude', nsmap, ard_ns=ard_ns),
                                             attrib={'uom': 'm'})
        orbitMeanAltitude.text = '{:.2e}'.format(meta.common.orbit_mean_altitude)
        if source.orbit.datatake_id is not None:
            dataTakeID = etree.SubElement(acquisition, _nsc('_:dataTakeID', nsmap, ard_ns=ard_ns))
            dataTakeID.text = str(source.orbit.datatake_id)
        majorCycleID = etree.SubElement(acquisition, _nsc('_:majorCycleID', nsmap, ard_ns=ard_ns))
        majorCycleID.text = str(source.orbit.major_cycle_id)


def _om_feature_of_interest(
        root: etree.Element,
        nsmap: dict[str, str],
        scene_id: str,
        extent: list[str],
        center: str
):
    """
    Creates the `om:featureOfInterest` XML elements.

    Parameters
    ----------
    root:
        Root XML element.
    nsmap:
        Dictionary listing abbreviation (key) and URI (value) of all necessary XML namespaces.
    scene_id:
        Scene basename.
    extent:
        A list containing one whitespace-separated ``latitude longitude``
        coordinate sequence for each polygon exterior ring.
    center:
        Center coordinate as a whitespace-separated ``latitude longitude``
        string.
    """
    featureOfInterest = etree.SubElement(root, _nsc('om:featureOfInterest', nsmap))
    footprint = etree.SubElement(
        featureOfInterest,
        _nsc('eop:Footprint', nsmap),
        attrib={_nsc('gml:id', nsmap): scene_id + '_5'}
    )
    
    multiExtentOf = etree.SubElement(
        footprint,
        _nsc('eop:multiExtentOf', nsmap)
    )
    
    multiSurface = etree.SubElement(
        multiExtentOf,
        _nsc('gml:MultiSurface', nsmap),
        attrib={_nsc('gml:id', nsmap): scene_id + '_6'}
    )
    
    for index, envelope in enumerate(extent, start=1):
        surfaceMember = etree.SubElement(
            multiSurface,
            _nsc('gml:surfaceMember', nsmap),
        )
        polygon = etree.SubElement(
            surfaceMember,
            _nsc('gml:Polygon', nsmap),
            attrib={_nsc('gml:id', nsmap): f'{scene_id}_7_{index}'},
        )
        exterior = etree.SubElement(
            polygon,
            _nsc('gml:exterior', nsmap),
        )
        linearRing = etree.SubElement(
            exterior,
            _nsc('gml:LinearRing', nsmap),
        )
        posList = etree.SubElement(
            linearRing,
            _nsc('gml:posList', nsmap),
        )
        posList.text = envelope
    
    centerOf = etree.SubElement(
        footprint,
        _nsc('eop:centerOf', nsmap)
    )
    point = etree.SubElement(
        centerOf,
        _nsc('gml:Point', nsmap),
        attrib={_nsc('gml:id', nsmap): scene_id + '_8'}
    )
    pos = etree.SubElement(point, _nsc('gml:pos', nsmap))
    pos.text = center
