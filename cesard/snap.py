import os
import re
from spatialist import Raster
from spatialist.envi import HDRobject
from spatialist.ancillary import finder
from pyroSAR import identify, identify_many
from pyroSAR.examine import ExamineSnap
from pyroSAR.snap.auxil import gpt, parse_recipe, parse_node, \
    mli_parametrize, geo_parametrize, \
    sub_parametrize, erode_edges
import cesard
import logging

log = logging.getLogger('cesard')


def find_datasets(scene, outdir, epsg):
    """
    Find processed datasets for a SAR scene.

    Parameters
    ----------
    scene: str
        the file name of the SAR scene
    outdir: str
        the output directory in which to search for results
    epsg: int
        the EPSG code defining the output projection of the processed products.

    Returns
    -------
    dict or None
        Either None if no datasets were found or a dictionary with the
        following keys and values pointing to the file names
        (polarization-specific keys depending on product availability):

         - hh-g-lin: gamma nought RTC backscatter HH polarization
         - hv-g-lin: gamma nought RTC backscatter HV polarization
         - vh-g-lin: gamma nought RTC backscatter VH polarization
         - vv-g-lin: gamma nought RTC backscatter VV polarization
         - hh-s-lin: sigma nought ellipsoidal backscatter HH polarization
         - hv-s-lin: sigma nought ellipsoidal backscatter HV polarization
         - vh-s-lin: sigma nought ellipsoidal backscatter VH polarization
         - vv-s-lin: sigma nought ellipsoidal backscatter VV polarization
         - dm: layover-shadow data mask
         - ei: ellipsoidal incident angle
         - gs: gamma-sigma ratio
         - lc: local contributing area (aka scattering area)
         - ld: range look direction angle
         - li: local incident angle
         - sg: sigma-gamma ratio
         - np-hh: NESZ HH polarization
         - np-hv: NESZ HV polarization
         - np-vh: NESZ VH polarization
         - np-vv: NESZ VV polarization
    """
    basename = os.path.splitext(os.path.basename(scene))[0]
    scenedir = os.path.join(outdir, basename)
    subdir = os.path.join(scenedir, basename + f'_geo_{epsg}.data')
    if not os.path.isdir(subdir):
        return
    lookup = {'dm': r'layoverShadowMask\.img$',
              'ei': r'incidenceAngleFromEllipsoid\.img$',
              'gs': r'gammaSigmaRatio_[VH]{2}\.img$',
              'lc': r'simulatedImage_[VH]{2}\.img$',
              'ld': r'lookDirection_[VH]{2}\.img$',
              'li': r'localIncidenceAngle\.img$',
              'sg': r'sigmaGammaRatio_[VH]{2}\.img$'}
    out = {}
    for key, pattern in lookup.items():
        match = finder(target=subdir, matchlist=[pattern], regex=True)
        if len(match) > 0:
            out[key] = match[0]
    pattern = r'(?P<bsc>Gamma0|Sigma0)_(?P<pol>[VH]{2})\.img$'
    backscatter = finder(target=subdir, matchlist=[pattern], regex=True)
    for item in backscatter:
        pol = re.search(pattern, item).group('pol').lower()
        bsc = re.search(pattern, item).group('bsc')[0].lower()
        out[f'{pol}-{bsc}-lin'] = item
    pattern = r'NESZ_(?P<pol>[VH]{2})\.img$'
    nesz = finder(target=subdir, matchlist=[pattern], regex=True)
    for item in nesz:
        pol = re.search(pattern, item).group('pol')
        out[f'np-{pol.lower()}'] = item
    if len(out) > 0:
        return out


def lsm_encoding() -> dict[str, int]:
    """
    Get the encoding of the layover shadow mask.
    """
    return {
        'not layover, not shadow': 0,
        'layover': 1,
        'shadow': 2,
        'layover in shadow': 3,
        'nodata': 255  # dummy value
    }


def version_dict() -> dict[str, str]:
    """
    Get processor software version information.

    Returns
    -------
        a dictionary with software components as keys and their versions as values.
    """
    try:
        snap_config = ExamineSnap()
        core = snap_config.get_version('core')
        microwavetbx = snap_config.get_version('microwavetbx')
        snap_core = f"{core['version']} | {core['date']}"
        snap_microwavetbx = f"{microwavetbx['version']} | {microwavetbx['date']}"
    except RuntimeError:
        snap_core = 'unknown'
        snap_microwavetbx = 'unknown'
    return ({'cesard': cesard.__version__,
             'snap_core': snap_core,
             'snap_microwavetbx': snap_microwavetbx})


###############################################################################

def geo(*src, dst, workflow, spacing, crs, geometry=None, buffer=0.01,
        export_extra=None, standard_grid_origin_x=0, standard_grid_origin_y=0,
        dem, dem_resampling_method='BILINEAR_INTERPOLATION',
        img_resampling_method='BILINEAR_INTERPOLATION', gpt_args=None, **bands):
    """
    Range-Doppler geocoding.

    Parameters
    ----------
    src: str or None
        variable number of input scene file names
    dst: str
        the file name of the target scene. Format is BEAM-DIMAP.
    workflow: str
        the target XML workflow file name
    spacing: int or float
        the target pixel spacing in meters
    crs: int or str
        the target coordinate reference system
    geometry: dict or spatialist.vector.Vector or str or None
        a vector geometry to limit the target product's extent
    buffer: int or float
        an additional buffer in degrees to add around `geometry`
    export_extra: list[str] or None
        a list of ancillary layers to write. Supported options:

         - DEM
         - incidenceAngleFromEllipsoid
         - layoverShadowMask
         - localIncidenceAngle
         - projectedLocalIncidenceAngle
    standard_grid_origin_x: int or float
        the X coordinate for pixel alignment
    standard_grid_origin_y: int or float
        the Y coordinate for pixel alignment
    dem: str
        the DEM file
    dem_resampling_method: str
        the DEM resampling method
    img_resampling_method: str
        the SAR image resampling method
    gpt_args: list[str] or None
        a list of additional arguments to be passed to the gpt call

        - e.g. ``['-x', '-c', '2048M']`` for increased tile cache size and intermediate clearing
    bands
        band ids for the input scenes in `src` as lists with keys bands<index>,
        e.g., ``bands1=['NESZ_VV'], bands2=['Gamma0_VV'], ...``

    Returns
    -------

    See Also
    --------
    pyroSAR.snap.auxil.sub_parametrize
    pyroSAR.snap.auxil.geo_parametrize
    """
    wf = parse_recipe('blank')
    ############################################
    scenes = identify_many(list(filter(None, src)))
    read_ids = []
    for i, scene in enumerate(scenes):
        read = parse_node('Read')
        read.parameters['file'] = scene.scene
        if f'bands{i}' in bands.keys():
            read.parameters['useAdvancedOptions'] = True
            read.parameters['sourceBands'] = bands[f'bands{i}']
        wf.insert_node(read)
        read_ids.append(read.id)
    ############################################
    if len(scenes) > 1:
        merge = parse_node('BandMerge')
        wf.insert_node(merge, before=read_ids)
        last = merge
    else:
        last = wf['Read']
    ############################################
    if geometry is not None:
        sub = sub_parametrize(scene=scenes[0], geometry=geometry, buffer=buffer)
        wf.insert_node(sub, before=last.id)
        last = sub
    ############################################
    tc = geo_parametrize(spacing=spacing, t_srs=crs,
                         export_extra=export_extra,
                         alignToStandardGrid=True,
                         externalDEMFile=dem,
                         externalDEMApplyEGM=False,
                         standardGridOriginX=standard_grid_origin_x,
                         standardGridOriginY=standard_grid_origin_y,
                         standardGridAreaOrPoint='area',
                         demResamplingMethod=dem_resampling_method,
                         imgResamplingMethod=img_resampling_method)
    wf.insert_node(tc, before=last.id)
    ############################################
    write = parse_node('Write')
    wf.insert_node(write, before=tc.id)
    write.parameters['file'] = dst
    write.parameters['formatName'] = 'BEAM-DIMAP'
    ############################################
    wf.write(workflow)
    gpt(xmlfile=workflow, tmpdir=os.path.dirname(dst),
        gpt_args=gpt_args)


def gsr(src, dst, workflow, src_sigma=None, gpt_args=None):
    """
    Gamma-sigma ratio computation for either ellipsoidal or RTC sigma nought.

    Parameters
    ----------
    src: str
        the file name of the source scene. Both gamma and sigma bands are expected unless `src_sigma` is defined.
    dst: str
        the file name of the target scene. Format is BEAM-DIMAP.
    workflow: str
        the output SNAP XML workflow filename.
    src_sigma: str or None
        the optional file name of a second source product from which to read the sigma band.
    gpt_args: list[str] or None
        a list of additional arguments to be passed to the gpt call

        - e.g. ``['-x', '-c', '2048M']`` for increased tile cache size and intermediate clearing

    Returns
    -------

    """
    scene = identify(src)
    pol = scene.polarizations[0]
    wf = parse_recipe('blank')
    ############################################
    read = parse_node('Read')
    read.parameters['file'] = scene.scene
    wf.insert_node(read)
    last = read
    ############################################
    if src_sigma is not None:
        read.parameters['sourceBands'] = f'Gamma_{pol}'
        read2 = parse_node('Read')
        read2.parameters['file'] = src_sigma
        read2.parameters['sourceBands'] = f'Sigma_{pol}'
        wf.insert_node(read2)
        ########################################
        merge = parse_node('BandMerge')
        wf.insert_node(merge, before=[read.id, read2.id])
        last = merge
    ############################################
    math = parse_node('BandMaths')
    wf.insert_node(math, before=last.id)
    ratio = 'gammaSigmaRatio'
    expression = f'Sigma0_{pol} / Gamma0_{pol}'
    
    math.parameters.clear_variables()
    exp = math.parameters['targetBands'][0]
    exp['name'] = ratio
    exp['type'] = 'float32'
    exp['expression'] = expression
    exp['noDataValue'] = 0.0
    ############################################
    write = parse_node('Write')
    wf.insert_node(write, before=math.id)
    write.parameters['file'] = dst
    write.parameters['formatName'] = 'BEAM-DIMAP'
    ############################################
    wf.write(workflow)
    gpt(xmlfile=workflow, tmpdir=os.path.dirname(dst),
        gpt_args=gpt_args)


def mli(src, dst, workflow, spacing=None, rlks=None, azlks=None, gpt_args=None):
    """
    Multi-looking.

    Parameters
    ----------
    src: str
        the file name of the source scene
    dst: str
        the file name of the target scene. Format is BEAM-DIMAP.
    workflow: str
        the output SNAP XML workflow filename.
    spacing: int or float
        the target pixel spacing for automatic determination of looks
        using function :func:`pyroSAR.ancillary.multilook_factors`.
        Overridden by arguments `rlks` and `azlks` if they are not None.
    rlks: int or None
        the number of range looks.
    azlks: int or None
        the number of azimuth looks.
    gpt_args: list[str] or None
        a list of additional arguments to be passed to the gpt call

        - e.g. ``['-x', '-c', '2048M']`` for increased tile cache size and intermediate clearing

    Returns
    -------

    See Also
    --------
    pyroSAR.snap.auxil.mli_parametrize
    pyroSAR.ancillary.multilook_factors
    """
    scene = identify(src)
    wf = parse_recipe('blank')
    ############################################
    read = parse_node('Read')
    read.parameters['file'] = scene.scene
    wf.insert_node(read)
    ############################################
    ml = mli_parametrize(scene=scene, spacing=spacing, rlks=rlks, azlks=azlks)
    if ml is not None:
        log.info('multi-looking')
        wf.insert_node(ml, before=read.id)
        ############################################
        write = parse_node('Write')
        wf.insert_node(write, before=ml.id)
        write.parameters['file'] = dst
        write.parameters['formatName'] = 'BEAM-DIMAP'
        ############################################
        wf.write(workflow)
        gpt(xmlfile=workflow, tmpdir=os.path.dirname(dst),
            gpt_args=gpt_args)


def postprocess(src, clean_edges=True, clean_edges_pixels=4):
    """
    Performs edge cleaning and sets the nodata value in the output ENVI HDR files.

    Parameters
    ----------
    src: str
        the file name of the source scene. Format is BEAM-DIMAP.
    clean_edges: bool
        perform edge cleaning?
    clean_edges_pixels: int
        the number of pixels to erode during edge cleaning.

    Returns
    -------

    """
    if clean_edges:
        erode_edges(src=src, only_boundary=True, pixels=clean_edges_pixels)
    datadir = src.replace('.dim', '.data')
    hdrfiles = finder(target=datadir, matchlist=['*.hdr'])
    for hdrfile in hdrfiles:
        if not 'layoverShadowMask' in hdrfile:
            with HDRobject(hdrfile) as hdr:
                hdr.data_ignore_value = 0
                hdr.write(hdrfile)


def rtc(src, dst, workflow, dem, dem_resampling_method='BILINEAR_INTERPOLATION',
        sigma0=True, scattering_area=True, dem_oversampling_multiple=2,
        gpt_args=None):
    """
    Radiometric Terrain Flattening.
    
    Parameters
    ----------
    src: str
        the file name of the source scene
    dst: str
        the file name of the target scene. Format is BEAM-DIMAP.
    workflow: str
        the output SNAP XML workflow filename.
    dem: str
        the input DEM file name.
    dem_resampling_method: str
        the DEM resampling method.
    sigma0: bool
        output sigma0 RTC backscatter?
    scattering_area: bool
        output scattering area image?
    dem_oversampling_multiple: int
        a factor to multiply the DEM oversampling factor computed by SNAP.
        The SNAP default of 1 has been found to be insufficient with stripe
        artifacts remaining in the image.
    gpt_args: list[str] or None
        a list of additional arguments to be passed to the gpt call
        
        - e.g. ``['-x', '-c', '2048M']`` for increased tile cache size and intermediate clearing
    
    Returns
    -------

    """
    scene = identify(src)
    wf = parse_recipe('blank')
    ############################################
    read = parse_node('Read')
    read.parameters['file'] = scene.scene
    wf.insert_node(read)
    ############################################
    tf = parse_node('Terrain-Flattening')
    polarizations = scene.polarizations
    bands = ['Beta0_{}'.format(pol) for pol in polarizations]
    wf.insert_node(tf, before=read.id)
    tf.parameters['sourceBands'] = bands
    if 'reGridMethod' in tf.parameters.keys():
        tf.parameters['reGridMethod'] = False
    tf.parameters['outputSigma0'] = sigma0
    tf.parameters['outputSimulatedImage'] = scattering_area
    tf.parameters['demName'] = 'External DEM'
    tf.parameters['externalDEMFile'] = dem
    tf.parameters['externalDEMApplyEGM'] = False
    with Raster(dem) as ras:
        tf.parameters['externalDEMNoDataValue'] = ras.nodata
    tf.parameters['demResamplingMethod'] = dem_resampling_method
    tf.parameters['oversamplingMultiple'] = dem_oversampling_multiple
    last = tf
    ############################################
    write = parse_node('Write')
    wf.insert_node(write, before=last.id)
    write.parameters['file'] = dst
    write.parameters['formatName'] = 'BEAM-DIMAP'
    ############################################
    wf.write(workflow)
    gpt(xmlfile=workflow, tmpdir=os.path.dirname(dst),
        gpt_args=gpt_args)


def sgr(src, dst, workflow, src_gamma=None, gpt_args=None):
    """
    Sigma-gamma ratio computation.

    Parameters
    ----------
    src: str
        the file name of the source scene. Both sigma and gamma bands are expected unless `src_gamma` is defined.
    dst: str
        the file name of the target scene. Format is BEAM-DIMAP.
    workflow: str
        the output SNAP XML workflow filename.
    src_gamma: str or None
        the optional file name of a second source product from which to read the gamma band.
    gpt_args: list[str] or None
        a list of additional arguments to be passed to the gpt call
        
        - e.g. ``['-x', '-c', '2048M']`` for increased tile cache size and intermediate clearing
    
    Returns
    -------

    """
    scene = identify(src)
    pol = scene.polarizations[0]
    wf = parse_recipe('blank')
    ############################################
    read = parse_node('Read')
    read.parameters['file'] = scene.scene
    wf.insert_node(read)
    last = read
    ############################################
    if src_gamma is not None:
        read.parameters['sourceBands'] = f'Sigma_{pol}'
        read2 = parse_node('Read')
        read2.parameters['file'] = src_gamma
        read2.parameters['sourceBands'] = f'Gamma_{pol}'
        wf.insert_node(read2)
        ########################################
        merge = parse_node('BandMerge')
        wf.insert_node(merge, before=[read.id, read2.id])
        last = merge
    ############################################
    math = parse_node('BandMaths')
    wf.insert_node(math, before=last.id)
    ratio = 'sigmaGammaRatio'
    expression = f'Gamma0_{pol} / Sigma0_{pol}'
    
    math.parameters.clear_variables()
    exp = math.parameters['targetBands'][0]
    exp['name'] = ratio
    exp['type'] = 'float32'
    exp['expression'] = expression
    exp['noDataValue'] = 0.0
    ############################################
    write = parse_node('Write')
    wf.insert_node(write, before=math.id)
    write.parameters['file'] = dst
    write.parameters['formatName'] = 'BEAM-DIMAP'
    ############################################
    wf.write(workflow)
    gpt(xmlfile=workflow, tmpdir=os.path.dirname(dst),
        gpt_args=gpt_args)
