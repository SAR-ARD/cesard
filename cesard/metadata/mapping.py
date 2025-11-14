# 'z_error': Maximum error threshold on values for LERC* compression.
# Will be ignored if a compression algorithm is used that isn't related to LERC.
LERC_ERR_THRES = {
    'vv-g-lin': 1e-4,
    'vh-g-lin': 1e-4,
    'hh-g-lin': 1e-4,
    'hv-g-lin': 1e-4,
    'vv-s-lin': 1e-4,
    'vh-s-lin': 1e-4,
    'hh-s-lin': 1e-4,
    'hv-s-lin': 1e-4,
    'ei': 1e-3,
    'em': 1e-3,
    'dm': 0.0,
    'li': 1e-2,
    'lc': 0.1,
    'ld': 1e-3,
    'gs': 1e-4,
    'id': 0.0,
    'np-vv': 2e-5,
    'np-vh': 2e-5,
    'np-hh': 2e-5,
    'np-hv': 2e-5,
    'sg': 1e-4,
    'wm': 2e-5
}

DEM_MAP = {
    'GETASSE30':
        {'access': 'https://step.esa.int/auxdata/dem/GETASSE30',
         'ref': 'https://seadas.gsfc.nasa.gov/help-8.1.0/desktop/GETASSE30ElevationModel.html',
         'type': 'elevation',
         'gsd': '30 arcsec',
         'egm': 'https://apps.dtic.mil/sti/citations/ADA166519'},
    'Copernicus 10m EEA DEM':
        {'access': 'ftps://cdsdata.copernicus.eu/DEM-datasets/COP-DEM_EEA-10-DGED/2021_1',
         'ref': 'https://spacedata.copernicus.eu/web/cscda/dataset-details?articleId=394198',
         'type': 'surface',
         'gsd': '10 m',
         'egm': 'https://bgi.obs-mip.fr/data-products/grids-and-models/egm2008-global-model/'},
    'Copernicus 30m Global DEM':
        {'access': 'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/',
         'ref': 'https://copernicus-dem-30m.s3.amazonaws.com/readme.html',
         'type': 'surface',
         'gsd': '30 m',
         'egm': 'https://bgi.obs-mip.fr/data-products/grids-and-models/egm2008-global-model/'},
    'Copernicus 30m Global DEM II':
        {'access': 'ftps://cdsdata.copernicus.eu/DEM-datasets/COP-DEM_GLO-30-DGED/2021_1',
         'ref': 'https://spacedata.copernicus.eu/web/cscda/dataset-details?articleId=394198',
         'type': 'surface',
         'gsd': '30 m',
         'egm': 'https://bgi.obs-mip.fr/data-products/grids-and-models/egm2008-global-model/'},
    'Copernicus 90m Global DEM II':
        {'access': 'ftps://cdsdata.copernicus.eu/DEM-datasets/COP-DEM_GLO-90-DGED/2021_1',
         'ref': 'https://spacedata.copernicus.eu/web/cscda/dataset-details?articleId=394198',
         'type': 'surface',
         'gsd': '90 m',
         'egm': 'https://bgi.obs-mip.fr/data-products/grids-and-models/egm2008-global-model/'}
}

# XML namespaces are identifiers, and it is not their goal to be directly usable for schema retrieval:
# https://stackoverflow.com/a/30761004
NS_MAP = {
    'placeholder': 'http://earth.esa.int/{0}/spec/role/1.0',
    'sar': 'http://www.opengis.net/sar/2.1',
    'eop': 'http://www.opengis.net/eop/2.1',
    'om': 'http://www.opengis.net/om/2.0',
    'gml': 'http://www.opengis.net/gml/3.2',
    'ows': 'http://www.opengis.net/ows/2.0',
    'xlink': 'http://www.w3.org/1999/xlink'
}

ASSET_MAP = {
    'dm': {'type': 'Mask',
           'unit': None,
           'role': 'data-mask',
           'title': 'Data Mask Image',
           'allowed': ['not layover, nor shadow',
                       'layover',
                       'shadow',
                       'layover and shadow',
                       'ocean',
                       'lakes',
                       'rivers']},
    'ei': {'type': 'Angle',
           'unit': 'deg',
           'role': 'ellipsoid-incidence-angle',
           'title': 'Ellipsoid Incidence Angle'},
    'em': {'type': 'Elevation',
           'unit': 'meters',
           'role': 'digital-elevation-model',
           'title': 'Digital Elevation Model'},
    'lc': {'type': 'Scattering Area',
           'unit': 'square_meters',
           'role': 'contributing-area',
           'title': 'Local Contributing Area'},
    'ld': {'type': 'Angle',
           'unit': 'deg',
           'role': 'range-look-direction-angle',
           'title': 'Range Look Direction Angle'},
    'li': {'type': 'Angle',
           'unit': 'deg',
           'role': 'local-incidence-angle',
           'title': 'Local Incidence Angle'},
    'gs': {'type': 'Ratio',
           'unit': None,
           'role': 'gamma-sigma-ratio',
           'title': 'Gamma0 RTC to sigma0 RTC ratio'},
    'id': {'type': 'AcqID',
           'unit': None,
           'role': 'acquisition-id',
           'title': 'Acquisition ID Image'},
    'np': {'type': 'Sigma-0',
           'unit': 'dB',
           'role': 'noise-power',
           'title': 'Noise Power'},
    'sg': {'type': 'Ratio',
           'unit': None,
           'role': 'sigma-gamma-ratio',
           'title': 'Sigma0 RTC to gamma0 RTC ratio'},
    'wm': {'type': 'Sigma-0',
           'unit': None,
           'role': 'wind-modelled-backscatter',
           'title': 'wind-modelled backscatter (OCN CMOD NRCS)'}
}

URL = {
    'ancillaryData_KML': 'https://sentiwiki.copernicus.eu/__attachments/1692737/'
                         'S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip',
    'card4l_nrb': 'https://ceos.org/ard/files/PFS/NRB/v5.5/CARD4L-PFS_NRB_v5.5.pdf',
    'card4l_orb': 'https://ceos.org/ard/files/PFS/ORB/v1.0/'
                  'CARD4L_Product_Family_Specification_Ocean_Radar_Backscatter-v1.0.pdf',
    'griddingConventionURL': 'https://www.mgrs-data.org/data/documents/nga_mgrs_doc.pdf',
    'platformReference': {
        'envisat-1': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=2',
        'ers-1': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=220',
        'ers-2': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=221',
        'sentinel-1a': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=575',
        'sentinel-1b': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=576',
        'sentinel-1c': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=577',
        'sentinel-1d': 'https://database.eohandbook.com/database/missionsummary.aspx?missionID=814'
    }
}
