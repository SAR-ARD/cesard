import os
import sys
from osgeo import gdal
import spatialist
import pyroSAR
import cesard
from datetime import datetime
from dateutil.parser import parse as dateparse

from typing import Any, Literal


def gdal_conf(
        config: dict[str, Any]
) -> dict[Literal['threads', 'threads_before', 'multithread'], Any]:
    """
    Stores GDAL configuration options for the current process.

    Parameters
    ----------
    config
        Dictionary of the parsed config parameters for the current process.

    Returns
    -------
        Dictionary containing GDAL configuration options for the current process.
    """
    threads = config['processing']['gdal_threads']
    threads_before = gdal.GetConfigOption('GDAL_NUM_THREADS')
    if not isinstance(threads, int):
        raise TypeError("'threads' must be of type int")
    if threads == 1:
        multithread = False
    elif threads > 1:
        multithread = True
        gdal.SetConfigOption('GDAL_NUM_THREADS', str(threads))
    else:
        raise ValueError("'threads' must be >= 1")
    
    return {'threads': threads, 'threads_before': threads_before,
            'multithread': multithread}


def keyval_check(
        key: str,
        val: str,
        allowed_keys: list[str]
) -> str | None:
    """
    Check and clean up key,value pairs while parsing a config file.
    
    Parameters
    ----------
    key:
        the parameter key
    val:
        the parameter value
    allowed_keys:
        a list of allowed keys
    """
    if key not in allowed_keys:
        msg = f"Parameter '{key}' is not allowed; should be one of {allowed_keys}"
        raise ValueError(msg)
    
    val = val.replace('"', '').replace("'", "")
    if val in ['None', 'none', '']:
        val = None
    return val


def parse_datetime(
        s: str
) -> datetime:
    """Custom converter for configparser:
    https://docs.python.org/3/library/configparser.html#customizing-parser-behaviour"""
    return dateparse(s)


def parse_list(
        s: str
) -> list[str] | None:
    """Custom converter for configparser:
    https://docs.python.org/3/library/configparser.html#customizing-parser-behaviour"""
    if s in ['', 'None']:
        return None
    else:
        return [x.strip() for x in s.split(',')]


def validate_options(
        k: str,
        v: Any,
        options: dict[str, list[str]]
) -> None:
    """
    Validate a configuration option against a set of allowed options.
    
    Parameters
    ----------
    k
        the configuration key
    v
        the configuration value(s)
    options:
        the configuration options
    """
    if k not in options:
        return
    if isinstance(v, list):
        for item in v:
            validate_options(k, item, options)
    else:
        msg = "Parameter '{}': expected value(s) to be one of {}; got '{}' instead"
        assert v in options[k], msg.format(k, options[k], v)


def validate_value(
        k: str,
        v: str | None | list[str]
) -> None:
    """
    Validate the value of a configuration option.
    
    Parameters
    ----------
    k:
        the configuration key
    v:
        the configuration value

    Returns
    -------

    """
    
    def val_aoi_geometry(x):
        return x is None or os.path.isfile(x)
    
    def val_aoi_tiles(x):
        return x is None or (isinstance(x, str) and len(x) == 5)
    
    def val_work_dir(x):
        return x is not None and os.path.isdir(v) and os.access(v, os.W_OK)
    
    validators = {'aoi_geometry': (val_aoi_geometry,
                                   'must be None or an existing file'),
                  'aoi_tiles': (val_aoi_tiles,
                                'must be None or a string of length 5'),
                  'work_dir': (val_work_dir,
                               'must be an existing, writable directory')}
    if k not in validators.keys():
        return
    if isinstance(v, list):
        for item in v:
            validate_value(k, item)
    else:
        validator, condition = validators[k]
        if not validator(v):
            msg = "Parameter '{}': value '{}' did not pass validation ({})."
            raise ValueError(msg.format(k, v, condition))


def version_dict() -> dict[str, str]:
    """
    Get the versions of used packages

    Returns
    -------
        a dictionary containing the versions of relevant python packages. Keys:
        
        - python
        - gdal
        - spatialist
        - pyrosar
        - cesard
    """
    out = {
        'python': sys.version,
        'gdal': gdal.__version__,
        'spatialist': spatialist.__version__,
        'pyrosar': pyroSAR.__version__,
        'cesard': cesard.__version__
    }
    return out
