import os


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


def validate_options(
        k: str,
        v: str,
        options: dict[str, list[str]]
) -> None:
    """
    Validate a configuration option against a set of allowed options.
    
    Parameters
    ----------
    k:
        the configuration key
    v:
        the configuration value
    options:
        the configuration options

    Returns
    -------

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
