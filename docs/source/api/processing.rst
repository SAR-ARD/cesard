Processing
----------

SAR
^^^
Implementation of full SAR processing is the responsibility of the satellite-specific processor packages like `s1ard`.
`cesard` however offers some general functions that these processors might make use of.
The :mod:`cesard.snap` module offers some functions to interface the SNAP processor with the respective ARD processor and some general processing functions.

SNAP
++++

.. automodule:: cesard.snap
    :members:
    :undoc-members:
    :show-inheritance:

    .. rubric:: main interface

    .. autosummary::
        :nosignatures:

        find_datasets
        lsm_encoding
        version_dict


    .. rubric:: processor functions

    .. autosummary::
        :nosignatures:

        geo
        gsr
        mli
        postprocess
        rtc
        sgr

ARD
^^^

.. automodule:: cesard.ard
    :members:
    :undoc-members:
    :show-inheritance:

    .. autosummary::
        :nosignatures:

        calc_product_start_stop
        create_acq_id_image
        create_data_mask
        create_rgb_vrt
        create_vrt

DEM
^^^

.. automodule:: cesard.dem
    :members:
    :undoc-members:
    :show-inheritance:

    .. autosummary::
        :nosignatures:

        authenticate
        mosaic
        prepare
        retile
        to_mgrs
