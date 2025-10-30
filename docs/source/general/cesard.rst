ARD Production
==============

The following sections give a brief overview of the major package components for creating an ARD product.

MGRS Gridding
-------------

The basis of the processing chain builds the Sentinel-2 Military Grid Reference System (MGRS) tiling system.
Hence, a reference file is needed containing the respective tile information for processing ARD products.
A KML file is available online that will be used in the following steps:

`S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml <https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip>`_

This file contains all relevant information about individual tiles, in particular the EPSG code of the respective UTM zone and the geometry of the tile in UTM coordinates.
This file is automatically downloaded to `~/.cesard` by the function :func:`cesard.ancillary.get_kml`.
The function :func:`cesard.tile_extraction.aoi_from_tile` can be used to extract one or multiple tiles as :class:`spatialist.vector.Vector` object.

Scene Management
----------------

The source products are managed in a local SQLite database to select scenes for processing (see pyroSAR's section on `Database Handling`_) or are directly queried from a STAC catalog (see :class:`s1ard.search.STACArchive`).

After loading an MGRS tile as an :class:`spatialist.vector.Vector` object and selecting all relevant overlapping scenes
from the database, processing can commence.

DEM Handling
------------

cesard offers a convenience function :func:`cesard.dem.mosaic` for creating scene-specific DEM files from various sources.
The function is based on :func:`pyroSAR.auxdata.dem_autoload` and :func:`pyroSAR.auxdata.dem_create` and will

- download all tiles of the selected source overlapping with a defined geometry
- create a GDAL VRT virtual mosaic from the tiles including gap filling over ocean areas
- create a new GeoTIFF from the VRT including geoid-ellipsoid height conversion if necessary
  (WGS84 heights are generally required for SAR processing but provided heights might be relative to a geoid like EGM2008).

ARD Formatting
--------------

During SAR processing, files covering a whole scene are created. In this last step, the scene-based structure is converted to the MGRS tile structure.
If one tile overlaps with multiple scenes, these scenes are first virtually mosaiced using VRT files.
The files are then subsetted to the actual tile extent, converted to Cloud Optimized GeoTIFFs (COG), and renamed to the ARD naming scheme.
All steps are performed by satellite-specific format functions, e.g. :func:`s1ard.ard.format`.

.. _Database Handling: https://pyrosar.readthedocs.io/en/latest/general/processing.html#database-handling
