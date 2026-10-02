"""Canonical metadata interface between SAR-ARD producers and CESARD writers.

The models in this module define the metadata contract consumed by the generic
CESARD STAC and XML writers. Satellite-specific packages such as s1ard and
asard should return :class:`ARDMetadata` rather than a free-form dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self, get_args, get_origin

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

BBox = tuple[float, float, float, float]
AffineTransform = tuple[float, float, float, float, float, float]
Polarization = Literal["HH", "HV", "VH", "VV"]

# Reserved values for mandatory metadata whose derivation is not yet
# implemented. These are valid interface values and are intentionally kept
# distinct from ``None``, which denotes genuinely optional/not-applicable
# metadata.
# Mind however that this is not rigorous. ``str | NotImplementedText``
# effectively allows every string, also empty ones.
NOT_IMPLEMENTED_NUMBER = -99999
NOT_IMPLEMENTED_TEXT = "TBD"

type NotImplementedNumber = Literal[-99999]
type NotImplementedText = Literal["TBD"]

type ARDNumber = float | NotImplementedNumber
type ARDPositiveNumber = PositiveFloat | NotImplementedNumber
type ARDNonNegativeNumber = NonNegativeFloat | NotImplementedNumber

type ARDInteger = int | NotImplementedNumber
type ARDPositiveInteger = PositiveInt | NotImplementedNumber
type ARDNonNegativeInteger = NonNegativeInt | NotImplementedNumber

type ARDText = str | NotImplementedText


class MetadataModel(BaseModel):
    """Base class for all metadata models."""
    
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        validate_default=True,
    )


class GeometryMetadata(MetadataModel):
    """Spatial footprint metadata in EPSG:4326 and, optionally, native CRS."""
    
    bbox: BBox = Field(
        description=(
            "WGS 84 bounding box of the footprint as (west, south, east, north), in "
            "longitude and latitude degrees. For footprints crossing the antimeridian, "
            "west may be greater than east."
        )
    )
    bbox_native: BBox | None = Field(
        default=None,
        description="Bounding box of the product footprint in the product's native coordinate reference system, ordered as (xmin, ymin, xmax, ymax).",
    )
    geometry: dict[str, Any] = Field(
        description=(
            "Acquisition footprint as a GeoJSON Polygon or MultiPolygon in WGS 84 "
            "(EPSG:4326). GeoJSON coordinates are ordered as longitude, latitude pairs; "
            "polygon rings are closed."
        )
    )
    
    @field_validator("geometry")
    @classmethod
    def _validate_geometry(cls, value: dict[str, Any]) -> dict[str, Any]:
        geom_type = value.get("type")
        if geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                "footprint geometry must be a GeoJSON Polygon or MultiPolygon"
            )
        if "coordinates" not in value:
            raise ValueError("footprint geometry is missing 'coordinates'")
        return value
    
    @property
    def center(self) -> tuple[float, float]:
        """Footprint center as ``(longitude, latitude)`` in EPSG:4326."""
        xmin, ymin, xmax, ymax = self.bbox
        if xmax < xmin:
            longitude = xmin + ((xmax + 360.0) - xmin) / 2.0
            if longitude > 180.0:
                longitude -= 360.0
        else:
            longitude = (xmin + xmax) / 2.0
        latitude = (ymin + ymax) / 2.0
        return longitude, latitude
    
    @property
    def exterior_rings(self) -> list[list[list[float]]]:
        """Exterior rings of the GeoJSON Polygon or MultiPolygon footprint."""
        coordinates = self.geometry["coordinates"]
        if self.geometry["type"] == "Polygon":
            return [coordinates[0]]
        return [polygon[0] for polygon in coordinates]
    
    @property
    def xml_center(self) -> str:
        """Center formatted as the ``latitude longitude`` pair used by XML."""
        longitude, latitude = self.center
        return f"{latitude} {longitude}"
    
    @property
    def xml_envelopes(self) -> list[str]:
        """Exterior rings formatted as XML ``gml:posList`` strings."""
        return [
            " ".join(f"{latitude} {longitude}" for longitude, latitude, *_ in ring)
            for ring in self.exterior_rings
        ]


class ProcessingMetadata(MetadataModel):
    """Information about the processor that created a product or source scene."""
    
    facility: ARDText | None = Field(
        default=None,
        description="Name or code of the facility that processed the source data or generated the ARD product.",
    )
    date: AwareDatetime | None = Field(
        default=None,
        description=(
            "UTC date and time at which the source data or ARD product was processed."
        ),
    )
    mode: Literal["PROTOTYPE", "NOMINAL"] = Field(
        description="Processing mode describing whether the processing configuration is nominal operational processing or a prototype."
    )
    processor: ARDText | None = Field(
        default=None,
        description="Name of the primary processor software used to generate the data.",
    )
    software: dict[str, str] = Field(
        default_factory=dict,
        description="Software used during processing, represented as a mapping from software name to version.",
    )
    
    @model_validator(mode="after")
    def _processor_has_version(self) -> ProcessingMetadata:
        if self.processor is not None and self.software and self.processor not in self.software:
            raise ValueError(
                f"processor {self.processor!r} is not present in software versions"
            )
        return self


class CommonMetadata(MetadataModel):
    """Metadata shared by the ARD product and all contributing source scenes."""
    
    antenna_look_direction: Literal["LEFT", "RIGHT"] = Field(
        description=(
            "Side of the satellite ground track illuminated by the SAR antenna, "
            "expressed as LEFT or RIGHT relative to flight direction."
        )
    )
    constellation: ARDText = Field(
        description="Name of the satellite constellation to which the platform belongs, where applicable.")
    instrument_short_name: ARDText = Field(
        description="Short name of the SAR instrument that acquired the source data."
    )
    operational_mode: ARDText = Field(
        description="Instrument acquisition mode used to collect the source data."
    )
    orbit_direction: Literal["ascending", "descending"] = Field(
        description="Direction of the platform ground track during acquisition: ascending or descending."
    )
    orbit_mean_altitude: float = Field(
        gt=0,
        description="Mean platform altitude during the acquisition, in metres.",
    )
    orbit_number_absolute: int = Field(
        ge=0,
        description="Absolute orbit number of the acquisition.",
    )
    orbit_number_relative: int = Field(
        ge=0,
        description="Relative orbit or track number of the acquisition.",
    )
    platform_full_name: ARDText = Field(
        description="Full name of the satellite platform."
    )
    platform_identifier: ARDText = Field(
        description="Platform serial identifier distinguishing a specific satellite within a mission or constellation."
    )
    platform_reference: ARDText = Field(
        description=(
            "Reference to the CEOS Missions, Instruments and Measurements database "
            "record describing the satellite mission."
        )
    )
    platform_short_name: ARDText = Field(
        description="Short mission or platform name used by the source metadata convention.")
    polarizations: tuple[Polarization, ...] = Field(
        min_length=1,
        description="Transmit/receive polarization channels contained in the acquisition, such as HH, HV, VH or VV.",
    )
    polarization_mode: ARDText = Field(
        description="Polarization acquisition mode describing the channel configuration of the SAR acquisition.")
    processing_level: Literal["L1C"] = Field(
        description="Processing level assigned to the source data used to generate the ARD product."
    )
    radar_band: Literal["X", "C", "L"] = Field(
        description="Common name of the radar frequency band used by the instrument, such as X, C or L band."
    )
    radar_center_frequency: float = Field(
        gt=0,
        description="Radar centre frequency of the instrument, in hertz.",
    )
    sensor_type: Literal["RADAR"] = Field(
        description="Type of remote-sensing instrument; RADAR for the SAR products represented by this model.")
    swath_identifier: ARDText = Field(description="Identifier of the acquisition swath or beam.")
    wrs_longitude_grid: int = Field(
        ge=0,
        description="Longitude-grid identifier of the reference system used to locate the acquisition, equivalent to a track-like identifier.",
    )


class Card4lMetadata(MetadataModel):
    """Reference to the applicable CARD4L product-family specification."""
    
    specification: Literal["NRB", "ORB"] = Field(
        description="Short identifier of the CEOS Analysis Ready Data product family specification implemented by the product."
    )
    version: ARDText = Field(description="Version of the implemented CEOS CARD4L product family specification.")
    document: ARDText = Field(
        description=(
            "URL or DOI referencing the implemented CEOS CARD4L Radar Backscatter "
            "product family specification."
        )
    )


class CompressionMetadata(MetadataModel):
    """Compression configuration used for product raster assets."""
    
    type: ARDText = Field(description="Compression algorithm or codec applied to product raster assets.")
    z_errors: dict[str, float] = Field(
        description=(
            "Maximum permitted compression error for applicable assets, represented as "
            "a mapping from asset identifier to MAX_Z_ERROR. A value of 0 denotes "
            "lossless LERC compression."
        )
    )


class GroundSamplingDistance(MetadataModel):
    """Ground sampling distance with an explicit unit."""
    
    value: float = Field(
        gt=0,
        description="Ground sampling distance of the elevation model in the unit given by 'unit'.",
    )
    unit: Literal["m", "arcsec"] = Field(
        description="Unit in which the elevation-model ground sampling distance is expressed: metres or arc-seconds."
    )


class DEMMetadata(MetadataModel):
    """Digital elevation model and Earth gravitational model metadata."""
    
    name: ARDText = Field(
        description="Human-readable name of the digital elevation or surface model used for geometric and radiometric terrain processing.")
    type: Literal["surface", "elevation"] = Field(
        description="Type of terrain-height model used during processing: an elevation model representing terrain height or a surface model including surface objects."
    )
    reference: ARDText = Field(
        description="URL, DOI or other persistent reference identifying the digital elevation or surface model used for terrain correction.")
    access: ARDText = Field(
        description="Location from which the digital elevation or surface model used for processing can be accessed.")
    gsd: GroundSamplingDistance = Field(
        description="Native ground sampling distance of the digital elevation or surface model used during processing."
    )
    resampling_method: ARDText = Field(
        description="Resampling method used to prepare the digital elevation or surface model for geometric and radiometric processing."
    )
    egm_reference: ARDText | None = Field(
        description="URL, DOI or other persistent reference identifying the Earth Gravitational Model used to relate elevation-model heights to the required vertical reference."
    )
    egm_resampling_method: ARDText | None = Field(
        description="Resampling method used when preparing the Earth Gravitational Model for processing."
    )
    
    @model_validator(mode='after')
    def validate_egm(self) -> Self:
        if (self.egm_reference is None) != (self.egm_resampling_method is None):
            raise ValueError(
                'egm_reference and egm_resampling_method must either both be set '
                'or both be None'
            )
        return self


class AxisAccuracy(MetadataModel):
    """Bias and standard deviation along one horizontal axis."""
    
    bias: ARDNumber = Field(
        description="Estimated systematic component (bias) of the absolute localisation error along this coordinate axis, in metres.")
    standard_deviation: ARDNonNegativeNumber = Field(
        description="Estimated standard deviation of the absolute localisation error along this coordinate axis, in metres."
    )


class GeometricAccuracyMetadata(MetadataModel):
    """Horizontal geolocation accuracy metadata."""
    
    # types defined by card4l:geometric_accuracy_type
    type: Literal["gtc", "slant-range"] = Field(
        description="Coordinate frame in which geometric accuracy is reported: 'gtc' uses northern/eastern axes for the geocoded terrain-corrected product, whereas 'slant-range' uses line/sample directions of the radar geometry."
    )
    
    eastern: AxisAccuracy = Field(
        description="Absolute localisation error statistics for the eastern axis when type='gtc', or the sample/range axis when type='slant-range'.")
    northern: AxisAccuracy = Field(
        description="Absolute localisation error statistics for the northern axis when type='gtc', or the line/azimuth axis when type='slant-range'.")
    radial_rmse: ARDNonNegativeNumber = Field(
        description="Radial root mean square error (rRMSE) summarising the horizontal absolute localisation error of the output product, in metres."
    )
    reference: ARDText = Field(
        description="URL or DOI referencing documentation or calibration/validation results used to derive the absolute localisation accuracy estimate.")


class GeometricCorrectionMetadata(MetadataModel):
    """Geometric correction algorithm and resulting accuracy."""
    
    algorithm: ARDText = Field(
        description="URL or DOI referencing the algorithm or technical documentation used for geometric terrain correction and geolocation."
    )
    resampling_method: ARDText = Field(
        description="Resampling method used when geometrically correcting and resampling the source measurements to the output grid."
    )
    accuracy: GeometricAccuracyMetadata = Field(
        description="Estimated absolute localisation accuracy of the geometrically corrected output product."
    )


class RadiometricAccuracyMetadata(MetadataModel):
    """Absolute and relative radiometric accuracy."""
    
    absolute: ARDNumber = Field(
        description="Estimated absolute radiometric accuracy of the terrain-corrected backscatter measurements, in decibels.")
    relative: ARDNumber = Field(
        description="Estimated relative radiometric accuracy of the terrain-corrected backscatter measurements, in decibels.")
    reference: ARDText = Field(
        description="URL or DOI referencing documentation that describes the radiometric accuracy or uncertainty estimate and its traceability.")


class NoiseRemovalMetadata(MetadataModel):
    """Thermal-noise removal metadata."""
    
    applied: bool = Field(
        description="Whether noise removal was applied during processing of the backscatter measurements.")
    algorithm: ARDText | None = Field(
        default=None,
        description="URL or DOI referencing the noise-removal algorithm when noise removal was applied; otherwise None.",
    )
    
    @model_validator(mode='after')
    def validate_tnr(self) -> Self:
        if self.applied and self.algorithm is None:
            raise ValueError(
                'if thermal noise removal is applied, an algorithm must be specified'
            )
        if not self.applied and self.algorithm is not None:
            raise ValueError(
                'if thermal noise removal is not applied, no algorithm may be specified'
            )
        return self


class BackscatterMetadata(MetadataModel):
    """Backscatter measurement representation and multilooking metadata."""
    
    measurement: Literal['sigma0', 'gamma0'] = Field(
        description="Backscatter coefficient represented by the measurement assets, for example sigma nought (sigma0) or gamma nought (gamma0)."
    )
    convention: Literal['linear power'] = Field(
        description="Numerical convention used to encode backscatter measurements; 'linear power' stores backscatter intensity on a linear scale."
    )
    conversion_equation: Literal['10*log10(DN)'] = Field(
        description=(
            "Equation used to convert linear backscatter values (DN) to logarithmic "
            "decibel values."
        )
    )
    range_number_of_looks: float = Field(
        gt=0,
        description=(
            "Number of statistically combined looks in the range direction for the "
            "generated product."
        ),
    )
    azimuth_number_of_looks: float = Field(
        gt=0,
        description=(
            "Number of statistically combined looks in the azimuth direction for the "
            "generated product."
        ),
    )
    equivalent_number_of_looks: ARDPositiveNumber | None = Field(
        default=None,
        description="Equivalent Number of Looks (ENL), describing the effective number of independent looks represented by the product.",
    )


class GridMetadata(MetadataModel):
    """Raster grid, CRS and MGRS metadata for the ARD product."""
    
    epsg: int = Field(
        gt=0,
        description="EPSG code of the native coordinate reference system used by the ARD product grid.",
    )
    wkt: ARDText = Field(
        description="Well-Known Text representation of the native coordinate reference system used by the ARD product grid.")
    rows: int = Field(
        gt=0,
        description="Number of raster rows (lines) in the default product grid.",
    )
    columns: int = Field(
        gt=0,
        description="Number of raster columns (pixels per line) in the default product grid.",
    )
    pixel_spacing_row: float = Field(
        gt=0,
        description="Sampling distance between adjacent raster rows in the native coordinate reference system.",
    )
    pixel_spacing_column: float = Field(
        gt=0,
        description="Sampling distance between adjacent raster columns in the native coordinate reference system.",
    )
    transform: AffineTransform = Field(
        description="Affine transformation coefficients mapping pixel coordinates of the default raster grid to coordinates in the native CRS."
    )
    mgrs_id: ARDText = Field(
        description=(
            "Military Grid Reference System (MGRS) tile identifier of the product. The "
            "identifier combines UTM zone, latitude band and 100 km grid square."
        )
    )
    pixel_coordinate_convention: Literal["upper-left"] = Field(
        description="Location within a raster pixel to which the grid coordinates refer; this model currently uses the upper-left pixel corner."
    )
    number_of_border_pixels: int = Field(
        ge=0,
        description=(
            "Number of no-data border pixels surrounding a backscatter measurement "
            "raster, if present."
        ),
    )
    definition_reference: ARDText = Field(
        description="URL or other persistent reference describing the grid or tiling-system definition used for the product.")
    convention_reference: ARDText = Field(
        description="Description, URL or DOI identifying the gridding convention used to align products to a consistent sampling frame.")


class WindNormalizationMetadata(MetadataModel):
    """Optional wind-normalization metadata used by ocean radar backscatter."""
    
    backscatter_measurement: Literal["sigma0"] = Field(
        description="Backscatter measurement to which wind normalization is referenced; sigma0 for the supported ORB representation."
    )
    backscatter_convention: Literal["intensity ratio"] = Field(
        description="Convention used to express the wind-normalized backscatter reference; represented as an intensity ratio."
    )
    reference_direction: float = Field(
        description="Reference wind direction used to evaluate the wind-normalization model, in degrees."
    )
    reference_model: ARDText = Field(
        description="Reference identifying the geophysical model function or model used to derive the wind-normalized backscatter reference."
    )
    reference_speed: NonNegativeFloat = Field(
        description="Reference wind speed used to evaluate the wind-normalization model, in metres per second."
    )
    reference_type: Literal["sigma0-ref"] = Field(
        description="Identifier of the modeled reference backscatter quantity used for wind normalization; sigma0-ref in the supported representation."
    )


class SpeckleFilterMetadata(MetadataModel):
    """Speckle-filter configuration applied during processing."""
    
    model_config = ConfigDict(extra='allow')
    type: str = Field(description="Name or type of the speckle-filtering algorithm applied to the product.")
    window_size_col: PositiveInt = Field(
        description="Width of the speckle-filter processing window in raster columns."
    )
    window_size_line: PositiveInt = Field(
        description="Height of the speckle-filter processing window in raster lines."
    )


class ProductMetadata(MetadataModel):
    """Metadata describing the generated ARD product."""
    
    name: Literal["Normalised Radar Backscatter", "Ocean Radar Backscatter"] = Field(
        description="Human-readable name of the implemented CEOS CARD4L Radar Backscatter product family."
    )
    product_type: Literal["NRB", "ORB"] = Field(
        description="Short CARD4L product-family identifier: NRB for Normalised Radar Backscatter or ORB for Ocean Radar Backscatter.")
    acquisition_type: Literal["NOMINAL", "CALIBRATION", "OTHER"] = Field(
        description=(
            "High-level classification of the acquisition as nominal data, calibration "
            "data or another acquisition type."
        )
    )
    status: Literal[
        "ARCHIVED", "ACQUIRED", "CANCELLED", "FAILED", "PLANNED",
        "POTENTIAL", "REJECTED", "QUALITYDEGRADED"
    ] = Field(description="Lifecycle or availability status assigned to the ARD product.")
    
    access: ARDText | None = Field(
        default=None,
        description="URL or other location from which the ARD product can be retrieved.",
    )
    doi: ARDText | None = Field(
        default=None,
        description="Digital Object Identifier assigned to the ARD product, if available.",
    )
    license: ARDText | None = Field(
        default=None,
        description="License or copyright identifier governing use and redistribution of the ARD product.",
    )
    
    time_start: AwareDatetime = Field(
        description="UTC start time of the earliest source acquisition contributing to the ARD product.")
    time_stop: AwareDatetime = Field(
        description="UTC end time of the latest source acquisition contributing to the ARD product.")
    time_created: AwareDatetime = Field(
        description=(
            "UTC date and time at which the ARD product was generated."
        )
    )
    
    geometry: GeometryMetadata = Field(
        description="Spatial footprint and bounding-box metadata of the generated ARD product.")
    grid: GridMetadata = Field(
        description="Dimensions, sampling, georeferencing, CRS and grid-convention metadata of the ARD product raster.")
    processing: ProcessingMetadata = Field(
        description="Facility, software, processing date and processing mode used to generate the ARD product.")
    card4l: Card4lMetadata = Field(
        description="Identification and reference of the CEOS CARD4L product family specification implemented by this product."
    )
    compression: CompressionMetadata = Field(
        description="Compression method and associated error thresholds used for product raster assets."
    )
    dem: DEMMetadata = Field(
        description="Digital elevation/surface model and Earth Gravitational Model metadata used for geometric and radiometric terrain processing."
    )
    backscatter: BackscatterMetadata = Field(
        description="Definition of the backscatter quantity, numerical convention, dB conversion and effective multilooking of the product."
    )
    geometric_correction: GeometricCorrectionMetadata = Field(
        description="Algorithm, resampling method and absolute localisation accuracy associated with geometric terrain correction."
    )
    radiometric_accuracy: RadiometricAccuracyMetadata = Field(
        description="Absolute and relative accuracy estimates for the radiometrically terrain-corrected backscatter measurements."
    )
    noise_removal: NoiseRemovalMetadata = Field(
        description="Whether noise removal was applied and, when applicable, the algorithm reference.")
    
    rtc_algorithm: ARDText | None = Field(
        default=None,
        description="URL or DOI referencing the radiometric terrain correction algorithm or its technical implementation documentation.",
    )
    
    number_of_acquisitions: int = Field(
        gt=0,
        description="Number of source acquisitions contributing to the generated ARD product.",
    )
    speckle_filter_applied: SpeckleFilterMetadata | None = Field(
        description="Speckle-filter configuration if filtering was applied; None indicates that no speckle filter was applied."
    )
    ellipsoidal_height: ARDNumber | None = Field(
        default=None,
        description="Ellipsoidal height, in metres, used when deriving ellipsoidal incidence-angle information, if applicable.",
    )
    wind_normalization: WindNormalizationMetadata | None = Field(
        default=None,
        description="Parameters defining the reference model and conditions used for wind normalization of ocean radar backscatter, if applicable.",
    )
    
    @model_validator(mode="after")
    def _validate_times(self) -> ProductMetadata:
        if self.time_stop < self.time_start:
            raise ValueError("product time_stop must not precede time_start")
        return self


class SwathAxisMetadata(MetadataModel):
    """Sampling and resolution metadata along one SAR image axis."""
    
    look_bandwidth: dict[str, ARDNumber | None] = Field(
        description="Processing look bandwidth for this image axis, keyed by swath or beam."
    )
    number_of_looks: dict[str, ARDPositiveInteger] = Field(
        description="Number of looks combined along this image axis, keyed by swath or beam."
    )
    pixel_spacing: dict[str, ARDPositiveNumber] = Field(
        description="Sampling distance between adjacent source-data pixels along this image axis, keyed by swath or beam."
    )
    resolution: dict[str, ARDPositiveNumber] = Field(
        description="Spatial resolution along this image axis, keyed by swath or beam; resolution describes the ability to distinguish adjacent targets."
    )


class IncidenceAngleMetadata(MetadataModel):
    """Near-, mid- and far-range incidence angles in degrees."""
    
    minimum: float = Field(description="Near-range incidence angle of the source acquisition, in degrees.")
    maximum: float = Field(description="Far-range incidence angle of the source acquisition, in degrees.")
    mid_swath: float = Field(description="Incidence angle at the centre of the source swath, in degrees.")
    
    @model_validator(mode="after")
    def _validate_order(self) -> IncidenceAngleMetadata:
        if not self.minimum <= self.mid_swath <= self.maximum:
            raise ValueError(
                "incidence angles must satisfy minimum <= mid_swath <= maximum"
            )
        return self


class PerformanceEstimate(MetadataModel):
    """Minimum, mean and maximum noise-equivalent intensity estimate."""
    
    minimum: float | None = Field(
        default=None,
        description="Minimum noise-equivalent backscatter intensity level for the corresponding polarization, in decibels.",
    )
    mean: float | None = Field(
        default=None,
        description="Mean noise-equivalent backscatter intensity level for the corresponding polarization, in decibels.",
    )
    maximum: float | None = Field(
        default=None,
        description="Maximum noise-equivalent backscatter intensity level for the corresponding polarization, in decibels.",
    )


class SourcePerformanceMetadata(MetadataModel):
    """CARD4L source-product performance indicators."""
    
    noise_equivalent_intensity_type: Literal['sigma0'] | None = Field(
        default=None,
        description="Backscatter convention to which the noise-equivalent intensity estimates refer, such as sigma0.",
    )
    estimates: dict[Polarization, PerformanceEstimate] = Field(
        description="Noise-equivalent intensity statistics for each available polarization channel."
    )
    equivalent_number_of_looks: ARDPositiveNumber | None = Field(
        default=None,
        description="Equivalent Number of Looks (ENL) of the source data, describing its effective number of independent looks.",
    )
    integrated_side_lobe_ratio: float | None = Field(
        default=None,
        description="Mean Integrated Side Lobe Ratio (ISLR) of the source data, in decibels.",
    )
    peak_side_lobe_ratio: float | None = Field(
        default=None,
        description="Mean Peak Side Lobe Ratio (PSLR) of the source data, in decibels.",
    )


class SourceOrbitMetadata(MetadataModel):
    """Orbit metadata specific to an input source scene."""
    
    ascending_node_date: AwareDatetime | None = Field(
        default=None,
        description="UTC date and time at which the platform crossed the ascending node for the orbit containing the acquisition.",
    )
    start_time_from_ascending_node: ARDNumber | None = Field(
        default=None,
        description="Elapsed time from the ascending-node crossing to acquisition start, in milliseconds.",
    )
    completion_time_from_ascending_node: ARDNumber | None = Field(
        default=None,
        description="Elapsed time from the ascending-node crossing to acquisition completion, in milliseconds.",
    )
    major_cycle_id: ARDPositiveInteger | None = Field(
        default=None,
        description="Mission-specific identifier of the major orbital cycle containing the acquisition.",
    )
    datatake_id: ARDPositiveInteger | None = Field(
        default=None,
        description="Mission-specific identifier of the data take to which the source acquisition belongs.",
    )
    data_access: ARDText = Field(
        description="Location from which orbit state-vector data used for processing can be accessed."
    )
    data_source: ARDText | None = Field(
        default=None,
        description="Classification of the orbit information used for processing, for example predicted, restituted or precise.",
    )
    state_vector: ARDText | None = Field(
        default=None,
        description=(
            "Reference to the external orbit state-vector file used during processing, "
            "if such a file was used."
        ),
    )


class SourceMetadata(MetadataModel):
    """Metadata describing one source product contributing to the ARD product."""
    
    filename: ARDText = Field(
        description="Identifier or filename of the source data product, without a file extension."
    )
    product_type: ARDText = Field(description="Product type or processing level of the source SAR data.")
    data_geometry: Literal["slant-range", "ground-range"] = Field(
        description="Radar geometry of the source image: slant-range or ground-range."
    )
    acquisition_type: Literal["NOMINAL", "CALIBRATION", "OTHER"] = Field(
        description=(
            "High-level classification of the source acquisition as nominal data, "
            "calibration data or another acquisition type."
        )
    )
    status: Literal[
        "ARCHIVED", "ACQUIRED", "CANCELLED", "FAILED", "PLANNED",
        "POTENTIAL", "REJECTED", "QUALITYDEGRADED"
    ] = Field(description="Lifecycle or availability status assigned to the source product.")
    
    access: ARDText | None = Field(
        default=None,
        description="URL, DOI or other location from which the source data product can be retrieved.",
    )
    doi: ARDText | None = Field(
        default=None,
        description="Digital Object Identifier assigned to the source data product, if available.",
    )
    
    time_start: AwareDatetime = Field(description="UTC start time of the source-data acquisition.")
    time_stop: AwareDatetime = Field(description="UTC end time of the source-data acquisition.")
    geometry: GeometryMetadata = Field(
        description="WGS 84 footprint and bounding-box metadata of the source acquisition.")
    processing: ProcessingMetadata = Field(
        description="Facility, software, processing date and processing mode associated with the source product.")
    orbit: SourceOrbitMetadata = Field(
        description="Orbit timing, source and state-vector metadata used for processing the source acquisition.")
    
    swaths: tuple[str, ...] = Field(
        min_length=1,
        description="Identifiers of the swaths or beams represented by the source acquisition.",
    )
    azimuth: SwathAxisMetadata = Field(
        description="Look bandwidth, number of looks, pixel spacing and resolution along the azimuth direction, keyed by swath or beam."
    )
    range: SwathAxisMetadata = Field(
        description="Look bandwidth, number of looks, pixel spacing and resolution along the range direction, keyed by swath or beam."
    )
    incidence_angle: IncidenceAngleMetadata = Field(
        description="Near-range, centre-swath and far-range incidence angles of the source acquisition."
    )
    instrument_azimuth_angle: float | None = Field(
        default=None,
        description="Mean platform heading or instrument azimuth angle during acquisition, in degrees clockwise from north.",
    )
    
    lut_applied: ARDText | None = Field(
        default=None,
        description="Name or identifier of the lookup table (LUT) applied during source-data processing, if applicable.",
    )
    sensor_calibration: ARDText | None = Field(
        default=None,
        description="Reference to sensor calibration parameters used for the source data, for example a URL, DOI or calibration-coefficient source.",
    )
    polarimetric_calibration_matrices: ARDText | None = Field(
        default=None,
        description=(
            "Reference or representation of the complex-valued polarimetric distortion "
            "matrices describing channel imbalance and cross-talk corrections."
        ),
    )
    faraday_mean_rotation_angle: ARDNumber | None = Field(
        default=None,
        description=(
            "Mean Faraday rotation angle estimated from the polarimetric source data or "
            "a model, in degrees."
        ),
    )
    faraday_rotation_reference: ARDText | None = Field(
        default=None,
        description=(
            "URL, DOI or other reference to the method or publication used to derive "
            "the mean Faraday rotation estimate."
        ),
    )
    ionosphere_indicator: bool | None = Field(
        default=None,
        description=(
            "Whether ionospheric effects are significant enough to exceed the "
            "radiometric calibration requirement or goal for the backscatter imagery."
        ),
    )
    
    performance: SourcePerformanceMetadata = Field(
        description="Noise-equivalent intensity and other SAR performance indicators reported for the source product."
    )
    
    @model_validator(mode="after")
    def _validate_source(self) -> SourceMetadata:
        if self.time_stop < self.time_start:
            raise ValueError("source time_stop must not precede time_start")
        
        expected = set(self.swaths)
        for axis_name, axis in (
                ("azimuth", self.azimuth),
                ("range", self.range)
        ):
            for field_name in (
                    "look_bandwidth",
                    "number_of_looks",
                    "resolution",
            ):
                actual = set(getattr(axis, field_name))
                if actual != expected:
                    raise ValueError(
                        f"{axis_name}.{field_name} keys {sorted(actual)!r} do not "
                        f"match swaths {sorted(expected)!r}"
                    )
        return self


class ARDMetadata(MetadataModel):
    """Canonical metadata transfer object for CESARD STAC and XML writers."""
    
    model_config = ConfigDict(
        title="CESARD ARD Metadata",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$comment": (
                "Reserved CESARD implementation markers: "
                "NOT_IMPLEMENTED_NUMBER = -99999; "
                "NOT_IMPLEMENTED_TEXT = 'TBD'. These values denote mandatory "
                "metadata whose derivation is not yet implemented and are "
                "distinct from null/None."
            ),
        },
    )
    
    schema_version: Literal["1.0.0"] = Field(
        default="1.0.0",  # so that it does not have to be supplied when model is constructed
        description="Version of the CESARD ARDMetadata interface schema used by this metadata object.",
    )
    common: CommonMetadata = Field(
        description="Mission, platform, instrument, orbit and acquisition metadata shared by the ARD product and its contributing source scenes."
    )
    product: ProductMetadata = Field(description="Metadata describing the generated analysis-ready radar product.")
    sources: dict[str, SourceMetadata] = Field(
        min_length=1,
        description="Source SAR products contributing to the generated ARD product, keyed by a stable source identifier.",
    )
    
    @model_validator(mode="after")
    def _validate_product_source_consistency(self) -> ARDMetadata:
        if self.product.number_of_acquisitions != len(self.sources):
            raise ValueError(
                "product.number_of_acquisitions does not match the number of sources"
            )
        return self
    
    @classmethod
    def get_schema_version(cls) -> str:
        """Return the ARD metadata schema version."""
        annotation = cls.model_fields["schema_version"].annotation
        versions = get_args(annotation)
        
        if get_origin(annotation) is not Literal or len(versions) != 1:
            raise TypeError("'schema_version' must be a Literal with exactly one value")
        
        return versions[0]
    
    @classmethod
    def get_schema_path(cls) -> Path:
        return (
                Path(__file__).parent
                / "schemas"
                / f"ard-metadata-{cls.get_schema_version()}.schema.json"
        )
    
    @classmethod
    def write_json_schema(cls, overwrite=False) -> None:
        """Write the model's versioned JSON Schema."""
        path = cls.get_schema_path()
        if path.exists() and not overwrite:
            raise FileExistsError(f"Schema file already exists: {path}."
                                  f"Consider updating the schema version.")
        path.write_text(
            json.dumps(cls.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
