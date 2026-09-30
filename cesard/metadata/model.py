"""Canonical metadata interface between SAR-ARD producers and CESARD writers.

The models in this module define the metadata contract consumed by the generic
CESARD STAC and XML writers. Satellite-specific packages such as s1ard and
asard should return :class:`ARDMetadata` rather than a free-form dictionary.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Self

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
    
    bbox: BBox
    bbox_native: BBox | None = None
    geometry: dict[str, Any]
    
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
    
    facility: ARDText | None = None
    date: AwareDatetime | None = None
    mode: Literal["PROTOTYPE", "NOMINAL"]
    processor: ARDText | None = None
    software: dict[str, str] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def _processor_has_version(self) -> ProcessingMetadata:
        if self.processor is not None and self.software and self.processor not in self.software:
            raise ValueError(
                f"processor {self.processor!r} is not present in software versions"
            )
        return self


class CommonMetadata(MetadataModel):
    """Metadata shared by the ARD product and all contributing source scenes."""
    
    antenna_look_direction: Literal["LEFT", "RIGHT"]
    constellation: ARDText
    instrument_short_name: ARDText
    operational_mode: ARDText
    orbit_direction: Literal["ascending", "descending"]
    orbit_mean_altitude: float = Field(gt=0)
    orbit_number_absolute: int = Field(ge=0)
    orbit_number_relative: int = Field(ge=0)
    platform_full_name: ARDText
    platform_identifier: ARDText
    platform_reference: ARDText
    platform_short_name: ARDText
    polarizations: tuple[Polarization, ...] = Field(min_length=1)
    polarization_mode: ARDText
    processing_level: Literal["L1C"]
    radar_band: Literal["X", "C", "L"]
    radar_center_frequency: float = Field(gt=0)
    sensor_type: Literal["RADAR"]
    swath_identifier: ARDText
    wrs_longitude_grid: int = Field(ge=0)


class Card4lMetadata(MetadataModel):
    """Reference to the applicable CARD4L product-family specification."""
    
    specification: Literal["NRB", "ORB"]
    version: ARDText
    document: ARDText


class CompressionMetadata(MetadataModel):
    """Compression configuration used for product raster assets."""
    
    type: ARDText
    z_errors: dict[str, float]


class GroundSamplingDistance(MetadataModel):
    """Ground sampling distance with an explicit unit."""
    
    value: float = Field(gt=0)
    unit: Literal["m", "arcsec"]


class DEMMetadata(MetadataModel):
    """Digital elevation model and Earth gravitational model metadata."""
    
    name: ARDText
    type: Literal["surface", "elevation"]
    reference: ARDText
    access: ARDText
    gsd: GroundSamplingDistance
    resampling_method: ARDText
    egm_reference: ARDText | None
    egm_resampling_method: ARDText | None
    
    @model_validator(mode='after')
    def validate_egm(self) -> Self:
        if (self.egm_reference is None) != (self.egm_resampling_method is None):
            raise ValueError(
                'egm_reference and egm_resampling_method must either both be set '
                'or both be None'
            )
        return self


class AxisAccuracy(MetadataModel):
    """Bias and standard deviation along one horizontal axis.
    """
    
    bias: ARDNumber = Field(
        description=(
            "Horizontal bias"
        )
    )
    standard_deviation: ARDNonNegativeNumber = Field(
        description=(
            "Horizontal standard deviation"
        )
    )


class GeometricAccuracyMetadata(MetadataModel):
    """Horizontal geolocation accuracy metadata."""
    
    # types defined by card4l:geometric_accuracy_type
    type: Literal["gtc", "slant-range"]
    
    eastern: AxisAccuracy
    northern: AxisAccuracy
    radial_rmse: ARDNonNegativeNumber = Field(
        description=(
            "Radial RMSE in metres"
        )
    )
    reference: ARDText = Field(
        description=(
            "Reference documenting the accuracy estimate"
        )
    )


class GeometricCorrectionMetadata(MetadataModel):
    """Geometric correction algorithm and resulting accuracy."""
    
    algorithm: ARDText
    resampling_method: ARDText
    accuracy: GeometricAccuracyMetadata


class RadiometricAccuracyMetadata(MetadataModel):
    """Absolute and relative radiometric accuracy.
    """
    
    absolute: ARDNumber = Field(
        description=(
            "Absolute radiometric accuracy"
        )
    )
    relative: ARDNumber = Field(
        description=(
            "Relative radiometric accuracy"
        )
    )
    reference: ARDText


class NoiseRemovalMetadata(MetadataModel):
    """Thermal-noise removal metadata."""
    
    applied: bool
    algorithm: ARDText | None = None


class BackscatterMetadata(MetadataModel):
    """Backscatter measurement representation and multilooking metadata."""
    
    measurement: Literal['sigma0', 'gamma0']
    convention: Literal['linear power']
    conversion_equation: Literal['10*log10(DN)']
    range_number_of_looks: float = Field(gt=0)
    azimuth_number_of_looks: float = Field(gt=0)
    equivalent_number_of_looks: float | None = Field(default=None, gt=0)


class GridMetadata(MetadataModel):
    """Raster grid, CRS and MGRS metadata for the ARD product."""
    
    epsg: int = Field(gt=0)
    wkt: ARDText
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    pixel_spacing_row: float = Field(gt=0)
    pixel_spacing_column: float = Field(gt=0)
    transform: AffineTransform
    mgrs_id: ARDText
    pixel_coordinate_convention: Literal["upper-left"]
    number_of_border_pixels: int = Field(ge=0)
    definition_reference: ARDText
    convention_reference: ARDText


class WindNormalizationMetadata(MetadataModel):
    """Optional wind-normalization metadata used by ocean radar backscatter."""
    
    backscatter_measurement: Literal["sigma0"] | None
    backscatter_convention: Literal["intensity ratio"] | None
    reference_direction: float | None
    reference_model: ARDText | None
    reference_speed: float | None = Field(ge=0)
    reference_type: Literal["sigma0-ref"] | None


class ProductMetadata(MetadataModel):
    """Metadata describing the generated ARD product."""
    
    name: Literal["Normalised Radar Backscatter", "Ocean Radar Backscatter"]
    product_type: Literal["NRB", "ORB"]
    acquisition_type: Literal["NOMINAL", "CALIBRATION", "OTHER"]
    status: Literal[
        "ARCHIVED", "ACQUIRED", "CANCELLED", "FAILED", "PLANNED",
        "POTENTIAL", "REJECTED", "QUALITYDEGRADED"
    ]
    
    access: ARDText | None = None
    doi: ARDText | None = None
    license: ARDText | None = None
    
    time_start: AwareDatetime
    time_stop: AwareDatetime
    time_created: AwareDatetime
    
    geometry: GeometryMetadata
    grid: GridMetadata
    processing: ProcessingMetadata
    card4l: Card4lMetadata
    compression: CompressionMetadata
    dem: DEMMetadata
    backscatter: BackscatterMetadata
    geometric_correction: GeometricCorrectionMetadata
    radiometric_accuracy: RadiometricAccuracyMetadata
    noise_removal: NoiseRemovalMetadata
    
    rtc_algorithm: ARDText | None = None
    
    number_of_acquisitions: int = Field(gt=0)
    speckle_filter_applied: SpeckleFilterMetadata | None
    ellipsoidal_height: ARDNumber | None = None
    wind_normalization: WindNormalizationMetadata | None = None
    
    @model_validator(mode="after")
    def _validate_times(self) -> ProductMetadata:
        if self.time_stop < self.time_start:
            raise ValueError("product time_stop must not precede time_start")
        return self


class SwathAxisMetadata(MetadataModel):
    """Sampling and resolution metadata along one SAR image axis."""
    
    look_bandwidth: dict[str, ARDNumber | None] = Field(
        description=(
            "Look bandwidth per swath"
        )
    )
    number_of_looks: dict[str, ARDPositiveInteger]
    pixel_spacing: dict[str, ARDPositiveNumber]
    resolution: dict[str, ARDPositiveNumber]


class SpeckleFilterMetadata(MetadataModel):
    model_config = ConfigDict(extra='allow')
    type: str
    window_size_col: str
    window_size_line: str


class IncidenceAngleMetadata(MetadataModel):
    """Near-, mid- and far-range incidence angles in degrees."""
    
    minimum: float
    maximum: float
    mid_swath: float
    
    @model_validator(mode="after")
    def _validate_order(self) -> IncidenceAngleMetadata:
        if not self.minimum <= self.mid_swath <= self.maximum:
            raise ValueError(
                "incidence angles must satisfy minimum <= mid_swath <= maximum"
            )
        return self


class PerformanceEstimate(MetadataModel):
    """Minimum, mean and maximum noise-equivalent intensity estimate."""
    
    minimum: float | None = None
    mean: float | None = None
    maximum: float | None = None


class SourcePerformanceMetadata(MetadataModel):
    """CARD4L source-product performance indicators."""
    
    noise_equivalent_intensity_type: Literal['sigma0'] | None = None
    estimates: dict[Polarization, PerformanceEstimate]
    equivalent_number_of_looks: float | None = Field(default=None, gt=0)
    integrated_side_lobe_ratio: float | None = None
    peak_side_lobe_ratio: float | None = None


class SourceOrbitMetadata(MetadataModel):
    """Orbit metadata specific to an input source scene."""
    
    ascending_node_date: AwareDatetime | None = None
    start_time_from_ascending_node: ARDNumber | None = None
    completion_time_from_ascending_node: ARDNumber | None = None
    major_cycle_id: int = Field(ge=0)
    datatake_id: int | None = None
    data_access: ARDText
    data_source: ARDText | None = None
    state_vector: ARDText | None = None


class SourceMetadata(MetadataModel):
    """Metadata describing one source product contributing to the ARD product."""
    
    filename: ARDText
    product_type: ARDText
    data_geometry: ARDText
    acquisition_type: Literal["NOMINAL", "CALIBRATION", "OTHER"]
    status: Literal[
        "ARCHIVED", "ACQUIRED", "CANCELLED", "FAILED", "PLANNED",
        "POTENTIAL", "REJECTED", "QUALITYDEGRADED"
    ]
    
    access: ARDText | None = None
    doi: ARDText | None = None
    
    time_start: AwareDatetime
    time_stop: AwareDatetime
    geometry: GeometryMetadata
    processing: ProcessingMetadata
    orbit: SourceOrbitMetadata
    
    swaths: tuple[str, ...] = Field(min_length=1)
    azimuth: SwathAxisMetadata
    range: SwathAxisMetadata
    incidence_angle: IncidenceAngleMetadata
    instrument_azimuth_angle: float | None = None
    
    lut_applied: ARDText | None = None
    sensor_calibration: ARDText | None = None
    polarimetric_calibration_matrices: ARDText | None = None
    faraday_mean_rotation_angle: ARDNumber | None = None
    faraday_rotation_reference: ARDText | None = None
    ionosphere_indicator: bool | None = None
    
    performance: SourcePerformanceMetadata
    
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
    
    schema_version: Literal["1.0"] = "1.0"
    common: CommonMetadata
    product: ProductMetadata
    sources: dict[str, SourceMetadata] = Field(min_length=1)
    
    @model_validator(mode="after")
    def _validate_product_source_consistency(self) -> ARDMetadata:
        if self.product.number_of_acquisitions != len(self.sources):
            raise ValueError(
                "product.number_of_acquisitions does not match the number of sources"
            )
        return self
    
    @classmethod
    def write_json_schema(cls, path: str | Path) -> None:
        """Write the model's JSON Schema to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cls.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
    
    @classmethod
    def from_legacy(cls, meta: Mapping[str, Any]) -> ARDMetadata:
        """Convert the current ``common/prod/source`` dictionaries to the model.

        This adapter is intended as a migration aid for the current s1ard and
        asard ``meta_dict`` implementations. Reserved implementation markers
        ``-99999`` and ``'TBD'`` are preserved. The legacy string ``'None'`` is
        converted to ``None`` only for genuinely optional metadata.
        """
        common = meta["common"]
        product = meta["prod"]
        sources = meta["source"]
        
        def optional_number(value: Any) -> float | None:
            if value is None:
                return None
            if value == NOT_IMPLEMENTED_NUMBER:
                return NOT_IMPLEMENTED_NUMBER
            return float(value)
        
        def optional_text(value: Any) -> str | None:
            if value is None or value == "None":
                return None
            if value == NOT_IMPLEMENTED_TEXT:
                return NOT_IMPLEMENTED_TEXT
            return str(value)
        
        def aware(value: datetime) -> datetime:
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        
        def bbox(value: Any) -> BBox | None:
            if value is None:
                return None
            if len(value) != 4:
                raise ValueError(f"expected a four-element bbox, got {value!r}")
            return tuple(float(x) for x in value)  # type: ignore[return-value]
        
        def transform(value: Any) -> AffineTransform:
            if len(value) != 6:
                raise ValueError(
                    f"expected a six-element affine transform, got {value!r}"
                )
            return tuple(float(x) for x in value)  # type: ignore[return-value]
        
        def geometry(container: Mapping[str, Any]) -> GeometryMetadata:
            return GeometryMetadata(
                bbox=bbox(container["geom_stac_bbox_4326"]),
                bbox_native=bbox(container.get("geom_stac_bbox_native")),
                geometry=dict(container["geom_stac_geometry_4326"]),
            )
        
        def per_swath_float(
                value: Mapping[str, Any], *, allow_none: bool = False
        ) -> dict[str, float | None] | dict[str, float]:
            if allow_none:
                return {
                    str(k): optional_number(v)
                    for k, v in value.items()
                }
            return {str(k): float(v) for k, v in value.items()}
        
        def gsd(value: Any) -> GroundSamplingDistance:
            if isinstance(value, str):
                parts = value.split()
                if len(parts) == 1:
                    return GroundSamplingDistance(value=float(parts[0]), unit="m")
                if len(parts) == 2:
                    return GroundSamplingDistance(
                        value=float(parts[0]), unit=parts[1]
                    )
                raise ValueError(f"cannot parse DEM GSD {value!r}")
            return GroundSamplingDistance(value=float(value), unit="m")
        
        wind = None
        if product.get("windNormBackscatterMeasurement") is not None:
            wind = WindNormalizationMetadata(
                backscatter_measurement=product["windNormBackscatterMeasurement"],
                backscatter_convention=product["windNormBackscatterConvention"],
                reference_direction=product["windNormReferenceDirection"],
                reference_model=product["windNormReferenceModel"],
                reference_speed=product["windNormReferenceSpeed"],
                reference_type=product["windNormReferenceType"],
            )
        
        product_model = ProductMetadata(
            name=product["productName"],
            product_type=product["productName-short"],
            acquisition_type=product["acquisitionType"],
            status=product["status"],
            access=optional_text(product.get("access")),
            doi=optional_text(product.get("doi")),
            license=optional_text(product.get("licence")),
            time_start=aware(product["timeStart"]),
            time_stop=aware(product["timeStop"]),
            time_created=aware(product["timeCreated"]),
            geometry=geometry(product),
            grid=GridMetadata(
                epsg=product["crsEPSG"],
                wkt=product["crsWKT"],
                rows=product["numLines"],
                columns=product["numPixelsPerLine"],
                pixel_spacing_row=product["pxSpacingRow"],
                pixel_spacing_column=product["pxSpacingColumn"],
                transform=transform(product["transform"]),
                mgrs_id=product["mgrsID"],
                pixel_coordinate_convention=product["pixelCoordinateConvention"],
                number_of_border_pixels=product["numBorderPixels"],
                definition_reference=product["grid_definition_url"],
                convention_reference=product["grid_convention_url"],
            ),
            processing=ProcessingMetadata(
                facility=optional_text(product.get("processingCenter")),
                date=aware(product["timeCreated"]),
                mode=product.get("processingMode"),
                processor=optional_text(product.get("processorName")),
                software={str(k): str(v) for k, v in (product.get("processorVersion") or {}).items()},
            ),
            card4l=Card4lMetadata(
                specification=product["productName-short"],
                version=product["card4l-version"],
                document=product["card4l-link"],
            ),
            compression=CompressionMetadata(
                type=product["compression_type"],
                z_errors=product["compression_zerrors"],
            ),
            dem=DEMMetadata(
                name=product["demName"],
                type=product["demType"],
                reference=product["demReference"],
                access=product["demAccess"],
                gsd=gsd(product["demGSD"]),
                resampling_method=product["demResamplingMethod"],
                egm_reference=product["demEGMReference"],
                egm_resampling_method=product["demEGMResamplingMethod"],
            ),
            backscatter=BackscatterMetadata(
                measurement=product["backscatterMeasurement"],
                convention=product["backscatterConvention"],
                conversion_equation=product["backscatterConversionEq"],
                range_number_of_looks=product["rangeNumberOfLooks"],
                azimuth_number_of_looks=product["azimuthNumberOfLooks"],
                equivalent_number_of_looks=optional_number(
                    product.get("equivalentNumberOfLooks")
                ),
            ),
            geometric_correction=GeometricCorrectionMetadata(
                algorithm=product["geoCorrAlgorithm"],
                resampling_method=product["geoCorrResamplingMethod"],
                accuracy=GeometricAccuracyMetadata(
                    type=product["geoCorrAccuracyType"],
                    eastern=AxisAccuracy(
                        bias=optional_number(product["geoCorrAccuracyEasternBias"]),
                        standard_deviation=optional_number(
                            product["geoCorrAccuracyEasternSTDev"]
                        ),
                    ),
                    northern=AxisAccuracy(
                        bias=optional_number(product["geoCorrAccuracyNorthernBias"]),
                        standard_deviation=optional_number(
                            product["geoCorrAccuracyNorthernSTDev"]
                        ),
                    ),
                    radial_rmse=optional_number(product["geoCorrAccuracy_rRMSE"]),
                    reference=str(product["geoCorrAccuracyReference"]),
                ),
            ),
            radiometric_accuracy=RadiometricAccuracyMetadata(
                absolute=optional_number(product["radiometricAccuracyAbsolute"]),
                relative=optional_number(product["radiometricAccuracyRelative"]),
                reference=str(product["radiometricAccuracyReference"]),
            ),
            noise_removal=NoiseRemovalMetadata(
                applied=bool(product["noiseRemovalApplied"]),
                algorithm=optional_text(product.get("noiseRemovalAlgorithm")),
            ),
            rtc_algorithm=optional_text(product.get("RTCAlgorithm")),
            number_of_acquisitions=product["numberOfAcquisitions"],
            speckle_filter_applied=(
                None if product.get("speckleFilterApplied") is None else bool(product["speckleFilterApplied"])),
            ellipsoidal_height=optional_number(product.get("ellipsoidalHeight")),
            wind_normalization=wind,
        )
        
        source_models: dict[str, SourceMetadata] = {}
        for uid, source in sources.items():
            perf_estimates = {
                str(pol): PerformanceEstimate(
                    minimum=optional_number(values.get("minimum")),
                    mean=optional_number(values.get("mean")),
                    maximum=optional_number(values.get("maximum")),
                )
                for pol, values in source["perfEstimates"].items()
            }
            
            source_models[str(uid)] = SourceMetadata(
                filename=source["filename"],
                product_type=source["productType"],
                data_geometry=source["dataGeometry"],
                acquisition_type=source["acquisitionType"],
                status=source["status"],
                access=optional_text(source.get("access")),
                doi=optional_text(source.get("doi")),
                time_start=aware(source["timeStart"]),
                time_stop=aware(source["timeStop"]),
                geometry=geometry(source),
                processing=ProcessingMetadata(
                    facility=optional_text(source.get("processingCenter")),
                    date=(
                        aware(source["processingDate"])
                        if source.get("processingDate") is not None
                        else None
                    ),
                    mode=source.get("processingMode"),
                    processor=optional_text(source.get("processorName")),
                    software={str(k): str(v) for k, v in (source.get("processorVersion") or {}).items()},
                ),
                orbit=SourceOrbitMetadata(
                    ascending_node_date=(
                        aware(source["ascendingNodeDate"])
                        if source.get("ascendingNodeDate") is not None
                        else None
                    ),
                    start_time_from_ascending_node=source["timeStartFromAscendingNode"],
                    completion_time_from_ascending_node=source["timeCompletionFromAscendingNode"],
                    major_cycle_id=source["majorCycleID"],
                    datatake_id=source.get("datatakeID"),
                    data_access=source["orbitDataAccess"],
                    data_source=optional_text(source.get("orbitDataSource")),
                    state_vector=optional_text(source.get("orbitStateVector")),
                ),
                swaths=tuple(str(x) for x in source["swaths"]),
                azimuth=SwathAxisMetadata(
                    look_bandwidth=per_swath_float(
                        source["azimuthLookBandwidth"], allow_none=True
                    ),
                    number_of_looks=per_swath_float(
                        source["azimuthNumberOfLooks"]
                    ),
                    pixel_spacing=per_swath_float(source["azimuthPixelSpacing"]),
                    resolution=per_swath_float(source["azimuthResolution"]),
                ),
                range=SwathAxisMetadata(
                    look_bandwidth=per_swath_float(
                        source["rangeLookBandwidth"], allow_none=True
                    ),
                    number_of_looks=per_swath_float(source["rangeNumberOfLooks"]),
                    pixel_spacing=per_swath_float(source["rangePixelSpacing"]),
                    resolution=per_swath_float(source["rangeResolution"]),
                ),
                incidence_angle=IncidenceAngleMetadata(
                    minimum=source["incidenceAngleMin"],
                    maximum=source["incidenceAngleMax"],
                    mid_swath=source["incidenceAngleMidSwath"],
                ),
                instrument_azimuth_angle=optional_number(
                    source.get("instrumentAzimuthAngle")
                ),
                lut_applied=optional_text(source.get("lutApplied")),
                sensor_calibration=optional_text(source.get("sensorCalibration")),
                polarimetric_calibration_matrices=optional_text(
                    source.get("polCalMatrices")
                ),
                faraday_mean_rotation_angle=optional_number(
                    source.get("faradayMeanRotationAngle")
                ),
                faraday_rotation_reference=optional_text(
                    source.get("faradayRotationReference")
                ),
                ionosphere_indicator=source.get("ionosphereIndicator"),
                performance=SourcePerformanceMetadata(
                    noise_equivalent_intensity_type=optional_text(
                        source.get("perfNoiseEquivalentIntensityType")
                    ),
                    estimates=perf_estimates,
                    equivalent_number_of_looks=optional_number(
                        source.get("perfEquivalentNumberOfLooks")
                    ),
                    integrated_side_lobe_ratio=optional_number(
                        source.get("perfIntegratedSideLobeRatio")
                    ),
                    peak_side_lobe_ratio=optional_number(
                        source.get("perfPeakSideLobeRatio")
                    ),
                ),
            )
        
        return cls(
            common=CommonMetadata(
                antenna_look_direction=common["antennaLookDirection"],
                constellation=common["constellation"],
                instrument_short_name=common["instrumentShortName"],
                operational_mode=common["operationalMode"],
                orbit_direction=common["orbitDirection"],
                orbit_mean_altitude=common["orbitMeanAltitude"],
                orbit_number_absolute=common["orbitNumber_abs"],
                orbit_number_relative=common["orbitNumber_rel"],
                platform_full_name=common["platformFullname"],
                platform_identifier=str(common["platformIdentifier"]),
                platform_reference=common["platformReference"],
                platform_short_name=common["platformShortName"],
                polarizations=tuple(common["polarisationChannels"]),
                polarization_mode=common["polarisationMode"],
                processing_level=common["processingLevel"],
                radar_band=common["radarBand"],
                radar_center_frequency=common["radarCenterFreq"],
                sensor_type=common["sensorType"],
                swath_identifier=common["swathIdentifier"],
                wrs_longitude_grid=common["wrsLongitudeGrid"],
            ),
            product=product_model,
            sources=source_models,
        )
