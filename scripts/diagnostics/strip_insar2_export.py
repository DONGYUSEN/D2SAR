from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from osgeo import gdal, osr

from common_processing import (
    compute_utm_output_shape,
    write_geocoded_geotiff,
    write_geocoded_png,
    write_wrapped_phase_geotiff,
    write_wrapped_phase_png,
)


def write_ground_overlay_kml(
    *,
    image_path: str | Path,
    output_kml: str | Path,
    west: float,
    east: float,
    south: float,
    north: float,
    overlay_name: str | None = None,
) -> str:
    image_path = Path(image_path)
    output_kml = Path(output_kml)
    output_kml.parent.mkdir(parents=True, exist_ok=True)
    name = overlay_name or image_path.stem
    href = escape(image_path.name)
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <GroundOverlay>
    <name>{escape(name)}</name>
    <Icon>
      <href>{href}</href>
    </Icon>
    <LatLonBox>
      <north>{float(north)}</north>
      <south>{float(south)}</south>
      <east>{float(east)}</east>
      <west>{float(west)}</west>
    </LatLonBox>
  </GroundOverlay>
</kml>
"""
    output_kml.write_text(content, encoding="utf-8")
    return str(output_kml)


def _to_geographic_bounds(
    *,
    projection_wkt: str,
    west: float,
    east: float,
    south: float,
    north: float,
) -> tuple[float, float, float, float]:
    src = osr.SpatialReference()
    src.ImportFromWkt(projection_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    tx = osr.CoordinateTransformation(src, dst)
    corners = [
        tx.TransformPoint(float(west), float(north)),
        tx.TransformPoint(float(east), float(north)),
        tx.TransformPoint(float(east), float(south)),
        tx.TransformPoint(float(west), float(south)),
    ]
    lons = [float(c[0]) for c in corners]
    lats = [float(c[1]) for c in corners]
    return min(lons), max(lons), min(lats), max(lats)


def write_ground_overlay_kml_from_geotiff(
    *,
    tif_path: str | Path,
    image_path: str | Path,
    output_kml: str | Path,
    overlay_name: str | None = None,
) -> str:
    tif_path = Path(tif_path)
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open GeoTIFF for KML bounds: {tif_path}")
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        ds = None
        raise RuntimeError(f"GeoTIFF missing geotransform: {tif_path}")
    projection = ds.GetProjectionRef()
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    ds = None

    west = float(gt[0])
    east = float(gt[0] + gt[1] * width)
    north = float(gt[3])
    south = float(gt[3] + gt[5] * height)
    west, east = sorted((west, east))
    south, north = sorted((south, north))

    if projection:
        west, east, south, north = _to_geographic_bounds(
            projection_wkt=projection,
            west=west,
            east=east,
            south=south,
            north=north,
        )

    return write_ground_overlay_kml(
        image_path=image_path,
        output_kml=output_kml,
        west=west,
        east=east,
        south=south,
        north=north,
        overlay_name=overlay_name,
    )


def export_insar_products(
    *,
    input_h5: str | Path,
    output_paths: dict[str, str],
    resolution_meters: float,
    block_rows: int = 64,
    generate_kml: bool = True,
) -> dict[str, str]:
    input_h5 = str(input_h5)
    target_width, target_height = compute_utm_output_shape(input_h5, resolution_meters)
    exported: dict[str, str] = {}

    scalar_datasets = (
        ("avg_amplitude", "avg_amplitude_tif", "avg_amplitude_png"),
        ("coherence", "coherence_tif", "coherence_png"),
        ("unwrapped_phase", "unwrapped_phase_tif", "unwrapped_phase_png"),
        ("los_displacement", "los_displacement_tif", "los_displacement_png"),
    )
    for dataset_name, tif_key, png_key in scalar_datasets:
        tif_path = output_paths[tif_key]
        png_path = output_paths[png_key]
        write_geocoded_geotiff(
            input_h5,
            tif_path,
            dataset_name=dataset_name,
            target_width=target_width,
            target_height=target_height,
            block_rows=block_rows,
        )
        write_geocoded_png(
            input_h5,
            png_path,
            dataset_name=dataset_name,
            target_width=target_width,
            target_height=target_height,
            block_rows=block_rows,
        )
        exported[tif_key] = tif_path
        exported[png_key] = png_path
        if generate_kml:
            kml_key = png_key.replace("_png", "_kml")
            exported[kml_key] = write_ground_overlay_kml_from_geotiff(
                tif_path=tif_path,
                image_path=png_path,
                output_kml=output_paths[kml_key],
                overlay_name=Path(png_path).stem,
            )

    wrapped_datasets = (
        ("interferogram", "interferogram_tif", "interferogram_png"),
        ("filtered_interferogram", "filtered_interferogram_tif", "filtered_interferogram_png"),
    )
    for dataset_name, tif_key, png_key in wrapped_datasets:
        tif_path = output_paths[tif_key]
        png_path = output_paths[png_key]
        write_wrapped_phase_geotiff(
            input_h5,
            tif_path,
            dataset_name=dataset_name,
            target_width=target_width,
            target_height=target_height,
            block_rows=block_rows,
        )
        write_wrapped_phase_png(
            input_h5,
            png_path,
            dataset_name=dataset_name,
            target_width=target_width,
            target_height=target_height,
            block_rows=block_rows,
        )
        exported[tif_key] = tif_path
        exported[png_key] = png_path
        if generate_kml:
            kml_key = png_key.replace("_png", "_kml")
            exported[kml_key] = write_ground_overlay_kml_from_geotiff(
                tif_path=tif_path,
                image_path=png_path,
                output_kml=output_paths[kml_key],
                overlay_name=Path(png_path).stem,
            )

    return exported
