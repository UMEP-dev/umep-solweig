__author__ = "xlinfr"

from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from osgeo.gdalconst import GDT_Float32
from pandas import read_csv, to_datetime
import os
import re

try:
    import pyproj
    import rasterio

    from rasterio.transform import Affine, from_origin
    from shapely import geometry

    GDAL_ENV = False

except:
    from osgeo import gdal
    GDAL_ENV = True

# Slope and aspect used in SEBE and Wall aspect
def get_ders(dsm, scale):
    # dem,_,_=read_dem_grid(dem_file)
    dx = 1 / scale
    # dx=0.5
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, "rad")
    slope = np.arctan(grad)
    asp = asp * -1
    asp = asp + (asp < 0) * (np.pi * 2)
    return slope, asp


def cart2pol(x, y, units="deg"):
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ["deg", "degs"]:
        theta = theta * 180 / np.pi
    return theta, radius


def saveraster(gdal_data, filename, raster):
    rows = gdal_data.RasterYSize
    cols = gdal_data.RasterXSize

    outDs = gdal.GetDriverByName("GTiff").Create(
        filename, cols, rows, int(1), GDT_Float32
    )
    outBand = outDs.GetRasterBand(1)

    # write the data
    outBand.WriteArray(raster, 0, 0)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-9999)

    # georeference the image and set the projection
    outDs.SetGeoTransform(gdal_data.GetGeoTransform())
    outDs.SetProjection(gdal_data.GetProjection())


def saverasternd(gdal_data, filename, raster):
    rows = gdal_data.RasterYSize
    cols = gdal_data.RasterXSize

    outDs = gdal.GetDriverByName("GTiff").Create(
        filename, cols, rows, int(1), GDT_Float32
    )
    outBand = outDs.GetRasterBand(1)

    # write the data
    outBand.WriteArray(raster, 0, 0)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    # outBand.SetNoDataValue(-9999)

    # georeference the image and set the projection
    outDs.SetGeoTransform(gdal_data.GetGeoTransform())
    outDs.SetProjection(gdal_data.GetProjection())


def xy2latlon(crsWtkIn, x, y):
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(crsWtkIn)

    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""

    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    transform = osr.CoordinateTransformation(old_cs, new_cs)

    lonlat = transform.TransformPoint(x, y)

    gdalver = float(gdal.__version__[0])
    if gdalver >= 3.0:
        lon = lonlat[1]  # changed to gdal 3
        lat = lonlat[0]  # changed to gdal 3
    else:
        lon = lonlat[0]  # changed to gdal 2
        lat = lonlat[1]  # changed to gdal 2

    return lat, lon


def createTSlist():
    from datetime import datetime, timezone
    from zoneinfo import available_timezones, ZoneInfo

    # Current aware UTC datetime
    now_utc = datetime.now(timezone.utc)

    timezones_by_offset = {}

    for tz_name in sorted(available_timezones()):
        try:
            tz = ZoneInfo(tz_name)

            # Convert UTC to local timezone
            local_time = now_utc.astimezone(tz)
            offset = local_time.utcoffset()

            if offset is None:
                continue

            total_minutes = int(offset.total_seconds() // 60)
            hours, minutes = divmod(abs(total_minutes), 60)
            sign = "+" if total_minutes >= 0 else "-"

            offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
            offset_hours = total_minutes / 60

            if offset_str not in timezones_by_offset:
                timezones_by_offset[offset_str] = {
                    "utc_offset": offset_hours,
                    "timezones": [],
                }

            # Add up to 3 example zones per offset
            if len(timezones_by_offset[offset_str]["timezones"]) < 3:
                timezones_by_offset[offset_str]["timezones"].append(tz_name)

        except Exception:
            # Some special zones may fail; ignore safely
            continue

    timezones_list = [
        {
            "utc_offset_str": offset,
            "utc_offset": data["utc_offset"],
            "timezones": data["timezones"],
        }
        for offset, data in timezones_by_offset.items()
    ]

    sorted_timezones = sorted(timezones_list, key=lambda x: x["utc_offset"])

    return sorted_timezones, timezones_by_offset


def check_path(path_str: str | Path, make_dir: bool = False) -> Path:
    # Ensure path exists
    path = Path(path_str).absolute()
    if not path.parent.exists():
        if make_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Parent directory {path} does not exist. Set make_dir=True to create it.")
    if not path.exists() and not path.suffix:
        if make_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Path {path} does not exist. Set make_dir=True to create it.")
    return path

def save_raster(
    out_path_str: str, data_arr: np.ndarray, trf_arr: list[float], crs_wkt: str, no_data_val: float = -9999
):
    attempts = 2
    while attempts > 0:
        attempts -= 1
        try:
            # Save raster using GDAL or rasterio
            out_path = check_path(out_path_str, make_dir=True)
            height, width = data_arr.shape
            if GDAL_ENV is False:
                trf = Affine.from_gdal(*trf_arr)
                crs = None
                if crs_wkt:
                    crs = pyproj.CRS(crs_wkt)
                with rasterio.open(
                    out_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=data_arr.dtype,
                    crs=crs,
                    transform=trf,
                    nodata=no_data_val,
                ) as dst:
                    dst.write(data_arr, 1)
            else:
                # Map numpy dtype to GDAL type
                dtype_map = {
                    np.dtype("uint8"): gdal.GDT_Byte,
                    np.dtype("int16"): gdal.GDT_Int16,
                    np.dtype("uint16"): gdal.GDT_UInt16,
                    np.dtype("int32"): gdal.GDT_Int32,
                    np.dtype("uint32"): gdal.GDT_UInt32,
                    np.dtype("float32"): gdal.GDT_Float32,
                    np.dtype("float64"): gdal.GDT_Float64,
                }
                gdal_dtype = dtype_map.get(data_arr.dtype, gdal.GDT_Float32)
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(out_path), width, height, 1, gdal_dtype)
                # trf is a list: [top left x, w-e pixel size, rotation, top left y, rotation, n-s pixel size]
                ds.SetGeoTransform(tuple(trf_arr))
                # GetProjection returns WKT (string)
                if crs_wkt:
                    ds.SetProjection(crs_wkt)
                band = ds.GetRasterBand(1)
                band.WriteArray(data_arr, 0, 0)
                band.SetNoDataValue(no_data_val)
                ds.FlushCache()
                ds = None
                return# Success, exit the function
        except Exception as e:
            print(f"Error saving raster, attempts left {attempts}: {e}")
            if attempts == 0:
                raise e


def get_resolution_from_file(folder_path):

    suews_pattern = re.compile(r".*_(\d{4})_SUEWS")

    # Step 1: Find the first matching SUEWS file
    suews_file = None
    for filename in os.listdir(folder_path):
        if suews_pattern.match(filename):
            suews_file = os.path.join(folder_path, filename)
            break

    # Step 2: Load the file and compute time resolution
    df_suews = read_csv(suews_file, delim_whitespace=True)

    # Combine Year, DOY, Hour, Min into a datetime index
    df_suews["Datetime"] = to_datetime(
        df_suews[["Year", "DOY", "Hour", "Min"]]
        .astype(str)
        .agg("-".join, axis=1),
        format="%Y-%j-%H-%M",
    )
    df_suews.set_index("Datetime", inplace=True)

    # Step 3: Compute resolution (in seconds)
    time1 = df_suews.index[1]
    time2 = df_suews.index[2]
    input_resolution = int((time2 - time1).total_seconds())

    return input_resolution


def SUEWS_txt_to_df(suews_output_path):
    df_output_suews = read_csv(suews_output_path, delim_whitespace=True)
    df_output_suews["Datetime"] = to_datetime(
        df_output_suews[["Year", "DOY", "Hour", "Min"]]
        .astype(str)
        .agg("-".join, axis=1),
        format="%Y-%j-%H-%M",
    )
    df_output_suews.set_index("Datetime", inplace=True)

    return df_output_suews


def SUEWS_met_txt_to_df(suews_met_path):
    df_met_forcing = read_csv(suews_met_path, delim_whitespace=True)
    df_met_forcing["Datetime"] = to_datetime(
        df_met_forcing[["iy", "id", "it", "imin"]]
        .astype(str)
        .agg("-".join, axis=1),
        format="%Y-%j-%H-%M",
    )
    df_met_forcing.set_index("Datetime", inplace=True)

    return df_met_forcing


def extract_suews_years(folder_path):
    # Match: anything + underscore + 4-digit year + _SUEWS
    suews_pattern = re.compile(r".*_(\d{4})_SUEWS")

    years = set()
    for filename in os.listdir(folder_path):
        match = suews_pattern.match(filename)
        if match:
            years.add(match.group(1))

    return sorted(years)


def xy2latlon_fromraster(crsWtkIn, gdal_dsm):
    # Get latlon from grid coordinate system
    old_cs = osr.SpatialReference()
    # dsm_ref = dsmlayer.crs().toWkt()
    old_cs.ImportFromWkt(crsWtkIn)

    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""

    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    transform = osr.CoordinateTransformation(old_cs, new_cs)
    widthx = gdal_dsm.RasterXSize
    heightx = gdal_dsm.RasterYSize
    geotransform = gdal_dsm.GetGeoTransform()
    minx = geotransform[0]
    miny = (
        geotransform[3] + widthx * geotransform[4] + heightx * geotransform[5]
    )

    lonlat = transform.TransformPoint(minx, miny)
    gdalver = float(gdal.__version__[0])
    if gdalver == 3.0:
        lon = lonlat[1]  # changed to gdal 3
        lat = lonlat[0]  # changed to gdal 3
    else:
        lon = lonlat[0]  # changed to gdal 2
        lat = lonlat[1]  # changed to gdal 2
    scale = 1 / geotransform[1]

    return lat, lon, scale, minx, miny
