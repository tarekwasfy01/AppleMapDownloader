#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Py Map Stitcher 3.

Use only with map/tile servers for which you have permission. Many public map
providers prohibit bulk downloading. The app intentionally uses a conservative
rate limit and requires user-supplied/custom URL templates.
"""

import concurrent.futures as cf
import dataclasses
import io
import json
import subprocess
import tempfile
import math
import os
import queue
import random
import shutil
import sys
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as exc:  # pragma: no cover
    requests = None

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:  # pragma: no cover
    Image = None

TILE_SIZE = 256
APPLE_LEFT_BAR_CROP_PX = 300
# Apple Frame keeps a fixed left control pane before the real map canvas.
# z13 seam checks show a remaining about-1px overlap with 284px, so use the
# slightly narrower measured map pane.
APPLE_MAP_PANE_LEFT_PX = 285.0
FIXED_FRAME_SHIFT_X_PX = 0.0
FIXED_FRAME_SHIFT_Y_PX = 0.0
APPLE_FRAME_MAX_EFFECTIVE_ZOOM = 19
USER_AGENT = "PyMapStitcher/1.0 (+local user tool)"
MAX_INFLIGHT_PER_WORKER = 4  # prevents millions of Futures in RAM
HARD_TILE_WARNING = 5_000_000
DEFAULT_CHUNK_SIZE = 64
MAX_DIRECT_TIFF_BYTES = 1_000_000_000_000  # 1 TB safety limit for sparse BigTIFF output
MAX_FRAME_SCREENSHOT_CELLS = 50_000
MAX_FRAME_SCREENSHOT_BYTES = 250_000_000_000
APPLE_SELECTOR_MIN_ZOOM = 8

MAP_PRESETS = {
    "Custom": {
        "url": "https://your-tile-server.example/{z}/{x}/{y}.png",
        "note": "Enter a custom URL template manually.",
        "preview": True,
    },
    "Own Frame Server / Screenshot TIFF": {
        "url": "http://127.0.0.1:8787/frame?center={center_lat},{center_lon}&span={lat_span},{lon_span}&z={z}",
        "note": "For your own/authorized frame server. Renders URLs in Qt WebEngine, crops UI, saves TIFF tiles, and writes a stitched GeoTIFF/BigTIFF.",
        "preview": "frame",
    },
    "Apple Frame Preview / center-span helper": {
        "url": "https://maps.apple.com/frame?map=satellite&center={center_lat}%2C{center_lon}&span={lat_span}%2C{lon_span}",
        "note": "Preview/helper only: shows the Apple-style center/span frame URL so you can inspect the same coordinate logic. Use downloading only with your own or otherwise authorized frame server by switching the URL to that server.",
        "preview": "frame",
    },
    "Google Satellite": {
        "url": "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&hl=de",
        "note": "Google Satellite. Respect the terms of use; no bulk downloading without permission.",
        "preview": True,
    },
    "Google Hybrid": {
        "url": "https://mt.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&hl=de",
        "note": "Google Satellite with labels. Respect the terms of use.",
        "preview": True,
    },
    "Bing Satellite": {
        "url": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=14574&mkt=de-DE&n=z",
        "note": "Bing Aerial/Satellite via QuadKey {q}. Respect the terms of use.",
        "preview": True,
    },
    "Bing Hybrid": {
        "url": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/h{q}.jpeg?g=14574&mkt=de-DE&n=z",
        "note": "Bing Hybrid via QuadKey {q}. Respect the terms of use.",
        "preview": True,
    },
    "Esri World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "note": "Satellite/aerial tiles. Respect Esri terms of use.",
        "preview": True,
    },
    "OpenStreetMap Mapnik": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "note": "OSM standard map. Respect the terms of use; no bulk downloading.",
        "preview": True,
    },
    "OpenTopoMap": {
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "note": "Topographic map. Respect the terms of use.",
        "preview": True,
    },
    "CartoDB Positron": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "note": "Light basemap. Respect the terms of use.",
        "preview": True,
    },
    "NoniMapView Legacy: Google Satellite": {
        "url": "http://khm{rnd}.google.com/kh/v=47&x={x}&y={y}&z={z}&s=&hl=de",
        "note": "Legacy NoniMapView profile; may be outdated or blocked today.",
        "preview": True,
    },
    "NoniMapView Legacy: Google Road": {
        "url": "http://mt{rnd}.google.com/vt/lyrs=m&hl=de&x={x}&y={y}&z={z}",
        "note": "Legacy NoniMapView profile; may be outdated or blocked today.",
        "preview": True,
    },
}



@dataclasses.dataclass(frozen=True)
class TileJob:
    x: int
    y: int
    z: int
    col: int
    row: int


@dataclasses.dataclass
class StitchConfig:
    url_template: str
    output_file: Path
    z: int
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    workers: int = 8
    rate_limit_ms: int = 50
    retries: int = 3
    timeout: int = 20
    headers: Optional[Dict[str, str]] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE


def clamp_lat(lat: float) -> float:
    return max(min(lat, 85.05112878), -85.05112878)


def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    lat = clamp_lat(lat)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_to_lonlat(x: float, y: float, z: int) -> Tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat



def lonlat_to_world_pixel(lon: float, lat: float, z: int) -> Tuple[float, float]:
    lat = clamp_lat(float(lat))
    world = float(TILE_SIZE) * (2 ** int(z))
    x = (float(lon) + 180.0) / 360.0 * world
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * world
    return x, y


def world_pixel_to_lonlat(px: float, py: float, z: int) -> Tuple[float, float]:
    world = float(TILE_SIZE) * (2 ** int(z))
    lon = float(px) / world * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(py) / world)))
    lat = math.degrees(lat_rad)
    return lon, lat


def world_pixel_bbox_to_lonlat(px_left: float, py_top: float, px_right: float, py_bottom: float, z: int) -> Tuple[float, float, float, float]:
    west, north = world_pixel_to_lonlat(px_left, py_top, z)
    east, south = world_pixel_to_lonlat(px_right, py_bottom, z)
    return west, south, east, north

def tile_bounds_for_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float, z: int):
    # NW and SE tile indices for Web Mercator XYZ.
    x1, y1 = lonlat_to_tile(min_lon, max_lat, z)
    x2, y2 = lonlat_to_tile(max_lon, min_lat, z)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def tile_to_quadkey(x: int, y: int, z: int) -> str:
    q = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        q.append(str(digit))
    return "".join(q)


def expand_url(template: str, x: int, y: int, z: int) -> str:
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)
    q = tile_to_quadkey(x, y, z)

    west_lon, north_lat = tile_to_lonlat(x, y, z)
    east_lon, south_lat = tile_to_lonlat(x + 1, y + 1, z)
    center_lon = (west_lon + east_lon) / 2.0
    center_lat = (north_lat + south_lat) / 2.0
    lon_span = abs(east_lon - west_lon)
    lat_span = abs(north_lat - south_lat)
    bbox = f"{west_lon:.12f},{south_lat:.12f},{east_lon:.12f},{north_lat:.12f}"

    # Supports normal XYZ URLs and custom frame/snapshot URLs for own servers.
    return (template.replace("{x}", str(x))
                    .replace("{y}", str(y))
                    .replace("{z}", str(z))
                    .replace("{c}", str(z))
                    .replace("{q}", q)
                    .replace("{quadkey}", q)
                    .replace("{rnd}", str(rnd))
                    .replace("{snum}", snum)
                    .replace("{s}", sub)
                    .replace("{west}", f"{west_lon:.12f}")
                    .replace("{south}", f"{south_lat:.12f}")
                    .replace("{east}", f"{east_lon:.12f}")
                    .replace("{north}", f"{north_lat:.12f}")
                    .replace("{center_lon}", f"{center_lon:.12f}")
                    .replace("{center_lat}", f"{center_lat:.12f}")
                    .replace("{lon_span}", f"{lon_span:.12f}")
                    .replace("{lat_span}", f"{lat_span:.12f}")
                    .replace("{span_lon}", f"{lon_span:.12f}")
                    .replace("{span_lat}", f"{lat_span:.12f}")
                    .replace("{bbox}", bbox)
                    .replace("*GMX*", str(x))
                    .replace("*GMY*", str(y))
                    .replace("*ZM1*", str(z))
                    .replace("*IZM*", str(z))
                    .replace("*RND*", str(rnd))
                    .replace("*LAN*", "de")
                    .replace("*LAN-LAN*", "de-DE"))






def expand_frame_url_for_screenshot_job(
    template: str,
    x: int,
    y: int,
    z: int,
    render_w: int,
    render_h: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    step_factor_x: float = 1.0,
    step_factor_y: float = 1.0,
) -> str:
    """Expand frame URL so the cropped screenshot lands on the intended mosaic tile.

    This version supports an additional tile step/offset factor. Some frame
    providers do not interpret span exactly like XYZ Web-Mercator tile size, or
    UI/canvas scaling makes neighbouring screenshots too close. In that case
    the requested center must move farther than one XYZ tile. step_factor_x/y
    multiplies the tile-center offset relative to the selected mosaic origin.
    """
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)
    q = tile_to_quadkey(x, y, z)

    west_lon, north_lat = tile_to_lonlat(x, y, z)
    east_lon, south_lat = tile_to_lonlat(x + 1, y + 1, z)

    tile_center_lon = (west_lon + east_lon) / 2.0
    tile_center_lat = (north_lat + south_lat) / 2.0
    base_lon_span = abs(east_lon - west_lon)
    base_lat_span = abs(north_lat - south_lat)

    render_w = max(1, int(render_w))
    render_h = max(1, int(render_h))
    crop_left = max(0, int(crop_left))
    crop_top = max(0, int(crop_top))
    crop_right = max(0, int(crop_right))
    crop_bottom = max(0, int(crop_bottom))

    visible_w = max(1, render_w - crop_left - crop_right)
    visible_h = max(1, render_h - crop_top - crop_bottom)

    scale_x = max(1.0, float(render_w) / float(visible_w))
    scale_y = max(1.0, float(render_h) / float(visible_h))

    requested_lon_span = base_lon_span * scale_x
    requested_lat_span = base_lat_span * scale_y

    # User-adjustable extra spacing between screenshot centers.
    # Values > 1 move centres farther apart. This fixes "too close together".
    step_factor_x = max(0.1, float(step_factor_x))
    step_factor_y = max(0.1, float(step_factor_y))

    # The local offset from this tile's normal center to its scaled spacing center
    # is calculated by the caller relative to a mosaic origin using placeholders
    # x/y. Since this function does not know x_min/y_min, it exposes the factors
    # by shifting from the integer tile coordinate origin:
    # center = tile origin lonlat + (x fractional centre * factor).
    # For regular WebMercator tiles this is equivalent to increasing the centre
    # distance between consecutive x/y jobs.
    west0, north0 = tile_to_lonlat(0, 0, z)
    # For lon, each tile is constant width.
    lon_tile_w = 360.0 / (2 ** int(z))
    center_lon = -180.0 + ((float(x) + 0.5) * lon_tile_w * step_factor_x)

    # For lat, tile height is not constant in degrees. Approximate step around
    # the current tile using local base_lat_span.
    # y increases downward, lat decreases downward.
    # Use normal tile center plus additional local offset from factor.
    center_lat = tile_center_lat - ((float(y) + 0.5) * 0.0)  # start normal
    if abs(step_factor_y - 1.0) > 1e-9:
        # Additional shift relative to tile index, local lat span approximation.
        # This deliberately moves rows farther apart when screenshots are too close.
        center_lat = tile_center_lat - ((float(y) - float(y)) * base_lat_span)  # no-op anchor
        # A local row spacing correction is applied by changing span and crop offset
        # unless mosaic-origin stepping is supplied by caller; here keep normal
        # latitude to avoid drifting globally. Caller can still use span multiplier.
        pass

    # The lon formula above is global and can jump if factor != 1; for actual
    # mosaic stepping we replace it below when origin placeholders are not used.
    # To avoid global drift, default to tile center and use explicit step shifts
    # injected by caller via optional placeholders when available.
    center_lon = tile_center_lon

    crop_center_x = (crop_left + (render_w - crop_right)) / 2.0
    crop_center_y = (crop_top + (render_h - crop_bottom)) / 2.0
    dx_px = crop_center_x - (render_w / 2.0)
    dy_px = crop_center_y - (render_h / 2.0)

    center_lon = center_lon - (dx_px / float(render_w)) * requested_lon_span
    center_lat = center_lat + (dy_px / float(render_h)) * requested_lat_span

    west_adj = center_lon - requested_lon_span / 2.0
    east_adj = center_lon + requested_lon_span / 2.0
    south_adj = center_lat - requested_lat_span / 2.0
    north_adj = center_lat + requested_lat_span / 2.0
    bbox = f"{west_adj:.12f},{south_adj:.12f},{east_adj:.12f},{north_adj:.12f}"

    return (template.replace("{x}", str(x))
                    .replace("{y}", str(y))
                    .replace("{z}", str(z))
                    .replace("{c}", str(z))
                    .replace("{q}", q)
                    .replace("{quadkey}", q)
                    .replace("{rnd}", str(rnd))
                    .replace("{snum}", snum)
                    .replace("{s}", sub)
                    .replace("{west}", f"{west_adj:.12f}")
                    .replace("{south}", f"{south_adj:.12f}")
                    .replace("{east}", f"{east_adj:.12f}")
                    .replace("{north}", f"{north_adj:.12f}")
                    .replace("{center_lon}", f"{center_lon:.12f}")
                    .replace("{center_lat}", f"{center_lat:.12f}")
                    .replace("{lon_span}", f"{requested_lon_span:.12f}")
                    .replace("{lat_span}", f"{requested_lat_span:.12f}")
                    .replace("{span_lon}", f"{requested_lon_span:.12f}")
                    .replace("{span_lat}", f"{requested_lat_span:.12f}")
                    .replace("{base_lon_span}", f"{base_lon_span:.12f}")
                    .replace("{base_lat_span}", f"{base_lat_span:.12f}")
                    .replace("{crop_scale_x}", f"{scale_x:.8f}")
                    .replace("{crop_scale_y}", f"{scale_y:.8f}")
                    .replace("{crop_dx_px}", f"{dx_px:.3f}")
                    .replace("{crop_dy_px}", f"{dy_px:.3f}")
                    .replace("{step_factor_x}", f"{step_factor_x:.8f}")
                    .replace("{step_factor_y}", f"{step_factor_y:.8f}")
                    .replace("{bbox}", bbox)
                    .replace("*GMX*", str(x))
                    .replace("*GMY*", str(y))
                    .replace("*ZM1*", str(z))
                    .replace("*IZM*", str(z))
                    .replace("*RND*", str(rnd))
                    .replace("*LAN*", "de")
                    .replace("*LAN-LAN*", "de-DE"))


def expand_frame_url_for_screenshot_job_with_origin(
    template: str,
    x: int,
    y: int,
    z: int,
    x_min: int,
    y_min: int,
    render_w: int,
    render_h: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    step_factor_x: float = 0.0,
    step_factor_y: float = 0.0,
) -> str:
    """Origin-aware frame URL expansion using screen/WebView size.

    Automatic math:
    - The final tile is the cropped visible part of the WebView.
    - If the full renderer is 1920 px wide and crop_left=300, crop_right=0,
      visible_w = 1620.
    - To make the cropped visible area cover exactly one output tile, the
      requested span must be enlarged by 1920 / 1620.
    - The next screenshot center must move by that enlarged requested span,
      not by the smaller XYZ tile span. Otherwise screenshots are too close.

    Therefore:
        scale_x = render_w / (render_w - crop_left - crop_right)
        scale_y = render_h / (render_h - crop_top - crop_bottom)

        requested_span_x = xyz_tile_span_x * scale_x
        requested_span_y = xyz_tile_span_y * scale_y

        center_x = origin_center_x + (x - x_min) * requested_span_x
        center_y = origin_center_y - (y - y_min) * requested_span_y

    If step_factor_x/y is > 0, it overrides the automatic scale. This keeps a
    manual emergency control, but default 0 means fully automatic.
    """
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)
    q = tile_to_quadkey(x, y, z)

    # Origin tile spans and center.
    ow, on = tile_to_lonlat(x_min, y_min, z)
    oe, os_ = tile_to_lonlat(x_min + 1, y_min + 1, z)
    origin_base_lon_span = abs(oe - ow)
    origin_base_lat_span = abs(on - os_)
    origin_center_lon = (ow + oe) / 2.0
    origin_center_lat = (on + os_) / 2.0

    # Current tile local span. Latitude degree size changes with y.
    west_lon, north_lat = tile_to_lonlat(x, y, z)
    east_lon, south_lat = tile_to_lonlat(x + 1, y + 1, z)
    base_lon_span = abs(east_lon - west_lon)
    base_lat_span = abs(north_lat - south_lat)

    render_w = max(1, int(render_w))
    render_h = max(1, int(render_h))
    crop_left = max(0, int(crop_left))
    crop_top = max(0, int(crop_top))
    crop_right = max(0, int(crop_right))
    crop_bottom = max(0, int(crop_bottom))

    visible_w = max(1, render_w - crop_left - crop_right)
    visible_h = max(1, render_h - crop_top - crop_bottom)

    auto_scale_x = max(1.0, float(render_w) / float(visible_w))
    auto_scale_y = max(1.0, float(render_h) / float(visible_h))

    # Manual override only if user sets > 0. Otherwise screen-size automatic.
    scale_x = float(step_factor_x) if float(step_factor_x) > 0.0 else auto_scale_x
    scale_y = float(step_factor_y) if float(step_factor_y) > 0.0 else auto_scale_y
    scale_x = max(0.1, scale_x)
    scale_y = max(0.1, scale_y)

    requested_lon_span = base_lon_span * scale_x
    requested_lat_span = base_lat_span * scale_y

    # The center step must be based on the requested frame coverage, not the
    # smaller cropped tile coverage.
    step_lon = origin_base_lon_span * scale_x
    # For latitude, use local span for current row to reduce row drift.
    # Move from origin by each row's approximate requested geographic coverage.
    step_lat = origin_base_lat_span * scale_y

    center_lon = origin_center_lon + (float(x - x_min) * step_lon)
    center_lat = origin_center_lat - (float(y - y_min) * step_lat)

    # Asymmetric crop shifts the center of the cropped rectangle inside the full
    # renderer. Move requested center in the opposite direction so the cropped
    # center lands on the intended mosaic cell.
    crop_center_x = (crop_left + (render_w - crop_right)) / 2.0
    crop_center_y = (crop_top + (render_h - crop_bottom)) / 2.0
    dx_px = crop_center_x - (render_w / 2.0)
    dy_px = crop_center_y - (render_h / 2.0)

    center_lon = center_lon - (dx_px / float(render_w)) * requested_lon_span
    center_lat = center_lat + (dy_px / float(render_h)) * requested_lat_span

    west_adj = center_lon - requested_lon_span / 2.0
    east_adj = center_lon + requested_lon_span / 2.0
    south_adj = center_lat - requested_lat_span / 2.0
    north_adj = center_lat + requested_lat_span / 2.0
    bbox = f"{west_adj:.12f},{south_adj:.12f},{east_adj:.12f},{north_adj:.12f}"

    return (template.replace("{x}", str(x))
                    .replace("{y}", str(y))
                    .replace("{z}", str(z))
                    .replace("{c}", str(z))
                    .replace("{q}", q)
                    .replace("{quadkey}", q)
                    .replace("{rnd}", str(rnd))
                    .replace("{snum}", snum)
                    .replace("{s}", sub)
                    .replace("{west}", f"{west_adj:.12f}")
                    .replace("{south}", f"{south_adj:.12f}")
                    .replace("{east}", f"{east_adj:.12f}")
                    .replace("{north}", f"{north_adj:.12f}")
                    .replace("{center_lon}", f"{center_lon:.12f}")
                    .replace("{center_lat}", f"{center_lat:.12f}")
                    .replace("{lon_span}", f"{requested_lon_span:.12f}")
                    .replace("{lat_span}", f"{requested_lat_span:.12f}")
                    .replace("{span_lon}", f"{requested_lon_span:.12f}")
                    .replace("{span_lat}", f"{requested_lat_span:.12f}")
                    .replace("{base_lon_span}", f"{base_lon_span:.12f}")
                    .replace("{base_lat_span}", f"{base_lat_span:.12f}")
                    .replace("{crop_scale_x}", f"{scale_x:.8f}")
                    .replace("{crop_scale_y}", f"{scale_y:.8f}")
                    .replace("{auto_crop_scale_x}", f"{auto_scale_x:.8f}")
                    .replace("{auto_crop_scale_y}", f"{auto_scale_y:.8f}")
                    .replace("{crop_dx_px}", f"{dx_px:.3f}")
                    .replace("{crop_dy_px}", f"{dy_px:.3f}")
                    .replace("{step_lon}", f"{step_lon:.12f}")
                    .replace("{step_lat}", f"{step_lat:.12f}")
                    .replace("{bbox}", bbox)
                    .replace("*GMX*", str(x))
                    .replace("*GMY*", str(y))
                    .replace("*ZM1*", str(z))
                    .replace("*IZM*", str(z))
                    .replace("*RND*", str(rnd))
                    .replace("*LAN*", "de")
                    .replace("*LAN-LAN*", "de-DE"))


def frame_span_for_center_zoom(lon: float, lat: float, z: int) -> Tuple[float, float]:
    """Return lat_span, lon_span for the Web-Mercator tile containing lon/lat at z."""
    x, y = lonlat_to_tile(lon, lat, z)
    west_lon, north_lat = tile_to_lonlat(x, y, z)
    east_lon, south_lat = tile_to_lonlat(x + 1, y + 1, z)
    return abs(north_lat - south_lat), abs(east_lon - west_lon)

def expand_frame_url_center_span(template: str, center_lon: float, center_lat: float, z: int) -> str:
    """Expand a frame/snapshot URL for an exact center coordinate.

    Unlike expand_url(), this does not snap the preview to the center of an XYZ
    tile. It keeps the current preview center exactly and only derives a useful
    Web-Mercator tile-sized span from the zoom level. This makes Apple-style
    preview frames and own frame-server previews land at the expected place.
    """
    x_float = (float(center_lon) + 180.0) / 360.0 * (2 ** int(z))
    lat = clamp_lat(float(center_lat))
    lat_rad = math.radians(lat)
    y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2 ** int(z))

    west_lon, north_lat = tile_to_lonlat(x_float - 0.5, y_float - 0.5, int(z))
    east_lon, south_lat = tile_to_lonlat(x_float + 0.5, y_float + 0.5, int(z))
    lon_span = abs(east_lon - west_lon)
    lat_span = abs(north_lat - south_lat)
    bbox = f"{west_lon:.12f},{south_lat:.12f},{east_lon:.12f},{north_lat:.12f}"
    x_int, y_int = lonlat_to_tile(float(center_lon), float(center_lat), int(z))

    return (template.replace("{x}", str(x_int))
                    .replace("{y}", str(y_int))
                    .replace("{z}", str(int(z)))
                    .replace("{c}", str(int(z)))
                    .replace("{west}", f"{west_lon:.12f}")
                    .replace("{south}", f"{south_lat:.12f}")
                    .replace("{east}", f"{east_lon:.12f}")
                    .replace("{north}", f"{north_lat:.12f}")
                    .replace("{center_lon}", f"{float(center_lon):.12f}")
                    .replace("{center_lat}", f"{float(center_lat):.12f}")
                    .replace("{lon_span}", f"{lon_span:.12f}")
                    .replace("{lat_span}", f"{lat_span:.12f}")
                    .replace("{span_lon}", f"{lon_span:.12f}")
                    .replace("{span_lat}", f"{lat_span:.12f}")
                    .replace("{bbox}", bbox))



def expand_frame_url_exact_bbox(
    template: str,
    west_lon: float,
    south_lat: float,
    east_lon: float,
    north_lat: float,
    z: int,
) -> str:
    """Expand a frame URL so Preview Selected BBox shows the exact bbox.

    The older preview helper used center+zoom and derived a one-tile span. That
    made Apple/frame preview jump back to a tiny start area when View/Preview BBox
    was clicked. This helper uses the left coordinate fields directly:
        center = bbox center
        span   = bbox size
    """
    west_lon = float(west_lon)
    south_lat = clamp_lat(float(south_lat))
    east_lon = float(east_lon)
    north_lat = clamp_lat(float(north_lat))
    if east_lon <= west_lon or north_lat <= south_lat:
        raise ValueError("invalid bbox for exact frame preview")

    center_lon = (west_lon + east_lon) / 2.0
    center_lat = (south_lat + north_lat) / 2.0
    lon_span = abs(east_lon - west_lon)
    lat_span = abs(north_lat - south_lat)
    bbox = f"{west_lon:.12f},{south_lat:.12f},{east_lon:.12f},{north_lat:.12f}"
    x_int, y_int = lonlat_to_tile(center_lon, center_lat, int(z))
    q = tile_to_quadkey(x_int, y_int, int(z))
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)

    return (template.replace("{x}", str(x_int))
                    .replace("{y}", str(y_int))
                    .replace("{z}", str(int(z)))
                    .replace("{c}", str(int(z)))
                    .replace("{q}", q)
                    .replace("{quadkey}", q)
                    .replace("{rnd}", str(rnd))
                    .replace("{snum}", snum)
                    .replace("{s}", sub)
                    .replace("{west}", f"{west_lon:.12f}")
                    .replace("{south}", f"{south_lat:.12f}")
                    .replace("{east}", f"{east_lon:.12f}")
                    .replace("{north}", f"{north_lat:.12f}")
                    .replace("{center_lon}", f"{center_lon:.12f}")
                    .replace("{center_lat}", f"{center_lat:.12f}")
                    .replace("{lon_span}", f"{lon_span:.12f}")
                    .replace("{lat_span}", f"{lat_span:.12f}")
                    .replace("{span_lon}", f"{lon_span:.12f}")
                    .replace("{span_lat}", f"{lat_span:.12f}")
                    .replace("{bbox}", bbox)
                    .replace("*GMX*", str(x_int))
                    .replace("*GMY*", str(y_int))
                    .replace("*ZM1*", str(int(z)))
                    .replace("*IZM*", str(int(z)))
                    .replace("*RND*", str(rnd))
                    .replace("*LAN*", "de")
                    .replace("*LAN-LAN*", "de-DE"))


def frame_view_bbox_for_center_zoom_pixels(
    center_lon: float,
    center_lat: float,
    z: int,
    width_px: float,
    height_px: float,
) -> Tuple[float, float, float, float]:
    """Return the lon/lat bbox covered by a Web-Mercator pixel viewport.

    This is the important Apple/frame fix: the preview fallback is no longer a
    single XYZ tile. A 1600 px renderer at z=18 should represent about 1600
    Web-Mercator pixels, not only 256. The optional UI multiplier can make the
    selectable preview cover several renderer-cells so Mark Area can create a
    multi-cell download instead of always collapsing to 1 x 1.
    """
    z = int(z)
    cx_px, cy_px = lonlat_to_world_pixel(float(center_lon), float(center_lat), z)
    half_w = max(1.0, float(width_px)) / 2.0
    half_h = max(1.0, float(height_px)) / 2.0
    return world_pixel_bbox_to_lonlat(cx_px - half_w, cy_px - half_h, cx_px + half_w, cy_px + half_h, z)

def is_frame_template(url_template: str) -> bool:
    markers = ("{center_lat}", "{center_lon}", "{lat_span}", "{lon_span}", "{span_lat}", "{span_lon}", "{bbox}")
    return any(m in url_template for m in markers)


def is_apple_frame_template(url_template: str) -> bool:
    return "maps.apple.com" in (url_template or "").lower()



def project_tiles_dir(output_file: Path) -> Path:
    return output_file.parent / f"{output_file.stem}_tiles"

def project_sqlite_dir(output_file: Path) -> Path:
    return output_file.parent / f"{output_file.stem}_sqlite"

def project_single_tiff_dir(output_file: Path) -> Path:
    return output_file.parent / f"{output_file.stem}_single_tiff_tiles"

def safe_cache_path(cache_dir: Path, z: int, x: int, y: int) -> Path:
    # Dateiname enthält jetzt ausdrücklich Zoom, X und Y.
    # Dadurch sieht man auch nach einem Abbruch sofort, welche Kachel vorhanden ist.
    return cache_dir / str(z) / f"z{z}_x{x}_y{y}.tile"


def default_tile_tif_dir(cfg: "StitchConfig") -> Path:
    base = cfg.output_file.parent if cfg.output_file.parent else Path.cwd()
    stem = cfg.output_file.stem or "map_output"
    return base / f"{stem}_einzelkacheln_tif_z{cfg.z}"


def safe_tile_tif_path(tile_tif_dir: Path, z: int, x: int, y: int) -> Path:
    return tile_tif_dir / f"z{z}_x{x}_y{y}.tif"


def lonlat_to_webmercator(lon: float, lat: float) -> Tuple[float, float]:
    lat = clamp_lat(lat)
    r = 6378137.0
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def tile_webmercator_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    west_lon, north_lat = tile_to_lonlat(x, y, z)
    east_lon, south_lat = tile_to_lonlat(x + 1, y + 1, z)
    west, north = lonlat_to_webmercator(west_lon, north_lat)
    east, south = lonlat_to_webmercator(east_lon, south_lat)
    return west, south, east, north


def mosaic_webmercator_bounds(x_min: int, y_min: int, x_max: int, y_max: int, z: int) -> Tuple[float, float, float, float]:
    west_lon, north_lat = tile_to_lonlat(x_min, y_min, z)
    east_lon, south_lat = tile_to_lonlat(x_max + 1, y_max + 1, z)
    west, north = lonlat_to_webmercator(west_lon, north_lat)
    east, south = lonlat_to_webmercator(east_lon, south_lat)
    return west, south, east, north



def lonlat_bbox_to_webmercator_bounds(west_lon: float, south_lat: float, east_lon: float, north_lat: float) -> Tuple[float, float, float, float]:
    west, north = lonlat_to_webmercator(float(west_lon), float(north_lat))
    east, south = lonlat_to_webmercator(float(east_lon), float(south_lat))
    return west, south, east, north


def expand_frame_url_grid(
    template: str,
    col: int,
    row: int,
    z: int,
    selected_west: float,
    selected_north: float,
    request_lon_span: float,
    request_lat_span: float,
    lon_per_px: float,
    lat_per_px: float,
    visible_w_px: int,
    visible_h_px: int,
    render_w: int,
    render_h: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    step_mult_x: float = 1.0,
    step_mult_y: float = 1.0,
    shift_x_px: float = 0.0,
    shift_y_px: float = 0.0,
    crop_correct_url: bool = False,
) -> Tuple[str, Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """World-pixel screenshot grid with explicit step multiplier.

    Default is NoDoubleCrop mode: the Apple/frame URL span is the visible
    output cell. Crop is only an image extraction step. This is the mode that
    keeps neighbouring Apple frame screenshots visually aligned on systems where
    Apple interprets span for the map canvas rather than the whole WebView.

    crop_correct_url=True keeps the experimental mode where the requested URL
    span is enlarged by the crop margins. That can be useful for some own frame
    servers, but it makes Apple frames drift on many setups.
    """
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)

    render_w = max(1, int(render_w))
    render_h = max(1, int(render_h))
    crop_left = max(0, int(crop_left))
    crop_top = max(0, int(crop_top))
    crop_right = max(0, int(crop_right))
    crop_bottom = max(0, int(crop_bottom))
    visible_w_px = max(1, int(visible_w_px))
    visible_h_px = max(1, int(visible_h_px))
    step_mult_x = max(0.01, float(step_mult_x))
    step_mult_y = max(0.01, float(step_mult_y))
    shift_x_px = float(shift_x_px)
    shift_y_px = float(shift_y_px)
    z = int(z)

    selected_px_x, selected_px_y = lonlat_to_world_pixel(float(selected_west), float(selected_north), z)

    # Manual fine tuning:
    # multiplier gives coarse overlap/spacing, shift_x/y adds/subtracts pixels per cell.
    # Negative shift = closer screenshots / more overlap. Positive shift = farther apart.
    effective_step_x_px = max(1.0, float(visible_w_px) * step_mult_x + shift_x_px)
    effective_step_y_px = max(1.0, float(visible_h_px) * step_mult_y + shift_y_px)

    # Target visible cell: THIS is the part that is written into the output raster.
    visible_left_px = selected_px_x + float(col) * effective_step_x_px
    visible_top_px = selected_px_y + float(row) * effective_step_y_px
    visible_right_px = visible_left_px + float(visible_w_px)
    visible_bottom_px = visible_top_px + float(visible_h_px)

    visible_west, visible_south, visible_east, visible_north = world_pixel_bbox_to_lonlat(
        visible_left_px, visible_top_px, visible_right_px, visible_bottom_px, z
    )

    # Unified crop-center correction.
    #
    # The visible output cell is the part that remains after PIL crops the
    # captured WebView. The URL center must therefore be shifted by the center
    # offset of that crop rectangle inside the full renderer:
    #
    #   x-shift = (crop_right  - crop_left) / 2
    #   y-shift = (crop_bottom - crop_top)  / 2
    #
    # This is the same principle that made left-crop work, but now it is applied
    # symmetrically to right, top and bottom as well. Example:
    #   left=300,right=100 -> center shifts -100 px
    #   left=0,right=100   -> center shifts +50 px
    #   top=0,bottom=100   -> center shifts +50 px downward
    visible_center_x_px = visible_left_px + (float(visible_w_px) / 2.0)
    visible_center_y_px = visible_top_px + (float(visible_h_px) / 2.0)
    crop_center_shift_x_px = (float(crop_right) - float(crop_left)) / 2.0
    crop_center_shift_y_px = (float(crop_bottom) - float(crop_top)) / 2.0

    if bool(crop_correct_url):
        # Apple/frame mode: the geographic map is not rendered across the full
        # WebView. It starts after the left Apple side pane. The measured z18/z20
        # outputs show an effective map pane of about 1316 px in a 1600 px
        # renderer, i.e. map-left ~= 284 px. Compute the URL center from the
        # center of the saved crop inside that real map pane.
        map_left_px = max(0.0, min(float(render_w - 1), float(APPLE_MAP_PANE_LEFT_PX)))
        map_span_px = max(1.0, min(float(render_w) - map_left_px, float(render_h)))
        crop_center_x_px = (float(crop_left) + (float(render_w) - float(crop_right))) / 2.0
        crop_center_y_px = (float(crop_top) + (float(render_h) - float(crop_bottom))) / 2.0
        map_center_x_px = map_left_px + (map_span_px / 2.0)
        map_center_y_px = float(render_h) / 2.0
        request_center_x_px = visible_center_x_px - (crop_center_x_px - map_center_x_px)
        request_center_y_px = visible_center_y_px - (crop_center_y_px - map_center_y_px)
        request_w_px = map_span_px
        request_h_px = map_span_px
        request_left_px = request_center_x_px - (request_w_px / 2.0)
        request_top_px = request_center_y_px - (request_h_px / 2.0)
        request_right_px = request_center_x_px + (request_w_px / 2.0)
        request_bottom_px = request_center_y_px + (request_h_px / 2.0)

        request_west, request_south, request_east, request_north = world_pixel_bbox_to_lonlat(
            request_left_px, request_top_px, request_right_px, request_bottom_px, z
        )
    else:
        # Legacy NoDoubleCrop mode kept as a fallback. It also uses the same
        # center-shift variables for diagnostics/placeholders, but keeps the URL
        # span equal to the output cell.
        request_west, request_south, request_east, request_north = (
            visible_west, visible_south, visible_east, visible_north
        )
        request_center_x_px = visible_center_x_px
        request_center_y_px = visible_center_y_px

    request_center_lon, request_center_lat = world_pixel_to_lonlat(request_center_x_px, request_center_y_px, z)

    real_request_lon_span = abs(request_east - request_west)
    real_request_lat_span = abs(request_north - request_south)

    next_center_lon, _ = world_pixel_to_lonlat(request_center_x_px + effective_step_x_px, request_center_y_px, z)
    _, next_center_lat = world_pixel_to_lonlat(request_center_x_px, request_center_y_px + effective_step_y_px, z)
    center_step_lon = abs(next_center_lon - request_center_lon)
    center_step_lat = abs(request_center_lat - next_center_lat)

    bbox = f"{request_west:.12f},{request_south:.12f},{request_east:.12f},{request_north:.12f}"
    q = ""
    try:
        q = tile_to_quadkey(max(0, int(col)), max(0, int(row)), z)
    except Exception:
        q = ""

    url = (template.replace("{x}", str(int(col)))
                   .replace("{y}", str(int(row)))
                   .replace("{z}", str(z))
                   .replace("{c}", str(z))
                   .replace("{q}", q)
                   .replace("{quadkey}", q)
                   .replace("{rnd}", str(rnd))
                   .replace("{snum}", snum)
                   .replace("{s}", sub)
                   .replace("{west}", f"{request_west:.12f}")
                   .replace("{south}", f"{request_south:.12f}")
                   .replace("{east}", f"{request_east:.12f}")
                   .replace("{north}", f"{request_north:.12f}")
                   .replace("{center_lon}", f"{request_center_lon:.12f}")
                   .replace("{center_lat}", f"{request_center_lat:.12f}")
                   .replace("{lon_span}", f"{real_request_lon_span:.12f}")
                   .replace("{lat_span}", f"{real_request_lat_span:.12f}")
                   .replace("{span_lon}", f"{real_request_lon_span:.12f}")
                   .replace("{span_lat}", f"{real_request_lat_span:.12f}")
                   .replace("{visible_west}", f"{visible_west:.12f}")
                   .replace("{visible_south}", f"{visible_south:.12f}")
                   .replace("{visible_east}", f"{visible_east:.12f}")
                   .replace("{visible_north}", f"{visible_north:.12f}")
                   .replace("{visible_center_lon}", f"{(visible_west + visible_east) / 2.0:.12f}")
                   .replace("{visible_center_lat}", f"{(visible_south + visible_north) / 2.0:.12f}")
                   .replace("{request_center_x_px}", f"{request_center_x_px:.3f}")
                   .replace("{request_center_y_px}", f"{request_center_y_px:.3f}")
                   .replace("{center_step_x_px}", f"{effective_step_x_px:.3f}")
                   .replace("{center_step_y_px}", f"{effective_step_y_px:.3f}")
                   .replace("{center_step_lon}", f"{center_step_lon:.12f}")
                   .replace("{center_step_lat}", f"{center_step_lat:.12f}")
                   .replace("{step_mult_x}", f"{step_mult_x:.4f}")
                   .replace("{step_mult_y}", f"{step_mult_y:.4f}")
                   .replace("{shift_x_px}", f"{shift_x_px:.3f}")
                   .replace("{shift_y_px}", f"{shift_y_px:.3f}")
                   .replace("{crop_center_shift_x_px}", f"{crop_center_shift_x_px:.3f}")
                   .replace("{crop_center_shift_y_px}", f"{crop_center_shift_y_px:.3f}")
                   .replace("{crop_correct_url}", "1" if bool(crop_correct_url) else "0")
                   .replace("{visible_w_px}", str(int(visible_w_px)))
                   .replace("{visible_h_px}", str(int(visible_h_px)))
                   .replace("{bbox}", bbox)
                   .replace("*GMX*", str(int(col)))
                   .replace("*GMY*", str(int(row)))
                   .replace("*ZM1*", str(z))
                   .replace("*IZM*", str(z))
                   .replace("*RND*", str(rnd))
                   .replace("*LAN*", "de")
                   .replace("*LAN-LAN*", "de-DE"))

    # QWebEngine/Apple frame can otherwise reuse an old frame. Force each cell to
    # be a unique URL while keeping the original center/span/bbox parameters intact.
    url = append_url_param(url, "__pymap_tile", f"{z}_{int(col)}_{int(row)}_{int(time.time()*1000)}_{rnd}")

    return url, (visible_west, visible_south, visible_east, visible_north), (request_west, request_south, request_east, request_north)





def cropped_frame_bounds_from_request_bounds(
    request_bounds: Tuple[float, float, float, float],
    z: int,
    render_w: int,
    render_h: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> Tuple[float, float, float, float]:
    """Return the geographic bbox of the actually cropped screenshot area.

    request_bounds is the full Apple/frame WebView request bbox in lon/lat
    order (west, south, east, north). This helper converts that full frame to
    Web-Mercator world pixels, removes the crop margins in the same pixel space,
    and converts the remaining cropped rectangle back to lon/lat.

    This is the safest georeferencing mode for Apple frame captures because the
    GeoTIFF is tied to the image that is actually written after crop, not to the
    marker bbox or to the synthetic screenshot step.
    """
    west, south, east, north = map(float, request_bounds)
    z = int(z)
    render_w = max(1, int(render_w))
    render_h = max(1, int(render_h))
    crop_left = max(0, int(crop_left))
    crop_top = max(0, int(crop_top))
    crop_right = max(0, int(crop_right))
    crop_bottom = max(0, int(crop_bottom))

    left_px, top_px = lonlat_to_world_pixel(west, north, z)
    right_px, bottom_px = lonlat_to_world_pixel(east, south, z)
    full_w_px = max(1e-9, right_px - left_px)
    full_h_px = max(1e-9, bottom_px - top_px)

    cropped_left_px = left_px + (float(crop_left) / float(render_w)) * full_w_px
    cropped_top_px = top_px + (float(crop_top) / float(render_h)) * full_h_px
    cropped_right_px = right_px - (float(crop_right) / float(render_w)) * full_w_px
    cropped_bottom_px = bottom_px - (float(crop_bottom) / float(render_h)) * full_h_px
    if cropped_right_px <= cropped_left_px or cropped_bottom_px <= cropped_top_px:
        return (west, south, east, north)
    return world_pixel_bbox_to_lonlat(cropped_left_px, cropped_top_px, cropped_right_px, cropped_bottom_px, z)


def center_georef_bounds_from_request_bounds(
    request_bounds: Tuple[float, float, float, float],
    z: int,
    visible_w_px: int,
    visible_h_px: int,
) -> Tuple[float, float, float, float]:
    """Georeference one cropped Apple screenshot from the Apple URL center.

    The Apple frame URL is center/span based. For georeferencing we therefore
    treat the URL center as the center of the saved cropped tile and derive the
    tile bounds from the final cropped output size in Web-Mercator pixels.
    This deliberately ignores the theoretical selector bbox and avoids
    double-applying crop offsets to the GeoTIFF.
    """
    west, south, east, north = map(float, request_bounds)
    z = int(z)
    visible_w_px = max(1, int(visible_w_px))
    visible_h_px = max(1, int(visible_h_px))

    left_px, top_px = lonlat_to_world_pixel(west, north, z)
    right_px, bottom_px = lonlat_to_world_pixel(east, south, z)
    center_x_px = (left_px + right_px) / 2.0
    center_y_px = (top_px + bottom_px) / 2.0

    return world_pixel_bbox_to_lonlat(
        center_x_px - (float(visible_w_px) / 2.0),
        center_y_px - (float(visible_h_px) / 2.0),
        center_x_px + (float(visible_w_px) / 2.0),
        center_y_px + (float(visible_h_px) / 2.0),
        z,
    )


def center_georef_grid_bounds_from_first_request(
    request_bounds: Tuple[float, float, float, float],
    z: int,
    cols: int,
    rows: int,
    visible_w_px: int,
    visible_h_px: int,
    effective_step_x_px: float,
    effective_step_y_px: float,
) -> Tuple[float, float, float, float]:
    """Return full mosaic bounds from the first Apple URL center.

    The screenshot URLs move by the effective center step
    (visible size * step multiplier + fixed frame shift). The GeoTIFF bounds
    must use the same center spacing; otherwise the mosaic is scaled/shifted
    when FrameShift is non-zero.
    """
    west, south, east, north = map(float, request_bounds)
    z = int(z)
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    visible_w_px = max(1, int(visible_w_px))
    visible_h_px = max(1, int(visible_h_px))
    effective_step_x_px = max(1.0, float(effective_step_x_px))
    effective_step_y_px = max(1.0, float(effective_step_y_px))

    left_px, top_px = lonlat_to_world_pixel(west, north, z)
    right_px, bottom_px = lonlat_to_world_pixel(east, south, z)
    center_x_px = (left_px + right_px) / 2.0
    center_y_px = (top_px + bottom_px) / 2.0

    grid_left_px = center_x_px - (float(visible_w_px) / 2.0)
    grid_top_px = center_y_px - (float(visible_h_px) / 2.0)
    grid_right_px = center_x_px + (float(cols - 1) * effective_step_x_px) + (float(visible_w_px) / 2.0)
    grid_bottom_px = center_y_px + (float(rows - 1) * effective_step_y_px) + (float(visible_h_px) / 2.0)

    return world_pixel_bbox_to_lonlat(grid_left_px, grid_top_px, grid_right_px, grid_bottom_px, z)


def ensure_python_package(import_name: str, pip_name: Optional[str] = None, log_cb=None):
    """Import a package, installing it with pip on demand.

    This keeps the single-file app convenient on Windows: when the new PyPI
    stitching merge is enabled, the missing package is installed automatically
    into the current Python environment.
    """
    import importlib
    pip_name = pip_name or import_name
    try:
        return importlib.import_module(import_name)
    except Exception as first_exc:
        if log_cb:
            log_cb(f"Missing Python package '{pip_name}'. Installing with pip...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", pip_name]
        try:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if log_cb and proc.stdout:
                for line in proc.stdout.splitlines()[-40:]:
                    log_cb(line)
            if proc.returncode != 0:
                raise RuntimeError(f"pip install {pip_name} failed with exit code {proc.returncode}")
            return importlib.import_module(import_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not import or install '{pip_name}'. First import error: {first_exc}; install error: {exc}"
            ) from exc


def sorted_frame_tile_paths(tile_dir: Path) -> List[Path]:
    """Return frame grid tiles in deterministic row-major order."""
    import re
    items = []
    for path in Path(tile_dir).glob("grid_z*_col*_row*.tif"):
        m = re.search(r"_col(\d+)_row(\d+)\.tif$", path.name)
        if not m:
            continue
        col = int(m.group(1))
        row = int(m.group(2))
        items.append((row, col, path))
    items.sort(key=lambda v: (v[0], v[1]))
    return [p for _row, _col, p in items]


def save_numpy_rgb_as_geotiff(out_file: Path, rgb_array, bounds_3857: Tuple[float, float, float, float], log_cb=None) -> None:
    """Save an RGB numpy array as embedded EPSG:3857 GeoTIFF/BigTIFF."""
    try:
        import numpy as np
        import tifffile
    except Exception as exc:
        raise RuntimeError("tifffile and numpy are required for stitched GeoTIFF output") from exc

    arr = np.asarray(rgb_array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected stitched image shape: {arr.shape}")
    if arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype("uint8", copy=False)

    height, width = int(arr.shape[0]), int(arr.shape[1])
    estimated = width * height * 3
    ensure_enough_disk_space(out_file, estimated, log_cb or (lambda _m: None))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp.tif")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    tifffile.imwrite(
        str(tmp),
        arr,
        bigtiff=estimated > 3_800_000_000,
        photometric="rgb",
        metadata=None,
        extratags=geotiff_extratags_epsg3857(width, height, bounds_3857),
        compression="deflate",
    )
    os.replace(tmp, out_file)
    write_worldfile_and_prj(out_file, width, height, bounds_3857)


def stitch_frame_tiles_with_pypi_stitching(
    tile_dir: Path,
    out_file: Path,
    bounds_3857: Tuple[float, float, float, float],
    log_cb=None,
    max_images: int = 500,
) -> bool:
    """Merge captured frame tiles with the PyPI 'stitching' package.

    Returns True if the panorama was produced. This is intentionally optional:
    if feature stitching fails because the map area has too little overlap or
    too few visual features, the existing grid BigTIFF remains available.
    """
    paths = sorted_frame_tile_paths(tile_dir)
    if len(paths) < 2:
        if log_cb:
            log_cb("PyPI stitching skipped: fewer than 2 frame tiles.")
        return False
    if len(paths) > int(max_images):
        if log_cb:
            log_cb(
                f"PyPI stitching skipped: {len(paths)} images is too much for OpenCV feature stitching. "
                f"Use a smaller area or raise max_images in code."
            )
        return False

    ensure_python_package("cv2", "opencv-python", log_cb)
    stitching_mod = ensure_python_package("stitching", "stitching", log_cb)

    # Prefer the affine stitcher for screenshot grids/linear map movement. It
    # tolerates planar translations better than pure panorama camera geometry.
    StitcherClass = getattr(stitching_mod, "AffineStitcher", None) or getattr(stitching_mod, "Stitcher")
    if log_cb:
        log_cb(f"PyPI stitching merge: {len(paths)} TIFF tiles, class={StitcherClass.__name__}")
        log_cb("Hinweis: Stitching braucht echte Überlappung/Features. Pixel step X/Y sollte eher 0.70-0.95 sein, nicht 1.00 oder 4.00.")

    # The package accepts filenames directly.
    stitcher = StitcherClass(detector="sift", confidence_threshold=0.2)
    panorama = stitcher.stitch([str(p) for p in paths])
    if panorama is None:
        raise RuntimeError("PyPI stitching returned no panorama")

    # stitching/OpenCV returns BGR. Convert to RGB before GeoTIFF.
    import cv2
    try:
        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    except Exception:
        panorama_rgb = panorama

    save_numpy_rgb_as_geotiff(out_file, panorama_rgb, bounds_3857, log_cb)
    if log_cb:
        log_cb(f"PyPI stitching finished and wrote georeferenced GeoTIFF/BigTIFF: {out_file}")
    return True


def write_worldfile_and_prj(tif_path: Path, width: int, height: int, bounds_3857: Tuple[float, float, float, float]) -> None:
    # Minimal-invasive Georeferenzierung: Der ursprüngliche TIFF/BigTIFF-Schreibweg bleibt unverändert.
    # QGIS/GIS liest die Georeferenz über .tfw + .prj neben der TIFF-Datei.
    west, south, east, north = bounds_3857
    px_w = (east - west) / float(width)
    px_h = (south - north) / float(height)
    tfw = tif_path.with_suffix(".tfw")
    prj = tif_path.with_suffix(".prj")
    tfw.write_text(
        f"{px_w:.12f}\n0.0\n0.0\n{px_h:.12f}\n{west + px_w / 2.0:.12f}\n{north + px_h / 2.0:.12f}\n",
        encoding="utf-8",
    )
    prj.write_text(
        'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3857"]]',
        encoding="utf-8",
    )


def save_tile_as_tif(data: Optional[bytes], out_path: Path, z: int, x: int, y: int) -> None:
    # Schreibt genau eine erzeugte Kachel sofort als TIFF.
    # Vorhandene TIFF-Tiles werden nicht erneut geschrieben.
    if out_path.exists() and out_path.stat().st_size > 100:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = decode_tile(data)
    tmp = out_path.with_suffix(".tmp.tif")
    im.save(tmp, format="TIFF", compression="tiff_deflate")
    os.replace(tmp, out_path)
    write_worldfile_and_prj(out_path, TILE_SIZE, TILE_SIZE, tile_webmercator_bounds(x, y, z))


def download_one(job: TileJob, cfg: StitchConfig, stop_event: threading.Event) -> Tuple[TileJob, Optional[bytes], Optional[str]]:
    """Download one tile without persistent cache.

    The tile bytes are returned to the stitcher and are never written to a raw
    tile cache folder. Resume/SQLite caching is intentionally disabled so the
    only persistent output is the streamed BigTIFF.
    """
    if stop_event.is_set():
        return job, None, "cancelled"
    if requests is None:
        return job, None, "requests is not installed"
    url = expand_url(cfg.url_template, job.x, job.y, job.z)
    headers = {"User-Agent": USER_AGENT}
    if cfg.headers:
        headers.update(cfg.headers)
    last_err = None
    for attempt in range(cfg.retries):
        if stop_event.is_set():
            return job, None, "cancelled"
        try:
            if cfg.rate_limit_ms:
                time.sleep(cfg.rate_limit_ms / 1000.0)
            r = requests.get(url, headers=headers, timeout=cfg.timeout, stream=True)
            r.raise_for_status()
            data = r.content
            if len(data) < 50:
                raise RuntimeError("empty/invalid tile")
            return job, data, None
        except Exception as exc:
            last_err = str(exc)
            time.sleep(0.5 * (attempt + 1))
    return job, None, last_err

def make_blank_tile() -> "Image.Image":
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))


def decode_tile(data: Optional[bytes]) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow is not installed")
    if not data:
        return make_blank_tile()
    try:
        im = Image.open(io.BytesIO(data))
        return im.convert("RGB").resize((TILE_SIZE, TILE_SIZE))
    except Exception:
        return make_blank_tile()




def iter_tile_jobs(x_min: int, y_min: int, x_max: int, y_max: int, z: int):
    # Generator statt Liste: selbst riesige Bereiche erzeugen keine RAM-Spitze.
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            yield TileJob(x, y, z, x - x_min, y - y_min)


def count_existing_tiles(cache_dir: Path, x_min: int, y_min: int, x_max: int, y_max: int, z: int) -> int:
    existing = 0
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            p = safe_cache_path(cache_dir, z, x, y)
            if p.exists() and p.stat().st_size > 100:
                existing += 1
    return existing



def sqlite_path_for(cfg: StitchConfig) -> Path:
    return cfg.cache_dir / f"download_state_z{cfg.z}.sqlite"


def init_state_db(cfg: StitchConfig):
    if not cfg.use_sqlite:
        return None
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(sqlite_path_for(cfg)), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS tiles (z INTEGER, x INTEGER, y INTEGER, status TEXT, updated REAL, error TEXT, PRIMARY KEY(z,x,y))")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (z INTEGER, x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER, status TEXT, updated REAL, PRIMARY KEY(z,x0,y0,x1,y1))")
    db.commit()
    return db


def db_tile_done(db, z: int, x: int, y: int) -> bool:
    if db is None:
        return False
    row = db.execute("SELECT status FROM tiles WHERE z=? AND x=? AND y=?", (z, x, y)).fetchone()
    return bool(row and row[0] == "done")


def db_mark_tile(db, z: int, x: int, y: int, status: str, error: Optional[str] = None):
    if db is None:
        return
    db.execute("INSERT OR REPLACE INTO tiles(z,x,y,status,updated,error) VALUES(?,?,?,?,?,?)", (z, x, y, status, time.time(), error))


def db_mark_chunk(db, z: int, x0: int, y0: int, x1: int, y1: int, status: str):
    if db is None:
        return
    db.execute("INSERT OR REPLACE INTO chunks(z,x0,y0,x1,y1,status,updated) VALUES(?,?,?,?,?,?,?)", (z, x0, y0, x1, y1, status, time.time()))
    db.commit()


def iter_chunks(x_min: int, y_min: int, x_max: int, y_max: int, chunk_size: int):
    """Spatial chunk scheduler. Yields chunk bounds only; never builds a global tile list."""
    chunk_size = max(1, int(chunk_size))
    for cy in range(y_min, y_max + 1, chunk_size):
        for cx in range(x_min, x_max + 1, chunk_size):
            yield cx, cy, min(cx + chunk_size - 1, x_max), min(cy + chunk_size - 1, y_max)


def iter_chunk_jobs(cx0: int, cy0: int, cx1: int, cy1: int, z: int, x_min: int, y_min: int):
    """Yields jobs for one chunk only."""
    for y in range(cy0, cy1 + 1):
        for x in range(cx0, cx1 + 1):
            yield TileJob(x, y, z, x - x_min, y - y_min)


def format_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def ensure_enough_disk_space(path: Path, required_bytes: int, log_cb) -> None:
    """Raise before creating the BigTIFF when the target drive is too small."""
    target_dir = path.expanduser().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(str(target_dir))
    except Exception as exc:
        raise RuntimeError(f"Could not check free disk space for {target_dir}: {exc}") from exc
    # tifffile metadata and filesystem allocation need a little headroom.
    required_with_margin = int(required_bytes * 1.03) + 512 * 1024 * 1024
    log_cb(f"Estimated raw BigTIFF payload: {format_bytes(required_bytes)}")
    log_cb(f"Free space on target drive: {format_bytes(usage.free)}")
    if usage.free < required_with_margin:
        raise RuntimeError(
            "Not enough free disk space for direct BigTIFF streaming. "
            f"Required with safety margin: {format_bytes(required_with_margin)}; "
            f"available: {format_bytes(usage.free)}. Choose a smaller area/zoom or another drive."
        )


def geotiff_extratags_epsg3857(width: int, height: int, bounds_3857: Tuple[float, float, float, float]):
    """Return embedded GeoTIFF tags for EPSG:3857 / Web Mercator.

    This writes georeferencing into the TIFF itself, so QGIS can place the
    BigTIFF without depending on .tfw/.prj sidecar files.
    """
    west, south, east, north = bounds_3857
    px_w = (east - west) / float(width)
    px_h = (north - south) / float(height)
    model_pixel_scale = (float(px_w), float(px_h), 0.0)
    # Raster coordinate (0,0,0) is tied to the top-left model coordinate.
    model_tiepoint = (0.0, 0.0, 0.0, float(west), float(north), 0.0)
    # GeoKeyDirectoryTag: header + GTModelTypeGeoKey(Projected),
    # GTRasterTypeGeoKey(PixelIsArea), ProjectedCSTypeGeoKey(EPSG:3857).
    geo_key_directory = (
        1, 1, 0, 3,
        1024, 0, 1, 1,
        1025, 0, 1, 1,
        3072, 0, 1, 3857,
    )
    return [
        (33550, "d", 3, model_pixel_scale, False),
        (33922, "d", 6, model_tiepoint, False),
        (34735, "H", len(geo_key_directory), geo_key_directory, False),
    ]


def open_direct_bigtiff(cfg: StitchConfig, width: int, height: int, bounds_3857: Tuple[float, float, float, float], log_cb):
    """Create a georeferenced on-disk BigTIFF memmap or raise a clear error.

    There is no cache fallback in this build. Georeferencing is embedded as
    GeoTIFF tags, not only written as .tfw/.prj sidecars.
    """
    estimated = int(width) * int(height) * 3
    ensure_enough_disk_space(cfg.output_file, estimated, log_cb)
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("tifffile is required for direct BigTIFF streaming. Install with: pip install tifffile") from exc
    try:
        cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
        bigtiff = estimated > 3_800_000_000
        extratags = geotiff_extratags_epsg3857(width, height, bounds_3857)
        mem = tifffile.memmap(
            str(cfg.output_file),
            shape=(height, width, 3),
            dtype="uint8",
            bigtiff=bigtiff,
            photometric="rgb",
            metadata=None,
            extratags=extratags,
        )
        log_cb(f"Direct GeoTIFF/BigTIFF writer opened: {cfg.output_file}")
        log_cb("Embedded GeoTIFF georeferencing written: EPSG:3857, ModelPixelScaleTag, ModelTiepointTag, GeoKeyDirectoryTag")
        return mem, "memmap"
    except OSError as exc:
        raise RuntimeError(
            "Direct BigTIFF output could not be created. This is usually caused by not enough disk space, "
            f"permission problems, or a path/drive limit. Target: {cfg.output_file}. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Direct GeoTIFF/BigTIFF writer failed: {exc}") from exc

def stitch_tiles(cfg: StitchConfig, progress_cb, log_cb, stop_event: threading.Event):
    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow requests")

    x_min, y_min, x_max, y_max = tile_bounds_for_bbox(cfg.min_lat, cfg.min_lon, cfg.max_lat, cfg.max_lon, cfg.z)
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    total = cols * rows
    width = cols * TILE_SIZE
    height = rows * TILE_SIZE
    chunk_size = max(1, int(cfg.chunk_size))

    log_cb(f"Tile range: x={x_min}..{x_max}, y={y_min}..{y_max}")
    log_cb(f"Tiles: {cols} x {rows} = {total:,}")
    log_cb(f"Image size: {width:,} x {height:,} px")
    log_cb(f"Direct BigTIFF streaming active: chunk size {chunk_size} x {chunk_size} tiles")
    log_cb("CPU-only HTTP tile stitching. CUDA/CuPy is removed.")
    log_cb("No raw tile cache, no SQLite resume database, and no separate TIFF tile output will be created.")

    if total > HARD_TILE_WARNING:
        log_cb(f"Warning: very large selection with {total:,} tiles.")

    bounds_3857 = mosaic_webmercator_bounds(x_min, y_min, x_max, y_max, cfg.z)
    direct_mem, direct_kind = open_direct_bigtiff(cfg, width, height, bounds_3857, log_cb)

    max_workers = max(1, cfg.workers)
    max_inflight = max_workers * MAX_INFLIGHT_PER_WORKER
    done = 0
    errors = 0

    try:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for cx0, cy0, cx1, cy1 in iter_chunks(x_min, y_min, x_max, y_max, chunk_size):
                if stop_event.is_set():
                    break
                log_cb(f"Chunk start: x={cx0}..{cx1}, y={cy0}..{cy1}")
                job_iter = iter_chunk_jobs(cx0, cy0, cx1, cy1, cfg.z, x_min, y_min)
                pending = set()
                while not stop_event.is_set():
                    while len(pending) < max_inflight:
                        try:
                            job = next(job_iter)
                        except StopIteration:
                            break
                        pending.add(pool.submit(download_one, job, cfg, stop_event))
                    if not pending:
                        break
                    done_set, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for fut in done_set:
                        job, data, err = fut.result()
                        done += 1
                        if err:
                            errors += 1
                            if errors <= 30:
                                log_cb(f"Error {job.z}/{job.x}/{job.y}: {err}")
                        else:
                            try:
                                tile_arr = tile_bytes_to_numpy_rgb(data)
                                r0 = job.row * TILE_SIZE
                                c0 = job.col * TILE_SIZE
                                direct_mem[r0:r0+TILE_SIZE, c0:c0+TILE_SIZE, :] = tile_arr
                            except Exception as exc:
                                errors += 1
                                if errors <= 30:
                                    log_cb(f"Write error {job.z}/{job.x}/{job.y}: {exc}")
                        if done % 25 == 0 or done == total:
                            progress_cb(done, total, "Stream")
                try:
                    direct_mem.flush()
                except Exception:
                    pass
    finally:
        if direct_mem is not None:
            try:
                direct_mem.flush()
                del direct_mem
            except Exception:
                pass

    if stop_event.is_set():
        log_cb("Stopped. Partial BigTIFF remains at the output path.")
        return

    log_cb(f"Finished direct BigTIFF streaming. Processed: {done:,}; errors: {errors:,}")
    log_cb(f"Finished BigTIFF/direct output: {cfg.output_file}")

def open_folder_in_file_manager(path: Path) -> None:
    """Open a folder in the OS file manager. Safe no-op if it cannot be opened."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass



# -----------------------------------------------------------------------------
# PySide6 integrated WebEngine GUI
# -----------------------------------------------------------------------------
try:
    from PySide6.QtCore import QObject, QTimer, Qt, QUrl, Slot, QEvent, QPoint, QRect, QSize
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
        QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
        QMessageBox, QPushButton, QProgressBar, QRubberBand, QSpinBox, QSplitter, QTextEdit,
        QVBoxLayout, QWidget
    )
    from PySide6.QtGui import QIcon
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
except Exception as _pyside_exc:  # pragma: no cover
    QObject = object  # type: ignore
    QMainWindow = object  # type: ignore
    QApplication = None  # type: ignore
    QTimer = None  # type: ignore
    QUrl = None  # type: ignore
    QEvent = None  # type: ignore
    QPoint = object  # type: ignore
    QRect = object  # type: ignore
    QWebEngineView = None  # type: ignore
    QWebChannel = None  # type: ignore
    QRubberBand = None  # type: ignore
    QIcon = None  # type: ignore
    QWebEngineProfile = None  # type: ignore
    QWebEngineSettings = None  # type: ignore

    class _DummyQt:
        class Orientation:
            Horizontal = 1
        class WindowType:
            Window = 1
        class WidgetAttribute:
            WA_DeleteOnClose = 1
            WA_TranslucentBackground = 2
        class CursorShape:
            CrossCursor = 1
        class MouseButton:
            LeftButton = 1
            RightButton = 2
        class KeyboardModifier:
            ShiftModifier = 1
        class Key:
            Key_Escape = 1
    Qt = _DummyQt()  # type: ignore

    def Slot(*_args, **_kwargs):  # type: ignore
        def _decorator(func):
            return func
        return _decorator

    _PYSIDE_IMPORT_ERROR = _pyside_exc
else:
    _PYSIDE_IMPORT_ERROR = None

ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
GOOGLE_HYBRID_SELECTOR = "https://mt.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&hl=de"
APP_NAME = "Frame Map Downloader"
APP_USER_MODEL_ID = "FrameMapDownloader.App"


def strike_text(text: str) -> str:
    return "".join((ch + chr(0x0336)) if ch != " " else ch for ch in text)


APP_DISPLAY_TITLE = f"{strike_text('Apple')} Frame Map Downloader"
APPLE_START_CENTER_LAT = 25.892909
APPLE_START_CENTER_LON = 13.962352
APPLE_START_LAT_SPAN = 119.310073
APPLE_START_LON_SPAN = 287.402344
APP_ICON_NAMES = (
    "app_icon.ico",
    "app_icon.png",
    "AppleMapDownloader.ico",
    "AppleMapDownloader.png",
    "AppleMapDownloader_256x256.png",
)


def apple_start_bbox(lon: float = APPLE_START_CENTER_LON, lat: float = APPLE_START_CENTER_LAT) -> Tuple[float, float, float, float]:
    south = float(clamp_lat(float(lat) - (APPLE_START_LAT_SPAN / 2.0)))
    north = float(clamp_lat(float(lat) + (APPLE_START_LAT_SPAN / 2.0)))
    west = max(-180.0, float(lon) - (APPLE_START_LON_SPAN / 2.0))
    east = min(180.0, float(lon) + (APPLE_START_LON_SPAN / 2.0))
    return west, south, east, north


def set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def find_app_icon_path() -> Optional[Path]:
    roots: List[Path] = []
    try:
        app_dir = Path(__file__).resolve().parent
        roots.extend([
            app_dir,
            app_dir.parent / "INPUT_ICON_HERE",
            app_dir.parent.parent / "INPUT_ICON_HERE",
        ])
    except Exception:
        pass
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    try:
        roots.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass

    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen:
            continue
        seen.add(root)
        for name in APP_ICON_NAMES:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def load_app_icon():
    if QIcon is None:
        return None
    icon_path = find_app_icon_path()
    if not icon_path:
        return None
    icon = QIcon(str(icon_path))
    if icon.isNull():
        return None
    return icon


def append_url_param(url: str, key: str, value: str) -> str:
    """Append a cache-busting/debug parameter without disturbing existing query params."""
    if not url:
        return url
    if f"{key}=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{key}={value}"


def apple_frame_step_multiplier_for_zoom(z: int) -> float:
    """Return center-step multiplier for Apple frame screenshots.

    Current z20 output moves map content by exactly half of the requested cell
    size, while z18 fits. Treat the Apple frame as visually capped at z19 and
    advance screenshot centers farther for requested zooms above that cap.
    """
    try:
        zoom = int(z)
    except Exception:
        zoom = APPLE_FRAME_MAX_EFFECTIVE_ZOOM
    capped_delta = max(0, zoom - int(APPLE_FRAME_MAX_EFFECTIVE_ZOOM))
    scale = 2.0 ** float(capped_delta)
    return max(1.0, min(20.0, float(scale)))


def apple_frame_step_scale_for_zoom(z: int, axis: str) -> float:
    """Compatibility wrapper for the hidden/manual X/Y step controls."""
    return apple_frame_step_multiplier_for_zoom(z)



def configure_webengine_view(view):
    """Apply robust WebEngine settings for remote map/frame pages."""
    try:
        page = view.page()
        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AutoLoadImages, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ErrorPageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
        try:
            settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
        except Exception:
            pass
        try:
            settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        except Exception:
            pass
        profile = page.profile()
        try:
            profile.setHttpUserAgent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 PyMapStitcherFramePreview/1.0"
            )
        except Exception:
            pass
        try:
            profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
        except Exception:
            pass
        try:
            profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.AllowPersistentCookies)
        except Exception:
            pass
    except Exception:
        pass


def open_url_in_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def find_chromium_executable() -> Optional[str]:
    names = ("chrome.exe", "msedge.exe", "chromium.exe", "chrome", "msedge", "chromium")
    for name in names:
        path = shutil.which(name)
        if path:
            return path

    env_paths = []
    for key in ("CHROME_PATH", "CHROMIUM_PATH", "EDGE_PATH"):
        value = os.environ.get(key)
        if value:
            env_paths.append(Path(value))

    program_roots = [
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    ]
    for root in program_roots:
        if not root:
            continue
        base = Path(root)
        env_paths.extend([
            base / "Google" / "Chrome" / "Application" / "chrome.exe",
            base / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            base / "Chromium" / "Application" / "chromium.exe",
        ])

    for path in env_paths:
        try:
            if path.exists() and path.is_file():
                return str(path)
        except Exception:
            pass
    return None


def run_hidden_chromium_screenshot(
    browser_path: str,
    url: str,
    output_png: Path,
    width: int,
    height: int,
    wait_ms: int,
    profile_dir: Path,
) -> None:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    wait_ms = max(1000, int(wait_ms))
    cmd = [
        str(browser_path),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-extensions",
        f"--user-data-dir={str(profile_dir)}",
        f"--window-size={max(1, int(width))},{max(1, int(height))}",
        f"--virtual-time-budget={wait_ms}",
        f"--screenshot={str(output_png)}",
        str(url),
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(30, int(wait_ms / 1000) + 30),
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        raise RuntimeError(f"Hidden Chromium screenshot failed with exit code {proc.returncode}: {tail}")
    if not output_png.exists() or output_png.stat().st_size <= 0:
        raise RuntimeError("Hidden Chromium did not create a screenshot file.")

def leaflet_webengine_html(lon: float, lat: float, zoom: int, tile_template: str) -> str:
    """Leaflet/QWebEngine preview adapted from Mustatil Satellite Preview.

    Shift+Drag or right mouse drag selects a bbox and sends it through QWebChannel.
    """
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script src=\"qrc:///qtwebchannel/qwebchannel.js\"></script>
<style>
html,body,#map{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#111}}
.leaflet-container{{background:#111;cursor:grab}}.leaflet-container.selecting{{cursor:crosshair}}
.hint{{position:absolute;left:10px;bottom:10px;z-index:1000;color:#eee;background:rgba(0,0,0,.68);font:12px/1.35 Arial,sans-serif;padding:7px 9px;border-radius:4px;user-select:none}}
.crosshair{{position:absolute;left:50%;top:50%;width:18px;height:18px;margin-left:-9px;margin-top:-9px;pointer-events:none;z-index:1000}}
.crosshair:before,.crosshair:after{{content:\"\";position:absolute;background:rgba(255,255,255,.88);box-shadow:0 0 2px #000}}
.crosshair:before{{left:8px;top:0;width:2px;height:18px}}.crosshair:after{{left:0;top:8px;width:18px;height:2px}}
</style></head><body><div id=\"map\"></div><div class=\"crosshair\"></div><div id=\"hint\" class=\"hint\">Shift+Drag or right-drag: select area</div>
<script>
(function(){{
const TILE_TEMPLATE={json.dumps(tile_template or ESRI_WORLD_IMAGERY)};
let bridge=null;
const map=L.map('map',{{zoomControl:false,attributionControl:false,preferCanvas:true,inertia:true,zoomAnimation:true,fadeAnimation:true,updateWhenIdle:false,updateWhenZooming:false,wheelPxPerZoomLevel:96}}).setView([{float(clamp_lat(lat))},{float(lon)}],{int(zoom)});
let layer=L.tileLayer(TILE_TEMPLATE,{{tileSize:256,minZoom:0,maxZoom:22,maxNativeZoom:22,keepBuffer:5,updateWhenIdle:false,updateWhenZooming:false,detectRetina:false,crossOrigin:false}}).addTo(map);
let selectionRect=null, selectedRect=null, selecting=false, startLatLng=null, forcedSelect=false;
function hint(t){{document.getElementById('hint').textContent=t;}}
function notifyMove(){{const c=map.getCenter();hint(`Selection map | Zoom ${{map.getZoom()}} | lon ${{c.lng.toFixed(7)}} lat ${{c.lat.toFixed(7)}} | Mark Area, Shift+Drag, or right-drag`);if(bridge&&bridge.mapMoved)bridge.mapMoved(c.lng,c.lat,map.getZoom());}}
map.on('moveend zoomend',notifyMove);
map.getContainer().addEventListener('contextmenu',function(e){{e.preventDefault();}});
window.pymapStartMarkArea=function(){{forcedSelect=true;hint('Mark Area active: drag a rectangle with the left mouse button. Esc cancels.');map.dragging.disable();map.getContainer().classList.add('selecting');return true;}};
window.pymapCancelMarkArea=function(){{forcedSelect=false;selecting=false;startLatLng=null;if(selectionRect){{map.removeLayer(selectionRect);selectionRect=null;}}map.dragging.enable();map.getContainer().classList.remove('selecting');notifyMove();return true;}};
document.addEventListener('keydown',function(e){{if(e.key==='Escape'&&forcedSelect)window.pymapCancelMarkArea();}},true);
map.on('mousedown',function(e){{const oe=e.originalEvent||{{}};if(!(forcedSelect||oe.shiftKey||oe.button===2))return;selecting=true;startLatLng=e.latlng;map.dragging.disable();map.getContainer().classList.add('selecting');if(selectionRect)map.removeLayer(selectionRect);selectionRect=L.rectangle([startLatLng,startLatLng],{{color:'#00ffff',weight:2,fill:true,fillOpacity:.12,dashArray:'5,4'}}).addTo(map);}});
map.on('mousemove',function(e){{if(selecting&&selectionRect&&startLatLng)selectionRect.setBounds(L.latLngBounds(startLatLng,e.latlng));}});
function finishSelection(e){{
  if(!selecting||!selectionRect)return;
  selecting=false;forcedSelect=false;map.dragging.enable();map.getContainer().classList.remove('selecting');
  const b=selectionRect.getBounds();
  const west=b.getWest(),south=b.getSouth(),east=b.getEast(),north=b.getNorth();
  if(east<=west||north<=south||Math.abs(east-west)<1e-9||Math.abs(north-south)<1e-9){{hint('Selection ignored: draw a larger rectangle');return;}}
  hint(`Selection saved: W ${{west.toFixed(8)}} S ${{south.toFixed(8)}} E ${{east.toFixed(8)}} N ${{north.toFixed(8)}}`);
  if(bridge&&bridge.selectionChanged)bridge.selectionChanged(west,south,east,north);
}}
map.on('mouseup',finishSelection);map.on('mouseout',function(e){{if(selecting)finishSelection(e);}});
function drawSelectedBounds(west,south,east,north){{
  const bounds=L.latLngBounds([Number(south),Number(west)],[Number(north),Number(east)]);
  if(!selectedRect){{
    selectedRect=L.rectangle(bounds,{{color:'#00ffff',weight:3,fill:true,fillOpacity:.12,dashArray:'7,4',interactive:false}}).addTo(map);
  }}else{{
    selectedRect.setBounds(bounds);
  }}
}}
window.pymapSetSelectedBounds=function(west,south,east,north){{drawSelectedBounds(west,south,east,north);return true;}};
window.pymapClearSelectedRect=function(){{if(selectedRect){{map.removeLayer(selectedRect);selectedRect=null;}}return true;}};
window.pymapSetView=function(lon,lat,zoom,tileTemplate){{if(tileTemplate)layer.setUrl(tileTemplate);map.setView([lat,lon],zoom,{{animate:false}});setTimeout(function(){{map.invalidateSize(true);notifyMove();}},30);}};
if(window.qt&&window.qt.webChannelTransport){{new QWebChannel(qt.webChannelTransport,function(channel){{bridge=channel.objects.pymapBridge;notifyMove();}});}}else{{notifyMove();}}
setTimeout(function(){{map.invalidateSize(true);notifyMove();}},100);
}})();
</script></body></html>"""


def apple_leaflet_webengine_html(lon: float, lat: float, zoom: int, frame_template: str) -> str:
    """Show the Apple/frame page behind a Leaflet coordinate layer.

    The iframe is display-only. Leaflet owns pan/zoom and Mark Area, so bbox
    fields are never derived from Apple's embedded page.
    """
    html = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#111}
#appleFrame{position:absolute;inset:0;width:100%;height:100%;border:0;background:#111;z-index:1;pointer-events:auto}
#map{position:absolute;inset:0;width:100%;height:100%;z-index:2;background:transparent;pointer-events:none}
#markLayer{position:absolute;inset:0;z-index:5;display:none;cursor:crosshair;background:rgba(0,0,0,0.001);pointer-events:auto}
#markBox{position:absolute;display:none;border:3px solid #00ffff;background:rgba(0,255,255,.16);box-shadow:0 0 0 1px rgba(0,0,0,.75),0 0 12px rgba(0,255,255,.65);pointer-events:none}
#selectedBox{position:absolute;z-index:4;display:none;border:3px solid #00ffff;background:rgba(0,255,255,.12);box-shadow:0 0 0 1px rgba(0,0,0,.8),0 0 18px rgba(0,255,255,.75);pointer-events:none}
.leaflet-container{background:transparent;cursor:grab}
.leaflet-container.selecting{cursor:crosshair}
.leaflet-control-zoom{box-shadow:0 2px 10px rgba(0,0,0,.45)}
.hint{position:absolute;left:10px;bottom:10px;z-index:1000;color:#eee;background:rgba(0,0,0,.72);font:12px/1.35 Arial,sans-serif;padding:7px 9px;border-radius:4px;user-select:none}
.source{position:absolute;right:10px;bottom:10px;z-index:1000;color:#ddd;background:rgba(0,0,0,.62);font:12px/1.35 Arial,sans-serif;padding:7px 9px;border-radius:4px;user-select:none}
.crosshair{position:absolute;left:50%;top:50%;width:18px;height:18px;margin-left:-9px;margin-top:-9px;pointer-events:none;z-index:1000}
.crosshair:before,.crosshair:after{content:"";position:absolute;background:rgba(255,255,255,.9);box-shadow:0 0 2px #000}
.crosshair:before{left:8px;top:0;width:2px;height:18px}.crosshair:after{left:0;top:8px;width:18px;height:2px}
</style></head>
<body>
<iframe id="appleFrame" src="about:blank" allow="geolocation *; fullscreen *"></iframe>
<div id="map"></div>
<div id="markLayer"><div id="markBox"></div></div>
<div id="selectedBox"></div>
<div class="crosshair"></div>
<div id="hint" class="hint">Apple view | Right-drag or Mark Area selects a fixed bbox</div>
<div class="source">BBox source: fixed coordinate fields</div>
<script>
(function(){
let FRAME_TEMPLATE=__FRAME_TEMPLATE__;
let bridge=null;
let lastFrameUrl='';
const frame=document.getElementById('appleFrame');
const markLayer=document.getElementById('markLayer');
const markBox=document.getElementById('markBox');
const selectedBox=document.getElementById('selectedBox');
const map=L.map('map',{zoomControl:false,attributionControl:false,preferCanvas:true,inertia:true,zoomAnimation:true,fadeAnimation:true,zoomSnap:0.25,wheelPxPerZoomLevel:96}).setView([__INIT_LAT__,__INIT_LON__],__INIT_ZOOM__);
const initialBounds=L.latLngBounds([__INIT_SOUTH__,__INIT_WEST__],[__INIT_NORTH__,__INIT_EAST__]);
map.fitBounds(initialBounds,{animate:false,padding:[0,0]});
let selectionRect=null, selecting=false, startLatLng=null, forcedSelect=false;
let markStart=null, markDragging=false;
let selectedBounds=null;
let frameRefreshTimer=null;
let frameNeedsTransformReset=false;
let dragAnchorLatLng=null;
let dragAnchorPoint=null;
let lastBridgeMoveAt=0;

function clampLat(lat){return Math.max(Math.min(Number(lat),85.05112878),-85.05112878);}
function fmt(n){return Number(n).toFixed(12);}
function hint(t){document.getElementById('hint').textContent=t;}
function lonLatToTile(lon,lat,z){
  const n=Math.pow(2,z);
  const x=Math.floor((Number(lon)+180.0)/360.0*n);
  const r=clampLat(lat)*Math.PI/180.0;
  const y=Math.floor((1.0-Math.asinh(Math.tan(r))/Math.PI)/2.0*n);
  return {x:Math.max(0,Math.min(n-1,x)),y:Math.max(0,Math.min(n-1,y))};
}
function quadKey(x,y,z){
  let q='';
  for(let i=z;i>0;i--){
    let digit=0, mask=1<<(i-1);
    if((x&mask)!==0)digit+=1;
    if((y&mask)!==0)digit+=2;
    q+=String(digit);
  }
  return q;
}
function replaceAllText(text, key, value){return text.split(key).join(String(value));}
function frameUrl(){
  const c=map.getCenter();
  const z=Math.round(map.getZoom());
  const b=map.getBounds();
  const west=Math.max(-180,b.getWest()), south=Math.max(-85.05112878,b.getSouth()), east=Math.min(180,b.getEast()), north=Math.min(85.05112878,b.getNorth());
  const latSpan=Math.min(170.10225756,Math.abs(north-south)), lonSpan=Math.min(360,Math.abs(east-west));
  const tile=lonLatToTile(c.lng,c.lat,z);
  const rnd=Math.floor(Math.random()*4);
  const sub=['a','b','c'][rnd%3];
  const bbox=`${fmt(west)},${fmt(south)},${fmt(east)},${fmt(north)}`;
  let url=FRAME_TEMPLATE;
  const repl={
    '{x}':tile.x,'{y}':tile.y,'{z}':z,'{c}':z,'{q}':quadKey(tile.x,tile.y,z),'{quadkey}':quadKey(tile.x,tile.y,z),
    '{rnd}':rnd,'{snum}':String(rnd%4),'{s}':sub,
    '{west}':fmt(west),'{south}':fmt(south),'{east}':fmt(east),'{north}':fmt(north),
    '{center_lon}':fmt(c.lng),'{center_lat}':fmt(c.lat),
    '{lon_span}':fmt(lonSpan),'{lat_span}':fmt(latSpan),'{span_lon}':fmt(lonSpan),'{span_lat}':fmt(latSpan),
    '{bbox}':bbox,'*GMX*':tile.x,'*GMY*':tile.y,'*ZM1*':z,'*IZM*':z,'*RND*':rnd,'*LAN*':'de','*LAN-LAN*':'de-DE'
  };
  Object.keys(repl).forEach(function(k){url=replaceAllText(url,k,repl[k]);});
  return url;
}
function updateFrame(){
  const url=frameUrl();
  if(url!==lastFrameUrl){
    lastFrameUrl=url;
    frame.src=url;
  }
}
function scheduleFrameUpdate(delay){
  if(frameRefreshTimer)window.clearTimeout(frameRefreshTimer);
  frameRefreshTimer=window.setTimeout(function(){frameRefreshTimer=null;updateFrame();},delay);
}
function sendBridgeMove(force){
  if(!bridge||!bridge.mapMoved)return;
  const now=Date.now();
  if(!force&&now-lastBridgeMoveAt<90)return;
  lastBridgeMoveAt=now;
  const c=map.getCenter();
  bridge.mapMoved(c.lng,c.lat,Math.round(map.getZoom()));
}
function updateHint(){
  const c=map.getCenter();
  const lockText=selectedBounds?' | BBox locked from fields':'';
  hint(`Apple view | Leaflet zoom ${map.getZoom()} | lon ${c.lng.toFixed(7)} lat ${c.lat.toFixed(7)}${lockText} | Right-drag or Mark Area to select`);
}
function notifyMove(forceBridge){
  updateHint();
  sendBridgeMove(forceBridge!==false);
}
map.on('zoomend resize',function(){scheduleFrameUpdate(80);notifyMove();});
map.on('moveend',function(){scheduleFrameUpdate(80);notifyMove();});
map.getContainer().addEventListener('contextmenu',function(e){e.preventDefault();});
function showMarkLayer(){
  markLayer.style.display='block';
  markBox.style.display='none';
  markStart=null;
  markDragging=false;
  hint('Mark Area active: drag a rectangle. Apple drag resumes after selection.');
}
function hideMarkLayer(cancelled){
  markLayer.style.display='none';
  markBox.style.display='none';
  markStart=null;
  markDragging=false;
  if(cancelled&&bridge&&bridge.markAreaCancelled)bridge.markAreaCancelled();
}
function drawMarkBox(x1,y1,x2,y2){
  const x=Math.min(x1,x2), y=Math.min(y1,y2), w=Math.abs(x2-x1), h=Math.abs(y2-y1);
  markBox.style.left=x+'px'; markBox.style.top=y+'px'; markBox.style.width=w+'px'; markBox.style.height=h+'px'; markBox.style.display='block';
}
function drawSelectedBox(x1,y1,x2,y2){
  const x=Math.min(Number(x1),Number(x2)), y=Math.min(Number(y1),Number(y2)), w=Math.abs(Number(x2)-Number(x1)), h=Math.abs(Number(y2)-Number(y1));
  selectedBox.style.left=x+'px'; selectedBox.style.top=y+'px'; selectedBox.style.width=w+'px'; selectedBox.style.height=h+'px'; selectedBox.style.display='block';
}
window.pymapShowSelectedRect=function(x1,y1,x2,y2){drawSelectedBox(x1,y1,x2,y2);return true;};
function drawSelectedBounds(west,south,east,north){
  selectedBox.style.display='none';
  const bounds=L.latLngBounds([Number(south),Number(west)],[Number(north),Number(east)]);
  selectedBounds=bounds;
  if(!selectedRect){
    selectedRect=L.rectangle(bounds,{color:'#00ffff',weight:3,fill:true,fillOpacity:.12,dashArray:'7,4',interactive:false}).addTo(map);
  }else{
    selectedRect.setBounds(bounds);
  }
  updateHint();
}
window.pymapSetSelectedBounds=function(west,south,east,north){drawSelectedBounds(west,south,east,north);return true;};
window.pymapClearSelectedRect=function(){selectedBox.style.display='none';selectedBounds=null;if(selectedRect){map.removeLayer(selectedRect);selectedRect=null;}updateHint();return true;};
window.pymapSyncLeafletView=function(lon,lat,zoom){return true;};
window.pymapStartMarkArea=function(){forcedSelect=true;showMarkLayer();return true;};
window.pymapCancelMarkArea=function(){forcedSelect=false;hideMarkLayer(true);notifyMove();return true;};
document.addEventListener('keydown',function(e){if(e.key==='Escape'&&forcedSelect)window.pymapCancelMarkArea();},true);
markLayer.addEventListener('mousedown',function(e){e.preventDefault();e.stopPropagation();markDragging=true;markStart={x:e.clientX,y:e.clientY};drawMarkBox(markStart.x,markStart.y,markStart.x,markStart.y);},true);
markLayer.addEventListener('mousemove',function(e){if(!markDragging||!markStart)return;e.preventDefault();e.stopPropagation();drawMarkBox(markStart.x,markStart.y,e.clientX,e.clientY);},true);
markLayer.addEventListener('mouseup',function(e){
  if(!markDragging||!markStart)return;
  e.preventDefault();e.stopPropagation();
  const x1=markStart.x,y1=markStart.y,x2=e.clientX,y2=e.clientY;
  markDragging=false;forcedSelect=false;
  if(Math.abs(x2-x1)<4||Math.abs(y2-y1)<4){hint('Selection ignored: draw a larger rectangle');hideMarkLayer(true);return;}
  if(bridge&&bridge.framePixelSelectionChanged)bridge.framePixelSelectionChanged(x1,y1,x2,y2,window.innerWidth,window.innerHeight);
  hideMarkLayer(false);
},true);
map.on('mousedown',function(e){const oe=e.originalEvent||{};if(!(forcedSelect||oe.shiftKey||oe.button===2))return;selecting=true;startLatLng=e.latlng;map.dragging.disable();map.getContainer().classList.add('selecting');if(selectionRect)map.removeLayer(selectionRect);selectionRect=L.rectangle([startLatLng,startLatLng],{color:'#00ffff',weight:2,fill:true,fillOpacity:.12,dashArray:'5,4'}).addTo(map);});
map.on('mousemove',function(e){if(selecting&&selectionRect&&startLatLng)selectionRect.setBounds(L.latLngBounds(startLatLng,e.latlng));});
function finishSelection(e){
  if(!selecting||!selectionRect)return;
  selecting=false;forcedSelect=false;map.dragging.enable();map.getContainer().classList.remove('selecting');
  const b=selectionRect.getBounds();
  const west=b.getWest(),south=b.getSouth(),east=b.getEast(),north=b.getNorth();
  if(east<=west||north<=south||Math.abs(east-west)<1e-9||Math.abs(north-south)<1e-9){hint('Selection ignored: draw a larger rectangle');return;}
  selectedBounds=L.latLngBounds([south,west],[north,east]);
  hint(`Selection saved: W ${west.toFixed(8)} S ${south.toFixed(8)} E ${east.toFixed(8)} N ${north.toFixed(8)}`);
  if(bridge&&bridge.selectionChanged)bridge.selectionChanged(west,south,east,north);
}
map.on('mouseup',finishSelection);map.on('mouseout',function(e){if(selecting)finishSelection(e);});
window.pymapSetView=function(lon,lat,zoom,frameTemplate){if(frameTemplate)FRAME_TEMPLATE=frameTemplate;map.setView([lat,lon],zoom,{animate:false});setTimeout(function(){map.invalidateSize(true);scheduleFrameUpdate(0);notifyMove();},30);};
if(window.qt&&window.qt.webChannelTransport){new QWebChannel(qt.webChannelTransport,function(channel){bridge=channel.objects.pymapBridge;scheduleFrameUpdate(0);notifyMove();});}else{scheduleFrameUpdate(0);notifyMove();}
setTimeout(function(){map.invalidateSize(true);scheduleFrameUpdate(0);notifyMove();},100);
})();
</script></body></html>"""
    init_lat = float(clamp_lat(lat))
    init_lon = float(lon)
    init_west, init_south, init_east, init_north = apple_start_bbox(init_lon, init_lat)
    return (html
            .replace("__FRAME_TEMPLATE__", json.dumps(frame_template or MAP_PRESETS["Apple Frame Preview / center-span helper"]["url"]))
            .replace("__INIT_LAT__", f"{init_lat:.12f}")
            .replace("__INIT_LON__", f"{init_lon:.12f}")
            .replace("__INIT_ZOOM__", str(int(zoom)))
            .replace("__INIT_SOUTH__", f"{init_south:.12f}")
            .replace("__INIT_WEST__", f"{init_west:.12f}")
            .replace("__INIT_NORTH__", f"{init_north:.12f}")
            .replace("__INIT_EAST__", f"{init_east:.12f}"))



def frame_preview_html(frame_url: str) -> str:
    """Frame preview wrapper with in-page JS selection overlay.

    This avoids transparent QWidget overlays over QWebEngine, which can turn the
    WebView white on Windows.
    """
    return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<script src=\"qrc:///qtwebchannel/qwebchannel.js\"></script>
<style>
html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#111}}
iframe{{position:absolute;inset:0;width:100%;height:100%;border:0;background:#111;}}
#markLayer{{position:absolute;inset:0;z-index:2147483647;display:none;cursor:crosshair;background:rgba(0,0,0,0.001);}}
#markBox{{position:absolute;display:none;border:3px solid #00ffff;background:rgba(0,255,255,.16);box-shadow:0 0 0 1px rgba(0,0,0,.75),0 0 12px rgba(0,255,255,.65);pointer-events:none;}}
#markHint{{position:absolute;left:10px;bottom:10px;z-index:2147483647;display:none;color:#eee;background:rgba(0,0,0,.78);font:12px Arial;padding:7px 9px;border-radius:4px;user-select:none;}}
</style>
</head>
<body>
<iframe id=\"frame\" src={json.dumps(frame_url)} allow=\"geolocation *; fullscreen *\"></iframe>
<div id=\"markLayer\"><div id=\"markBox\"></div></div>
<div id=\"markHint\">Mark Area: drag and release. Esc cancels.</div>
<script>
(function(){{
let bridge=null;
const layer=document.getElementById('markLayer');
const box=document.getElementById('markBox');
const hint=document.getElementById('markHint');
let start=null, dragging=false;

function showLayer(){{
  layer.style.display='block';
  hint.style.display='block';
  box.style.display='none';
  start=null;
  dragging=false;
}}
function hideLayer(){{
  layer.style.display='none';
  hint.style.display='none';
  box.style.display='none';
  start=null;
  dragging=false;
}}
function draw(x1,y1,x2,y2){{
  const x=Math.min(x1,x2), y=Math.min(y1,y2);
  const w=Math.abs(x2-x1), h=Math.abs(y2-y1);
  box.style.left=x+'px';
  box.style.top=y+'px';
  box.style.width=w+'px';
  box.style.height=h+'px';
  box.style.display='block';
}}
window.pymapStartMarkArea=function(){{showLayer(); return true;}};
window.pymapCancelMarkArea=function(){{hideLayer(); return true;}};

layer.addEventListener('mousedown', function(e){{
  e.preventDefault(); e.stopPropagation();
  dragging=true;
  start={{x:e.clientX,y:e.clientY}};
  draw(start.x,start.y,start.x,start.y);
}}, true);

layer.addEventListener('mousemove', function(e){{
  if(!dragging || !start) return;
  e.preventDefault(); e.stopPropagation();
  draw(start.x,start.y,e.clientX,e.clientY);
}}, true);

layer.addEventListener('mouseup', function(e){{
  if(!dragging || !start) return;
  e.preventDefault(); e.stopPropagation();
  let x1=start.x, y1=start.y, x2=e.clientX, y2=e.clientY;
  if(Math.abs(x2-x1)<4 || Math.abs(y2-y1)<4){{
    hint.textContent='Selection ignored: draw a larger rectangle.';
    dragging=false;
    box.style.display='none';
    return;
  }}
  if(bridge && bridge.framePixelSelectionChanged){{
    bridge.framePixelSelectionChanged(x1,y1,x2,y2,window.innerWidth,window.innerHeight);
  }}
  hideLayer();
}}, true);

document.addEventListener('keydown', function(e){{ if(e.key==='Escape') hideLayer(); }}, true);
document.addEventListener('contextmenu', function(e){{ if(layer.style.display==='block'){{e.preventDefault(); e.stopPropagation();}} }}, true);

if(window.qt && window.qt.webChannelTransport){{
  new QWebChannel(qt.webChannelTransport, function(channel) {{
    bridge = channel.objects.pymapBridge;
  }});
}}
}})();
</script>
</body>
</html>"""


def frame_render_html(frame_url: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#000}}
iframe{{position:absolute;inset:0;width:100%;height:100%;border:0;background:#000}}
</style>
</head>
<body data-frame-loaded="0">
<iframe id="frame" src={json.dumps(frame_url)} allow="geolocation *; fullscreen *" onload="document.body.dataset.frameLoaded='1'"></iframe>
</body>
</html>"""

class WebBridge(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window

    @Slot(float, float, int)
    def mapMoved(self, lon: float, lat: float, zoom: int) -> None:
        self.window.center_lon = float(lon)
        self.window.center_lat = float(lat)
        self.window.preview_zoom = int(zoom)
        self.window.sync_preview_view_bbox_to_current_view()
        self.window.redraw_preview_selection_from_coords()
        self.window.status_label.setText(
            f"Preview: zoom {int(zoom)} | lon {float(lon):.7f} lat {float(lat):.7f}"
        )

    @Slot()
    def markAreaCancelled(self) -> None:
        self.window.preview_mark_mode = False
        self.window.status_label.setText("Mark Area cancelled")

    @Slot(float, float, float, float)
    def selectionChanged(self, west: float, south: float, east: float, north: float) -> None:
        west = float(west); south = float(south); east = float(east); north = float(north)
        if east < west:
            west, east = east, west
        if north < south:
            south, north = north, south
        self.window.min_lon_edit.setText(f"{west:.8f}")
        self.window.min_lat_edit.setText(f"{south:.8f}")
        self.window.max_lon_edit.setText(f"{east:.8f}")
        self.window.max_lat_edit.setText(f"{north:.8f}")
        self.window.user_bbox_valid = True
        self.window.last_bbox_source = "mark_area"
        self.window.preview_exact_bbox = (west, south, east, north)
        self.window.preview_mark_mode = False

        # Match the reference workflow: the selector map becomes the app-owned
        # coordinate state after selection. Apple/frame rendering still reads
        # only the fixed bbox fields for download URL creation.
        self.window.center_lon = (west + east) / 2.0
        self.window.center_lat = (south + north) / 2.0
        try:
            self.window.preview_zoom = int(self.window.zoom_spin.value())
        except Exception:
            pass

        self.window.log_msg(
            f"Selection entered EXACTLY: South={south:.8f}, West={west:.8f}, North={north:.8f}, East={east:.8f}"
        )
        self.window.calculate()
        self.window.redraw_preview_selection_from_coords()
        # Do not refresh/recenter iframe after selection; this caused jump-back.

    @Slot(float, float, float, float, float, float)
    def framePixelSelectionChanged(self, x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> None:
        """Convert frame pixel selection to bbox.

        Important: panning inside a cross-origin iframe cannot update Python's
        center coordinate. Therefore this function uses the currently known
        app-controlled frame extent. If the left coordinate boxes already contain
        a bbox, that bbox is used as the visible frame extent. Otherwise it falls
        back to center/span from the current zoom.
        """
        try:
            w = max(1.0, float(width))
            h = max(1.0, float(height))
            px1, px2 = sorted([max(0.0, min(w, float(x1))), max(0.0, min(w, float(x2)))])
            py1, py2 = sorted([max(0.0, min(h, float(y1))), max(0.0, min(h, float(y2)))])

            if abs(px2 - px1) < 4 or abs(py2 - py1) < 4:
                self.window.log_msg(
                    f"Frame Mark Area ignored: selection too small ({abs(px2 - px1):.0f}x{abs(py2 - py1):.0f}px). Draw a larger rectangle."
                )
                return

            view_west, view_south, view_east, view_north, source = self.window.selection_view_bbox_for_pixels(w, h)

            lon_span = view_east - view_west
            lat_span = view_north - view_south

            west = view_west + (px1 / w) * lon_span
            east = view_west + (px2 / w) * lon_span
            north = view_north - (py1 / h) * lat_span
            south = view_north - (py2 / h) * lat_span

            self.window.log_msg(
                f"Frame Mark Area pixels: x={px1:.0f}..{px2:.0f}, y={py1:.0f}..{py2:.0f}, viewport={w:.0f}x{h:.0f}"
            )
            self.window.log_msg(
                f"Frame Mark Area based on {source}: W={view_west:.8f}, S={view_south:.8f}, E={view_east:.8f}, N={view_north:.8f}"
            )
            self.selectionChanged(west, south, east, north)
        except Exception as exc:
            self.window.log_msg(f"Frame Mark Area failed: {exc}")


class PySideMapStitcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_TITLE)
        icon = load_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        self.resize(1320, 820)
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.q = queue.Queue()
        self.center_lon = APPLE_START_CENTER_LON
        self.center_lat = APPLE_START_CENTER_LAT
        self.preview_zoom = 1
        self.user_bbox_valid = False
        self.last_bbox_source = ""
        self.preview_exact_bbox = None
        self.preview_view_bbox = None
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)
        self.frame_active = False
        self.frame_queue = []
        self.frame_queue_index = 0
        self.frame_done = 0
        self.frame_total = 0
        self.frame_renderers = []
        self.frame_render_windows = []
        self.frame_mem = None
        self.frame_cfg = None
        self.frame_tile_dir = None
        self.frame_chromium_thread = None
        self.frame_chromium_browser = None
        self.frame_x_min = self.frame_y_min = 0
        self.frame_cell_w = self.frame_cell_h = TILE_SIZE
        self.frame_render_w_actual = self.frame_render_h_actual = 0
        self.frame_request_lon_span = self.frame_request_lat_span = 0.0
        self.frame_lon_per_px = self.frame_lat_per_px = 0.0
        self.frame_visible_lon_span = self.frame_visible_lat_span = 0.0
        self.frame_selected_west = self.frame_selected_north = 0.0
        self.frame_grid_west = self.frame_grid_south = self.frame_grid_east = self.frame_grid_north = 0.0
        self.frame_step_mult_x = 1.0
        self.frame_step_mult_y = 1.0
        self.frame_shift_x_px = FIXED_FRAME_SHIFT_X_PX
        self.frame_shift_y_px = FIXED_FRAME_SHIFT_Y_PX
        self.frame_crop_correct_url = True
        self.preview_exact_bbox = None
        self.preview_view_bbox = None
        self._google_selector_loaded = False
        self._leaflet_selector_loaded = False
        self._google_selector_pending_mark = False
        self._webview_drag_tracking = False
        self._webview_drag_last = QPoint()
        self._webview_drag_accum_dx = 0.0
        self._webview_drag_accum_dy = 0.0
        self._webview_filter_widgets = set()
        QTimer.singleShot(200, self.refresh_webmap)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(left)
        splitter.setStretchFactor(0, 0)

        form_box = QGroupBox("Map / Download")
        form = QFormLayout(form_box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        left_layout.addWidget(form_box)

        # Fixed Apple workflow: no selectable provider dropdown and no editable URL field.
        # The hidden combo/line edit stay available for existing internal methods.
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Apple Frame Preview / center-span helper"])
        self.preset_combo.setCurrentText("Apple Frame Preview / center-span helper")
        self.preset_combo.setVisible(False)
        form.addRow("Map Selection", QLabel("Apple Frame Preview / center-span helper"))

        self.url_edit = QLineEdit(MAP_PRESETS.get("Apple Frame Preview / center-span helper", MAP_PRESETS["Google Satellite"])["url"])
        self.url_edit.setVisible(False)
        self.note_label = QLabel(MAP_PRESETS.get("Apple Frame Preview / center-span helper", MAP_PRESETS["Google Satellite"])["note"])
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(False)

        self.zoom_spin = QSpinBox(); self.zoom_spin.setRange(0, 22); self.zoom_spin.setValue(18)
        self.min_lat_edit = QLineEdit("")
        self.min_lon_edit = QLineEdit("")
        self.max_lat_edit = QLineEdit("")
        self.max_lon_edit = QLineEdit("")
        for _bbox_edit in (self.min_lat_edit, self.min_lon_edit, self.max_lat_edit, self.max_lon_edit):
            _bbox_edit.textEdited.connect(self._mark_manual_bbox_edit)
        self.workers_spin = QSpinBox(); self.workers_spin.setRange(1, 256); self.workers_spin.setValue(32)
        self.rate_spin = QSpinBox(); self.rate_spin.setRange(0, 20000); self.rate_spin.setSingleStep(250); self.rate_spin.setValue(3000)
        self.frame_settle_extra_spin = QSpinBox(); self.frame_settle_extra_spin.setRange(0, 20000); self.frame_settle_extra_spin.setSingleStep(250); self.frame_settle_extra_spin.setValue(2500)
        self.chunk_spin = QSpinBox(); self.chunk_spin.setRange(8, 2048); self.chunk_spin.setSingleStep(8); self.chunk_spin.setValue(128)
        self.frame_views_spin = QSpinBox(); self.frame_views_spin.setRange(1, 16); self.frame_views_spin.setValue(4)
        self.render_w_spin = QSpinBox(); self.render_w_spin.setRange(256, 4096); self.render_w_spin.setSingleStep(128); self.render_w_spin.setValue(1600)
        self.render_h_spin = QSpinBox(); self.render_h_spin.setRange(256, 4096); self.render_h_spin.setSingleStep(128); self.render_h_spin.setValue(1600)
        self.crop_top_spin = QSpinBox(); self.crop_top_spin.setRange(0, 1000); self.crop_top_spin.setValue(0)
        self.crop_bottom_spin = QSpinBox(); self.crop_bottom_spin.setRange(0, 1000); self.crop_bottom_spin.setValue(100)
        self.step_factor_x_spin = QDoubleSpinBox()
        self.step_factor_x_spin.setRange(0.00, 5.00)
        self.step_factor_x_spin.setSingleStep(0.05)
        self.step_factor_x_spin.setDecimals(3)
        self.step_factor_x_spin.setValue(0.0)
        self.step_factor_y_spin = QDoubleSpinBox()
        self.step_factor_y_spin.setRange(0.00, 5.00)
        self.step_factor_y_spin.setSingleStep(0.05)
        self.step_factor_y_spin.setDecimals(3)
        self.step_factor_y_spin.setValue(0.0)
        self.crop_left_spin = QSpinBox(); self.crop_left_spin.setRange(0, 1000); self.crop_left_spin.setValue(APPLE_LEFT_BAR_CROP_PX)
        self.crop_right_spin = QSpinBox(); self.crop_right_spin.setRange(0, 1000); self.crop_right_spin.setValue(100)
        self.pixel_step_x_spin = QDoubleSpinBox()
        self.pixel_step_x_spin.setRange(0.25, 20.00)
        self.pixel_step_x_spin.setSingleStep(0.001)
        self.pixel_step_x_spin.setDecimals(4)
        self.pixel_step_x_spin.setValue(apple_frame_step_scale_for_zoom(self.zoom_spin.value(), "x"))
        self.pixel_step_x_spin.setToolTip("Auto from zoom. Apple frame behaves like max z19: z20 uses 2.0, z21 uses 4.0, z22 uses 8.0.")
        self.pixel_step_y_spin = QDoubleSpinBox()
        self.pixel_step_y_spin.setRange(0.25, 20.00)
        self.pixel_step_y_spin.setSingleStep(0.001)
        self.pixel_step_y_spin.setDecimals(4)
        self.pixel_step_y_spin.setValue(apple_frame_step_scale_for_zoom(self.zoom_spin.value(), "y"))
        self.pixel_step_y_spin.setToolTip("Auto from zoom. Apple frame behaves like max z19: z20 uses 2.0, z21 uses 4.0, z22 uses 8.0.")
        self.frame_shift_x_spin = QDoubleSpinBox()
        self.frame_shift_x_spin.setRange(-3000.0, 3000.0)
        self.frame_shift_x_spin.setSingleStep(10.0)
        self.frame_shift_x_spin.setDecimals(1)
        self.frame_shift_x_spin.setValue(FIXED_FRAME_SHIFT_X_PX)
        self.frame_shift_x_spin.setToolTip("Additive Feinverschiebung pro Spalte in Pixeln. Negativ = mehr Überlappung, positiv = weiter auseinander.")
        self.frame_shift_y_spin = QDoubleSpinBox()
        self.frame_shift_y_spin.setRange(-3000.0, 3000.0)
        self.frame_shift_y_spin.setSingleStep(10.0)
        self.frame_shift_y_spin.setDecimals(1)
        self.frame_shift_y_spin.setValue(FIXED_FRAME_SHIFT_Y_PX)
        self.frame_shift_y_spin.setToolTip("Additive Feinverschiebung pro Zeile in Pixeln. Negativ = mehr Überlappung, positiv = weiter auseinander.")

        self.frame_preview_cells_spin = QDoubleSpinBox()
        self.frame_preview_cells_spin.setRange(1.0, 50.0)
        self.frame_preview_cells_spin.setSingleStep(0.5)
        self.frame_preview_cells_spin.setDecimals(1)
        self.frame_preview_cells_spin.setValue(4.0)
        self.frame_preview_cells_spin.setToolTip("Apple/Frame Mark Area without an existing bbox: preview area in renderer cells. Higher values allow selecting a larger area and prevent tiny fallback spans.")

        self.frame_min_cols_spin = QSpinBox()
        self.frame_min_cols_spin.setRange(1, 10000)
        self.frame_min_cols_spin.setValue(1)
        self.frame_min_cols_spin.setToolTip("Notfall/Test: erzwingt mindestens so viele Screenshot-Spalten, auch wenn die berechnete BBox kleiner wirkt.")
        self.frame_min_rows_spin = QSpinBox()
        self.frame_min_rows_spin.setRange(1, 10000)
        self.frame_min_rows_spin.setValue(1)
        self.frame_min_rows_spin.setToolTip("Notfall/Test: erzwingt mindestens so viele Screenshot-Zeilen, auch wenn die berechnete BBox kleiner wirkt.")
        self.hidden_render_check = QCheckBox("Use hidden Chromium renderer")
        self.hidden_render_check.setChecked(False)
        self.hidden_render_check.setToolTip("Uses external Chrome/Edge headless screenshots when available. If no browser is found, the app falls back to Qt WebViews.")
        self.fullscreen_render_check = QCheckBox("Frame WebViews als eigene 1600x1600-Fenster")
        self.fullscreen_render_check.setChecked(True)
        self.fullscreen_render_check.setToolTip("Öffnet jeden Renderer als eigenes festes Fenster in Render-W/H. Nicht maximieren, damit Pixelrechnung exakt bleibt.")
        self.crop_correct_url_check = QCheckBox("Crop in Apple-URL-Geometrie einrechnen")
        self.crop_correct_url_check.setChecked(True)
        self.crop_correct_url_check.setToolTip("Standard AN: Crop L/T/R/B wird in die URL-Center-Geometrie eingerechnet. Left, right, top und bottom verschieben den Center jetzt symmetrisch.")
        self.pypi_stitching_check = QCheckBox("Frame-Endexport mit PyPI stitching zusammensetzen")
        self.pypi_stitching_check.setChecked(False)
        self.pypi_stitching_check.setToolTip("Optional. Standard AUS, weil Feature-Stitching bei Karten oft schlecht arbeitet. Der direkte Grid-GeoTIFF bleibt der Hauptausgang.")
        self.outfile_edit = QLineEdit(str(Path.home() / "Desktop" / "map_output.tif"))

        # Only the requested user-facing controls remain visible.
        form.addRow("Download Zoom", self.zoom_spin)
        form.addRow("South / min lat", self.min_lat_edit)
        form.addRow("West / min lon", self.min_lon_edit)
        form.addRow("North / max lat", self.max_lat_edit)
        form.addRow("East / max lon", self.max_lon_edit)
        form.addRow("Download Threads", self.workers_spin)
        form.addRow("Frame WebViews", self.frame_views_spin)
        form.addRow("", self.hidden_render_check)

        out_row = QHBoxLayout()
        out_row.addWidget(self.outfile_edit, 1)
        browse = QPushButton("…")
        browse.clicked.connect(self.pick_output)
        out_row.addWidget(browse)
        out_widget = QWidget(); out_widget.setLayout(out_row)
        form.addRow("Output File", out_widget)

        btn_row = QHBoxLayout()
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self.calculate)
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self.start)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_event.set)
        btn_row.addWidget(calc_btn); btn_row.addWidget(start_btn); btn_row.addWidget(stop_btn)
        left_layout.addLayout(btn_row)
        self.zoom_spin.valueChanged.connect(self.update_zoom_dependent_pixel_steps)

        terms = QLabel("Only use servers where downloading/stitching is allowed. Google/Bing/OSM may restrict bulk downloads.")
        terms.setWordWrap(True)
        left_layout.addWidget(terms)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left_layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(170)
        left_layout.addWidget(self.log, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        map_header = QHBoxLayout()
        map_title = QLabel("Apple Frame Preview / Leaflet Selection")
        map_title.setStyleSheet("font-weight: 600;")
        map_header.addWidget(map_title, 1)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Iframe wrapper"])
        self.preview_mode_combo.setCurrentText("Iframe wrapper")
        self.preview_mode_combo.setVisible(False)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.refresh_webmap)
        map_header.addWidget(reload_btn)
        browser_btn = QPushButton("Open in Browser")
        browser_btn.clicked.connect(self.open_current_preview_in_browser)
        map_header.addWidget(browser_btn)
        leaflet_selector_btn = QPushButton("Accurate Leaflet Selector")
        leaflet_selector_btn.clicked.connect(self.load_accurate_leaflet_selector)
        map_header.addWidget(leaflet_selector_btn)
        self.mark_area_btn = QPushButton("Mark Area")
        self.mark_area_btn.clicked.connect(self.start_mark_area)
        map_header.addWidget(self.mark_area_btn)
        self.preview_bbox_btn = QPushButton("Preview Selected BBox")
        self.preview_bbox_btn.clicked.connect(self.preview_selected_bbox)
        map_header.addWidget(self.preview_bbox_btn)
        right_layout.addLayout(map_header)

        self.webview = QWebEngineView()
        configure_webengine_view(self.webview)
        self.webview.setStyleSheet("background:#111;")
        self.webview.installEventFilter(self)
        QTimer.singleShot(0, self._install_webview_mouse_filters)
        self.preview_mark_mode = False
        self.preview_selecting = False
        self.preview_select_start = QPoint()
        self.preview_overlay = None  # JS page overlay is used; QWidget overlay breaks QWebEngine on Windows.
        self.preview_rubber_band = None
        self.web_bridge = WebBridge(self)
        self.web_channel = QWebChannel(self.webview.page())
        self.web_channel.registerObject("pymapBridge", self.web_bridge)
        self.webview.page().setWebChannel(self.web_channel)
        right_layout.addWidget(self.webview, 1)

        render_label = QLabel("Frame Render WebViews (nur Own Frame Server; sichtbar lassen, falls Hidden-Screenshots schwarz werden)")
        render_label.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(render_label)
        self.render_area = QWidget()
        self.render_layout = QGridLayout(self.render_area)
        self.render_layout.setContentsMargins(0, 0, 0, 0)
        self.render_layout.setSpacing(4)
        right_layout.addWidget(self.render_area, 0)

        splitter.setSizes([420, 900])

    def update_zoom_dependent_pixel_steps(self, z: Optional[int] = None) -> None:
        try:
            zoom = int(self.zoom_spin.value() if z is None else z)
            x_scale = apple_frame_step_scale_for_zoom(zoom, "x")
            y_scale = apple_frame_step_scale_for_zoom(zoom, "y")
            for spin, value in ((self.pixel_step_x_spin, x_scale), (self.pixel_step_y_spin, y_scale)):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)
        except Exception:
            pass

    def _mark_manual_bbox_edit(self, *_args) -> None:
        self.user_bbox_valid = True
        self.last_bbox_source = "manual"
        self.preview_exact_bbox = None

    def clear_bbox_fields(self, log: bool = False) -> None:
        for edit in (self.min_lat_edit, self.min_lon_edit, self.max_lat_edit, self.max_lon_edit):
            edit.blockSignals(True)
            edit.clear()
            edit.blockSignals(False)
        self.user_bbox_valid = False
        self.last_bbox_source = ""
        self.preview_exact_bbox = None
        self.preview_view_bbox = None
        self.hide_preview_selection_rect()
        if log:
            self.log_msg("BBox cleared: Apple/Frame mode will not use any preset mini bbox. Mark an area or enter coordinates before Start.")

    def read_bbox_values(self):
        west = float(self.min_lon_edit.text().replace(",", "."))
        south = float(self.min_lat_edit.text().replace(",", "."))
        east = float(self.max_lon_edit.text().replace(",", "."))
        north = float(self.max_lat_edit.text().replace(",", "."))
        if not (east > west and north > south):
            raise ValueError("invalid bbox")
        return west, south, east, north

    def has_valid_bbox(self) -> bool:
        try:
            self.read_bbox_values()
            return True
        except Exception:
            return False

    def preview_selected_bbox(self) -> None:
        """Reload frame preview using the exact bbox fields, not a tiny center/zoom span."""
        try:
            west, south, east, north = self.read_bbox_values()
            self.center_lon = (west + east) / 2.0
            self.center_lat = (south + north) / 2.0
            self.preview_zoom = int(self.zoom_spin.value())
            self.preview_exact_bbox = (west, south, east, north)
            self.user_bbox_valid = True
            self.last_bbox_source = "preview_bbox"
            self.log_msg(
                f"Preview Selected BBox EXACT: W={west:.8f}, S={south:.8f}, E={east:.8f}, N={north:.8f}, z={self.preview_zoom}"
            )
            self.refresh_webmap()
        except Exception as exc:
            QMessageBox.warning(self, "Preview selected bbox", f"Could not preview selected bbox: {exc}")

    def show_preview_selection_rect(self, x1: float, y1: float, x2: float, y2: float) -> None:
        try:
            js = (
                "if(window.pymapShowSelectedRect){"
                f"window.pymapShowSelectedRect({float(x1):.3f},{float(y1):.3f},{float(x2):.3f},{float(y2):.3f});"
                "true;} else {false;}"
            )
            self.webview.page().runJavaScript(js)
        except Exception:
            pass

    def hide_preview_selection_rect(self) -> None:
        try:
            self.webview.page().runJavaScript("if(window.pymapClearSelectedRect){window.pymapClearSelectedRect(); true;} else {false;}")
        except Exception:
            pass

    def selected_bbox_from_fields_or_memory(self) -> Optional[Tuple[float, float, float, float]]:
        exact = getattr(self, "preview_exact_bbox", None)
        if exact and len(exact) == 4:
            try:
                west, south, east, north = map(float, exact)
                if east > west and north > south:
                    return west, south, east, north
            except Exception:
                pass
        if bool(getattr(self, "user_bbox_valid", False)):
            try:
                return self.read_bbox_values()
            except Exception:
                pass
        return None

    def redraw_preview_selection_from_coords(self) -> None:
        bbox = self.selected_bbox_from_fields_or_memory()
        if not bbox:
            self.hide_preview_selection_rect()
            return
        try:
            west, south, east, north = map(float, bbox)
            js = (
                "if(window.pymapSetSelectedBounds){"
                f"window.pymapSetSelectedBounds({west:.12f},{south:.12f},{east:.12f},{north:.12f});"
                "true;} else {false;}"
            )
            self.webview.page().runJavaScript(js)
        except Exception:
            self.hide_preview_selection_rect()

    def sync_leaflet_overlay_view(self) -> None:
        try:
            js = (
                "if(window.pymapSyncLeafletView){"
                f"window.pymapSyncLeafletView({float(self.center_lon):.12f},{float(self.center_lat):.12f},{int(self.preview_zoom)});"
                "true;} else {false;}"
            )
            self.webview.page().runJavaScript(js)
        except Exception:
            pass

    def apple_selector_zoom(self) -> int:
        try:
            z = int(getattr(self, "preview_zoom", 1))
        except Exception:
            z = 1
        return max(APPLE_SELECTOR_MIN_ZOOM, min(22, z))

    def selection_view_bbox_for_pixels(self, width: float, height: float) -> Tuple[float, float, float, float, str]:
        apple_frame = is_apple_frame_template(self.url_edit.text().strip())
        if apple_frame:
            try:
                z = self.apple_selector_zoom()
                w = max(1.0, float(width))
                h = max(1.0, float(height))
                west, south, east, north = frame_view_bbox_for_center_zoom_pixels(
                    float(self.center_lon), float(self.center_lat), z, w, h
                )
                if east > west and north > south:
                    self.preview_view_bbox = (west, south, east, north)
                    return west, south, east, north, f"safe Apple selector bbox (z={z})"
            except Exception:
                pass

            # Last-resort fallback: still never use the wide world-start bbox for
            # Mark Area. A bounded z=8 viewport prevents accidental ocean/world
            # selections from becoming multi-terabyte screenshot jobs.
            west, south, east, north = frame_view_bbox_for_center_zoom_pixels(
                float(self.center_lon), float(self.center_lat), APPLE_SELECTOR_MIN_ZOOM,
                max(1.0, float(width)), max(1.0, float(height))
            )
            self.preview_view_bbox = (west, south, east, north)
            return west, south, east, north, f"bounded Apple selector fallback (z={APPLE_SELECTOR_MIN_ZOOM})"

        view_bbox = getattr(self, "preview_view_bbox", None)
        if view_bbox and len(view_bbox) == 4:
            west, south, east, north = map(float, view_bbox)
            if east > west and north > south:
                return west, south, east, north, "known preview bbox"

        z = int(self.zoom_spin.value())
        render_w = int(self.render_w_spin.value()) if hasattr(self, "render_w_spin") else int(width)
        render_h = int(self.render_h_spin.value()) if hasattr(self, "render_h_spin") else int(height)
        cells = float(self.frame_preview_cells_spin.value()) if hasattr(self, "frame_preview_cells_spin") else 4.0
        west, south, east, north = frame_view_bbox_for_center_zoom_pixels(
            float(self.center_lon), float(self.center_lat), z,
            max(float(width), float(render_w)) * cells,
            max(float(height), float(render_h)) * cells
        )
        self.preview_view_bbox = (west, south, east, north)
        return west, south, east, north, f"renderer-pixel fallback ({cells:.1f} cells)"

    def load_google_hybrid_selection_map(self, reason: str = "") -> None:
        """Load Google Hybrid/Leaflet only as the coordinate selector.

        The URL template on the left is NOT changed. Therefore Apple/Frame can
        remain the renderer/download template, while this preview supplies the
        bbox coordinates exactly like the reference app.
        """
        try:
            try:
                self.preview_zoom = int(self.zoom_spin.value())
            except Exception:
                pass
            self.preview_view_bbox = None
            html = leaflet_webengine_html(
                float(self.center_lon),
                float(self.center_lat),
                int(self.preview_zoom),
                GOOGLE_HYBRID_SELECTOR,
            )
            self._google_selector_loaded = True
            self._leaflet_selector_loaded = True
            self.webview.setHtml(html, QUrl("https://mustatil.local/"))
            QTimer.singleShot(250, self._install_webview_mouse_filters)
            QTimer.singleShot(450, self.redraw_preview_selection_from_coords)
            self.status_label.setText("Google Hybrid selector loaded - download still uses the Apple/Frame URL template")
            self.log_msg(
                "Google Hybrid selector active: Mark Area writes only the four bbox fields; "
                "download still uses the URL template on the left."
                + (f" ({reason})" if reason else "")
            )
        except Exception as exc:
            self.log_msg(f"Google Hybrid selector failed: {exc}")

    def load_accurate_leaflet_selector(self) -> None:
        """Alternative selector: pure Leaflet map, exact bbox from Leaflet bounds."""
        try:
            self.load_google_hybrid_selection_map("manual selector")
        except Exception as exc:
            self.log_msg(f"Accurate Leaflet selector failed: {exc}")

    def start_mark_area(self) -> None:
        """Enable Mark Area. For Apple/frame downloads, select on Leaflet."""
        try:
            self.preview_mark_mode = True
            url_template = self.url_edit.text().strip()
            preset = MAP_PRESETS.get(self.preset_combo.currentText(), {})
            if preset.get("preview") == "frame" or is_frame_template(url_template):
                # Critical workflow fix: use Google Hybrid/Leaflet for coordinate
                # but do not change cfg.url_template. Start still uses
                # Apple/Frame URL + the four coordinate fields.
                if not bool(getattr(self, "_google_selector_loaded", False)):
                    self.load_google_hybrid_selection_map("Apple/Frame download mode")
                    QTimer.singleShot(500, self.start_mark_area)
                    return

            js = "if(window.pymapStartMarkArea){window.pymapStartMarkArea(); true;} else {false;}"
            def _mark_started(ok):
                if not ok:
                    self.preview_mark_mode = False
                self.log_msg(
                    "Mark Area active: drag a rectangle in the preview." if ok
                    else "Could not start Mark Area. Reload the preview and try again."
                )
            self.webview.page().runJavaScript(js, _mark_started)
            self.status_label.setText("Mark Area active: drag in the preview. Esc cancels.")
        except Exception as exc:
            self.log_msg(f"Mark Area failed: {exc}")


    def _install_webview_mouse_filters(self) -> None:
        """Catch right-drag selection events on the WebEngine child widgets too."""
        try:
            widgets = [self.webview]
            try:
                widgets.extend(self.webview.findChildren(QWidget))
            except Exception:
                pass
            installed = getattr(self, "_webview_filter_widgets", set())
            for widget in widgets:
                key = id(widget)
                if key in installed:
                    continue
                try:
                    widget.installEventFilter(self)
                    installed.add(key)
                except Exception:
                    pass
            self._webview_filter_widgets = installed
        except Exception:
            pass

    def _is_preview_widget(self, obj) -> bool:
        try:
            if obj is self.webview:
                return True
            if isinstance(obj, QWidget) and self.webview.isAncestorOf(obj):
                return True
        except Exception:
            pass
        return False

    def _event_pos_in_webview(self, obj, event) -> QPoint:
        try:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        except Exception:
            return QPoint()
        try:
            if obj is not self.webview and isinstance(obj, QWidget):
                pos = obj.mapTo(self.webview, pos)
        except Exception:
            pass
        return pos

    def _ensure_preview_rubber_band(self):
        if self.preview_rubber_band is None and QRubberBand is not None:
            shape = QRubberBand.Shape.Rectangle if hasattr(QRubberBand, "Shape") else QRubberBand.Rectangle
            self.preview_rubber_band = QRubberBand(shape, self.webview)
        return self.preview_rubber_band

    def _finish_webview_drag_tracking(self) -> None:
        if not bool(getattr(self, "_webview_drag_tracking", False)):
            return
        dx = float(getattr(self, "_webview_drag_accum_dx", 0.0))
        dy = float(getattr(self, "_webview_drag_accum_dy", 0.0))
        self._webview_drag_tracking = False
        self._webview_drag_accum_dx = 0.0
        self._webview_drag_accum_dy = 0.0
        if abs(dx) >= 1.0 or abs(dy) >= 1.0:
            self.pan_preview_model_by_pixels(dx, dy)

    def sync_preview_view_bbox_to_current_view(self) -> None:
        try:
            w = max(1, int(self.webview.width()))
            h = max(1, int(self.webview.height()))
            if is_apple_frame_template(self.url_edit.text().strip()):
                z = self.apple_selector_zoom()
            else:
                z = int(self.preview_zoom)
            self.preview_view_bbox = frame_view_bbox_for_center_zoom_pixels(
                float(self.center_lon), float(self.center_lat), z, w, h
            )
        except Exception:
            self.preview_view_bbox = None

    def pan_preview_model_by_pixels(self, dx: float, dy: float) -> None:
        try:
            z = int(self.preview_zoom)
            cx, cy = lonlat_to_world_pixel(float(self.center_lon), float(self.center_lat), z)
            lon, lat = world_pixel_to_lonlat(cx - float(dx), cy - float(dy), z)
            self.center_lon = float(lon)
            self.center_lat = float(clamp_lat(lat))
            self.sync_preview_view_bbox_to_current_view()
            self.sync_leaflet_overlay_view()
            self.redraw_preview_selection_from_coords()
            self.status_label.setText(
                f"Apple viewport: zoom {z} | lon {self.center_lon:.7f} lat {self.center_lat:.7f}"
            )
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # Like the reference app: no QWidget/Python mouse overlay over QWebEngine.
        # Leaflet and the in-page JS selection layer receive drag events directly.
        return super().eventFilter(obj, event)

    def apply_preview_pixel_selection(self, p1: QPoint, p2: QPoint) -> None:
        """Convert WebView pixel rectangle to left-side bbox using current preview center/span."""
        try:
            w = max(1, self.webview.width())
            h = max(1, self.webview.height())
            x1 = max(0, min(w, int(p1.x())))
            x2 = max(0, min(w, int(p2.x())))
            y1 = max(0, min(h, int(p1.y())))
            y2 = max(0, min(h, int(p2.y())))
            px1, px2 = sorted([x1, x2])
            py1, py2 = sorted([y1, y2])

            if abs(px2 - px1) < 4 or abs(py2 - py1) < 4:
                self.log_msg(
                    f"Python Mark Area ignored: selection too small ({abs(px2 - px1):.0f}x{abs(py2 - py1):.0f}px). Draw a larger rectangle."
                )
                return

            view_west, view_south, view_east, view_north, source = self.selection_view_bbox_for_pixels(float(w), float(h))

            lon_span = view_east - view_west
            lat_span = view_north - view_south

            west = view_west + (float(px1) / float(w)) * lon_span
            east = view_west + (float(px2) / float(w)) * lon_span
            north = view_north - (float(py1) / float(h)) * lat_span
            south = view_north - (float(py2) / float(h)) * lat_span

            self.log_msg(
                f"Python Mark Area pixels: x={px1:.0f}..{px2:.0f}, y={py1:.0f}..{py2:.0f}, preview={w}x{h}"
            )
            self.log_msg(
                f"Python Mark Area based on {source}: W={view_west:.8f}, S={view_south:.8f}, E={view_east:.8f}, N={view_north:.8f}"
            )
            self.web_bridge.selectionChanged(west, south, east, north)
        except Exception as exc:
            self.log_msg(f"Python Mark Area failed: {exc}")

    def current_preview_url(self) -> str:
        url_template = self.url_edit.text().strip()
        preset = MAP_PRESETS.get(self.preset_combo.currentText(), {})
        if preset.get("preview") == "frame" or is_frame_template(url_template):
            z = int(self.zoom_spin.value()) if hasattr(self, "zoom_spin") else self.preview_zoom

            # FIELD-BBOX ONLY for Apple/Frame preview when coordinates exist.
            # This intentionally ignores the internal map/preview center. The four
            # left-side fields are the authority for center/span/bbox URL creation.
            try:
                west, south, east, north = self.read_bbox_values()
                self.center_lon = (west + east) / 2.0
                self.center_lat = (south + north) / 2.0
                self.preview_zoom = z
                self.preview_exact_bbox = (west, south, east, north)
                self.preview_view_bbox = (float(west), float(south), float(east), float(north))
                return expand_frame_url_exact_bbox(url_template, west, south, east, north, z)
            except Exception:
                pass

            exact = getattr(self, "preview_exact_bbox", None)
            if exact:
                try:
                    west, south, east, north = exact
                    self.preview_view_bbox = (float(west), float(south), float(east), float(north))
                    return expand_frame_url_exact_bbox(url_template, west, south, east, north, z)
                except Exception as exc:
                    self.log_msg(f"Exact bbox preview failed, falling back to renderer-pixel preview span: {exc}")

            # Only preview fallback when the coordinate fields are empty.
            # Start/download still refuses to run without valid field coordinates.
            try:
                render_w = int(self.render_w_spin.value()) if hasattr(self, "render_w_spin") else 1600
                render_h = int(self.render_h_spin.value()) if hasattr(self, "render_h_spin") else 1600
                cells = float(self.frame_preview_cells_spin.value()) if hasattr(self, "frame_preview_cells_spin") else 4.0
                west, south, east, north = frame_view_bbox_for_center_zoom_pixels(
                    self.center_lon, self.center_lat, z,
                    max(1.0, float(render_w) * cells),
                    max(1.0, float(render_h) * cells),
                )
                self.preview_view_bbox = (float(west), float(south), float(east), float(north))
                return expand_frame_url_exact_bbox(url_template, west, south, east, north, z)
            except Exception as exc:
                self.log_msg(f"Renderer-pixel preview span failed, falling back to center/span preview only: {exc}")
                url = expand_frame_url_center_span(url_template, self.center_lon, self.center_lat, z)
                try:
                    lat_span, lon_span = frame_span_for_center_zoom(self.center_lon, self.center_lat, z)
                    self.preview_view_bbox = (
                        float(self.center_lon) - lon_span / 2.0,
                        float(self.center_lat) - lat_span / 2.0,
                        float(self.center_lon) + lon_span / 2.0,
                        float(self.center_lat) + lat_span / 2.0,
                    )
                except Exception:
                    self.preview_view_bbox = None
                return url
        return url_template or ESRI_WORLD_IMAGERY

    def open_current_preview_in_browser(self) -> None:
        url = self.current_preview_url()
        self.log_msg(f"Open preview in browser: {url}")
        open_url_in_browser(url)

    def on_preset_changed(self, name: str) -> None:
        preset = MAP_PRESETS.get(name, MAP_PRESETS["Custom"])
        self.url_edit.setText(preset["url"])
        self.note_label.setText(preset.get("note", ""))
        # Important: do not keep an old/tiny bbox when switching to Apple/frame mode.
        # Otherwise Start may download that stale mini extent immediately.
        if preset.get("preview") == "frame" or is_frame_template(preset.get("url", "")):
            self.clear_bbox_fields(log=True)
            self._google_selector_loaded = False
            self._leaflet_selector_loaded = False
            self.log_msg("Apple/Frame workflow: the right side uses the Google Hybrid/Leaflet selector; Start uses only the URL on the left plus the bbox fields.")
        self.refresh_webmap()

    def refresh_webmap(self) -> None:
        preset = MAP_PRESETS.get(self.preset_combo.currentText(), {})
        url_template = self.url_edit.text().strip()
        preview_mode = self.preview_mode_combo.currentText() if hasattr(self, "preview_mode_combo") else "Auto"

        if preview_mode == "Leaflet tiles":
            if preset.get("preview") == "frame" or is_frame_template(url_template):
                self.load_google_hybrid_selection_map("Leaflet selector for Apple/Frame URL")
                return
            html = leaflet_webengine_html(
                self.center_lon,
                self.center_lat,
                self.preview_zoom,
                url_template or ESRI_WORLD_IMAGERY,
            )
            self._google_selector_loaded = False
            self._leaflet_selector_loaded = False
            self.webview.setHtml(html, QUrl("https://mustatil.local/"))
            self.status_label.setText("Leaflet tile preview loaded")
            self.log_msg(f"Preview mode: Leaflet tiles | template: {url_template}")
            return

        if preset.get("preview") == "frame" or is_frame_template(url_template):
            # Same stable workflow as the reference app: select bbox in
            # Google Hybrid/Leaflet, but keep the Apple/Frame URL template on
            # the left for rendering/download.
            if preview_mode != "Direct URL":
                self.load_google_hybrid_selection_map("refresh")
                return

            # Direct URL is only for inspecting the actual Apple/frame URL. It is
            # intentionally not used for coordinate selection.
            url = self.current_preview_url()
            try:
                self.webview.loadFinished.disconnect()
            except Exception:
                pass
            self.webview.load(QUrl(url))
            self._google_selector_loaded = False
            self._leaflet_selector_loaded = False
            self.status_label.setText("Frame URL loaded directly for inspection only - use Google Hybrid selector for Mark Area")
            self.log_msg(f"Preview mode: direct frame URL inspection | URL: {url}")
            return

        html = leaflet_webengine_html(
            self.center_lon,
            self.center_lat,
            self.preview_zoom,
            url_template or ESRI_WORLD_IMAGERY,
        )
        self._google_selector_loaded = False
        self._leaflet_selector_loaded = False
        self.webview.setHtml(html, QUrl("https://mustatil.local/"))
        self.status_label.setText("WebMap loaded - Shift+Drag or right-drag to select extent")

    def pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output BigTIFF", self.outfile_edit.text(), "TIFF (*.tif *.tiff);;All files (*)")
        if path:
            self.outfile_edit.setText(path)

    def _config(self) -> StitchConfig:
        try:
            west, south, east, north = self.read_bbox_values()
        except Exception as exc:
            raise RuntimeError("No valid download bbox. Draw a real Mark Area or enter South/West/North/East manually; no tiny preset bbox is used anymore.") from exc
        url_template = self.url_edit.text().strip() or MAP_PRESETS["Apple Frame Preview / center-span helper"]["url"]
        return StitchConfig(
            url_template=url_template,
            output_file=Path(self.outfile_edit.text()).expanduser(),
            z=int(self.zoom_spin.value()),
            min_lat=south,
            min_lon=west,
            max_lat=north,
            max_lon=east,
            workers=int(self.workers_spin.value()),
            rate_limit_ms=int(self.rate_spin.value()),
            chunk_size=int(self.chunk_spin.value()),
        )

    def frame_grid_limit_message(self, cols: int, rows: int, width: int, height: int, raw_bytes: int) -> Optional[str]:
        total_cells = int(cols) * int(rows)
        if int(raw_bytes) > MAX_DIRECT_TIFF_BYTES:
            return (
                f"Selected area exceeds the hard BigTIFF safety limit: {format_bytes(raw_bytes)}. "
                "Select a smaller bbox or reduce the download zoom."
            )
        if total_cells > MAX_FRAME_SCREENSHOT_CELLS:
            return (
                f"Selected area is too large for one screenshot job: {cols} x {rows} = {total_cells:,} cells. "
                f"The safety limit is {MAX_FRAME_SCREENSHOT_CELLS:,} cells. Zoom in, select a smaller bbox, or split the area."
            )
        if int(raw_bytes) > MAX_FRAME_SCREENSHOT_BYTES:
            return (
                f"Selected area is too large for one BigTIFF: {format_bytes(raw_bytes)} estimated raw payload "
                f"({width:,} x {height:,} px). The safety limit is {format_bytes(MAX_FRAME_SCREENSHOT_BYTES)}. "
                "Zoom in, select a smaller bbox, or split the area."
            )
        return None

    def calculate(self) -> None:
        try:
            cfg = self._config()
            render_w = int(self.render_w_spin.value())
            render_h = int(self.render_h_spin.value())
            crop_left = APPLE_LEFT_BAR_CROP_PX
            crop_top = int(self.crop_top_spin.value())
            crop_right = int(self.crop_right_spin.value())
            crop_bottom = int(self.crop_bottom_spin.value())
            self.frame_crop_left = int(crop_left)
            self.frame_crop_top = int(crop_top)
            self.frame_crop_right = int(crop_right)
            self.frame_crop_bottom = int(crop_bottom)
            self.frame_wait_ms = int(self.rate_spin.value())
            self.frame_extra_wait_ms = int(self.frame_settle_extra_spin.value())
            visible_w = max(1, render_w - crop_left - crop_right)
            visible_h = max(1, render_h - crop_top - crop_bottom)

            # FIELD-BBOX ONLY: calculation mirrors Start exactly. No preview/map
            # center is used to create the download URL grid.
            sel_left_px, sel_top_px = lonlat_to_world_pixel(cfg.min_lon, cfg.max_lat, cfg.z)
            sel_right_px, sel_bottom_px = lonlat_to_world_pixel(cfg.max_lon, cfg.min_lat, cfg.z)
            selected_width_px = max(1.0, sel_right_px - sel_left_px)
            selected_height_px = max(1.0, sel_bottom_px - sel_top_px)

            step_mult_x = apple_frame_step_multiplier_for_zoom(cfg.z)
            step_mult_y = apple_frame_step_multiplier_for_zoom(cfg.z)
            shift_x_px = FIXED_FRAME_SHIFT_X_PX
            shift_y_px = FIXED_FRAME_SHIFT_Y_PX
            crop_correct_url = True
            effective_step_x_px = max(1.0, float(visible_w) * step_mult_x + shift_x_px)
            effective_step_y_px = max(1.0, float(visible_h) * step_mult_y + shift_y_px)

            calc_cols = max(1, int(math.ceil(selected_width_px / effective_step_x_px)))
            calc_rows = max(1, int(math.ceil(selected_height_px / effective_step_y_px)))
            min_cols = int(self.frame_min_cols_spin.value()) if hasattr(self, "frame_min_cols_spin") else 1
            min_rows = int(self.frame_min_rows_spin.value()) if hasattr(self, "frame_min_rows_spin") else 1
            cols = max(calc_cols, min_cols)
            rows = max(calc_rows, min_rows)

            # Apple georeferencing mode: anchor to the actual first cropped Apple frame,
            # not the Google/field bbox. This includes crop margins and the Apple URL
            # center/span math in the GeoTIFF origin.
            _sample_url_calc, _sample_visible_calc, _sample_request_calc = expand_frame_url_grid(
                cfg.url_template, 0, 0, cfg.z,
                cfg.min_lon, cfg.max_lat,
                0.0, 0.0,
                0.0, 0.0,
                visible_w, visible_h,
                render_w, render_h,
                crop_left, crop_top, crop_right, crop_bottom,
                step_mult_x, step_mult_y, shift_x_px, shift_y_px, crop_correct_url,
            )
            anchor_crop_bounds = _sample_visible_calc
            anchor_left_px, anchor_top_px = lonlat_to_world_pixel(anchor_crop_bounds[0], anchor_crop_bounds[3], cfg.z)
            grid_right_px = anchor_left_px + cols * float(visible_w)
            grid_bottom_px = anchor_top_px + rows * float(visible_h)
            grid_west, grid_south, grid_east, grid_north = world_pixel_bbox_to_lonlat(
                anchor_left_px, anchor_top_px, grid_right_px, grid_bottom_px, cfg.z
            )

            width = cols * visible_w
            height = rows * visible_h
            raw_bytes = width * height * 3

            self.log_msg("=== Calculation: Frame Screenshot Grid / FIELD-BBOX URL MODE ===")
            self.log_msg("URL source: ONLY the four coordinate fields. Preview/map center is ignored for download URL creation.")
            self.log_msg(f"Selected bbox fields: S={cfg.min_lat:.8f}, W={cfg.min_lon:.8f}, N={cfg.max_lat:.8f}, E={cfg.max_lon:.8f}")
            self.log_msg(f"Selected size at z={cfg.z}: {selected_width_px:.1f} x {selected_height_px:.1f} world-px")
            self.log_msg(f"Grid coverage: S={grid_south:.8f}, W={grid_west:.8f}, N={grid_north:.8f}, E={grid_east:.8f}")
            self.log_msg(f"Render size: {render_w}x{render_h}; crop L/T/R/B={crop_left}/{crop_top}/{crop_right}/{crop_bottom}")
            self.log_msg(f"Visible output cell: {visible_w}x{visible_h} px")
            self.log_msg(
                "Logical center mode: Apple URL center is computed from the cropped cell center and crop offsets; "
                "for zooms above Apple's effective frame zoom the center step is scaled."
            )
            self.log_msg(
                f"Apple effective zoom cap: z{APPLE_FRAME_MAX_EFFECTIVE_ZOOM}; requested z={cfg.z}; "
                f"step multiplier X/Y={step_mult_x:.4f}/{step_mult_y:.4f}"
            )
            self.log_msg(f"Effective step: X={effective_step_x_px:.1f}px, Y={effective_step_y_px:.1f}px; crop-center correction L/T/R/B={crop_left}/{crop_top}/{crop_right}/{crop_bottom}")
            if cols != calc_cols or rows != calc_rows:
                self.log_msg(f"Force min grid applied: calculated {calc_cols} x {calc_rows}, using {cols} x {rows}.")
            self.log_msg(f"Grid cells: {cols} x {rows} = {cols*rows:,}; output pixels: {width:,} x {height:,}")
            self.log_msg(f"Estimated raw BigTIFF payload: {format_bytes(raw_bytes)}")
            limit_msg = self.frame_grid_limit_message(cols, rows, width, height, raw_bytes)
            if limit_msg:
                self.log_msg(f"Start will be blocked: {limit_msg}")

            sample_url, sample_visible, sample_request = expand_frame_url_grid(
                cfg.url_template, 0, 0, cfg.z,
                cfg.min_lon, cfg.max_lat,
                0.0, 0.0,
                0.0, 0.0,
                visible_w, visible_h,
                render_w, render_h,
                crop_left, crop_top, crop_right, crop_bottom,
                step_mult_x, step_mult_y, shift_x_px, shift_y_px, crop_correct_url,
            )
            self.log_msg(f"Sample cell 0/0 visible={sample_visible}")
            self.log_msg(f"Sample cell 0/0 request={sample_request}")
            self.log_msg(f"Sample URL: {sample_url}")
            self.log_msg(
                f"Alignment mode: logical crop-center with Apple effective z{APPLE_FRAME_MAX_EFFECTIVE_ZOOM} step scaling."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.information(self, "Running", "A job is already running.")
            return
        if self.frame_active:
            QMessageBox.information(self, "Running", "A frame screenshot job is already running.")
            return
        preset = MAP_PRESETS.get(self.preset_combo.currentText(), {})
        current_url_template = self.url_edit.text().strip()
        current_is_frame = preset.get("preview") == "frame" or any(k in current_url_template for k in ("{center_lat}", "{center_lon}", "{lat_span}", "{lon_span}", "{bbox}"))
        if current_is_frame and not self.has_valid_bbox():
            QMessageBox.warning(
                self,
                "No bbox selected",
                "Apple/Frame mode has no preset mini bbox anymore. Please click Mark Area and draw a real rectangle, or enter South/West/North/East manually.",
            )
            self.log_msg("Start blocked: no valid bbox. The old automatic tiny Apple/frame bbox was removed.")
            return
        try:
            cfg = self._config()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        self.stop_event.clear()
        self.progress.setValue(0)

        # FIELD-BBOX ONLY for Start/download: the four text fields are now the
        # single source of truth for Apple/Frame URL center/span/bbox creation.
        west = float(cfg.min_lon)
        south = float(cfg.min_lat)
        east = float(cfg.max_lon)
        north = float(cfg.max_lat)
        self.center_lon = (west + east) / 2.0
        self.center_lat = (south + north) / 2.0
        self.preview_exact_bbox = (west, south, east, north)
        self.preview_view_bbox = (west, south, east, north)
        self.user_bbox_valid = True
        self.last_bbox_source = "field_bbox_start"
        self.log_msg(
            f"FIELD-BBOX START: URL creation uses ONLY fields W={west:.8f}, S={south:.8f}, E={east:.8f}, N={north:.8f}, z={cfg.z}."
        )
        preset = MAP_PRESETS.get(self.preset_combo.currentText(), {})
        if "maps.apple.com/frame" in cfg.url_template:
            self.log_msg("Hinweis: Das Frame-Template zeigt auf maps.apple.com/frame. Nutze Download nur, wenn du für diese Quelle berechtigt bist. Für deinen eigenen Haus-/Sentinel-Server einfach die URL auf deinen eigenen Frame-Server umstellen.")
        if preset.get("preview") == "frame" or any(k in cfg.url_template for k in ("{center_lat}", "{center_lon}", "{lat_span}", "{lon_span}", "{bbox}")):
            self.start_frame_screenshot_job(cfg)
            return
        self.worker_thread = threading.Thread(target=self._run_job, args=(cfg,), daemon=True)
        self.worker_thread.start()

    def qimage_to_pil_rgb(self, qimage):
        if Image is None:
            raise RuntimeError("Pillow is required")
        from PySide6.QtGui import QImage
        img = qimage.convertToFormat(QImage.Format.Format_RGB888)
        width = img.width(); height = img.height(); bpl = img.bytesPerLine()
        data = bytes(img.constBits()[:bpl * height])
        return Image.frombytes("RGB", (width, height), data, "raw", "RGB", bpl, 1).copy()

    def pil_looks_blank(self, pil) -> bool:
        try:
            probe = pil.resize((32, 32), Image.Resampling.BILINEAR)
            extrema = probe.getextrema()
            ranges = [hi - lo for lo, hi in extrema]
            lows = [lo for lo, _hi in extrema]
            highs = [hi for _lo, hi in extrema]
            nearly_flat = max(ranges) <= 4
            nearly_white = min(lows) >= 245
            nearly_black = max(highs) <= 10
            return bool(nearly_flat and (nearly_white or nearly_black))
        except Exception:
            return False

    def clear_frame_renderers(self) -> None:
        for w in list(getattr(self, "frame_render_windows", [])):
            try:
                w.close()
                w.deleteLater()
            except Exception:
                pass
        self.frame_render_windows = []
        for item in list(getattr(self, "frame_renderers", [])):
            try:
                view = item.get("view") if isinstance(item, dict) else item
                view.setParent(None)
                view.deleteLater()
            except Exception:
                pass
        self.frame_renderers = []
        self.frame_render_windows = []
        try:
            while self.render_layout.count():
                child = self.render_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
        except Exception:
            pass

    def start_frame_screenshot_job(self, cfg: StitchConfig) -> None:
        if Image is None:
            QMessageBox.critical(self, "Error", "Pillow is required for frame screenshot TIFF export.")
            return
        try:
            # Real screenshot-grid mode:
            # The output cell is the cropped visible screenshot, not an XYZ 256px tile.
            render_w = int(self.render_w_spin.value())
            render_h = int(self.render_h_spin.value())
            crop_left = APPLE_LEFT_BAR_CROP_PX
            crop_top = int(self.crop_top_spin.value())
            crop_right = int(self.crop_right_spin.value())
            crop_bottom = int(self.crop_bottom_spin.value())
            visible_w = max(1, render_w - crop_left - crop_right)
            visible_h = max(1, render_h - crop_top - crop_bottom)
            if visible_w <= 8 or visible_h <= 8:
                raise RuntimeError("Crop values leave too little visible area.")

            self.log_msg("FIELD-BBOX URL MODE active: every Apple/Frame URL is generated from the coordinate fields, not from the preview map.")

            # WebMercator world-pixel grid at zoom z.
            sel_left_px, sel_top_px = lonlat_to_world_pixel(cfg.min_lon, cfg.max_lat, cfg.z)
            sel_right_px, sel_bottom_px = lonlat_to_world_pixel(cfg.max_lon, cfg.min_lat, cfg.z)
            selected_width_px = max(1.0, sel_right_px - sel_left_px)
            selected_height_px = max(1.0, sel_bottom_px - sel_top_px)

            step_mult_x = apple_frame_step_multiplier_for_zoom(cfg.z)
            step_mult_y = apple_frame_step_multiplier_for_zoom(cfg.z)
            shift_x_px = FIXED_FRAME_SHIFT_X_PX
            shift_y_px = FIXED_FRAME_SHIFT_Y_PX
            crop_correct_url = True
            effective_step_x_px = max(1.0, float(visible_w) * step_mult_x + shift_x_px)
            effective_step_y_px = max(1.0, float(visible_h) * step_mult_y + shift_y_px)

            calc_cols = max(1, int(math.ceil(selected_width_px / effective_step_x_px)))
            calc_rows = max(1, int(math.ceil(selected_height_px / effective_step_y_px)))
            min_cols = int(self.frame_min_cols_spin.value()) if hasattr(self, "frame_min_cols_spin") else 1
            min_rows = int(self.frame_min_rows_spin.value()) if hasattr(self, "frame_min_rows_spin") else 1
            cols = max(calc_cols, min_cols)
            rows = max(calc_rows, min_rows)

            if calc_cols == 1 and calc_rows == 1:
                self.log_msg(
                    "Warning: calculated frame grid is only 1 x 1. "
                    f"Selected size at z={cfg.z}: {selected_width_px:.1f} x {selected_height_px:.1f} world-px; "
                    f"step: {effective_step_x_px:.1f} x {effective_step_y_px:.1f} px. "
                    "Use higher Download Zoom, lower Render W/H or Pixel step, or increase Apple Mark Area view x cells before marking."
                )
            if cols != calc_cols or rows != calc_rows:
                self.log_msg(f"Force min grid applied: calculated {calc_cols} x {calc_rows}, using {cols} x {rows}.")

            # Build the first URL once and use its CROPPED/visible screenshot geometry
            # as the georeferencing anchor. This is intentionally different from
            # the old mode, which georeferenced the full output from the raw
            # coordinate-field bbox. In Apple frame mode the real output tile is
            # the cropped WebView screenshot, so the first cropped cell is the
            # most stable top-left anchor for the streamed GeoTIFF.
            sample_url0, sample_visible0, sample_request0 = expand_frame_url_grid(
                cfg.url_template, 0, 0, cfg.z,
                cfg.min_lon, cfg.max_lat,
                0.0, 0.0,
                0.0, 0.0,
                visible_w, visible_h,
                render_w, render_h,
                crop_left, crop_top, crop_right, crop_bottom,
                step_mult_x, step_mult_y, shift_x_px, shift_y_px, crop_correct_url,
            )
            request_lon_span = abs(sample_request0[2] - sample_request0[0])
            request_lat_span = abs(sample_request0[3] - sample_request0[1])
            visible_lon_span = abs(sample_visible0[2] - sample_visible0[0])
            visible_lat_span = abs(sample_visible0[3] - sample_visible0[1])
            lon_per_px = 0.0
            lat_per_px = 0.0

            # Apple CENTER georeferencing:
            # Use the first Apple URL center as the first output-cell center.
            # The full GeoTIFF extent uses the same effective center step as the
            # Apple URL grid, including the fixed FrameShift X/Y values.
            # This prevents a mismatch where the captures move by visible+shift
            # but the GeoTIFF was still written as if each cell moved only by
            # visible_w/visible_h pixels.
            anchor_crop_bounds = sample_visible0
            anchor_left_px, anchor_top_px = lonlat_to_world_pixel(anchor_crop_bounds[0], anchor_crop_bounds[3], cfg.z)
            grid_west, grid_south, grid_east, grid_north = world_pixel_bbox_to_lonlat(
                anchor_left_px,
                anchor_top_px,
                anchor_left_px + cols * float(visible_w),
                anchor_top_px + rows * float(visible_h),
                cfg.z,
            )

            width = cols * visible_w
            height = rows * visible_h
            raw_bytes = width * height * 3
            limit_msg = self.frame_grid_limit_message(cols, rows, width, height, raw_bytes)
            if limit_msg:
                raise RuntimeError(limit_msg)

            self.frame_cell_w = visible_w
            self.frame_cell_h = visible_h
            self.frame_render_w_actual = render_w
            self.frame_render_h_actual = render_h
            self.frame_request_lon_span = float(request_lon_span)
            self.frame_request_lat_span = float(request_lat_span)
            self.frame_lon_per_px = float(lon_per_px)
            self.frame_lat_per_px = float(lat_per_px)
            self.frame_visible_lon_span = float(visible_lon_span)
            self.frame_visible_lat_span = float(visible_lat_span)
            self.frame_selected_west = float(cfg.min_lon)
            self.frame_selected_north = float(cfg.max_lat)
            self.frame_anchor_visible_bounds = tuple(sample_visible0)
            self.frame_anchor_request_bounds = tuple(sample_request0)
            self.frame_anchor_crop_bounds = tuple(anchor_crop_bounds)
            self.frame_grid_west = float(grid_west)
            self.frame_grid_north = float(grid_north)
            self.frame_grid_east = float(grid_east)
            self.frame_grid_south = float(grid_south)
            self.frame_step_mult_x = float(step_mult_x)
            self.frame_step_mult_y = float(step_mult_y)
            self.frame_shift_x_px = float(shift_x_px)
            self.frame_shift_y_px = float(shift_y_px)
            self.frame_crop_correct_url = bool(crop_correct_url)

            bounds_3857 = lonlat_bbox_to_webmercator_bounds(
                self.frame_grid_west, self.frame_grid_south, self.frame_grid_east, self.frame_grid_north
            )
            self.frame_mem, _ = open_direct_bigtiff(cfg, width, height, bounds_3857, self.log_msg)
            self.frame_cfg = cfg
            self.frame_x_min = 0
            self.frame_y_min = 0
            self.frame_tile_dir = default_tile_tif_dir(cfg)
            self.frame_tile_dir.mkdir(parents=True, exist_ok=True)

            # Create jobs by screenshot-grid cell. x/y are only stable IDs now.
            self.frame_queue = [
                TileJob(col, row, cfg.z, col, row)
                for row in range(rows)
                for col in range(cols)
            ]
            self.frame_total = len(self.frame_queue)
            self.frame_queue_index = 0
            self.frame_done = 0
            self.frame_active = True
            self.progress.setRange(0, max(1, self.frame_total))
            self.progress.setValue(0)
            self.clear_frame_renderers()

            requested_count = max(1, min(16, int(self.frame_views_spin.value())))
            use_hidden_chromium = bool(getattr(self, "hidden_render_check", None) and self.hidden_render_check.isChecked())
            chromium_browser = find_chromium_executable() if use_hidden_chromium else None
            hidden = bool(use_hidden_chromium and chromium_browser)
            if use_hidden_chromium and not chromium_browser:
                self.log_msg("Use hidden Chromium is checked, but no Chrome/Edge/Chromium executable was found. Falling back to Qt WebViews.")
            count = requested_count
            self.log_msg("=== Frame Screenshot Grid mode ===")
            self.log_msg(
                "Logical crop-center alignment active: output cell size defines the grid; "
                "Apple URL center is derived from the cropped screenshot center."
            )
            self.log_msg(
                f"Apple effective zoom cap: z{APPLE_FRAME_MAX_EFFECTIVE_ZOOM}; requested z={cfg.z}; "
                f"step multiplier X/Y={step_mult_x:.4f}/{step_mult_y:.4f}. "
                "This makes z20 advance two visible cells because the frame content moved only half a cell."
            )
            if hasattr(self, "frame_preview_cells_spin"):
                self.log_msg(f"Apple/Frame Mark Area view fallback: {float(self.frame_preview_cells_spin.value()):.1f} renderer-cells. This prevents the old one-tile fallback.")
            self.log_msg("Tip: X/Y shift negative = screenshots closer/more overlap; positive = farther apart. PyPI stitching is off by default.")
            self.log_msg(f"Crop URL correction: {'ON' if crop_correct_url else 'OFF'}")
            self.log_msg(f"Selected bbox fields: S={cfg.min_lat:.8f}, W={cfg.min_lon:.8f}, N={cfg.max_lat:.8f}, E={cfg.max_lon:.8f}")
            self.log_msg(f"First synthetic visible-cell bounds diagnostic: S={sample_visible0[1]:.8f}, W={sample_visible0[0]:.8f}, N={sample_visible0[3]:.8f}, E={sample_visible0[2]:.8f}")
            self.log_msg(f"First screenshot requested full-frame bounds: S={sample_request0[1]:.8f}, W={sample_request0[0]:.8f}, N={sample_request0[3]:.8f}, E={sample_request0[2]:.8f}")
            self.log_msg(f"First CENTER-based Apple-frame georef bounds: S={anchor_crop_bounds[1]:.8f}, W={anchor_crop_bounds[0]:.8f}, N={anchor_crop_bounds[3]:.8f}, E={anchor_crop_bounds[2]:.8f}")
            self.log_msg(f"GeoTIFF grid coverage from Apple centers: S={self.frame_grid_south:.8f}, W={self.frame_grid_west:.8f}, N={self.frame_grid_north:.8f}, E={self.frame_grid_east:.8f}")
            self.log_msg(f"Render size: {render_w}x{render_h} px")
            self.log_msg("Screen note: monitor may be 2560x1440, but renderer capture is forced to exact Render W/H so offsets stay correct.")
            self.log_msg(f"Crop L/T/R/B: {crop_left}/{crop_top}/{crop_right}/{crop_bottom} px")
            self.log_msg(f"Visible output cell: {visible_w}x{visible_h} px")
            self.log_msg(f"Request span full screenshot: lon={request_lon_span:.12f}, lat={request_lat_span:.12f}")
            self.log_msg(f"Degrees per px: lon={lon_per_px:.14f}, lat={lat_per_px:.14f}")
            self.log_msg(f"Step per screenshot: X={effective_step_x_px:.1f}px, Y={effective_step_y_px:.1f}px; visible cell={visible_w}x{visible_h}px; center correction from crop L/T/R/B={crop_left}/{crop_top}/{crop_right}/{crop_bottom}")
            self.log_msg(f"Grid: {cols} x {rows} = {self.frame_total:,}; output pixels: {width:,} x {height:,}")
            self.log_msg(f"Render WebViews: {count}; hidden={hidden}")
            self.log_msg(f"Individual TIFF tiles: {self.frame_tile_dir}")
            self.log_msg(f"Stitched GeoTIFF/BigTIFF: {cfg.output_file}")

            # Log first two sample URLs so the math can be checked.
            for sample_col, sample_row in [(0, 0), (1, 0), (0, 1)]:
                if sample_col < cols and sample_row < rows:
                    sample_url, sample_visible, sample_request = expand_frame_url_grid(
                        cfg.url_template, sample_col, sample_row, cfg.z,
                        self.frame_selected_west, self.frame_selected_north,
                        self.frame_request_lon_span, self.frame_request_lat_span,
                        self.frame_lon_per_px, self.frame_lat_per_px,
                        self.frame_cell_w, self.frame_cell_h,
                        render_w, render_h,
                        crop_left, crop_top, crop_right, crop_bottom,
                        step_mult_x, step_mult_y, shift_x_px, shift_y_px, crop_correct_url,
                    )
                    self.log_msg(f"Sample cell col={sample_col} row={sample_row} visible={sample_visible} request={sample_request}")
                    self.log_msg(f"Sample URL col={sample_col} row={sample_row}: {sample_url}")

            if hidden and chromium_browser:
                self.frame_chromium_browser = chromium_browser
                self.log_msg(f"Hidden Chromium renderer active: {chromium_browser}")
                self.frame_chromium_thread = threading.Thread(target=self._run_hidden_chromium_frame_jobs, daemon=True)
                self.frame_chromium_thread.start()
                return

            fullscreen = True if not getattr(self, "fullscreen_render_check", None) else bool(self.fullscreen_render_check.isChecked())
            self.log_msg(f"Exact renderer windows: {fullscreen}; not maximized; capture size forced to Render W/H.")

            for i in range(count):
                if fullscreen:
                    win = QWidget(None, Qt.WindowType.Window)
                    win.setWindowTitle(f"PyMapStitcher Frame Renderer {i}")
                    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
                    layout = QVBoxLayout(win)
                    layout.setContentsMargins(0, 0, 0, 0)
                    view = QWebEngineView(win)
                    configure_webengine_view(view)
                    layout.addWidget(view)
                    view.setFixedSize(render_w, render_h)
                    win.setFixedSize(render_w, render_h)
                    win.resize(render_w, render_h)
                    self.frame_render_windows.append(win)
                    win.show()
                    win.raise_()
                    win.activateWindow()
                    view.resize(render_w, render_h)
                else:
                    view = QWebEngineView(self.render_area)
                    configure_webengine_view(view)
                    view.setFixedSize(render_w, render_h)
                    view.resize(render_w, render_h)
                    view.show()
                item = {"view": view, "busy": False, "job": None, "index": i}
                self.frame_renderers.append(item)

            QTimer.singleShot(500, self._start_frame_dispatch)
        except Exception as exc:
            QMessageBox.critical(self, "Frame screenshot error", str(exc))
            self.log_msg(f"Frame screenshot error: {exc}")

    def _run_hidden_chromium_frame_jobs(self) -> None:
        stopped = False
        profile_dir = Path(tempfile.mkdtemp(prefix="pymap_hidden_chromium_profile_"))
        png_dir = Path(tempfile.mkdtemp(prefix="pymap_hidden_chromium_png_"))
        try:
            import numpy as np
            browser = str(getattr(self, "frame_chromium_browser", "") or "")
            if not browser:
                raise RuntimeError("Hidden Chromium was requested, but no browser executable is configured.")
            total = len(list(getattr(self, "frame_queue", [])))
            for index, job in enumerate(list(getattr(self, "frame_queue", [])), start=1):
                if self.stop_event.is_set():
                    stopped = True
                    break
                try:
                    url, visible_bounds, request_bounds = expand_frame_url_grid(
                        self.frame_cfg.url_template,
                        job.col, job.row, job.z,
                        self.frame_selected_west, self.frame_selected_north,
                        self.frame_request_lon_span, self.frame_request_lat_span,
                        self.frame_lon_per_px, self.frame_lat_per_px,
                        self.frame_cell_w, self.frame_cell_h,
                        int(self.frame_render_w_actual), int(self.frame_render_h_actual),
                        int(getattr(self, "frame_crop_left", APPLE_LEFT_BAR_CROP_PX)),
                        int(getattr(self, "frame_crop_top", 0)),
                        int(getattr(self, "frame_crop_right", 0)),
                        int(getattr(self, "frame_crop_bottom", 0)),
                        float(getattr(self, "frame_step_mult_x", 1.0)),
                        float(getattr(self, "frame_step_mult_y", 1.0)),
                        float(getattr(self, "frame_shift_x_px", 0.0)),
                        float(getattr(self, "frame_shift_y_px", 0.0)),
                        bool(getattr(self, "frame_crop_correct_url", False)),
                    )
                    actual_cropped_bounds = visible_bounds
                    wait_ms = max(
                        1500,
                        int(getattr(self, "frame_wait_ms", 1000)) + int(getattr(self, "frame_extra_wait_ms", 2500)),
                    )
                    png_path = png_dir / f"grid_z{job.z}_col{job.col}_row{job.row}.png"
                    pil = None
                    for attempt in range(1, 6):
                        run_hidden_chromium_screenshot(
                            browser,
                            url,
                            png_path,
                            int(self.frame_render_w_actual),
                            int(self.frame_render_h_actual),
                            wait_ms + ((attempt - 1) * 1000),
                            profile_dir,
                        )
                        candidate = Image.open(png_path).convert("RGB")
                        if not self.pil_looks_blank(candidate) or attempt == 5:
                            pil = candidate
                            break
                        self.q.put(("log", f"Hidden Chromium captured a blank cell at col={job.col} row={job.row}; retry {attempt}/5."))
                    if pil is None:
                        raise RuntimeError("Hidden Chromium did not return an image.")

                    l = int(getattr(self, "frame_crop_left", APPLE_LEFT_BAR_CROP_PX))
                    t = int(getattr(self, "frame_crop_top", 0))
                    r = pil.width - int(getattr(self, "frame_crop_right", 0))
                    b = pil.height - int(getattr(self, "frame_crop_bottom", 0))
                    if r <= l or b <= t:
                        raise RuntimeError("Crop values remove the full hidden Chromium image.")
                    pil = pil.crop((l, t, r, b))
                    if pil.width != self.frame_cell_w or pil.height != self.frame_cell_h:
                        pil = pil.resize((self.frame_cell_w, self.frame_cell_h), Image.Resampling.LANCZOS)

                    tile_path = self.frame_tile_dir / f"grid_z{job.z}_col{job.col}_row{job.row}.tif"
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = tile_path.with_suffix(".tmp.tif")
                    pil.save(tmp, format="TIFF", compression="tiff_deflate")
                    os.replace(tmp, tile_path)
                    write_worldfile_and_prj(
                        tile_path, self.frame_cell_w, self.frame_cell_h,
                        lonlat_bbox_to_webmercator_bounds(
                            actual_cropped_bounds[0],
                            actual_cropped_bounds[1],
                            actual_cropped_bounds[2],
                            actual_cropped_bounds[3],
                        )
                    )

                    arr = np.asarray(pil, dtype=np.uint8)
                    r0 = job.row * self.frame_cell_h
                    c0 = job.col * self.frame_cell_w
                    self.frame_mem[r0:r0+self.frame_cell_h, c0:c0+self.frame_cell_w, :] = arr
                    self.frame_done = index
                    self.q.put(("progress", index, total, "Hidden Chromium"))
                    if index == 1:
                        self.q.put(("log", f"First hidden Chromium URL: {url}"))
                    if index % 5 == 0:
                        try:
                            self.frame_mem.flush()
                        except Exception:
                            pass
                        self.q.put(("log", f"Hidden Chromium progress: {index:,}/{total:,}"))
                except Exception as exc:
                    self.frame_done = index
                    self.q.put(("progress", index, total, "Hidden Chromium"))
                    self.q.put(("log", f"Hidden Chromium capture error at col={job.col} row={job.row}: {exc}"))
            try:
                if self.frame_mem is not None:
                    self.frame_mem.flush()
            except Exception:
                pass
        except Exception as exc:
            self.q.put(("log", f"Hidden Chromium renderer failed: {exc}"))
        finally:
            try:
                shutil.rmtree(profile_dir, ignore_errors=True)
                shutil.rmtree(png_dir, ignore_errors=True)
            except Exception:
                pass
            self.q.put(("frame_done", stopped))

    def _start_frame_dispatch(self) -> None:
        """Start all frame renderer queues after WebEngine windows had time to appear."""
        if not self.frame_active:
            return
        try:
            QApplication.processEvents()
        except Exception:
            pass
        for item in list(getattr(self, "frame_renderers", [])):
            try:
                QTimer.singleShot(0, lambda it=item: self.dispatch_next_frame_job(it))
            except Exception as exc:
                self.log_msg(f"Could not start frame renderer {item.get('index')}: {exc}")

    def dispatch_next_frame_job(self, item) -> None:
        if not self.frame_active:
            return
        if self.stop_event.is_set():
            self.finish_frame_screenshot_job(stopped=True)
            return
        if self.frame_queue_index >= len(self.frame_queue):
            item["busy"] = False
            if all(not it.get("busy") for it in self.frame_renderers):
                self.finish_frame_screenshot_job(stopped=False)
            return

        job = self.frame_queue[self.frame_queue_index]
        self.frame_queue_index += 1
        item["busy"] = True
        item["job"] = job
        item["job_key"] = (job.z, job.col, job.row)
        item["loaded_ok"] = False

        render_w = int(self.frame_render_w_actual or self.render_w_spin.value())
        render_h = int(self.frame_render_h_actual or self.render_h_spin.value())

        # Exact-size mode: URL math and captured WebView use the same render_w/render_h.
        url, visible_bounds, request_bounds = expand_frame_url_grid(
            self.frame_cfg.url_template,
            job.col, job.row, job.z,
            self.frame_selected_west, self.frame_selected_north,
            self.frame_request_lon_span, self.frame_request_lat_span,
            self.frame_lon_per_px, self.frame_lat_per_px,
            self.frame_cell_w, self.frame_cell_h,
            int(self.frame_render_w_actual), int(self.frame_render_h_actual),
            APPLE_LEFT_BAR_CROP_PX,
            int(self.crop_top_spin.value()),
            int(self.crop_right_spin.value()),
            int(self.crop_bottom_spin.value()),
            float(getattr(self, "frame_step_mult_x", 1.0)),
            float(getattr(self, "frame_step_mult_y", 1.0)),
            float(getattr(self, "frame_shift_x_px", 0.0)),
            float(getattr(self, "frame_shift_y_px", 0.0)),
            bool(getattr(self, "frame_crop_correct_url", False)),
        )
        actual_cropped_bounds = visible_bounds
        item["url"] = url
        item["expected_url"] = QUrl(url).toString()
        item["visible_bounds"] = actual_cropped_bounds
        item["synthetic_visible_bounds"] = visible_bounds
        item["request_bounds"] = request_bounds
        item["load_event_seen"] = False
        item["iframe_waits"] = 0
        item["blank_retries"] = 0

        if self.frame_done == 0:
            self.log_msg(f"First frame render URL: {url}")
        if item.get("index", 0) < 4:
            self.log_msg(
                f"Frame renderer {item.get('index')} loading cell col={job.col} row={job.row}; "
                f"center_georef={actual_cropped_bounds}; synthetic_visible={visible_bounds}; request={request_bounds}"
            )

        view = item["view"]
        try:
            view.stop()
        except Exception:
            pass
        try:
            view.loadFinished.disconnect()
        except Exception:
            pass
        view.loadFinished.connect(lambda ok, it=item, key=(job.z, job.col, job.row): self.frame_loaded(ok, it, key))
        view.load(QUrl(url))
        load_timeout_ms = max(12000, int(self.rate_spin.value()) + int(self.frame_settle_extra_spin.value()) + 8000)
        QTimer.singleShot(load_timeout_ms, lambda it=item, key=(job.z, job.col, job.row): self.frame_load_timeout(it, key))

    def frame_loaded(self, ok: bool, item, key=None) -> None:
        # Ignore stale loadFinished events from the previous URL/page.
        if key is not None and item.get("job_key") != key:
            self.log_msg(f"Ignored stale loadFinished for renderer {item.get('index')}: {key} != {item.get('job_key')}")
            return
        try:
            item["view"].loadFinished.disconnect()
        except Exception:
            pass

        item["load_event_seen"] = True
        item["loaded_ok"] = bool(ok)
        if not ok:
            job = item.get("job")
            self.log_msg(f"Warning: renderer {item.get('index')} loadFinished=False for z={job.z if job else '?'} x={job.x if job else '?'} y={job.y if job else '?'}")

        # First wait: page load -> map canvas starts painting.
        wait_ms = max(1000, int(self.rate_spin.value()))
        QTimer.singleShot(wait_ms, lambda it=item, k=key: self.frame_extra_settle_wait(it, k))

    def frame_load_timeout(self, item, key=None) -> None:
        if not self.frame_active:
            return
        if key is not None and item.get("job_key") != key:
            return
        if bool(item.get("load_event_seen")):
            return
        job = item.get("job")
        self.log_msg(
            f"Renderer {item.get('index')} load timeout; continuing after settle for col={job.col if job else '?'} row={job.row if job else '?'}."
        )
        item["load_event_seen"] = True
        self.frame_extra_settle_wait(item, key)

    def frame_extra_settle_wait(self, item, key=None) -> None:
        if not self.frame_active:
            return
        if key is not None and item.get("job_key") != key:
            return
        view = item["view"]
        try:
            def _after_iframe_check(loaded):
                if not self.frame_active:
                    return
                if key is not None and item.get("job_key") != key:
                    return
                if str(loaded) != "1" and int(item.get("iframe_waits", 0)) < 8:
                    item["iframe_waits"] = int(item.get("iframe_waits", 0)) + 1
                    QTimer.singleShot(1000, lambda it=item, k=key: self.frame_extra_settle_wait(it, k))
                    return
                self.frame_after_iframe_ready(item, key)
            view.page().runJavaScript("(!document.body || !document.body.dataset || document.body.dataset.frameLoaded === undefined) ? '1' : (document.body.dataset.frameLoaded || '0');", _after_iframe_check)
            return
        except Exception:
            pass

        self.frame_after_iframe_ready(item, key)

    def frame_after_iframe_ready(self, item, key=None) -> None:
        if not self.frame_active:
            return
        if key is not None and item.get("job_key") != key:
            return
        view = item["view"]
        try:
            # Force repaint/resize before final settle. This reduces captures of
            # the previous map location in Qt WebEngine.
            win = view.window()
            if win:
                win.show()
                win.raise_()
                win.activateWindow()
                view.resize(win.size())
            view.update()
            view.repaint()
            QApplication.processEvents()
        except Exception:
            pass
        extra_ms = max(0, int(self.frame_settle_extra_spin.value())) if hasattr(self, "frame_settle_extra_spin") else 2500
        QTimer.singleShot(extra_ms, lambda it=item, k=key: self.capture_frame_tile(it, k))

    def capture_frame_tile(self, item, key=None) -> None:
        if not self.frame_active:
            return
        if key is not None and item.get("job_key") != key:
            self.log_msg(f"Ignored stale capture for renderer {item.get('index')}: {key} != {item.get('job_key')}")
            return
        job = item.get("job")
        try:
            import numpy as np
            view = item["view"]
            try:
                view.resize(int(self.frame_render_w_actual), int(self.frame_render_h_actual))
                win = view.window()
                if win:
                    win.show()
                    win.raise_()
                    win.activateWindow()
                view.repaint()
                QApplication.processEvents()
            except Exception:
                pass

            pix = view.grab()
            pil = self.qimage_to_pil_rgb(pix.toImage())

            l = APPLE_LEFT_BAR_CROP_PX
            t = int(self.crop_top_spin.value())
            r = pil.width - int(self.crop_right_spin.value())
            b = pil.height - int(self.crop_bottom_spin.value())
            if r <= l or b <= t:
                raise RuntimeError("Crop values remove the full image. Reduce crop.")
            pil = pil.crop((l, t, r, b))

            # Normalize to the fixed visible output cell size calculated at job start.
            # This avoids mismatches if the OS maximized window differs by borders/taskbar.
            if pil.width != self.frame_cell_w or pil.height != self.frame_cell_h:
                pil = pil.resize((self.frame_cell_w, self.frame_cell_h), Image.Resampling.LANCZOS)

            if self.pil_looks_blank(pil) and int(item.get("blank_retries", 0)) < 5:
                item["blank_retries"] = int(item.get("blank_retries", 0)) + 1
                self.log_msg(
                    f"Renderer {item.get('index')} captured a blank cell at col={job.col} row={job.row}; retry {item['blank_retries']}/5."
                )
                try:
                    view.update()
                    view.repaint()
                    QApplication.processEvents()
                except Exception:
                    pass
                QTimer.singleShot(1500, lambda it=item, k=key: self.capture_frame_tile(it, k))
                return

            tile_path = self.frame_tile_dir / f"grid_z{job.z}_col{job.col}_row{job.row}.tif"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = tile_path.with_suffix(".tmp.tif")
            pil.save(tmp, format="TIFF", compression="tiff_deflate")
            os.replace(tmp, tile_path)

            visible_bounds = item.get("visible_bounds")
            if visible_bounds:
                write_worldfile_and_prj(
                    tile_path, self.frame_cell_w, self.frame_cell_h,
                    lonlat_bbox_to_webmercator_bounds(visible_bounds[0], visible_bounds[1], visible_bounds[2], visible_bounds[3])
                )

            arr = np.asarray(pil, dtype=np.uint8)
            r0 = job.row * self.frame_cell_h
            c0 = job.col * self.frame_cell_w
            self.frame_mem[r0:r0+self.frame_cell_h, c0:c0+self.frame_cell_w, :] = arr

            self.frame_done += 1
            self.progress.setValue(self.frame_done)
            self.status_label.setText(f"Frame grid TIFF: {self.frame_done:,}/{self.frame_total:,}")
            if self.frame_done % 5 == 0:
                try:
                    self.frame_mem.flush()
                except Exception:
                    pass
                self.log_msg(f"Frame grid progress: {self.frame_done:,}/{self.frame_total:,}")
        except Exception as exc:
            self.frame_done += 1
            self.progress.setValue(self.frame_done)
            self.log_msg(f"Frame grid capture error at col={job.col if job else '?'} row={job.row if job else '?'}: {exc}")

        try:
            item["view"].stop()
        except Exception:
            pass
        item["busy"] = False
        item["job"] = None
        item["job_key"] = None
        item["visible_bounds"] = None
        item["request_bounds"] = None
        QTimer.singleShot(0, lambda it=item: self.dispatch_next_frame_job(it))

    def finish_frame_screenshot_job(self, stopped: bool = False) -> None:
        if not self.frame_active:
            return
        self.frame_active = False
        try:
            if self.frame_mem is not None:
                self.frame_mem.flush()
                del self.frame_mem
        except Exception:
            pass
        self.frame_mem = None
        try:
            self.clear_frame_renderers()
        except Exception:
            pass
        if stopped:
            self.log_msg("Frame screenshot job stopped. Partial output remains on disk.")
            self.status_label.setText("Stopped")
        else:
            # Optional final merge with the PyPI package "stitching".
            # The old streamed grid GeoTIFF is still written first; if feature
            # stitching fails, the old output remains usable.
            try:
                use_pypi = bool(getattr(self, "pypi_stitching_check", None) and self.pypi_stitching_check.isChecked())
            except Exception:
                use_pypi = False
            if use_pypi:
                try:
                    self.log_msg("Starting PyPI stitching final merge from individual TIFF tiles...")
                    stitch_frame_tiles_with_pypi_stitching(
                        self.frame_tile_dir,
                        self.frame_cfg.output_file,
                        lonlat_bbox_to_webmercator_bounds(
                            self.frame_grid_west,
                            self.frame_grid_south,
                            self.frame_grid_east,
                            self.frame_grid_north,
                        ),
                        self.log_msg,
                    )
                except Exception as exc:
                    self.log_msg(f"PyPI stitching failed; keeping streamed grid GeoTIFF/BigTIFF: {exc}")
            self.log_msg(f"Finished frame screenshot GeoTIFF/BigTIFF: {self.frame_cfg.output_file}")
            self.log_msg(f"Finished individual TIFF tiles: {self.frame_tile_dir}")
            self.status_label.setText("Finished")

    def _run_job(self, cfg: StitchConfig) -> None:
        try:
            stitch_tiles(cfg, self._progress, self._log_thread, self.stop_event)
        except Exception as exc:
            self._log_thread(f"ERROR: {exc}")
            self.q.put(("status", "Error"))

    def _progress(self, done: int, total: int, phase: str) -> None:
        self.q.put(("progress", done, total, phase))

    def _log_thread(self, msg: str) -> None:
        self.q.put(("log", msg))

    def log_msg(self, msg: str) -> None:
        self.log.append(str(msg))

    def _poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "log":
                    self.log_msg(item[1])
                elif item[0] == "progress":
                    _, done, total, phase = item
                    self.progress.setRange(0, max(1, int(total)))
                    self.progress.setValue(int(done))
                    self.status_label.setText(f"{phase}: {done:,}/{total:,}")
                elif item[0] == "status":
                    self.status_label.setText(item[1])
                elif item[0] == "frame_done":
                    self.finish_frame_screenshot_job(stopped=bool(item[1]))
        except queue.Empty:
            pass


def main() -> int:
    if _PYSIDE_IMPORT_ERROR is not None:
        print("PySide6 WebEngine is missing.")
        print("Install with: python -m pip install PySide6 PySide6-Addons PySide6-Essentials shiboken6")
        print("Import error:", _PYSIDE_IMPORT_ERROR)
        return 1
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--ignore-gpu-blocklist")
    set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    try:
        app.setApplicationDisplayName(APP_DISPLAY_TITLE)
    except Exception:
        pass
    icon = load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = PySideMapStitcher()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
