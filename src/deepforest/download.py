import json
import aiohttp
from pyproj import CRS
import geopandas as gpd
import shapely
import math
import io
from PIL import Image
import os
from tqdm import tqdm
import asyncio
import itertools
import re

Image.MAX_IMAGE_PIXELS = None


def deg2num(lat, lon, zoom):
    n = 2**zoom
    xtile = ((lon + 180) / 360 * n)
    ytile = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) * n / 2
    return (xtile, ytile)


async def fetch_tile(url, session, x, y):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                content_type = resp.headers.get('Content-Type', '').lower()
                if 'image' in content_type:
                    return await resp.read(), (x, y)
                else:
                    print(f"Warning: Non-image content received for tile ({x}, {y})")
    except Exception as e:
        print(f"Error fetching tile ({x}, {y}): {e}")
    return None, (x, y)


def is_empty(im):
    extrema = im.getextrema()
    if len(extrema) >= 3:
        if len(extrema) > 3 and extrema[-1] == (0, 0):
            return True
        for ext in extrema[:3]:
            if ext != (0, 0):
                return False
        return True
    else:
        return extrema[0] == (0, 0)


def paste_tile(bigim, base_size, tile, corner_xy, bbox):
    if tile is None:
        return bigim
    im = Image.open(io.BytesIO(tile))
    mode = 'RGB' if im.mode == 'RGB' else 'RGBA'
    size = im.size
    if bigim is None:
        base_size[0] = size[0]
        base_size[1] = size[1]
        newim = Image.new(mode,
                          (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1])))
    else:
        newim = bigim

    dx = abs(corner_xy[0] - bbox[0])
    dy = abs(corner_xy[1] - bbox[1])
    xy0 = (size[0] * dx, size[1] * dy)
    if mode == 'RGB':
        newim.paste(im, xy0)
    else:
        if im.mode != mode:
            im = im.convert(mode)
        if not is_empty(im):
            newim.paste(im, xy0)
    im.close()
    return newim


async def download_ArcGIS_REST(semaphore,
                               limiter,
                               url,
                               xmin,
                               ymin,
                               xmax,
                               ymax,
                               bbox_crs,
                               savedir,
                               additional_params=None,
                               image_name="image.tiff",
                               download_service="exportImage"):
    """Fetch data from a web server using geographic boundaries and save it as
    a GeoTIFF file. This function is used to download data from an ArcGIS REST
    service, not WMTS or WMS services. Example url: https://gis.calgary.ca/arcg
    is/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/

    Parameters:
    - semaphore: An asyncio.Semaphore instance to limit concurrent downloads.
    - limiter: An asyncio-based rate limiter to control the download rate.
    - url: The base URL of the ArcGIS REST service
    - xmin: The minimum x-coordinate (longitude).
    - ymin: The minimum y-coordinate (latitude).
    - xmax: The maximum x-coordinate (longitude).
    - ymax: The maximum y-coordinate (latitude).
    - bbox_crs: The coordinate reference system (CRS) of the bounding box.
    - savedir: The directory to save the downloaded image.
    - additional_params: Additional query parameters to include in the request (default is None).
    - image_name: The name of the image file to be saved (default is "image.tiff").
    - download_service: The specific service to use for downloading the image (default is "exportImage").

    Returns:
    - The file path of the saved image if the download is successful.
    - None if the download fails.

    Function usage:
        import asyncio
        from asyncio import Semaphore
        from aiolimiter import AsyncLimiter
        from deepforest import utilities

        async def main():
            semaphore = Semaphore(10)
            limiter = AsyncLimiter(1, 0.5)
            url = "https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2020_CIR/ImageServer/
            xmin, ymin, xmax, ymax = -114.0719, 51.0447, -114.001, 51.075
            bbox_crs = "EPSG:4326"
            savedir = "/path/to/save"
            image_name = "example_image.tiff"
            await utilities.download_ArcGIS_REST(semaphore, limiter, url, xmin, ymin, xmax, ymax, bbox_crs, savedir, image_name=image_name)

        asyncio.run(main())

    For more details on the function usage, refer to https://deepforest.readthedocs.io/en/latest/annotation.html.
    """

    params = {"f": "json"}

    async with aiohttp.ClientSession() as session:
        await semaphore.acquire()
        async with limiter:
            try:
                async with session.get(url, params=params) as resp:
                    response = await resp.read()
                    response_dict = json.loads(response)
                    spatialReference = response_dict["spatialReference"]
                    if "latestWkid" in spatialReference:
                        wkid = spatialReference["latestWkid"]
                        crs = CRS.from_epsg(wkid)
                    elif 'wkt' in spatialReference:
                        crs = CRS.from_wkt(spatialReference['wkt'])

                bbox = f"{xmin},{ymin},{xmax},{ymax}"
                bounds = gpd.GeoDataFrame(geometry=[
                    shapely.geometry.box(xmin, ymin, xmax, ymax)
                ],
                                          crs=bbox_crs).to_crs(crs).bounds

                params.update({
                    "bbox":
                        f"{bounds.minx[0]},{bounds.miny[0]},{bounds.maxx[0]},{bounds.maxy[0]}",
                    "f":
                        "image",
                    'format':
                        'tiff',
                })

                if additional_params:
                    params.update(additional_params)

                download_url_service = f"{url}/{download_service}"
                async with session.get(download_url_service, params=params) as resp:
                    response = await resp.read()
                    if resp.status == 200:
                        filename = f"{savedir}/{image_name}"
                        with open(filename, "wb") as f:
                            f.write(response)
                        return filename
                    else:
                        raise Exception(f"Failed to fetch data: {resp.status}")

            except Exception as e:
                print(f"Error downloading image {image_name}: {e}")
            finally:
                semaphore.release()


async def download_TileMapServer(semaphore,
                                 limiter,
                                 source,
                                 lat0,
                                 lon0,
                                 lat1,
                                 lon1,
                                 zoom,
                                 save_image=True,
                                 save_dir=None,
                                 image_name='image.tiff'):
    """Download map tiles from a Tile Map Server (TMS) and optionally save them
    as a single image.

    This function uses a Tile Map Server to download individual map tiles, stitches them together,
    and optionally saves the resulting image as a GeoTIFF file. It supports concurrent downloads
    with rate limiting and semaphore control.

    Parameters:
    - semaphore: Semaphore to limit the number of concurrent downloads.
    - limiter: Rate limiter to control the download rate.
    - source: URL template for the Tile Map Server, with placeholders for zoom level (z), and tile coordinates (x, y).
    - lat0: Latitude of the first point defining the bounding box.
    - lon0: Longitude of the first point defining the bounding box.
    - lat1: Latitude of the second point defining the bounding box.
    - lon1: Longitude of the second point defining the bounding box.
    - zoom: Zoom level for the map tiles.
    - save_image: Whether to save the downloaded tiles as a single image. Default is True.
    - save_dir: Directory where the image will be saved. Required if save_image is True.
    - image_name: Name of the output image file. Default is 'image.tiff'.

    Returns:
    - retim: The stitched image of map tiles, or None if save_image is False.

    Raises:
    - Exception: If there is an error during the download process.

    Example:
    --------
    async def main():
        semaphore = asyncio.Semaphore(10)
        limiter = aiolimiter.AsyncLimiter(1, 0.5)
        source = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        lat0, lon0, lat1, lon1 = 34.0522, -118.2437, 34.0523, -118.2436
        zoom = 15
        save_dir = "./maps"
        image_name = "la_area.tiff"
        await download_TileMapServer(semaphore, limiter, source, lat0, lon0, lat1, lon1, zoom, save_image=True, save_dir=save_dir, image_name=image_name)

    asyncio.run(main())
    """
    async with aiohttp.ClientSession() as session:
        await semaphore.acquire()
        async with limiter:
            try:
                session.headers.update({
                    "Accept":
                        "*/*",
                    "Accept-Encoding":
                        "gzip, deflate",
                    "User-Agent":
                        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
                })
                x0, y0 = deg2num(lat0, lon0, zoom)
                x1, y1 = deg2num(lat1, lon1, zoom)
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0

                corners = tuple(
                    itertools.product(range(math.floor(x0), math.ceil(x1)),
                                      range(math.floor(y0), math.ceil(y1))))
                totalnum = len(corners)
                done_num = 0
                tasks = []

                for x, y in corners:
                    task = fetch_tile(source.format(z=zoom, x=x, y=y), session, x, y)
                    tasks.append(task)
                results = await asyncio.gather(*tasks)
                bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
                bigim = None
                base_size = [256, 256]

                with tqdm(total=totalnum, desc="Downloading tiles") as pbar:
                    for result in results:
                        img_data, xy = result
                        if save_image:
                            bigim = paste_tile(bigim, base_size, img_data, xy, bbox)
                        done_num += 1
                        pbar.update(1)

                if not save_image:
                    return None, None

                xfrac = x0 - bbox[0]
                yfrac = y0 - bbox[1]
                x2 = round(base_size[0] * xfrac)
                y2 = round(base_size[1] * yfrac)
                imgw = round(base_size[0] * (x1 - x0))
                imgh = round(base_size[1] * (y1 - y0))
                retim = bigim.crop((x2, y2, x2 + imgw, y2 + imgh))
                if retim.mode == 'RGBA' and retim.getextrema()[3] == (255, 255):
                    retim = retim.convert('RGB')
                bigim.close()

                os.makedirs(save_dir, exist_ok=True)
                retim.save(os.path.join(save_dir, image_name), format='TIFF')
                return retim

            except Exception as e:
                print(f"Error: {e}")
            finally:
                semaphore.release()


async def download_web_server(semaphore, limiter, url, *args, **kwargs):
    """Wrapper function to determine the appropriate download method based on
    the URL.

    Parameters:
    - semaphore: An asyncio.Semaphore instance to limit concurrent downloads.
    - limiter: An asyncio-based rate limiter to control the download rate.
    - url: The base URL of the web server to download data from.
    - *args: Additional positional arguments for the specific download function.
    - **kwargs: Additional keyword arguments for the specific download function.

    Returns:
    - The result of the specific download function.
    """
    if re.search(r'/arcgis/rest/services/', url, re.IGNORECASE):
        return await download_ArcGIS_REST(semaphore, limiter, url, *args, **kwargs)
    elif re.search(r'/{z}/{x}/{y}', url, re.IGNORECASE):
        return await download_TileMapServer(semaphore, limiter, url, *args, **kwargs)
    else:
        raise ValueError("Unsupported URL pattern for download.")
