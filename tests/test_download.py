from deepforest import download
import os
import asyncio
from aiolimiter import AsyncLimiter
import matplotlib.pyplot as plt
import cv2
import rasterio as rio
import pytest


def url():
    return [
        "https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2020_CIR/ImageServer/",
        "https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/",
        "https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer"
    ]


def boxes():
    return [
        (-124.112622, 40.493891, -124.111536, 40.49457),
        (-114.12529, 51.072134, -114.12117, 51.07332),
        (-73.763941, 41.111032, -73.763447, 41.111626)
    ]


def additional_params():
    return [None, None, {"format": "png"}]


def download_service():
    return ["exportImage", "exportImage", "export"]


# Pair each URL with its corresponding box
url_box_pairs = list(zip(["CA.tif", "MA.tif", "NY.png"], url(), boxes(), additional_params(), download_service()))


@pytest.mark.parametrize("image_name, url, box, params, download_service_name", url_box_pairs)
def test_download_arcgis_rest(tmpdir, image_name, url, box, params, download_service_name):
    async def run_test():
        semaphore = asyncio.Semaphore(20)
        limiter = AsyncLimiter(1, 0.05)
        xmin, ymin, xmax, ymax = box
        bbox_crs = "EPSG:4326"  # Assuming WGS84 for bounding box CRS
        savedir = tmpdir
        filename = await download.download_web_server(semaphore, limiter, url, xmin, ymin, xmax, ymax, bbox_crs,
                                                      savedir, additional_params=params, image_name=image_name,
                                                      download_service=download_service_name)
        # Check the saved file
        assert os.path.exists(filename)

        # Confirm file has CRS
        with rio.open(filename) as src:
            if image_name.endswith('.tif'):
                assert src.crs is not None
                plt.imshow(src.read().transpose(1, 2, 0))
            else:
                assert src.crs is None
                plt.imshow(cv2.imread(filename)[:, :, ::-1])
    asyncio.run(run_test())


locations = [
    [
        'https://aerial.openstreetmap.org.za/layer/ngi-aerial/{z}/{x}/{y}.jpg',
        -33.9249, 18.4241,
        -30.0000, 22.0000,
        6,
        True,
        'dataset3',
        'CapeTown.tiff'
    ],
    [
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        45.699, 127,  # From (latitude, longitude)
        30, 148.492,  # To (latitude, longitude)
        6,  # Zoom level
        True,
        'dataset',
        'output.tiff'
    ],
]


# Parametrize test cases with different locations
@pytest.mark.parametrize("source, lat0, lon0, lat1, lon1, zoom, save_image, save_dir, image_name", locations)
def test_download_tile_mapserver(tmpdir, source, lat0, lon0, lat1, lon1, zoom, save_image, save_dir, image_name):
    async def run_test():
        semaphore = asyncio.Semaphore(20)
        limiter = AsyncLimiter(1, 0.05)
        save_path = os.path.join(tmpdir, image_name)
        await download.download_web_server(semaphore, limiter, source, lat0, lon0, lat1, lon1, zoom, save_image=True,
                                           save_dir=tmpdir, image_name=image_name)
        # Check if the image file is saved
        assert os.path.exists(save_path)
        # Confirm file format and load image
        img = cv2.imread(save_path)
        assert img is not None

    asyncio.run(run_test())
