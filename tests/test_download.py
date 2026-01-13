import asyncio
import os
from unittest.mock import patch, AsyncMock

import cv2
import matplotlib.pyplot as plt
import pytest
import rasterio as rio
from aiolimiter import AsyncLimiter
from rasterio.errors import RasterioIOError

from deepforest import download


def url():
    return [
        "https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer"
    ]


def boxes():
    return [(-73.763941, 41.111032, -73.763447, 41.111626)]


def additional_params():
    return [{"format": "png"}]


def download_service():
    return ["export"]


# Pair each URL with its corresponding box
url_box_pairs = list(zip(["NY.png"], url(), boxes(), additional_params(), download_service()))


def mock_arcgis_response_json():
    mock_json_resp = AsyncMock()
    mock_json_resp.__aenter__.return_value = mock_json_resp
    mock_json_resp.read = AsyncMock(return_value=b'{"spatialReference": {"latestWkid": 4326}}')
    mock_json_resp.headers = {"Content-Type": "application/json"}
    return mock_json_resp


def mock_arcgis_response_image():
    mock_img_resp = AsyncMock()
    mock_img_resp.__aenter__.return_value = mock_img_resp
    mock_img_resp.status = 200
    with open(os.path.join(os.path.dirname(__file__), "data", "NY.png"), "rb") as f:
        mock_img_resp.read = AsyncMock(return_value=f.read())

    return mock_img_resp


@pytest.mark.parametrize("image_name, url, box, params, download_service_name", url_box_pairs)
def test_download_arcgis_rest(tmp_path, image_name, url, box, params, download_service_name):
    async def run_test():
        semaphore = asyncio.Semaphore(20)
        limiter = AsyncLimiter(1, 0.05)
        xmin, ymin, xmax, ymax = box
        bbox_crs = "EPSG:4326"  # Assuming WGS84 for bounding box CRS
        savedir = tmp_path

        # Mock network requests to prevent flakes
        with patch("aiohttp.ClientSession.get", side_effect=[mock_arcgis_response_json(), mock_arcgis_response_image()]):
            filename = await download.download_web_server(
                semaphore, limiter, url, xmin, ymin, xmax, ymax, bbox_crs,
                savedir=savedir, additional_params=params, image_name=image_name,
                download_service=download_service_name
            )

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
def test_download_tile_mapserver(tmp_path, source, lat0, lon0, lat1, lon1, zoom, save_image, save_dir, image_name):
    async def run_test():
        semaphore = asyncio.Semaphore(20)
        limiter = AsyncLimiter(1, 0.05)
        save_path = tmp_path / image_name
        await download.download_web_server(semaphore, limiter, source, lat0, lon0, lat1, lon1, zoom, save_image=True,
                                           save_dir=tmp_path, image_name=image_name)
        try:
            # Check if the image file is saved
            assert os.path.exists(save_path)
            # Confirm file format and load image
            img = cv2.imread(save_path)
            assert img is not None
        except RasterioIOError:
            pytest.skip("Rasterio IO Error - likely due to download failure")
        except asyncio.TimeoutError:
            pytest.skip("Timeout error from download web server")

    asyncio.run(run_test())
