
from dataclasses import dataclass
import datetime
import pandas as pd
import os
from urllib.parse import urlencode
import json
import requests
import numpy as np
import shapely
import functools
import io
from rasterio.io import MemoryFile, DatasetReader
from PIL import Image
import boto3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from sqlalchemy.dialects.postgresql import insert
# from sqlalchemy.orm import DeclarativeBase
from sqlmodel import SQLModel, Field

from pprint import pprint

# from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

import ee

import zipfile
from pprint import pprint

from typing import ClassVar, Callable

LOOKBACK_DAYS = 30
IMAGE_STORAGE_BUCKET = "public-zone-117819748843-us-east-1"
IMAGE_STORAGE_PREFIX = "water_body_satellite_images/"
THUMBNAIL_STORAGE_PREFIX = "water_body_satellite_thumbnails/"
SATELLITE_IMAGE_TABLE = "waterbody_satellite_images"
MAX_IMAGE_SIDE_PIXELS = "2100"
THUMBNAIL_SCALE_FACTOR = 7
IMAGE_NODATA_VALUE = int(0)
PARALLELISM = 10
check_existing_images = True

end_date = datetime.date.today()
start_date = datetime.date.today() - datetime.timedelta(days=LOOKBACK_DAYS)
waterbody_limit = 1100
area_limit = 900

from .database import engine

# ee_key_file = "waterbodyweather-key.json"

# with open(ee_key_file, "r") as f:
#     key_data = f.read()

# ""

secret_arn = os.environ.get("DB_CREDS_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:117819748843:secret:google-earth-engine-api-key")


key_data = boto3.client("secretsmanager", 'us-east-1').get_secret_value(SecretId=secret_arn)["SecretString"]


import ee
service_account = 'service-google-earth-api@waterbodyweather.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)

ee.Initialize(credentials)


@dataclass
class ImageBand:
    band: str
    color_name: str
    # pixel_value_factor: float
    # pixel_value_offset: float
    # pixel_value_transform: Callable = lambda x: x  # how to transform pixel values for all bands


satellite_dataset_configs = [
        # {
        #     "ee_dataset_name": "LANDSAT/LC09/C02/T1_L2",
        #     "bands": (
        #         ImageBand('SR_B4', 'red', 0.0000275, -0.2), 
        #         ImageBand('SR_B3', 'green', 0.0000275, -0.2), 
        #         ImageBand('SR_B2', 'blue', 0.0000275, -0.2),
        #     )
        # },
        {
            "ee_dataset_name": "COPERNICUS/S2_SR_HARMONIZED",
            "ee_filter": ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20),
            "bands": (
                ImageBand('TCI_R', 'red'), 
                ImageBand('TCI_G', 'green'), 
                ImageBand('TCI_B', 'blue'),
            )
        }
    ]


@dataclass
class GoogleEarthImageLayer:
    band: ImageBand
    image: DatasetReader = None
    image_array: np.ndarray = None
    clipped_image: DatasetReader = None


class WaterBodySatelliteImage(SQLModel, table=True):    
    __tablename__ = "waterbody_satellite_images"

    waterbody_id: int = Field(primary_key=True)
    captured_ts: datetime.datetime = Field(primary_key=True)
    ee_id: str 
    satellite_dataset: str
    properties: str
    filename: str
    thumbnail_filename: str
    red_average: float
    green_average: float
    blue_average: float
    white_fraction: float


@dataclass
class GoogleEarthImageReference:
    ee_id: str
    waterbody_id: int
    captured_ts: datetime.datetime
    properties: str
    filename: str
    thumbnail_filename: str
    channel_means: "list[float]"

@dataclass
class GoogleEarthImage:
    query: "GoogleEarthImageQuery"
    ee_id: str
    captured_ts: datetime.datetime
    properties: dict
    ee_image: ee.Image = None
    layers: "list[GoogleEarthImageLayer]" = None
    image_filename: str = None
    thumbnail_filename: str = None
    clipped_image_filename: str = None

    def __post_init__(self):
        filename = f"{self.ee_id}/{self.query.id}_{self.captured_ts.strftime('%Y%M%d%H%m%S')}"

        print(f"creating {filename}")

        self.image_filename = f"{filename}.tif"
        self.thumbnail_filename = f"{filename}_thumbnail.png"
        self.clipped_image_filename = f"{filename}_clipped.tif"

        self.ee_image = (
            ee.Image(self.ee_id)
            .select(opt_selectors=[b.band for b in self.query.bands], opt_names=[b.color_name for b in self.query.bands])
        )

        self.layers = [
            GoogleEarthImageLayer(band)
            for band in self.query.bands
        ]


    def download_layers(self):
        download_props = {
            # 'scale': meters_per_pixel, # meters per pixel
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'region': self.query.bounding_box,
            'dimensions': MAX_IMAGE_SIDE_PIXELS
        }

        if "image" in download_props.keys():
            del download_props["image"]

        url = self.ee_image.getDownloadURL(download_props)
        
        img_response = requests.get(url, stream=True)

        with zipfile.ZipFile(io.BytesIO(img_response.content)) as z:
            # There is a separate tif file for each band
            assert len(z.namelist()) == len(self.query.bands)

            for layer in self.layers:
                
                image_filename = f"{self.properties['system:index']}.{layer.band.color_name}.tif"
                if image_filename not in z.namelist():
                    image_filename = f"download.{layer.band.color_name}.tif"
                img_data = z.read(image_filename)

                with MemoryFile() as memfile:
                    memfile.write(img_data)

                    layer.image = memfile.open()

                    raw_image_array = layer.image.read()
                    assert len(raw_image_array) == 1

                    layer.image_array = raw_image_array[0] #

                    # print(layer.image_array)
                    # print(np.mean(layer.image_array))
                    # print(np.max(layer.image_array))
                    # print(layer.image.profile)
                    # layer.image_array = (raw_image_array[0] * layer.band.pixel_value_factor) + layer.band.pixel_value_offset
        print(f"waterbody id: {self.query.id}")
        print(self.ee_id)

    def combine_image_layers(self):

        self.tif_profile = {}
        for layer in self.layers:
            self.tif_profile.update(layer.image.profile)

        self.tif_profile['count'] = len(self.layers)

        self.tif_profile['photometric'] = "RGB"

        # The 'tiled' propertty causes issues with the image
        del self.tif_profile['tiled']

        self.tif_memfile = MemoryFile()

        with self.tif_memfile.open(**self.tif_profile) as tif_bytes:

            self.image_array = np.array([layer.image_array for layer in self.layers])

            for layer in self.layers:
                assert layer.image.crs.data["init"] == 'epsg:4326'

            self.layers = None # Attempt at reducing memory usage

            # for i, layer in enumerate(self.layers):
                # print(layer)
            tif_bytes.write(self.image_array)

            from rasterio.plot import show
            from rasterio.mask import mask
            # show(layer.image)

            
            
            # boundary_img_coords = [json.loads(shapely.to_geojson(self.query.boundary))]
            boundary_img_coords = [shapely.geometry.mapping(self.query.boundary)]

            self.clipped_image_array, out_transform = mask(dataset=tif_bytes, shapes=boundary_img_coords, crop=True, nodata=IMAGE_NODATA_VALUE)

        

        self.image_channel_means = []
        for channel in self.clipped_image_array:
            self.image_channel_means.append(
                np.mean(channel[channel != IMAGE_NODATA_VALUE])
            )

        # print(self.clipped_image_array)

        self.clipped_tif_memfile = MemoryFile()
        with self.clipped_tif_memfile.open(**self.tif_profile) as tif_bytes:
            tif_bytes.write(self.clipped_image_array)

    def run_image_calculations(self):

        self.image_channel_means = []
        for channel in self.clipped_image_array:
            self.image_channel_means.append(
                np.mean(channel[channel != IMAGE_NODATA_VALUE])
            )

        # Definition of white is rgb channels are all above 128 (out of 255) 
        # and all colors are within 15% of each other
        # for 8 bit color/sRGB
        assert self.clipped_image_array.dtype == np.uint8

        img = self.clipped_image_array.transpose((2, 1, 0))
        
        channel_max = 255 # for unint8

        is_out_of_bounds = img.max(2) == IMAGE_NODATA_VALUE
        is_white = (img.min(2) >= int(0.6 * channel_max)) & ((img.max(2) - img.min(2)) <= int(0.1 * channel_max))

        num_in_bounds_pixels = is_out_of_bounds.size - is_out_of_bounds.sum()

        self.white_fraction = is_white.sum() / num_in_bounds_pixels

        print("white:")
        print(
            self.white_fraction
        )

    def create_thumbnail_image(self):
  
        _, height, width = np.shape(self.image_array)
        thumbnail_width = int(width / THUMBNAIL_SCALE_FACTOR)
        thumbnail_height = int(height / THUMBNAIL_SCALE_FACTOR)

        self.thumbnail_tif_memfile = MemoryFile()
        with self.thumbnail_tif_memfile.open(**self.tif_profile) as tif_bytes:
            tif_bytes.write(self.image_array)

        assert self.image_array.dtype == np.uint8  # Image.fromarray only works with 8 bit colors in RGB mode
        
        thumbnail_img = (
            Image.open(self.thumbnail_tif_memfile, formats=["TIFF"]) # .fromarray(self.image_array) # .transpose((2,1,0)), 'RGB' #
            # Downsample image for thumbnail
            .resize((thumbnail_width, thumbnail_height), Image.BICUBIC)
        )

        self.thumbnail_png_bytes = io.BytesIO()
        # thumbnail_img.save('test.png')
        thumbnail_img.save(self.thumbnail_png_bytes, format="PNG")


    def write_images_to_s3(self):

        files_to_upload = [
            {"Body": self.tif_memfile, "Bucket": IMAGE_STORAGE_BUCKET , "Key": f'{IMAGE_STORAGE_PREFIX}{self.image_filename}'},
            {"Body": self.thumbnail_png_bytes.getvalue(), "Bucket": IMAGE_STORAGE_BUCKET , "Key": f'{IMAGE_STORAGE_PREFIX}{self.thumbnail_filename}'},
            {"Body": self.clipped_tif_memfile, "Bucket": IMAGE_STORAGE_BUCKET , "Key": f'{IMAGE_STORAGE_PREFIX}{self.clipped_image_filename}'},
        ]

        s3 = boto3.client('s3')

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(s3.put_object, **file_to_upload) for file_to_upload in files_to_upload]
        
        for future in futures:
            future.result()


    def to_image_reference(self):
        """This will be stored in the DB"""
        return WaterBodySatelliteImage(
            waterbody_id=self.query.id,
            captured_ts=self.captured_ts,
            satellite_dataset=self.query.ee_dataset_name,
            ee_id=self.ee_id,
            properties=json.dumps(self.properties),
            filename=self.image_filename,
            thumbnail_filename=self.thumbnail_filename,
            red_average=self.image_channel_means[0],
            green_average=self.image_channel_means[1],
            blue_average=self.image_channel_means[2],
            white_fraction=self.white_fraction
        )


@dataclass
class GoogleEarthImageQuery:
    id: int
    areasqkm: float
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float
    latitude: float
    longitude: float
    start_date: datetime.date
    end_date: datetime.date
    geometry: "list[list[list[float]]]"
    exclude_ee_ids: "list[str]"
    ee_dataset_name: str
    bands: "tuple[ImageBand]"
    ee_filter: ee.Filter = None

    images: "tuple[GoogleEarthImage]" = None

    # api_url: ClassVar[str] = 'earthengine.googleapis.com/v1alpha/projects/earthengine-public/assets/COPERNICUS/S2'

    def __post_init__(self):
        self.latitude = float(self.latitude)
        self.longitude = float(self.longitude)

        if self.exclude_ee_ids is None:
            self.exclude_ee_ids = []

        # remove z element
        for polygon in self.geometry:
            for point in polygon:
                del point[2]

        self.width_degrees = abs(self.max_latitude - self.min_latitude)
        self.height_degrees = abs(self.max_longitude - self.min_longitude)

        self.bounding_box = ee.Geometry.BBox(self.min_longitude, self.min_latitude, self.max_longitude, self.max_latitude) #ee.Geometry.BBox(-121.55, 39.01, -120.57, 39.38)

        self.ee_centerpoint = ee.Geometry.Point((self.longitude, self.latitude)) # Note: latitude and longitude are reversed in the args here
        
        polygons = [shapely.Polygon(polygon) for polygon in self.geometry]

        self.boundary = functools.reduce(lambda x, y: x.union(y), polygons)

        self.ee_boundary = ee.Geometry.Polygon(self.boundary.__geo_interface__['coordinates'])

        print(f"Initialized {self.id}")

    
    def get_image_list(self):

        img_collection: ee.ImageCollection = (
            ee.ImageCollection(self.ee_dataset_name) #dataset catalog: https://developers.google.com/earth-engine/datasets/
            .filterBounds(self.ee_centerpoint)
            .filterDate(ee.Date(self.start_date.isoformat()), ee.Date(self.end_date.isoformat()))
            .select(selectors=[b.band for b in self.bands], opt_names=[b.color_name for b in self.bands])
            # .addBands(ee.Image.pixelLonLat())
        )

        if self.ee_filter:
            img_collection = img_collection.filter(self.ee_filter)

        # print(img_collection.getInfo())

        info = img_collection.getInfo()
        if info is None:
            raise Exception(f"No images found for {self}")

        self.images: list[GoogleEarthImage] = []
        for img in info['features']:
            # Exclude ee ids that already have been downloaded
            if not check_existing_images or img['id'] not in self.exclude_ee_ids: #not check_existing_images or 
                capture_ts_unix = int(int(img['properties']['system:time_start']) / 1000)

                print(datetime.datetime.fromtimestamp(capture_ts_unix))

                self.images.append(
                    GoogleEarthImage(
                        query=self,
                        ee_id=img['id'],
                        captured_ts=datetime.datetime.fromtimestamp(capture_ts_unix),
                        properties=img['properties']
                    )
                )

        return self.images


def run_image_query(row):
    for dataset_config in satellite_dataset_configs:

        image_query = GoogleEarthImageQuery(
            start_date=start_date, #datetime.date(2023, 1, 1),
            end_date=end_date, #datetime.date(2023, 1, 30), #
            **dataset_config,
            **dict(row.items())
        )

        images = image_query.get_image_list()

        for image in images:
            image.download_layers()
            image.combine_image_layers()
            image.run_image_calculations()
            image.create_thumbnail_image()
            image.write_images_to_s3()

            table_record = image.to_image_reference()

            with engine.connect() as conn:
                stmt = insert(WaterBodySatelliteImage).values(table_record.dict())
                stmt = stmt.on_conflict_do_nothing()
                result = conn.execute(stmt)
                print(f"query result: {result}")

def main():
    import sys
    print(sys.version)
    # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2#bands

    water_bodies_table_name = "water_bodies"

    water_bodies_df = pd.read_sql(sql=f"""
                                WITH already_downloaded_images AS (
                                    SELECT waterbody_id, ARRAY_AGG(ee_id) as exclude_ee_ids
                                    FROM waterbody_satellite_images
                                    WHERE captured_ts >= '{start_date.isoformat()}'::DATE
                                    GROUP BY waterbody_id
                                )
                                
                                SELECT b.*, g.geometry, d.exclude_ee_ids
                                FROM {water_bodies_table_name} b
                                LEFT JOIN water_body_geometries g
                                ON b.id = g.id
                                LEFT JOIN already_downloaded_images d
                                ON b.id = d.waterbody_id
                                --WHERE b.id = 4342 --9725
                                WHERE b.areasqkm < {area_limit}
                                order by b.areasqkm desc
                                LIMIT {waterbody_limit};""", 
                                con=engine.connect()
                            )

    water_bodies_df = water_bodies_df[["id", "areasqkm", "min_longitude", "max_longitude", "min_latitude", "max_latitude", "latitude", "longitude", "geometry", "exclude_ee_ids"]]

    print(water_bodies_df)

    futures = []
    with ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
        for i, row in water_bodies_df.iterrows():
            # futures.append(
            #     executor.submit(run_image_query, row)
            # )
            run_image_query(row)

    for future in futures:
        res = future.result()
        if res is not None:
            print(
                future.result()
            )


if __name__ == "__main__":
    main()