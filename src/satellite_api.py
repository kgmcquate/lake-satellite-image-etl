
from dataclasses import dataclass
import datetime
import pandas as pd



from urllib.parse import urlencode
import json
import requests
import rasterio
import os
import cv2
import numpy as np
import math
import shapely
import functools
import io
from rasterio.io import MemoryFile, DatasetReader
import geopandas as gpd

from pprint import pprint

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

import ee

import zipfile
from pprint import pprint
# https://api.nasa.gov/

from typing import ClassVar, Callable

IMAGE_STORAGE_BUCKET = "public-zone-117819748843-us-east-1"
IMAGE_STORAGE_PREFIX = "water_body_satellite_images/"
THUMBNAIL_STORAGE_PREFIX = "water_body_satellite_thumbnails/"
MAX_IMAGE_SIDE_PIXELS = "2048"
THUMBNAIL_SCALE_FACTOR = 16



def apply_crs_to_shapely(shape: shapely.Polygon, crs_data):
    from fiona.crs import from_epsg
    return [
        json.loads(
            gpd.GeoDataFrame(
                {'geometry': shape}, 
                index=[0], 
                crs=from_epsg(4326)
            )
            .to_crs(crs=crs_data)
            .to_json()
        )
        ['features'][0]['geometry']
    ]

@dataclass
class ImageBand:
    band: str
    color_name: str
    pixel_value_factor: float
    pixel_value_offset: float
    # pixel_value_transform: Callable = lambda x: x  # how to transform pixel values for all bands

@dataclass
class GoogleEarthImageLayer:
    band: ImageBand
    image: DatasetReader = None
    image_array: np.ndarray = None
    clipped_image: DatasetReader = None


@dataclass
class GoogleEarthImageReference:
    ee_id: str
    waterbody_id: int
    captured_ts: datetime.datetime
    properties: str
    filename: str
    thumbnail_filename: str
    channel_means: list[float]

@dataclass
class GoogleEarthImage:
    query: "GoogleEarthImageQuery"
    ee_id: str
    captured_ts: datetime.datetime
    properties: dict
    ee_image: ee.Image = None
    layers: list[GoogleEarthImageLayer] = None
    image_filename: str = None
    thumbnail_filename: str = None
    clipped_image_filename: str = None

    def __post_init__(self):
        filename = f"{self.ee_id}/{self.query.id}_{self.captured_ts.strftime('%Y%M%d%H%m%S')}"

        self.image_filename = f"{filename}.tif"
        self.thumbnail_filename = f"{filename}_thumbnail.tif"
        self.clipped_image_filename = f"{filename}_clipped.tif"

        self.ee_image = (
            ee.Image(self.ee_id)
            .select(opt_selectors=[b.band for b in self.query.bands], opt_names=[b.color_name for b in self.query.bands])
            # Values from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
            # .multiply(0.0000275)
            # .add(-0.2)
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

            # print(self.properties)
            
            for layer in self.layers:
                image_filename = f"{self.properties['system:index']}.{layer.band.color_name}.tif"
                img_data = z.read(image_filename)

                with MemoryFile() as memfile:
                    memfile.write(img_data)

                    layer.image = memfile.open()

                    raw_image_array = layer.image.read()
                    assert len(raw_image_array) == 1

                    layer.image_array = raw_image_array[0] #
                    # layer.image_array = (raw_image_array[0] * layer.band.pixel_value_factor) + layer.band.pixel_value_offset
        

        # self.clipped_image_array =     
        # out_meta=src.meta.copy() # copy the metadata of the source DEM

        print(self.ee_id)
        print(self.captured_ts)

    def combine_image_layers(self):

        tif_profile = {}
        for layer in self.layers:
            tif_profile.update(layer.image.profile)

        # for layer in self.layers:
        #     print(layer.image_array)

        tif_profile['count'] = len(self.layers)

        tif_profile['photometric'] = "RGB"

        # The 'tiled' propertty causes issues with the image
        del tif_profile['tiled']

        self.tif_memfile = MemoryFile()

        with self.tif_memfile.open(**tif_profile) as tif_bytes:

            self.image_array = np.array([layer.image_array for layer in self.layers])
            # for i, layer in enumerate(self.layers):
                # print(layer)
            tif_bytes.write(self.image_array)

            from rasterio.plot import show
            from rasterio.mask import mask
            # show(layer.image)
            print(type(tif_bytes))
            
            img_coords = apply_crs_to_shapely(self.query.boundary, self.layers[0].image.crs.data)

            self.clipped_image_array, out_transform = mask(dataset=tif_bytes, shapes=img_coords, crop=True, nodata=-1)


        self.image_channel_means = []
        for channel in self.clipped_image_array:
            self.image_channel_means.append(
                np.mean(channel[channel != -1])
            )

        # print(self.clipped_image_array)

        self.clipped_tif_memfile = MemoryFile()

        with self.clipped_tif_memfile.open(**tif_profile) as tif_bytes:
            tif_bytes.write(self.clipped_image_array)

        # Downsample image for thumbnail
        self.thumbnail_image_array = self.image_array[:, int(THUMBNAIL_SCALE_FACTOR/2)::THUMBNAIL_SCALE_FACTOR, int(THUMBNAIL_SCALE_FACTOR/2)::THUMBNAIL_SCALE_FACTOR]

        _, thumbnail_height, thumbnail_width = np.shape(self.thumbnail_image_array)
        thumbnail_tif_profile = tif_profile.copy()
        thumbnail_tif_profile['width'] = thumbnail_width
        thumbnail_tif_profile['height'] = thumbnail_height

        # print(thumbnail_tif_profile)

        self.thumbnail_tif_memfile = MemoryFile()
        with self.thumbnail_tif_memfile.open(**thumbnail_tif_profile) as tif_bytes:
            tif_bytes.write(self.thumbnail_image_array)


    def write_images_to_s3(self):
        import boto3

        s3 = boto3.client('s3')

        s3.put_object(Body=self.tif_memfile, Bucket=IMAGE_STORAGE_BUCKET , Key=f'{IMAGE_STORAGE_PREFIX}{self.image_filename}')
        s3.put_object(Body=self.thumbnail_tif_memfile, Bucket=IMAGE_STORAGE_BUCKET , Key=f'{IMAGE_STORAGE_PREFIX}{self.thumbnail_filename}')
        s3.put_object(Body=self.clipped_tif_memfile, Bucket=IMAGE_STORAGE_BUCKET , Key=f'{IMAGE_STORAGE_PREFIX}{self.clipped_image_filename}')


    def to_image_reference(self):
        """This will be stored in the DB"""
        return GoogleEarthImageReference(
            ee_id=self.ee_id,
            waterbody_id=self.query.id,
            captured_ts=self.captured_ts,
            properties=json.dumps(self.properties),
            filename=self.image_filename,
            thumbnail_filename=self.thumbnail_filename,
            channel_means=self.image_channel_means
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
    geometry: list[list[list[float]]]

    ee_dataset_name: str = "LANDSAT/LC09/C02/T1_L2"

    bands: tuple[ImageBand] = (
        ImageBand('SR_B4', 'red', 0.0000275, -0.2), 
        ImageBand('SR_B3', 'green', 0.0000275, -0.2), 
        ImageBand('SR_B2', 'blue', 0.0000275, -0.2),
    )

    images: tuple[GoogleEarthImage] = None

    # api_url: ClassVar[str] = 'earthengine.googleapis.com/v1alpha/projects/earthengine-public/assets/COPERNICUS/S2'

    def __post_init__(self):
        self.latitude = float(self.latitude)
        self.longitude = float(self.longitude)

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

    
    def get_image_list(self):

        img_collection: ee.ImageCollection = (
            ee.ImageCollection(self.ee_dataset_name) #dataset catalog: https://developers.google.com/earth-engine/datasets/
            .filterBounds(self.ee_centerpoint)
            .filterDate(ee.Date(self.start_date.isoformat()), ee.Date(self.end_date.isoformat()))
            .select(selectors=[b.band for b in self.bands], opt_names=[b.color_name for b in self.bands])
            # .addBands(ee.Image.pixelLonLat())
        )

        if img_collection.getInfo() is None:
            raise Exception(f"No images found for {self}")

        self.images = []
        for img in img_collection.getInfo()['features']:
 
            capture_ts_unix = int(int(img['properties']['system:time_start']) / 1000)

            self.images.append(
                GoogleEarthImage(
                    query=self,
                    ee_id=img['id'],
                    captured_ts=datetime.datetime.fromtimestamp(capture_ts_unix),
                    properties=img['properties']
                )
            )

        return self.images


ee_key_file = "waterbodyweather-key.json"

import ee
service_account = 'service-google-earth-api@waterbodyweather.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, ee_key_file)

ee.Initialize(credentials)

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2#bands

from database import engine

water_bodies_table_name = "water_bodies"

water_bodies_df = pd.read_sql(sql=f"""SELECT b.*, g.geometry 
                              FROM {water_bodies_table_name} b
                              LEFT JOIN water_body_geometries g
                              ON b.id = g.id
                              WHERE b.id = 9725
                                --order by b.areasqkm asc
                              LIMIT 1;""", 
                              con=engine.connect()
                        )

water_bodies_df = water_bodies_df[["id", "areasqkm", "min_longitude", "max_longitude", "min_latitude", "max_latitude", "latitude", "longitude", "geometry"]]

image_refs: list[GoogleEarthImageReference] = []

for i, row in water_bodies_df.iterrows():

    image_query = GoogleEarthImageQuery(
        start_date=datetime.date.today() - datetime.timedelta(days=80), #datetime.date(2023, 1, 1),
        end_date=datetime.date.today(), #datetime.date(2023, 1, 30), #
        **dict(row.items())
    )

    images = image_query.get_image_list()

    for image in images[:1]:
        image.download_layers()
        image.combine_image_layers()
        image.write_images_to_s3()
        image_refs.append(
            image.to_image_reference()
        )

print(image_refs)

