
from dataclasses import dataclass
import datetime
import pandas as pd

from urllib.parse import urlencode
import json
import requests
import rasterio
import os
import numpy as np

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

import ee

import zipfile

# https://api.nasa.gov/

from typing import ClassVar

@dataclass
class GoogleEarthImageLayer:
    band: str
    color_name: str

    def __post_init__(self):
        pass

    def get_image(self, ee_image: ee.Image, download_props: dict):
        task = ee.batch.Export.image.toDrive(
            image=ee_image,
            fileNamePrefix=self.band,
            **download_props
        )

        task.start()

        url = ee_image.select(self.band).getDownloadURL(download_props)
        
        r = requests.get(url, stream=True)

        # filenameZip = band+'.zip'
        # filenameTif = band+'.tif'

        # # unzip and write the tif file, then remove the original zip file
        # with open(filenameZip, "wb") as fd:
        #     for chunk in r.iter_content(chunk_size=1024):
        #         fd.write(chunk)

        # zipdata = zipfile.ZipFile(filenameZip)
        # zipinfos = zipdata.infolist()

        # # iterate through each file (there should be only one)
        # for zipinfo in zipinfos:
        #     zipinfo.filename = filenameTif
        #     zipdata.extract(zipinfo)

        # zipdata.close()

        # B2 = rasterio.open('B2.tif')
        # B3 = rasterio.open('B3.tif')
        # B4 = rasterio.open('B4.tif')

        # #get the scaling
        # image = np.array([B2.read(1), B3.read(1), B4.read(1)]).transpose(1,2,0)
        # p2, p98 = np.percentile(image, (2,98))

        # # use the B2 image as a starting point so that I keep the same parameters
        # B2_geo = B2.profile
        # B2_geo.update({'count': 3})

        # with rasterio.open("test.tif", 'w', **B2_geo) as dest:
        #     dest.write( (np.clip(B4.read(1), p2, p98) - p2)/(p98 - p2)*255, 1)
        #     dest.write( (np.clip(B3.read(1), p2, p98) - p2)/(p98 - p2)*255, 2)
        #     dest.write( (np.clip(B2.read(1), p2, p98) - p2)/(p98 - p2)*255, 3)

        # B2.close()
        # B3.close()
        # B4.close()

        # remove the intermediate files
        # for selection in bands:
        #     os.remove(selection + '.tif')
        #     os.remove(selection + '.zip')


@dataclass
class GoogleEarthImage:
    id: int
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float
    latitude: float
    longitude: float
    start_date: datetime.date
    end_date: datetime.date

    ee_image: ee.Image = None

    api_url: ClassVar[str] = 'earthengine.googleapis.com/v1alpha/projects/earthengine-public/assets/COPERNICUS/S2'

    def __post_init__(self):

        self.width_degrees = abs(self.max_latitude - self.min_latitude)

        self.height_degrees = abs(self.max_longitude - self.min_longitude)

        # The API returns square images, so use the max of the langth and width to get the whole body in the image
        self.image_side_length_degrees = max(self.width_degrees, self.height_degrees)

        self.bounding_box = [
                    [self.min_latitude, self.max_longitude],
                    [self.max_latitude, self.min_longitude],
                    [self.min_latitude, self.max_longitude],
                    [self.max_latitude, self.min_longitude], 
                ]
        
        self.layers = [
            GoogleEarthImageLayer('SR_B4', 'red'), 
            GoogleEarthImageLayer('SR_B3', 'green'),
            GoogleEarthImageLayer('SR_B2', 'blue'),
        ]

        self.bands = [l.name for l in self.layers]

        self.ee_boundary = ee.Geometry.Polygon(self.bounding_box)

    
    def get_image_list(self):

        img_collection: ee.ImageCollection = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(self.ee_boundary)
            .filterDate(ee.Date(self.start_date.isoformat()), ee.Date(self.end_date.isoformat()))
            .select(*self.bands)
            .multiply(0.0000275) # Values from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
            .add(-0.2)
        )

        self.ee_image = ee.Image(img_collection
                        #  .sort('system:time_start', False)
                         .sort('CLOUDY_PIXEL_PERCENTAGE')
                         .first()
                         ).addBands(ee.Image.pixelLonLat())
        

    def download_layers(self):
        download_props = {
            'scale': 30,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'region': self.ee_boundary
        }

        for layer in self.layers:
            layer.get_image(
                ee_image=self.ee_image, 
                download_props=download_props
            )




ee_key_file = "waterbodyweather-key.json"

import ee
service_account = 'service-google-earth-api@waterbodyweather.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, ee_key_file)

ee.Initialize(credentials)

# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2#bands
# img_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
# "SR_B2"



from database import engine

water_bodies_table_name = "water_bodies"

water_bodies_df = pd.read_sql(sql=f"SELECT * FROM {water_bodies_table_name} WHERE id = 9725 LIMIT 1;", con=engine.connect())

water_bodies_df = water_bodies_df[["id", "min_longitude", "max_longitude", "min_latitude", "max_latitude"]]

for i, row in water_bodies_df.iterrows():

    image_asset_list = GoogleEarthImage(
        start_date=datetime.date.today() - datetime.timedelta(days=7),
        end_date=datetime.date.today(),
        **dict(row.items())
    )
    from pprint import pprint

    pprint(image_asset_list.get_image_list())

