
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
from pprint import pprint
# https://api.nasa.gov/

from typing import ClassVar

data_folder = "data"

@dataclass
class GoogleEarthImageLayer:
    band: str
    color_name: str

    def __post_init__(self):
        self.zip_filename = os.path.join(data_folder, f"{self.band}.zip")

    def get_image(self, ee_image: ee.Image, download_props: dict):
        if "image" in download_props.keys():
            del download_props["image"]

        print(download_props)
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

        # unzip and write the tif file, then remove the original zip file
        with open(self.zip_filename, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

        image_filename = f"download.{self.band}.tif"

        with zipfile.ZipFile(self.zip_filename, 'r') as f:
            f.extract(image_filename)

        with rasterio.open(image_filename) as src:
            self.image = src

        return self.image



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
    ee_dataset_name: str = "LANDSAT/LC09/C02/T1_L2"
    layers: tuple[GoogleEarthImageLayer] = (
        GoogleEarthImageLayer('SR_B4', 'red'), 
        GoogleEarthImageLayer('SR_B3', 'green'),
        GoogleEarthImageLayer('SR_B2', 'blue'),
    )

    # api_url: ClassVar[str] = 'earthengine.googleapis.com/v1alpha/projects/earthengine-public/assets/COPERNICUS/S2'

    def __post_init__(self):

        self.latitude = float(self.latitude)
        self.longitude = float(self.longitude)

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
        


        self.bands = [l.band for l in self.layers]

        self.ee_boundary = ee.Geometry.Polygon(self.bounding_box)

        self.ee_centerpoint = ee.Geometry.Point((self.longitude, self.latitude)) # Note: latitude and longitude are reversed in the args here

    
    def get_image_list(self):

        img_collection: ee.ImageCollection = (
            ee.ImageCollection(self.ee_dataset_name) #dataset catalog: https://developers.google.com/earth-engine/datasets/
            .filterBounds(self.ee_centerpoint)
            .filterDate(ee.Date(self.start_date.isoformat()), ee.Date(self.end_date.isoformat()))
            .select(*self.bands)
            
            # .addBands(ee.Image.pixelLonLat())
        )

        # pprint(img_collection.limit(5).getInfo())

        self.ee_image = (
            ee.Image(
                img_collection
                #  .sort('system:time_start', False)
                    .sort('CLOUDY_PIXEL_PERCENTAGE')
                    .first()
                    
            )
            .multiply(0.0000275) # Values from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
            .add(-0.2)
        )

        pprint(self.ee_image.getInfo())
        

    def download_layers(self):
        download_props = {
            'scale': 5000, # meters per pixel
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'region': self.ee_boundary,
            # 'dimensions': "5000x5000"
        }

        for layer in self.layers:
            layer.get_image(
                ee_image=self.ee_image, 
                download_props=download_props
            )

    def combine_image_layers(self):
        
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

        profile = {}
        for layer in self.layers:
            profile.update(layer.image.profile)

        profile.update({'count': len(self.layers)})

        with rasterio.open("test.tif", 'w', **profile) as dest:
            for i, layer in enumerate(self.layers):
                dest.write(layer.image.read(1) , i)
            




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

water_bodies_df = water_bodies_df[["id", "min_longitude", "max_longitude", "min_latitude", "max_latitude", "latitude", "longitude"]]

for i, row in water_bodies_df.iterrows():


    image_asset_list = GoogleEarthImage(
        start_date=datetime.date.today() - datetime.timedelta(days=80),
        end_date=datetime.date.today(),
        **dict(row.items())
    )
    from pprint import pprint

    pprint(image_asset_list)

    image_asset_list.get_image_list()

    image_asset_list.download_layers()



# image = GoogleEarthImage(id=9725,
#                  min_latitude=44.9,
#                  max_latitude=0,
#                  min_longitude=0,
#                  max_longitude=0,
#                  latitude=44.9812577,
#                  longitude=-93.2716135,
#                  start_date=datetime.date(2022, 1, 1),
#                  end_date=datetime.date(2022, 2, 1),
#                  ee_image=None,
#                  ee_dataset_name='LANDSAT/LC09/C02/T1_L2')

# image.get_image_list()

# image.download_layers()