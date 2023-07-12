import ee
import datetime

ee_key_file = "waterbodyweather-key.json"

import ee
service_account = 'service-google-earth-api@waterbodyweather.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, ee_key_file)

ee.Initialize(credentials)

# define the area of interest, using the Earth Engines geometry object
coords = [
     [lon - sze/2., lat - sze/2.],
     [lon + sze/2., lat - sze/2.],
     [lon + sze/2., lat + sze/2.],
     [lon - sze/2., lat + sze/2.],
     [lon - sze/2., lat - sze/2.]
]
aoi = ee.Geometry.Polygon(coords)

# get the image using Google's Earth Engine
db = ee.Image(ee.ImageCollection('COPERNICUS/S2_SR')\
                   .filterBounds(aoi)\
                   .filterDate(ee.Date(dateMin), ee.Date(dateMax))\
                   .sort('CLOUDY_PIXEL_PERCENTAGE')\
                   .first())

# add the latitude and longitude
db = db.addBands(ee.Image.pixelLonLat())