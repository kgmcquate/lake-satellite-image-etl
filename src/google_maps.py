import requests

#pytesseract to get text

from ascii_magic import AsciiArt
from PIL import Image
import urllib

base_url = "https://maps.googleapis.com/maps/api/staticmap"

with open("api_key", "r") as f:
    api_key = f.read()

water_only_map_styling = [
                        # 'feature:administrative|element:all|visibility:off', 
                        # 'feature:landscape|element:all|visibility:off', 
                        # 'feature:poi|element:all|visibility:off', 
                        # 'feature:road|element:all|visibility:off', 
                        # 'feature:transit|element:all|visibility:off', 
                        # 'feature:water|element:labels|visibility:on', 
                    #   'feature:water|element:labels|hue:0xff0000', 
                        # 'feature:water|element:labels|color:0xff0000', 
                        # 'feature:water|element:geometry|color:0xffffff'
                    ]


def get_satellite_image(latitude, longitude, zoom, location_names: list[str]):
    resp = requests.get(
        url=base_url,
        params=[
            ("key", api_key),
            ("center", f"{latitude},{longitude}"),
            ("zoom", zoom),
            ("size", f"100x100"),
            #("scale", 10),
            ("maptype", "roadmap"),
            #( "visible", ",".join(location_names))
            
        ] + [("style", style) for style in water_only_map_styling]
    )
    # print(
    # resp.content
    # )
    resp.raise_for_status()

    import io
    img = Image.open(io.BytesIO(resp.content))
    img.show()

get_satellite_image(46.274007,-93.5727791, 8, ["Abbey Lake"])
# print(
# build_styles(water_only_map_styling)
# )