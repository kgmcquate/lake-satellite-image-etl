"""
Data sources:
1. lake water color
    - satellite imagery from google earth for each water body over time, the averages of each channel (RGB) are stored in a table (waterbody_satellite_images)
2. daily weather history
    - daily weather information for each water body


    
Notes:
    Satellite imagery is not available for every day, so will likely have to impute daily RGB values 

References:
https://www.epa.gov/climate-indicators/climate-change-indicators-lake-ice

"""
# 

# import pandas as pd
import connectorx
import polars as pl
from database import connectorx_url #engine
import datetime


start_date = datetime.date(2022, 8, 6)
end_date = datetime.date(2023, 8, 6)



# images_df = pd.read_sql(sql=f"""
                            
#                             SELECT waterbody_id, 
#                             captured_ts::DATE as captured_date, 
#                             red_average, green_average, blue_average
#                             FROM waterbody_satellite_images im
#                             INNER JOIN training_waterbody_ids ids
#                             ON im.waterbody_id = ids.id
                            
#                         ;
#                         """, 
#                         con=engine.connect()
                    # )

weather_df = pl.read_database(f"""
                    SELECT "date", 
                         w.latitude, 
                         w.longitude, 
                         timezone, 
                         temperature_2m_max, 
                         temperature_2m_min, 
                         sunrise, 
                         sunset, 
                         --uv_index_max, --too many nulls
                         --uv_index_clear_sky_max, --too many nulls
                         precipitation_sum, 
                         rain_sum, 
                         showers_sum, 
                         snowfall_sum, 
                         precipitation_hours, 
                         --precipitation_probability_max, --too many nulls
                         windspeed_10m_max, 
                         windgusts_10m_max, 
                         winddirection_10m_dominant, 
                         shortwave_radiation_sum, 
                         et0_fao_evapotranspiration
                    FROM public.daily_weather w
                    INNER JOIN training_waterbody_ids ids
                    ON w.latitude = ids.latitude AND w.longitude = ids.longitude
                    WHERE "date" BETWEEN '{start_date.strftime('%Y-%m-%d')}'::date AND '{end_date.strftime('%Y-%m-%d')}'::date
                    LIMIT 10
                """, 
                connectorx_url
            )

weather_df = (
        weather_df
        .with_columns(
            pl.col("date").dt.iso_year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
            pl.col("longitude").cast(float).alias("longitude"),
            pl.col("latitude").cast(float).alias("latitude"),
        )
        .drop(
            "timezone", "date"
        )
    )

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)

print(
weather_df
)


# Dimensionality reduction on weather data
# correlation_matrix = weather_df.corr()
# print(correlation_matrix)


# stacked_corr = (
#     correlation_matrix
#     .with_columns(index = pl.lit(correlation_matrix.columns))
#     .melt(id_vars = "index")
#     .filter(pl.col('index') != pl.col('variable'))
# )

# print(stacked_corr)

# Remove the columns with high corre
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(weather_df.to_numpy())

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
