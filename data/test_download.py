import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
from datetime import date

path_to_shapefile = "shape/POLYGON.shp"
path_to_download = "downloads"

USERNAME = "renateask"
PASSWORD = "#Voff123"

### 1. Read Shapefile using Geopandas
gdf = gpd.read_file(path_to_shapefile)
### 2. Extract the Bounding Box Coordinates
bounds = gdf.total_bounds
### 3. Create GeoDataFrame of the Bounding Box 
gdf_bounds = gpd.GeoSeries([box(*bounds)])
### 4. Get WKT Coordinates
wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0]

results = asf.search(
    processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
    start = date(2020, 9, 10),
    end = date(2020, 9, 11),
    intersectsWith = wkt_aoi
    )
print(f'Total Images Found: {len(results)}')
### Save Metadata to a Dictionary
metadata = results.geojson()

session = asf.ASFSession().auth_with_creds(USERNAME, PASSWORD)

results.download(
     path = path_to_download,
     session = session, 
     processes = 2 
  )