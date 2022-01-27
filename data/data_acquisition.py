import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
from datetime import date

gdf = gpd.read_file('polygon_trondheim.shp')
bounds = gdf.total_bounds
gdf_bounds = gpd.GeoSeries([box(*bounds)])
wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0]

results = asf.search(
    platform= asf.PLATFORM.SENTINEL1A,
    processingLevel=[asf.PRODUCT_TYPE.SLC],
    start = date(2022, 1, 12),
    end = date(2022, 1, 18),
    intersectsWith = wkt_aoi
    )

print(f'Total Images Found: {len(results)}')
metadata = results.geojson()

session = asf.ASFSession().auth_with_creds('mikaelvagen','Drage127')

results.download(
    path='data/test_data_download/',
    session=session,
    processes=2
)
