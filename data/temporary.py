import os
import geopandas as gpd
from numpy import source



def clip_shapefile(source_shp, mask_shps, destination):
    src = gpd.read_file(source_shp)
    for mask in mask_shps:
        output_name = os.path.split(mask)[1]
        out_path = destination+output_name.split('.')[0]+'/'+output_name
        if not os.path.exists(out_path):
            extent = gpd.read_file(mask)
            # Check CRS:
            if not src.crs == extent.crs:
                raise Exception("[ABORTING] CRS of source SHP and mask SHP are not the same.")
            clipped = gpd.clip(src, extent)
            clipped.to_file(out_path)
        else:
            print("[SKIPPING] ... Clipped SHP already exists.")

if __name__ == '__main__':
    src = '/localhome/studenter/mikaellv/Project/data/FBK_vann/FKB_vann.shp'
    mask = ['/localhome/studenter/mikaellv/Project/data/shapefiles/åndalsnes/åndalsnes.shp']
    dest = '/localhome/studenter/mikaellv/Project/data/untiled_masks/'
    clip_shapefile(src,mask,dest)