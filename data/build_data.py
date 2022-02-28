import os
from itertools import product
import rasterio
from rasterio import windows


def get_tiles(ds, width=64, height=64):
    """
    Provide tile window and transform.
    Used by internal methods only, even though access-restriction is not provided.
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def tile_write(image,output_path,size=(256,256)):
    """
    Tile large satellite image and save to specified output location.
    ----- Parameters: -----
    image (opened rasterio dataset object)
    output_path (str) :: path to write to
    size (tuple) :: (height,width) of desired tiles
    """
    output_filename = 'tile_{}-{}.tif'
    meta = image.meta.copy()
    
    for window, transform in get_tiles(image,size[0],size[1]):
        print(window)
        meta['transform'] = transform
        if window.width == size[1] and window.height == size[0]:
            meta['width'],meta['height'] = window.width,window.height
            outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
            with rasterio.open(outpath, 'w', **meta) as outds:
                outds.write(image.read(window=window))