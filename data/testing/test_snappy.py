# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import snappy
from snappy import ProductIO

# Set Path to Input Satellite Data
# miniconda users
path = "/home/studenter/renatask/.snap/snap-python/snappy/testdata"

filename = 'MER_FRS_L1B_SUBSET.dim'

# Read File
df = ProductIO.readProduct(os.path.join(path, filename))
# Get the list of Band Names
list(df.getBandNames())
# Using "radiance_3" band as an example
band = df.getBand('radiance_3') # Assign Band to a variable
w = df.getSceneRasterWidth() # Get Band Width
h = df.getSceneRasterHeight() # Get Band Height
# Create an empty array
band_data = np.zeros(w * h, np.float32)
# Populate array with pixel value
band.readPixels(0, 0, w, h, band_data) 
# Reshape
band_data.shape = h, w
# Plot the band  
plt.figure(figsize=(18,10))
plt.imshow(band_data, cmap = plt.cm.binary)
plt.savefig("test_image.png")