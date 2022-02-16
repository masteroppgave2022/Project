#!/usr/bin/bash

eval "python -m pip install --upgrade pip"
PACKAGES=(
    "numpy" 
    "matplotlib" 
    "tensorflow==2.7.0" 
    "keras" "pandas" 
    "segmentation_models" 
    "seaborn" 
    "opencv-python" 
    "scikit-learn" 
    "datetime" 
    "sentinelsat" 
    "asf_search" 
    "geopandas" 
    "pyshp" 
    "shapely"
    "pygeoif"
    "PyQt5")

PREFIX="python -m pip install "

for package in ${PACKAGES[@]}; do
	COMMAND=$PREFIX$package
	eval $COMMAND
done

eval "export CPLUS_INCLUDE_PATH=/usr/include/gdal"
eval "export C_INCLUDE_PATH=/usr/include/gdal"
PACKAGE="GDAL==3.3.2"
eval $PREFIX$PACKAGE
