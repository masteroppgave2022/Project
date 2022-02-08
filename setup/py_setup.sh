#!/usr/bin/bash

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
    "pygeoif")

PREFIX="python -m pip install "

for package in ${PACKAGES[@]}; do
	COMMAND=$PREFIX$package
	eval $COMMAND
done
