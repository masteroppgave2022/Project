#!/usr/bin/bash
user=$USER
root_path="/localhome/studenter/${user}/"

# Python installation
PYTHON_COMMANDS=( 
    "git clone https://github.com/pyenv/pyenv.git ${root_path}.pyenv" 
    "cd /home/studenter/${user}/" 
    "sed -Ei -e '/^([^#]|$)/ {a \ 
    export PYENV_ROOT='${root_path}.pyenv'
    a \ 
    export PATH='$PYENV_ROOT/bin:$PATH'
    a \ 
    ' -e ':a' -e '$!{n;ba};}' ~/.profile"   
    "echo 'eval \"$(pyenv init --path)\"' >>~/.profile"
    ">.bashrc" 
    "echo 'export PYENV_ROOT=\"${root_path}.pyenv\"' >> ~/.bashrc" 
    "echo 'export PATH=\"\$PYENV_ROOT/bin:\$PATH\"' >> ~/.bashrc"
    "echo 'eval \"\$(pyenv init --path)\"' >> ~/.bashrc"
    "echo 'eval \"\$(pyenv init -)\"' >> ~/.bashrc"
    "pyenv install 3.9.7" 
    "pyenv global 3.9.7"
)
if [[ -f "${root_path}.pyenv" ]]; then
    echo "${root_path}.pyenv"
    echo "Pyenv and Python already installed... skipping..."
else
    for c in "${PYTHON_COMMANDS[@]}"; do
        eval $c
    done
fi

python_installation=$(pyenv which python)
echo "Pyenv Python installed: ${python_installation}"

PACKAGES=(
    "numpy" 
    "matplotlib" 
    "tensorflow==2.7.0" 
    "keras" 
    "pandas" 
    "segmentation_models" 
    "seaborn" 
    "opencv-python" 
    "scikit-learn" 
    "datetime" 
    "asf_search" 
    "geopandas" 
    "pyshp" 
    "shapely"
    "pygeoif"
    "PyQt5"
    "configparser"
    "rasterio" 
    "xarray" 
    "rioxarray" 
    "albumentations")

PREFIX="python -m pip install "
eval "python -m pip install --upgrade pip"

for package in ${PACKAGES[@]}; do
	COMMAND=$PREFIX$package
	eval $COMMAND
done
echo "All generic packages installed."

eval "export CPLUS_INCLUDE_PATH=/usr/include/gdal"
eval "export C_INCLUDE_PATH=/usr/include/gdal"
PACKAGE="GDAL==3.3.2"
eval $PREFIX$PACKAGE
echo "GDAL Python-binding complete."

# Snappy setup:
SNAPPY_COMMANDS=(
    "cd ${root_path}" 
    "wget https://download.esa.int/step/snap/8.0/installers/esa-snap_sentinel_unix_8_0.sh" 
    "bash esa-snap_sentinel_unix_8_0.sh" 
    "git clone https://github.com/bcdev/jpy.git" 
    "cd jpy/" 
    javac_installation=$(readlink -f `which javac` | sed "s:/bin/javac::") 
    "export JDK_HOME=\${javac_installation}" 
    "export JAVA_HOME=\${javac_installation}" 
    "python setup.py build maven bdist_wheel" 
    "cp dist/*.whl /home/studenter/${user}/.snap/snap-python/snappy" 
    "cd ${root_path}snap/bin/" 
    "./snappy-conf ${python_installation}"
)
if [[ -f "${root_path}esa-snap_sentinel_unix_8_0.sh" ]]; then
    echo "Snap already downloaded and installed... skipping..."
    exit 0
fi

for c in "${SNAPPY_COMMANDS[@]}"; do
    eval $c
done
echo "Snappy is hopefully configured..."




