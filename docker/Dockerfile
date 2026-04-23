# =============================================================================
# D2SAR: Build isce3 + dolphin + plant-isce3 from source (GPU version)
# Base: nvidia/cuda:12.4.1-devel-ubuntu22.04
# Strategy: Build GDAL 3.8.5 from source to /opt/gdal38, then rebuild
#           isce3 linked against GDAL 3.8. This enables plant_isce3 which
#           requires gdal>=3.8 and gdal_array support.
# =============================================================================

ARG CACHE_VERSION=2
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    MAKEFLAGS="-j4"

# Tsinghua apt mirror + install system dependencies
# Keep libgdal-dev (3.4.1): needed for building GDAL 3.8 from source
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
        curl \
        ca-certificates \
        wget \
        git \
        cmake \
        make \
        gcc \
        g++ \
        gfortran \
        unzip \
        pkg-config \
        libglib2.0-dev \
        libsqlite3-dev \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libproj-dev \
        libgeos-dev \
        libhdf5-dev \
        libnetcdf-dev \
        libfftw3-dev \
        libopenmpi-dev \
        openmpi-bin \
        libboost-all-dev \
        libarmadillo-dev \
        libgtest-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        libgdal-dev \
        pybind11-dev \
        ccache \
        swig \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# pip mirror
RUN mkdir -p /root/.config/pip && \
    echo "[global]" > /root/.config/pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.config/pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> /root/.config/pip/pip.conf

# =============================================================================
# STEP 1: Build GDAL 3.8.5 from source to /opt/gdal38
# - Installs libgdal.so.36 (SONAME different from system libgdal.so.31)
# - Includes Python bindings with gdal_array support
# - Builds GDAL command-line apps such as gdalinfo
# - Enables GTiff, ENVI, HDF5, NetCDF drivers        
#  -DGDAL_BUILD_OPTIONAL_DRIVERS=OFF \
#        -DOGR_BUILD_OPTIONAL_DRIVERS=OFF \
#
# =============================================================================
WORKDIR /tmp
ARG GDAL_VERSION=3.8.5
RUN curl -sSL "https://github.com/OSGeo/gdal/releases/download/v${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz" -o gdal.tar.gz && \
    tar xzf gdal.tar.gz && \
    mkdir -p /tmp/gdal_build && \
    cmake -S gdal-${GDAL_VERSION} -B /tmp/gdal_build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/gdal38 \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_APPS=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPython_EXECUTABLE=/usr/bin/python3 \
        -DPython_VERSION=3.10 \
        -DGDAL_BUILD_OPTIONAL_DRIVERS=ON \
        -DOGR_BUILD_OPTIONAL_DRIVERS=ON \
        -DGDAL_ENABLE_DRIVER_GTIFF=ON \
        -DGDAL_ENABLE_DRIVER_RAW=ON \
        -DGDAL_ENABLE_DRIVER_HDF5=ON \
        -DGDAL_ENABLE_DRIVER_NETCDF=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_Arrow=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_OpenJPEG=ON && \
    cmake --build /tmp/gdal_build -j$(nproc) && \
    cmake --install /tmp/gdal_build && \
    LD_LIBRARY_PATH=/opt/gdal38/lib PYTHONPATH=/opt/gdal38/lib/python3/dist-packages \
        python3 -c "from osgeo import gdal_array; print('gdal_array:', gdal_array.__file__)" && \
    rm -rf /tmp/gdal.tar.gz /tmp/gdal-${GDAL_VERSION} /tmp/gdal_build

ENV PATH=/opt/gdal38/bin:$PATH \
    LD_LIBRARY_PATH=/opt/gdal38/lib:/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH \
    GDAL_DATA=/opt/gdal38/share/gdal \
    PROJ_DATA=/usr/share/proj \
    GDAL_DIR=/opt/gdal38/lib/cmake/gdal \
    PYTHONPATH=/opt/gdal38/lib/python3/dist-packages:${PYTHONPATH}

# =============================================================================
# STEP 2: Build pyre from source (needed by isce3 cmake)
# =============================================================================
WORKDIR /opt/build/pyre
ENV GIT_TERMINAL_PROMPT=0
RUN git clone -b v1.12.5 https://github.com/pyre/pyre /opt/build/pyre/src && \
    cmake -S src -B build \
        -DCMAKE_INSTALL_PREFIX=/opt/isce3 \
        -DPYRE_BUILD_TESTING=OFF \
        -DWITH_CUDA=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    rm -rf /opt/build/pyre

ENV CMAKE_PREFIX_PATH=/opt/isce3:/opt/gdal38:${CMAKE_PREFIX_PATH} \
    LD_LIBRARY_PATH=/opt/isce3/lib:${LD_LIBRARY_PATH}

# =============================================================================
# STEP 3: Build isce3 C++ library with CUDA (linked to GDAL 3.8)
# Use local isce3 source code (with GPU crossmul fixes)
# =============================================================================
WORKDIR /opt/build/isce3
COPY isce3 /opt/build/isce3/src
RUN cmake -S src -B build \
        -DCMAKE_INSTALL_PREFIX=/opt/isce3 \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DISCE3_FETCH_EIGEN=ON \
        -DISCE3_FETCH_GTEST=ON \
        -DISCE3_FETCH_PYBIND11=ON \
        -DISCE3_FETCH_PYRE=OFF \
        -DPython_EXECUTABLE=/usr/bin/python3 && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    rm -rf /opt/build/isce3

# =============================================================================
# STEP 4: Python dependencies (must be before isce3 Python bindings)
# =============================================================================
RUN python3 -m pip install --upgrade --no-cache-dir \
        "numpy>=1.26,<2" \
        "scipy>=1.11,<1.14" \
        "matplotlib>=3.8,<3.10" \
        h5py \
        numba \
        pydantic \
        tyro \
        tqdm \
        threadpoolctl \
        pyproj \
        rasterio \
        affine \
        snuggs \
        ruamel.yaml \
        shapely \
        geopandas \
        boto3 \
        yamale \
        opera-utils \
        snaphu \
        pytest \
        scikit-build-core \
        pysolid

RUN python3 -m pip install --prefix=/opt/isce3 backoff requests

# =============================================================================
# STEP 5: Install isce3 Python bindings (use local source)
# =============================================================================
ENV PYTHONPATH=/opt/gdal38/lib/python3/dist-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3.10/dist-packages:/opt/isce3/packages:/opt/isce3/local/lib/python3.10/dist-packages:/opt/isce3/lib/python3.10/dist-packages \
    LD_LIBRARY_PATH=/opt/gdal38/lib:/opt/isce3/lib:${LD_LIBRARY_PATH} \
    GDAL_DIR=/opt/gdal38/lib/cmake/gdal

WORKDIR /opt/build/isce3_py
COPY isce3 /opt/build/isce3_py/src
RUN pip install /opt/build/isce3_py/src --prefix=/opt/isce3 --no-build-isolation && \
    rm -rf /opt/build/isce3_py

# =============================================================================
# STEP 6: plant, dolphin, plant-isce3
# =============================================================================
WORKDIR /opt/build/plant
RUN git clone https://gitlab.com/plant/plant /opt/build/plant/src && \
    pip install /opt/build/plant/src --prefix=/opt/isce3 --no-deps && \
    rm -rf /opt/build/plant

WORKDIR /opt/build/plant-isce3
COPY plant-isce3/ /opt/build/plant-isce3/src/
RUN cp -r src/src/plant_isce3 /opt/isce3/local/lib/python3.10/dist-packages/ && \
    mkdir -p /opt/isce3/local/lib/python3.10/dist-packages/plant_isce3-0.3.2.egg-info && \
    cp -r src/src/plant_isce3.egg-info/* /opt/isce3/local/lib/python3.10/dist-packages/plant_isce3-0.3.2.egg-info/ && \
    rm -rf /opt/build/plant-isce3

WORKDIR /opt/build/dolphin
COPY dolphin /opt/dolphin_src/
RUN cp -r /opt/dolphin_src/src/dolphin /opt/isce3/local/lib/python3.10/dist-packages/ && \
    rm -rf /opt/dolphin_src /opt/build/dolphin

# Final cleanup: remove conflicting 'io' namespace package from opera-utils
RUN rm -rf /opt/isce3/local/lib/python3.10/dist-packages/io

WORKDIR /work
COPY scripts/ ./scripts/
COPY *.md ./

CMD ["bash"]
