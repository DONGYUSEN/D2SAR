#!/bin/bash
# Wrapper script for running tops_insar2.py in Docker with proper PYTHONPATH
#
# Usage:
#   ./run_tops_insar2.sh <output_dir> <master_safe> <slave_safe> [options]
#
# Example:
#   ./run_tops_insar2.sh /temp/tops_output \
#       /temp/s1/raw/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.SAFE \
#       /temp/s1/raw/S1A_IW_SLC__1SDV_20230719T114147_20230719T114214_049492_05F38A_3C77.SAFE \
#       --swath IW1 --dem /temp/dem/dem_clip_wgs84.tif --start-stage preprocess --end-stage publish

set -e

# Volume mappings
WORK_DIR="/home/ysdong/Software/D2SAR"
RESULTS_DIR="/home/ysdong/Software/D2SAR/results"
TEMP_DIR="/home/ysdong/Temp"

# Python path order matters! Put /work last to avoid shadowing installed packages
# - /opt/isce3/packages: isce3 with C++ bindings
# - /opt/gdal38/lib/python3/dist-packages: GDAL/osgeo
# - /work/scripts: scripts/ modules (common_processing, tops_*, etc.)
# - /work: D2SAR root (for scripts.tops_* imports)
PYTHONPATH="/opt/isce3/packages:/opt/gdal38/lib/python3/dist-packages:/work/scripts:/work"

docker run --rm --user $(id -u):$(id -g) \
    -v "$WORK_DIR:/work" \
    -v "$RESULTS_DIR:/results" \
    -v "$TEMP_DIR:/temp" \
    d2sar:cuda \
    bash -c "cd /work && PYTHONPATH='$PYTHONPATH' python3 scripts/tops_insar2.py $@"