#!/bin/bash

cd $BUILD_DIR || exit 1
echo "Building Hongo: ${BUILD_HONGO}"

if [ "${BUILD_HONGO}" != "true" ]; then
  echo "BUILD_HONGO is not set. Exiting."
  exit 1
fi

rm -rf *
cmake ..
cmake --build .
cp ../setup.py .
python3 setup.py build_ext --inplace