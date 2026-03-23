#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/build"
cmake ../src -DCMAKE_INSTALL_PREFIX=$(pwd)/install
make -j$(nproc)
make install