# CT Volume Reconstruction - OpenCL GPU & C++ CPU

GPU-accelerated and CPU-parallelized cone-beam CT volume reconstruction using OpenCL and C++.

## Prerequisites

- OpenCL runtime and development headers
- HDF5 C++ library
- FFTW3 (single precision)
- OpenMP
- Meson build system
- Ninja
- clFFT (built locally, see build instructions below)

## Installing Dependencies (Ubuntu)

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    meson \
    ninja-build \
    libhdf5-dev \
    libfftw3-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo
```

## Configuration

Before building and running, edit the configuration section at the top of `src/CTSVolumeReconstruction.cpp`:

```cpp
const std::string dataPath = "/lgrp/edu-2025-2-gpulab/Data/proj_shepplogan128.hdf5";
const bool USE_BUFFER_BACKPROJECTION = false;  // true = buffer, false = image3d
const bool RUN_CPU = true;                     // true = also run CPU reconstruction
```

- **`dataPath`**: Path to the input HDF5 projection data. Several example paths are provided as comments in the source file..
- **`USE_BUFFER_BACKPROJECTION`**: Selects the GPU back-projection variant. `false` uses `image3d` with hardware interpolation (faster), `true` uses a buffer with software interpolation (more accurate).
- **`RUN_CPU`**: Whether to also run the C++ CPU reconstruction after the GPU reconstruction.

## Building

All commands are run from the project root directory.

### 1. Build clFFT (first time only) (required)

clFFT is not installed via package manager and must be built locally.

Using the provided script:
```bash
./clFFT/build_clfft.sh
```

Or manually:
```bash
cd clFFT/build
cmake ../src -DCMAKE_INSTALL_PREFIX=$(pwd)/install
make -j$(nproc)
make install
cd ../..
```

### 2. Set up the Meson build directory (first time only)

```bash
meson setup build
```

### 3. Compile

```bash
meson compile -C build
```

To reconfigure after changing `meson.build`:
```bash
meson setup --reconfigure build
```

## Running

```bash
./build/CTSVolumeReconstruction
```

## Output

Reconstructed volumes are saved as HDF5 files in the `reconstructed/` directory:

| File | Description | Dataset name |
|------|-------------|-------------|
| `reconstructed/volume_gpu.hdf5` | GPU reconstruction output | `ReconstructedVolume` |
| `reconstructed/volume_cpu.hdf5` | CPU reconstruction output (if `RUN_CPU = true`) | `ReconstructedVolume` |

The output volume has shape `(N_xz, N_xz, N_y)` stored as 32-bit float.

## Project Structure

```
.
├── meson.build                          # Build configuration
├── README.md
├── clFFT/                               # clFFT library (built locally)
│   ├── build_clfft.sh                   # Build script for clFFT
│   └── ...
├── data/                                # Input projection data (local copies)
│   └── proj_shepplogan128.hdf5
├── reconstructed/                       # Output reconstructed volumes
├── src/
│   ├── CTSVolumeReconstruction.cpp      # Main entry point (GPU reconstruction)
│   ├── CTSVolumeReconstruction.cl       # OpenCL kernels
│   ├── cpu/
│   │   ├── cpu_recon.hpp                # CPU reconstruction header
│   │   └── cpu_recon.cpp                # CPU reconstruction implementation
│   ├── gpu_fft/
│   │   ├── hilbert_fft.hpp              # GPU Hilbert transform header
│   │   └── hilbert_fft.cpp              # GPU Hilbert transform implementation
│   └── io/
│       ├── HDF5IO.hpp                   # HDF5 data loading/saving header
│       └── HDF5IO.cpp                   # HDF5 data loading/saving implementation
├── lib/                                 # Provided utility libraries
│   ├── Core/                            # Time, assertions, etc.
│   ├── CT/                              # CT data structures
│   ├── HDF5/                            # HDF5 utilities
│   └── OpenCL/                          # OpenCL utilities
└── python/                              # Python reference implementation
```