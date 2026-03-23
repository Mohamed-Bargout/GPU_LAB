project(
    'CTSVolumeReconstruction',
    ['cpp', 'c'],
    default_options: [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

cc = meson.get_compiler('cpp')

# Include directories
inc_dirs = include_directories('src')  # your source folder

# HDF5 libraries
hdf5_inc = include_directories('/usr/lib/x86_64-linux-gnu/hdf5/serial/include')
hdf5_cpp_lib = cc.find_library('hdf5_cpp', dirs: ['/usr/lib/x86_64-linux-gnu/hdf5/serial'])
hdf5_c_lib   = cc.find_library('hdf5', dirs: ['/usr/lib/x86_64-linux-gnu/hdf5/serial'])
hdf5_hl_cpp  = cc.find_library('hdf5_hl_cpp', dirs: ['/usr/lib/x86_64-linux-gnu/hdf5/serial'])
hdf5_hl      = cc.find_library('hdf5_hl', dirs: ['/usr/lib/x86_64-linux-gnu/hdf5/serial'])


# clFFT library (built locally)
clfft_dep = declare_dependency(
  include_directories: include_directories('clFFT/build/install/include'),
  link_args: [
    '-L' + meson.project_source_root() + '/clFFT/build/install/lib64',
    '-Wl,-rpath,' + meson.project_source_root() + '/clFFT/build/install/lib64',
    '-lclFFT',
  ]
)
# FFTW library (for CPU reconstruction)
fftw_dep = dependency('fftw3f', required: true)
fftw_threads_dep = cc.find_library('fftw3f_threads', required: true)

# OpenMP
omp_dep = dependency('openmp', required: true)

# Main executable
executable('CTSVolumeReconstruction',
    [
        'src/CTSVolumeReconstruction.cpp',
        'src/gpu_fft/hilbert_fft.cpp',
        'src/io/HDF5IO.cpp',
        'src/cpu/cpu_recon.cpp',
        xxd.process('src/CTSVolumeReconstruction.cl'),
    ],
    include_directories: [inc_dirs, hdf5_inc],
    dependencies: [
        hdf5_cpp_lib,
        hdf5_c_lib,
        hdf5_hl_cpp,
        hdf5_hl,
        core_dep,
        opencl_dep,
        ct_dep,
        clfft_dep,
        fftw_dep,
        fftw_threads_dep,
        omp_dep,
    ],
)
