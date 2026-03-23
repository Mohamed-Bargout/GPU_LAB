#ifndef HILBERT_FFT_HPP
#define HILBERT_FFT_HPP

#include <OpenCL/cl-patched.hpp>
#include <clFFT.h>
#include <vector>

// Perform the Hilbert transform on the volume data using GPU FFT
// Volume layout: [x, z, y] with x slowest
// Steps: transpose -> pack complex (padded) -> FFT -> multiply h -> IFFT -> unpack -> transpose back
void hilbertTransformGPU(
    cl::Buffer& d_volumeData,
    cl::Context& context,
    cl::CommandQueue& queue,
    cl::Program& program,
    int Volumen_num_xz,
    int Volumen_num_y
);

#endif // HILBERT_FFT_HPP