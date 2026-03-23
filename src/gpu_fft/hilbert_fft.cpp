#include "hilbert_fft.hpp"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void hilbertTransformGPU(
    cl::Buffer& d_volumeData,
    cl::Context& context,
    cl::CommandQueue& queue,
    cl::Program& program,
    int Volumen_num_xz,
    int Volumen_num_y
)
{
    size_t countVolumeData = (size_t)Volumen_num_xz * Volumen_num_y * Volumen_num_xz;
    size_t sizeVolumeData = countVolumeData * sizeof(float);
    size_t numLines = (size_t)Volumen_num_y * Volumen_num_xz;
    int n_padded = 3 * Volumen_num_xz;

    // --- Step 1: Transpose [x, z, y] -> [y, z, x] ---
    cl::Buffer d_volumeTransposed(context, CL_MEM_READ_WRITE, sizeVolumeData);

    cl::Kernel kernelTranspose(program, "kernel_transpose_volume");
    kernelTranspose.setArg(0, d_volumeData);
    kernelTranspose.setArg(1, d_volumeTransposed);
    kernelTranspose.setArg(2, (cl_int)Volumen_num_xz);
    kernelTranspose.setArg(3, (cl_int)Volumen_num_y);
    kernelTranspose.setArg(4, (cl_int)Volumen_num_xz);

    queue.enqueueNDRangeKernel(
        kernelTranspose, cl::NullRange,
        cl::NDRange(Volumen_num_xz, Volumen_num_y, Volumen_num_xz),
        cl::NDRange(8, 8, 8)
    );

    d_volumeData = cl::Buffer(); // release original volume

    // --- Step 2: Pack into padded complex buffer ---
    cl::Buffer d_complexBuf(context, CL_MEM_READ_WRITE, 2 * 3 * sizeVolumeData);

    cl::Kernel kernelPack(program, "kernel_pack_complex_padded");
    kernelPack.setArg(0, d_volumeTransposed);
    kernelPack.setArg(1, d_complexBuf);
    kernelPack.setArg(2, (cl_int)Volumen_num_xz);

    queue.enqueueNDRangeKernel(
        kernelPack, cl::NullRange,
        cl::NDRange(numLines, n_padded),
        cl::NullRange
    );

    d_volumeTransposed = cl::Buffer(); // release transposed buffer
    queue.finish();

    // --- Step 3: clFFT setup and forward FFT ---
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);

    size_t fftLengths[1] = {(size_t)n_padded};
    clfftPlanHandle fftPlan;
    clfftCreateDefaultPlan(&fftPlan, context(), CLFFT_1D, fftLengths);
    clfftSetPlanPrecision(fftPlan, CLFFT_SINGLE);
    clfftSetLayout(fftPlan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(fftPlan, CLFFT_INPLACE);
    clfftSetPlanBatchSize(fftPlan, numLines);
    clfftSetPlanDistance(fftPlan, (size_t)n_padded, (size_t)n_padded);

    cl_command_queue rawQueue = queue();
    clfftBakePlan(fftPlan, 1, &rawQueue, NULL, NULL);

    cl_mem rawBuf = d_complexBuf();
    clfftEnqueueTransform(fftPlan, CLFFT_FORWARD, 1, &rawQueue, 0, NULL, NULL, &rawBuf, NULL, NULL);
    queue.finish();

    // --- Step 4: Multiply by Hilbert filter h ---
    std::vector<float> h_hilbert(n_padded, 0.0f);
    h_hilbert[0] = 1.0f;
    for (int i = 1; i < n_padded / 2; i++){
        h_hilbert[i] = 2.0f;
    }
    h_hilbert[n_padded / 2] = 1.0f;

    cl::Buffer d_hBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n_padded * sizeof(float), h_hilbert.data());

    cl::Kernel kernelHilbert(program, "kernel_hilbert_multiply");
    kernelHilbert.setArg(0, d_complexBuf);
    kernelHilbert.setArg(1, d_hBuf);
    kernelHilbert.setArg(2, (cl_int)n_padded);

    queue.enqueueNDRangeKernel(
        kernelHilbert, cl::NullRange,
        cl::NDRange(numLines, n_padded),
        cl::NullRange
    );
    queue.finish();

    // --- Step 5: Inverse FFT ---
    clfftEnqueueTransform(fftPlan, CLFFT_BACKWARD, 1, &rawQueue, 0, NULL, NULL, &rawBuf, NULL, NULL);
    queue.finish();

    // --- Step 6: Unpack imaginary part and scale ---
    cl::Buffer d_volumeHilbert(context, CL_MEM_READ_WRITE, sizeVolumeData);
    cl_float hilbertScale = 1.0f / (-2.0f * (float)M_PI);

    cl::Kernel kernelUnpack(program, "kernel_unpack_hilbert_padded");
    kernelUnpack.setArg(0, d_complexBuf);
    kernelUnpack.setArg(1, d_volumeHilbert);
    kernelUnpack.setArg(2, (cl_int)Volumen_num_xz);
    kernelUnpack.setArg(3, hilbertScale);

    queue.enqueueNDRangeKernel(
        kernelUnpack, cl::NullRange,
        cl::NDRange(numLines, Volumen_num_xz),
        cl::NDRange(16, 16)
    );

    d_complexBuf = cl::Buffer(); // release complex buffer

    // --- Step 7: Transpose back [y, z, x] -> [x, z, y] ---
    d_volumeData = cl::Buffer(context, CL_MEM_READ_WRITE, sizeVolumeData);

    kernelTranspose.setArg(0, d_volumeHilbert);
    kernelTranspose.setArg(1, d_volumeData);
    kernelTranspose.setArg(2, (cl_int)Volumen_num_y);
    kernelTranspose.setArg(3, (cl_int)Volumen_num_xz);
    kernelTranspose.setArg(4, (cl_int)Volumen_num_xz);

    queue.enqueueNDRangeKernel(
        kernelTranspose, cl::NullRange,
        cl::NDRange(Volumen_num_y, Volumen_num_xz, Volumen_num_xz),
        cl::NDRange(8, 8, 8)
    );

    d_volumeHilbert = cl::Buffer(); // release

    queue.finish();

    // --- Cleanup clFFT ---
    clfftDestroyPlan(&fftPlan);
    clfftTeardown();
}