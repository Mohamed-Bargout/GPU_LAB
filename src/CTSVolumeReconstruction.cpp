// includes
#include <stdio.h>

#include <CT/DataFiles.hpp>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include <H5Cpp.h>
#include <vector>

// my includes
#include "cpu/cpu_recon.hpp"
#include "io/HDF5IO.hpp"
#include "gpu_fft/hilbert_fft.hpp"

//////////////////////////////////////////////////////////////////////////////
// Configuration
//////////////////////////////////////////////////////////////////////////////
const std::string dataPath = "/lgrp/edu-2025-2-gpulab/Data/proj_shepplogan128.hdf5";
// const std::string dataPath = "/lgrp/edu-2025-2-gpulab/Data/proj_shepplogan512.hdf5";
// const std::string dataPath = "data/proj_shepplogan128.hdf5";

// const std::string dataPath = "data/proj_shepplogan512.hdf5";

const bool USE_BUFFER_BACKPROJECTION = false;  // true = buffer, false = image3d
const bool RUN_CPU = true;                     // true = also run CPU reconstruction

//////////////////////////////////////////////////////////////////////////////
// Utility functions
//////////////////////////////////////////////////////////////////////////////
void printEventTime(const cl::Event& event, const std::string& name) {
    Core::TimeSpan elapsed = OpenCL::getElapsedTime(event);
    std::string strTime = elapsed.toString();
    if (!strTime.empty() && strTime.back() == 's')
        strTime.pop_back();
    float timeMs = std::stof(strTime) * 1000.0f;
    std::cout << name << ": " << timeMs << " ms" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    /* ----- GPU ----- */
    /* --- Load Data --- */
    CTProjectionData data = loadProjectionData(dataPath);

    size_t num_projs       = data.num_projs;
    size_t detector_width  = data.detector_width;
    size_t detector_height = data.detector_height;

    float SDD       = data.SDD;
    float SOD       = data.SOD;
    float pixelSize = data.pixelSize;
    float voxelSize = data.voxelSize;

    int Volumen_num_xz = data.Volumen_num_xz;
    int Volumen_num_y  = data.Volumen_num_y;

    std::vector<float> angles    = data.angles;
    std::vector<float> projection = data.projection;

    /* --- Preprocessing on Host --- */
    // shift angles by -270 degrees
    float angle_shift = -3.0f * (float)M_PI / 2.0f;
    for (size_t i = 0; i < angles.size(); ++i)
        angles[i] += angle_shift;

    /* --- OpenCL Setup --- */
    // Calculate buffer sizes
    size_t countProjectionData = num_projs * detector_width * detector_height;
    size_t sizeProjectionData  = sizeof(float) * countProjectionData;

    size_t countAngles         = angles.size();
    size_t sizeAngles          = sizeof(float) * countAngles;

    size_t countABuffer        = detector_width * detector_height;
    size_t sizeABuffer         = countABuffer * sizeof(float);

    size_t countVolumeData     = (size_t)Volumen_num_xz * Volumen_num_y * Volumen_num_xz;
    size_t sizeVolumeData      = countVolumeData * sizeof(float);

    // Host buffer for final volume data
    std::vector<float> h_volumeData(countVolumeData);

    // Create OpenCL context
    cl::Context context(CL_DEVICE_TYPE_GPU);

    // Get a device of the context
    int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
    std::cout << "Using device " << deviceNr << " / "
                << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
    ASSERT(deviceNr > 0);
    ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
    std::vector<cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cout, device);

    // CPU name
    FILE* cpuinfo = popen("grep 'model name' /proc/cpuinfo | head -n 1", "r");
    if (cpuinfo) {
        char buf[256];
        if (fgets(buf, sizeof(buf), cpuinfo))
            std::cout << "CPU: " << buf;
        pclose(cpuinfo);
    }

    // Create OpenCL command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Build OpenCL program from source
    extern unsigned char CTSVolumeReconstruction_cl[];
    extern unsigned int CTSVolumeReconstruction_cl_len;
    cl::Program program(context, std::string((const char*)CTSVolumeReconstruction_cl, CTSVolumeReconstruction_cl_len));
    OpenCL::buildProgram(program, devices);

    /* --- Allocate Buffers --- */
    cl::Buffer d_A_Buffer(context, CL_MEM_READ_WRITE, sizeABuffer);
    cl::Buffer d_ProjectionData(context, CL_MEM_READ_WRITE, sizeProjectionData);
    cl::Buffer d_PreprocessedProjection(context, CL_MEM_READ_WRITE, sizeProjectionData);
    cl::Buffer d_anglesBuffer(context, CL_MEM_READ_ONLY, sizeAngles);

    /* --- Copy input data to device --- */
    cl::Event eventCopyProj, eventCopyAngles;
    queue.enqueueWriteBuffer(d_ProjectionData, CL_TRUE, 0, sizeProjectionData, projection.data(), NULL, &eventCopyProj);
    queue.enqueueWriteBuffer(d_anglesBuffer, CL_TRUE, 0, sizeAngles, angles.data(), NULL, &eventCopyAngles);

    /* --- Workgroup sizes --- */
    size_t wgSizeP = 1;
    size_t wgSizeX = 16;
    size_t wgSizeY = 16;
    while (detector_width % wgSizeX != 0) wgSizeX--;
    while (detector_height % wgSizeY != 0) wgSizeY--;
    
    size_t wgSizeVX = 8;
    size_t wgSizeVY = 8;
    size_t wgSizeVZ = 8;
    while (Volumen_num_xz % wgSizeVX != 0) wgSizeVX--;
    while (Volumen_num_y % wgSizeVY != 0) wgSizeVY--;
    while (Volumen_num_xz % wgSizeVZ != 0) wgSizeVZ--;
    //////////////////////////////////////////////////////////////////////////
    // Kernel 0: Compute A array
    //////////////////////////////////////////////////////////////////////////
    cl::Kernel kernelA(program, "Kernel_A_Array");
    kernelA.setArg(0, d_A_Buffer);
    kernelA.setArg(1, (cl_float)SDD);
    kernelA.setArg(2, (cl_float)pixelSize);

    cl::Event eventKernelA;
    queue.enqueueNDRangeKernel(kernelA,
        cl::NullRange,
        cl::NDRange(detector_width, detector_height),
        cl::NDRange(wgSizeX, wgSizeY),
        NULL,
        &eventKernelA);
    //////////////////////////////////////////////////////////////////////////
    // Kernel 1: Preprocess projections (apply weight1, compute derivative, and apply weight2)
    //////////////////////////////////////////////////////////////////////////
    cl::Kernel kernelPreprocess(program, "kernel_preprocess");
    kernelPreprocess.setArg(0, d_ProjectionData);
    kernelPreprocess.setArg(1, d_PreprocessedProjection);
    kernelPreprocess.setArg(2, d_A_Buffer);
    kernelPreprocess.setArg(3, (cl_float)SOD);
    kernelPreprocess.setArg(4, (cl_float)pixelSize);

    cl::Event eventKernelPre;
    queue.enqueueNDRangeKernel(kernelPreprocess,
        cl::NullRange,
        cl::NDRange(num_projs, detector_width, detector_height),
        cl::NDRange(wgSizeP, wgSizeX, wgSizeY),
        NULL,
        &eventKernelPre);

    d_ProjectionData = cl::Buffer();  // release
    d_A_Buffer = cl::Buffer();        // release

    //////////////////////////////////////////////////////////////////////////
    // Kernel 2: Backprojection (image or buffer variant)
    //////////////////////////////////////////////////////////////////////////
    cl::Buffer d_volumeData;
    cl::Event eventKernelBP;
    cl::Event eventBufferToImage;

    if (USE_BUFFER_BACKPROJECTION) {
        // Buffer-based backprojection
        d_volumeData = cl::Buffer(context, CL_MEM_READ_WRITE, sizeVolumeData);

        cl::Kernel kernelBP(program, "kernel_backprojection_buffer");
        kernelBP.setArg(0, d_PreprocessedProjection);
        kernelBP.setArg(1, d_anglesBuffer);
        kernelBP.setArg(2, d_volumeData);
        kernelBP.setArg(3, (cl_int)num_projs);
        kernelBP.setArg(4, (cl_float)voxelSize);
        kernelBP.setArg(5, (cl_float)pixelSize);
        kernelBP.setArg(6, (cl_float)SDD);
        kernelBP.setArg(7, (cl_float)SOD);
        kernelBP.setArg(8, (cl_int)detector_width);
        kernelBP.setArg(9, (cl_int)detector_height);

        queue.enqueueNDRangeKernel(kernelBP,
            cl::NullRange,
            cl::NDRange(Volumen_num_xz, Volumen_num_y, Volumen_num_xz),
            cl::NDRange(wgSizeVX, wgSizeVY, wgSizeVZ),
            NULL,
            &eventKernelBP);

        d_PreprocessedProjection = cl::Buffer(); // release

    } else {
        // Image-based backprojection 
        cl::ImageFormat format(CL_R, CL_FLOAT);
        cl::Image3D d_projImage(context, CL_MEM_READ_ONLY, format,
            detector_height, detector_width, num_projs, 0, 0, NULL);

        cl::size_t<3> origin; origin[0] = 0; origin[1] = 0; origin[2] = 0;
        cl::size_t<3> region; region[0] = detector_height; region[1] = detector_width; region[2] = num_projs;

        queue.enqueueCopyBufferToImage(d_PreprocessedProjection, d_projImage, 0, origin, region, NULL, &eventBufferToImage);
        d_PreprocessedProjection = cl::Buffer(); // release

        d_volumeData = cl::Buffer(context, CL_MEM_READ_WRITE, sizeVolumeData); // allocate output buffer

        cl::Kernel kernelBP(program, "kernel_backprojection");
        kernelBP.setArg(0, d_projImage);
        kernelBP.setArg(1, d_anglesBuffer);
        kernelBP.setArg(2, d_volumeData);
        kernelBP.setArg(3, (cl_int)num_projs);
        kernelBP.setArg(4, (cl_float)voxelSize);
        kernelBP.setArg(5, (cl_float)pixelSize);
        kernelBP.setArg(6, (cl_float)SDD);
        kernelBP.setArg(7, (cl_float)SOD);
        kernelBP.setArg(8, (cl_int)detector_width);
        kernelBP.setArg(9, (cl_int)detector_height);

        queue.enqueueNDRangeKernel(kernelBP,
            cl::NullRange,
            cl::NDRange(Volumen_num_xz, Volumen_num_y, Volumen_num_xz),
            cl::NDRange(wgSizeVX, wgSizeVY, wgSizeVZ),
            NULL,
            &eventKernelBP);

        d_projImage = cl::Image3D(); // release
    }

    d_anglesBuffer = cl::Buffer(); // release
    queue.finish();

    //////////////////////////////////////////////////////////////////////////
    // Hilbert Transform (GPU FFT)
    //////////////////////////////////////////////////////////////////////////
    Core::TimeSpan hilbertStart = Core::getCurrentTime();

    hilbertTransformGPU(d_volumeData, context, queue, program,
        Volumen_num_xz, Volumen_num_y);

    Core::TimeSpan hilbertEnd = Core::getCurrentTime();
    Core::TimeSpan hilbertTime = hilbertEnd - hilbertStart;

    //////////////////////////////////////////////////////////////////////////
    // Read result back to host
    //////////////////////////////////////////////////////////////////////////
    cl::Event eventCopyToHost;
    queue.enqueueReadBuffer(d_volumeData, CL_TRUE, 0, sizeVolumeData, h_volumeData.data(), NULL, &eventCopyToHost);
    d_volumeData = cl::Buffer(); // release
    
    // Save Volume as HDF5
    saveVolumeData("reconstructed/volume_gpu.hdf5", h_volumeData, Volumen_num_xz, Volumen_num_y);
    std::cout << "Saved volume to reconstructed/volume_gpu.hdf5" << std::endl;

    //////////////////////////////////////////////////////////////////////////
    // Timing
    //////////////////////////////////////////////////////////////////////////
    Core::TimeSpan timePreprocess = OpenCL::getElapsedTime(eventKernelA) + OpenCL::getElapsedTime(eventKernelPre);
    Core::TimeSpan timeBP        = OpenCL::getElapsedTime(eventKernelBP);
    Core::TimeSpan totalGpuTime  = timePreprocess + timeBP + hilbertTime;

    Core::TimeSpan copyToDevTime  = OpenCL::getElapsedTime(eventCopyProj);
    Core::TimeSpan copyToHostTime = OpenCL::getElapsedTime(eventCopyToHost);
    Core::TimeSpan totalCopyTime  = copyToDevTime + copyToHostTime;
    
    std::cout << "\n------- GPU OpenCL Reconstruction -------" << std::endl;
    std::cout << "------- Kernel Timing -------" << std::endl;
    std::cout << "Preprocessing:          " << timePreprocess.toString() << std::endl;
    std::cout << "Backprojection:         " << timeBP.toString() << std::endl;
    std::cout << "Hilbert Transform:      " << hilbertTime.toString() << std::endl;
    std::cout << "Total GPU Time:         " << totalGpuTime.toString() << std::endl;
    std::cout << "------- Memory Transfer -------" << std::endl;
    std::cout << "Copy proj to device:    " << copyToDevTime.toString() << std::endl;
    if (!USE_BUFFER_BACKPROJECTION) {
        Core::TimeSpan bufToImgTime = OpenCL::getElapsedTime(eventBufferToImage);
        totalCopyTime = totalCopyTime + bufToImgTime;
        std::cout << "Buffer to Image copy:   " << bufToImgTime.toString() << std::endl;
    }
    std::cout << "Copy volume to host:    " << copyToHostTime.toString() << std::endl;
    std::cout << "Total memory transfer:  " << totalCopyTime.toString() << std::endl;
    std::cout << "------- Total -------" << std::endl;
    std::cout << "Total w/o mem transfer: " << totalGpuTime.toString() << std::endl;
    std::cout << "Total with mem transfer:" << (totalGpuTime + totalCopyTime).toString() << std::endl;

    if (RUN_CPU) {
        std::cout << "\n------- CPU Reconstruction -------" << std::endl;
        runCPUReconstruction(dataPath, "reconstructed/volume_cpu.hdf5");
    }
    
    return 0;
}