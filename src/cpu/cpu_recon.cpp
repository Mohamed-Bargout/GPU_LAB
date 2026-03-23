#include "cpu_recon.hpp"

#include <H5Cpp.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <fftw3.h>
#include <complex>
#include <iostream>

#include <Core/Time.hpp>

using namespace H5;

namespace {

constexpr float PI_CPU = 3.14159265358979323846f;

struct CBParameters {
    float voxel_size, SDD, SOD, pixel_size;
    int volume_num_xz, volume_num_y, num_projs, detector_width, detector_height;
};

class HDF5Manager {
public:
    static CBParameters loadParameters(const std::string& filename) {
        H5File file(filename, H5F_ACC_RDONLY);
        CBParameters params;
        double val;
        DataSet(file.openDataSet("SDD")).read(&params.SDD, PredType::NATIVE_FLOAT);
        DataSet(file.openDataSet("SOD")).read(&params.SOD, PredType::NATIVE_FLOAT);
        DataSet(file.openDataSet("voxelSize")).read(&params.voxel_size, PredType::NATIVE_FLOAT);
        DataSet(file.openDataSet("pixelSize")).read(&params.pixel_size, PredType::NATIVE_FLOAT);
        DataSet(file.openDataSet("Volumen_num_xz")).read(&val, PredType::NATIVE_DOUBLE);
        params.volume_num_xz = static_cast<int>(val);
        DataSet(file.openDataSet("Volumen_num_y")).read(&val, PredType::NATIVE_DOUBLE);
        params.volume_num_y = static_cast<int>(val);
        DataSet(file.openDataSet("num_projs")).read(&val, PredType::NATIVE_DOUBLE);
        params.num_projs = static_cast<int>(val);
        DataSet(file.openDataSet("detector_width")).read(&val, PredType::NATIVE_DOUBLE);
        params.detector_width = static_cast<int>(val);
        DataSet(file.openDataSet("detector_height")).read(&val, PredType::NATIVE_DOUBLE);
        params.detector_height = static_cast<int>(val);
        file.close();
        return params;
    }

    static std::vector<float> loadProjections(const std::string& filename, const CBParameters& params) {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet proj_dataset = file.openDataSet("Projection");
        size_t total_size = (size_t)params.num_projs * params.detector_width * params.detector_height;
        std::vector<float> projections(total_size);
        proj_dataset.read(projections.data(), PredType::NATIVE_FLOAT);
        proj_dataset.close();
        file.close();
        return projections;
    }

    static void saveVolume(const std::string& filename, const std::vector<float>& volume, const CBParameters& params) {
        H5File file(filename, H5F_ACC_TRUNC);
        hsize_t dims[3] = {static_cast<hsize_t>(params.volume_num_xz),
                           static_cast<hsize_t>(params.volume_num_xz),
                           static_cast<hsize_t>(params.volume_num_y)};
        DataSpace dataspace(3, dims);
        DataSet dataset = file.createDataSet("ReconstructedVolume", PredType::NATIVE_FLOAT, dataspace);
        dataset.write(volume.data(), PredType::NATIVE_FLOAT);
        dataset.close();
        dataspace.close();
        file.close();
    }
};

class Preprocessing {
public:
    static std::vector<float> getPixelPosition(int width) {
        std::vector<float> positions(width);
        float center = (width - 1) / 2.0f;
        for (int i = 0; i < width; i++) positions[i] = i - center;
        return positions;
    }

    static std::vector<float> gradient(const std::vector<float>& x, float dx = 1.0f) {
        std::vector<float> grad(x.size());
        if (x.size() < 2) return grad;
        grad[0] = (x[1] - x[0]) / dx;
        for (size_t i = 1; i < x.size() - 1; i++) grad[i] = (x[i + 1] - x[i - 1]) / (2.0f * dx);
        grad[x.size() - 1] = (x[x.size() - 1] - x[x.size() - 2]) / dx;
        return grad;
    }

    static void applyWeight1AndDerivAndWeight2(std::vector<float>& projections, const CBParameters& params) {
        auto u_pos = getPixelPosition(params.detector_width);
        auto v_pos = getPixelPosition(params.detector_height);
        for (auto& u : u_pos) u *= -params.pixel_size;
        for (auto& v : v_pos) v *= -params.pixel_size;

        std::vector<float> w1(params.detector_width * params.detector_height);
        std::vector<float> w2(params.detector_width * params.detector_height);
        for (int u_idx = 0; u_idx < params.detector_width; u_idx++) {
            for (int v_idx = 0; v_idx < params.detector_height; v_idx++) {
                float A = std::sqrt(params.SDD * params.SDD + u_pos[u_idx] * u_pos[u_idx] + v_pos[v_idx] * v_pos[v_idx]);
                w1[u_idx * params.detector_height + v_idx] = params.SOD / A;
                w2[u_idx * params.detector_height + v_idx] = A * A;
            }
        }

        #pragma omp parallel for
        for (int p = 0; p < params.num_projs; p++) {
            for (int i = 0; i < params.detector_width * params.detector_height; i++) {
                projections[p * params.detector_width * params.detector_height + i] *= w1[i];
            }
            for (int v = 0; v < params.detector_height; v++) {
                std::vector<float> line(params.detector_width);
                for (int u = 0; u < params.detector_width; u++) {
                    line[u] = projections[p * params.detector_width * params.detector_height + u * params.detector_height + v];
                }
                auto grad = gradient(line, 1.0f);
                for (int u = 0; u < params.detector_width; u++) {
                    projections[p * params.detector_width * params.detector_height + u * params.detector_height + v] = grad[u] / params.pixel_size;
                }
            }
            for (int i = 0; i < params.detector_width * params.detector_height; i++) {
                projections[p * params.detector_width * params.detector_height + i] *= w2[i];
            }
        }
    }
};

class Interpolator {
public:
    static float bilinearInterpolate(const float* grid, int width, int height,
                                      float u, float v,
                                      float u_first, float u_last,
                                      float v_first, float v_last) {
        float u_min = std::min(u_first, u_last);
        float u_max = std::max(u_first, u_last);
        float v_min = std::min(v_first, v_last);
        float v_max = std::max(v_first, v_last);
        if (u < u_min || u > u_max || v < v_min || v > v_max) return 0.0f;
        float u_norm = (u - u_first) / (u_last - u_first) * (width - 1);
        float v_norm = (v - v_first) / (v_last - v_first) * (height - 1);
        int u0 = std::max(0, std::min((int)std::floor(u_norm), width - 2));
        int v0 = std::max(0, std::min((int)std::floor(v_norm), height - 2));
        float du = u_norm - u0;
        float dv = v_norm - v0;
        float f00 = grid[u0 * height + v0];
        float f10 = grid[(u0 + 1) * height + v0];
        float f01 = grid[u0 * height + (v0 + 1)];
        float f11 = grid[(u0 + 1) * height + (v0 + 1)];
        return (1-du)*(1-dv)*f00 + du*(1-dv)*f10 + (1-du)*dv*f01 + du*dv*f11;
    }
};

class Backprojector {
public:
    static std::vector<float> reconstruct(const std::vector<float>& projections, const CBParameters& params) {
        int vol_xz = params.volume_num_xz;
        int vol_y = params.volume_num_y;
        int width = params.detector_width;
        int height = params.detector_height;
        std::vector<float> volume(vol_xz * vol_xz * vol_y, 0.0f);
        auto u_pos = Preprocessing::getPixelPosition(width);
        auto v_pos = Preprocessing::getPixelPosition(height);
        float u_first = u_pos[0] * (-params.pixel_size);
        float u_last = u_pos[width - 1] * (-params.pixel_size);
        float v_first = v_pos[0] * (-params.pixel_size);
        float v_last = v_pos[height - 1] * (-params.pixel_size);
        float radius_xz = vol_xz / 2.0f - 0.5f;
        float radius_y = vol_y / 2.0f - 0.5f;
        float angle_shift = -3.0f * PI_CPU / 2.0f;

        for (int p_idx = 0; p_idx < params.num_projs; p_idx++) {
            float theta = angle_shift + (2.0f * PI_CPU * p_idx) / params.num_projs;
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            const float* proj_data = projections.data() + p_idx * width * height;

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < vol_xz; i++) {
                for (int j = 0; j < vol_xz; j++) {
                    float xpr = (i - radius_xz) * params.voxel_size;
                    float ypr = (j - radius_xz) * params.voxel_size;
                    float t = ypr * cos_t - xpr * sin_t;
                    float U = params.SOD + ypr * sin_t + cos_t * xpr;
                    for (int k = 0; k < vol_y; k++) {
                        float zpr = (k - radius_y) * params.voxel_size;
                        float ai = params.SDD * t / U;
                        float bi = zpr * params.SDD / U;
                        float weight_3 = U * U + t * t + zpr * zpr;
                        float weight_sin = (std::sin(theta + std::atan2(ai, params.SDD)) >= 0.0f) ? 1.0f : -1.0f;
                        float sample = Interpolator::bilinearInterpolate(proj_data, width, height, ai, bi, u_first, u_last, v_first, v_last);
                        volume[i * vol_xz * vol_y + j * vol_y + k] += sample * weight_sin / weight_3;
                    }
                }
            }
        }
        return volume;
    }
};

class HilbertTransform {
public:
    static void applyAlongX(std::vector<float>& volume, int vol_xz, int vol_y) {
        int n = vol_xz;
        int n_padded = 3 * n;
        fftwf_init_threads();
        fftwf_complex* plan_buf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n_padded);
        fftwf_plan plan_fwd = fftwf_plan_dft_1d(n_padded, plan_buf, plan_buf, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_plan plan_inv = fftwf_plan_dft_1d(n_padded, plan_buf, plan_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_free(plan_buf);
        std::vector<float> h(n_padded, 0.0f);
        h[0] = 1.0f;
        for (int i = 1; i < n_padded / 2; i++) h[i] = 2.0f;
        if (n_padded % 2 == 0) h[n_padded / 2] = 1.0f;

        #pragma omp parallel
        {
            fftwf_complex* fft_buf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n_padded);
            #pragma omp for collapse(2)
            for (int j = 0; j < vol_xz; j++) {
                for (int k = 0; k < vol_y; k++) {
                    float left_edge = volume[0 * vol_xz * vol_y + j * vol_y + k];
                    float right_edge = volume[(n-1) * vol_xz * vol_y + j * vol_y + k];
                    for (int i = 0; i < n; i++) { fft_buf[i][0] = left_edge; fft_buf[i][1] = 0.0f; }
                    for (int i = 0; i < n; i++) { fft_buf[n + i][0] = volume[i * vol_xz * vol_y + j * vol_y + k]; fft_buf[n + i][1] = 0.0f; }
                    for (int i = 0; i < n; i++) { fft_buf[2*n + i][0] = right_edge; fft_buf[2*n + i][1] = 0.0f; }
                    fftwf_execute_dft(plan_fwd, fft_buf, fft_buf);
                    for (int i = 0; i < n_padded; i++) { fft_buf[i][0] *= h[i]; fft_buf[i][1] *= h[i]; }
                    fftwf_execute_dft(plan_inv, fft_buf, fft_buf);
                    for (int i = 0; i < n; i++) { volume[i * vol_xz * vol_y + j * vol_y + k] = fft_buf[n + i][1] / n_padded; }
                }
            }
            fftwf_free(fft_buf);
        }
        fftwf_destroy_plan(plan_fwd);
        fftwf_destroy_plan(plan_inv);
        fftwf_cleanup_threads();
    }
};

} 

void runCPUReconstruction(const std::string& inputPath, const std::string& outputPath) {
    CBParameters params = HDF5Manager::loadParameters(inputPath);
    auto projections = HDF5Manager::loadProjections(inputPath, params);

    Core::TimeSpan startTotal = Core::getCurrentTime();

    Core::TimeSpan startPre = Core::getCurrentTime();
    Preprocessing::applyWeight1AndDerivAndWeight2(projections, params);
    Core::TimeSpan timePre = Core::getCurrentTime() - startPre;

    Core::TimeSpan startBP = Core::getCurrentTime();
    auto volume = Backprojector::reconstruct(projections, params);
    Core::TimeSpan timeBP = Core::getCurrentTime() - startBP;

    Core::TimeSpan startPost = Core::getCurrentTime();
    float scale1 = -PI_CPU / params.num_projs;
    #pragma omp parallel for
    for (size_t i = 0; i < volume.size(); i++) {
        volume[i] *= scale1;
    }

    HilbertTransform::applyAlongX(volume, params.volume_num_xz, params.volume_num_y);

    float scale2 = 1.0f / (-2.0f * PI_CPU);
    #pragma omp parallel for
    for (size_t i = 0; i < volume.size(); i++) {
        volume[i] *= scale2;
    }
    Core::TimeSpan timePost = Core::getCurrentTime() - startPost;

    Core::TimeSpan timeTotal = Core::getCurrentTime() - startTotal;

    std::cout << "------- CPU Timing -------" << std::endl;
    std::cout << "Preprocessing:          " << timePre.toString() << std::endl;
    std::cout << "Backprojection:         " << timeBP.toString() << std::endl;
    std::cout << "Post-processing (DHT):  " << timePost.toString() << std::endl;
    std::cout << "Total CPU Time:         " << timeTotal.toString() << std::endl;

    HDF5Manager::saveVolume(outputPath, volume, params);
    std::cout << "Saved CPU volume to " << outputPath << std::endl;
}