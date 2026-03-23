#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

// Helper Functions
inline int proj_index(int proj, int x, int y, int width, int height ) {
    return (proj * width * height) + height * x + y;
}

inline int image_index(int x, int y, int height){
  return (height * x + y);
}

inline int volume_index(int x, int y, int z, int numY, int numZ) {
    return x * numY * numZ + y * numZ + z;
}

// 3D over projections (proj,x,y)
__kernel void Kernel_A_Array(__global float* d_A_Buffer,const float SDD, const float pixelSize) {
    uint x_id = get_global_id(0); // Detector X-index
    uint y_id = get_global_id(1); // Detector Y-index
    
    uint width = get_global_size(0);
    uint height = get_global_size(1);
    
    float u = -(x_id - (width - 1) * 0.5f) * pixelSize; // detector position width
    float v = -(y_id - (height - 1) * 0.5f) * pixelSize; // detector position height
    
    float A = sqrt(SDD * SDD + u * u + v * v);
    
    int index = image_index(x_id, y_id, height);
    d_A_Buffer[index] = A;
    
}

// 3D over projections (proj,x,y)
__kernel void kernel_preprocess(
    __global const float* d_ProjectionData,
    __global float* d_ProjectionDataPreprocessed,
    __global const float* d_A_Buffer,
    float SOD, float pixelSize
)
{
    uint p_id = get_global_id(0);
    uint x_id = get_global_id(1);
    uint y_id = get_global_id(2);

    uint width  = get_global_size(1);
    uint height = get_global_size(2);

    float A = d_A_Buffer[image_index(x_id, y_id, height)]; // Get A value for current pixel

    // Derivative of weight1-applied data
    float derivative;
    if (x_id == 0) {
        float A_next = d_A_Buffer[image_index(1, y_id, height)]; // Get A value for next pixel
        float w_curr = d_ProjectionData[proj_index(p_id, 0, y_id, width, height)] * (SOD / A); // Apply Weight 1
        float w_next = d_ProjectionData[proj_index(p_id, 1, y_id, width, height)] * (SOD / A_next); // Apply Weight 1
        derivative = w_next - w_curr; // Calculate Derivative
    }
    else if (x_id == width - 1) { 
        float A_prev = d_A_Buffer[image_index(x_id - 1, y_id, height)]; // Get A value for previous pixel
        float w_curr = d_ProjectionData[proj_index(p_id, x_id, y_id, width, height)] * (SOD / A); // Apply Weight 1
        float w_prev = d_ProjectionData[proj_index(p_id, x_id - 1, y_id, width, height)] * (SOD / A_prev); // Apply Weight 1
        derivative = w_curr - w_prev; // Calculate Derivative
    }
    else {
        float A_prev = d_A_Buffer[image_index(x_id - 1, y_id, height)];
        float A_next = d_A_Buffer[image_index(x_id + 1, y_id, height)];
        float w_prev = d_ProjectionData[proj_index(p_id, x_id - 1, y_id, width, height)] * (SOD / A_prev); // Apply Weight 1
        float w_next = d_ProjectionData[proj_index(p_id, x_id + 1, y_id, width, height)] * (SOD / A_next); // Apply Weight 1
        derivative = (w_next - w_prev) / 2.0f; // Calculate Derivative
    }

    // Apply derivative scaling and weight2
    d_ProjectionDataPreprocessed[proj_index(p_id, x_id, y_id, width, height)] = derivative / pixelSize * A * A;
}

// 3D Input = (proj,x,y)
// 3D Output = (x,y,z)
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP |
                               CLK_FILTER_LINEAR;

__kernel void kernel_backprojection(
    __read_only image3d_t d_projections, __global const float* d_angles, __global float* d_volume, 
    uint num_projs,
    float voxelSize, float pixelSize,
    float SDD, float SOD,
    uint detector_width, uint detector_height
)
{
  uint x_id = get_global_id(0); // Volume X-index
  uint y_id = get_global_id(1); // Volume Y-index
  uint z_id = get_global_id(2); // Volume Z-index

  uint outX = get_global_size(0); // Volume_num_xz
  uint outY = get_global_size(1); // Volume_num_y
  uint outZ = get_global_size(2); // Volume_num_xz

  float radius_x = outX * 0.5f - 0.5f;
  float radius_y = outY * 0.5f - 0.5f;
  float radius_z = outZ * 0.5f - 0.5f;

  float xpr = (x_id - radius_x) * voxelSize;
  float ypr = (y_id - radius_y) * voxelSize;
  float zpr = (z_id - radius_z) * voxelSize;

  float voxelValue = 0;
  for(uint i = 0; i < num_projs; i++){
    float angle = d_angles[i];
    float sin_a = native_sin(angle);
    float cos_a = native_cos(angle);

    float t = ypr * cos_a - xpr * sin_a;
    float U = SOD + ypr * sin_a + xpr * cos_a;

    float ai = SDD * t / U;
    float bi = zpr * SDD / U;

    float weight_3 = U*U + t*t + zpr*zpr;
    float weight_sin = sign(native_sin(angle + atan(ai / SDD)));

    float ai_pixel = -ai / pixelSize + (detector_width - 1) * 0.5f; // x-axis
    float bi_pixel = -bi / pixelSize + (detector_height - 1) * 0.5f; // y-axis

    voxelValue += read_imagef(d_projections, sampler, (float4)(bi_pixel+0.5, ai_pixel+0.5, i + 0.5f, 0.0f)).x * weight_sin / (weight_3); // order: fastest, middle, slowest moving axis

  }
  d_volume[volume_index(x_id, y_id, z_id, outY, outZ)] = -voxelValue * M_PI_F / num_projs;

}

// Software 2D Interpolation
inline float bilinear_interpolate(
    __global const float* data,
    int proj, int width, int height,
    float u, float v
)
{
    int x0 = (int)floor(u);
    int y0 = (int)floor(v);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Out of bounds: return 0
    if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height)
        return 0.0f;

    float fx = u - x0;
    float fy = v - y0;

    int base = proj * width * height;
    float f00 = data[base + x0 * height + y0];
    float f10 = data[base + x1 * height + y0];
    float f01 = data[base + x0 * height + y1];
    float f11 = data[base + x1 * height + y1];

    return (1 - fx) * (1 - fy) * f00
         + fx * (1 - fy) * f10
         + (1 - fx) * fy * f01
         + fx * fy * f11;
}

__kernel void kernel_backprojection_buffer(
    __global const float* d_projections, __global const float* d_angles, __global float* d_volume, 
    uint num_projs,
    float voxelSize, float pixelSize,
    float SDD, float SOD,
    uint detector_width, uint detector_height
)
{
    uint x_id = get_global_id(0);
    uint y_id = get_global_id(1);
    uint z_id = get_global_id(2);

    uint outX = get_global_size(0);
    uint outY = get_global_size(1);
    uint outZ = get_global_size(2);

    float radius_x = outX * 0.5f - 0.5f;
    float radius_y = outY * 0.5f - 0.5f;
    float radius_z = outZ * 0.5f - 0.5f;

    float xpr = (x_id - radius_x) * voxelSize;
    float ypr = (y_id - radius_y) * voxelSize;
    float zpr = (z_id - radius_z) * voxelSize;

    float voxelValue = 0;
    for (uint i = 0; i < num_projs; i++) {
        float angle = d_angles[i];
        float sin_a = native_sin(angle);
        float cos_a = native_cos(angle);
        
        float t = ypr * cos_a - xpr * sin_a;
        float U = SOD + ypr * sin_a + xpr * cos_a;
        
        float ai = SDD * t / U;
        float bi = zpr * SDD / U;
        
        float weight_3 = U * U + t * t + zpr * zpr;
        float weight_sin = sign(native_sin(angle + atan(ai / SDD)));

        float ai_pixel = -ai / pixelSize + (detector_width - 1) * 0.5f;
        float bi_pixel = -bi / pixelSize + (detector_height - 1) * 0.5f;

        float val = bilinear_interpolate(d_projections, i, detector_width, detector_height, ai_pixel, bi_pixel);

        voxelValue += val * weight_sin / weight_3;
    }
    d_volume[volume_index(x_id, y_id, z_id, outY, outZ)] = -voxelValue * M_PI_F / num_projs;
}

/* --- Hilbert Transform ---*/

// Swap first and last axes: [x, y, z] -> [z, y, x]
__kernel void kernel_transpose_volume(
    __global const float* input,
    __global float* output,
    int numX, int numY, int numZ
)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);
    
    if (x >= numX || y >= numY || z >= numZ) return;
    
    uint in_idx  = x * numY * numZ + y * numZ + z;
    uint out_idx = z * numY * numX + y * numX + x;
    
    output[out_idx] = input[in_idx];
}

// Pack real data into padded complex buffer for Hilbert transform
// Input:  [z, y, x] layout, x contiguous, line length = numX
// Output: complex interleaved, padded to 3*numX per line
//         [left_edge * numX | signal * numX | right_edge * numX]
__kernel void kernel_pack_complex_padded(
    __global const float* input,
    __global float* complex_buf,
    int numX
)
{
    int line = get_global_id(0);
    int i    = get_global_id(1);  // position in padded line (0 to 3*numX-1)
    
    uint n_padded = 3 * numX;
    if (i >= n_padded) return;
    
    uint in_base = line * numX;
    uint out_idx = line * n_padded * 2 + i * 2;
    
    float val;
    if (i < numX) {
        val = input[in_base];              // left edge padding
    } else if (i < 2 * numX) {
        val = input[in_base + (i - numX)]; // actual value
    } else {
        val = input[in_base + numX - 1];   // right edge padding
    }
    
    complex_buf[out_idx]     = val;   // real
    complex_buf[out_idx + 1] = 0.0f;  // imag
}

// Multiply complex FFT result by h array for Hilbert transform
// numX is the padded line length (3 * original numX)
__kernel void kernel_hilbert_multiply(
    __global float* complex_buf,
    __global const float* h_buf,
    uint numX
)
{
    uint line = get_global_id(0);
    uint x    = get_global_id(1);
    
    if (x >= numX) return;
    
    uint idx = line * numX * 2 + x * 2;
    float h = h_buf[x];
    
    complex_buf[idx]     *= h;  // real
    complex_buf[idx + 1] *= h;  // imag
}

// Extract imaginary part from complex buffer, scale by 1/(-2*PI)
__kernel void kernel_unpack_hilbert_padded(
    __global const float* complex_buf,
    __global float* output,
    int numX,
    float scale
)
{
    uint line = get_global_id(0);
    uint x    = get_global_id(1);
    
    if (x >= numX) return;
    
    uint n_padded = 3 * numX;
    uint complex_idx = line * n_padded * 2 + (numX + x) * 2 + 1;  // middle section, +1 for imag
    uint out_idx     = line * numX + x;
    
    output[out_idx] = complex_buf[complex_idx] * scale;
}