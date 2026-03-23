#pragma once
#include <Core/Assert.hpp>
#include <H5Cpp.h>
#include <string>
#include <vector>

// #include <HDF5/BaseTypes.hpp>
// #include <HDF5/DataSet.hpp>
// #include <HDF5/DataSpace.hpp>
// #include <HDF5/DataType.hpp>
// #include <HDF5/File.hpp>
// #include <HDF5/Group.hpp>
// #include <HDF5/SerializationKey.hpp>
// #include <HDF5/Type.hpp>
// #include <HDF5/Util.hpp>

struct CTProjectionData {
    int num_projs;
    int detector_width;
    int detector_height;

    float SDD;
    float SOD;
    float pixelSize;
    float voxelSize;

    int Volumen_num_xz;
    int Volumen_num_y;

    std::vector<float> angles;
    std::vector<float> projection;
};

CTProjectionData loadProjectionData(const std::string& filename);

void saveVolumeData(const std::string& filename, const std::vector<float>& volume, int Volumen_num_xz, int Volumen_num_y);
