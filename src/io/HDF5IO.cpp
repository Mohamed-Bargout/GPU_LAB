#include "io/HDF5IO.hpp"

// #include <H5Cpp.h>
// #include <Core/Assert.hpp>

using namespace H5;

CTProjectionData loadProjectionData(const std::string& filename)
{
    CTProjectionData data;

    H5File file(filename, H5F_ACC_RDONLY);

    file.openDataSet("num_projs").read(&data.num_projs, PredType::NATIVE_INT);
    file.openDataSet("detector_width").read(&data.detector_width, PredType::NATIVE_INT);
    file.openDataSet("detector_height").read(&data.detector_height, PredType::NATIVE_INT);

    file.openDataSet("Volumen_num_xz").read(&data.Volumen_num_xz, PredType::NATIVE_INT);
    file.openDataSet("Volumen_num_y").read(&data.Volumen_num_y, PredType::NATIVE_INT);

    file.openDataSet("SDD").read(&data.SDD, PredType::NATIVE_FLOAT);
    file.openDataSet("SOD").read(&data.SOD, PredType::NATIVE_FLOAT);
    file.openDataSet("pixelSize").read(&data.pixelSize, PredType::NATIVE_FLOAT);
    file.openDataSet("voxelSize").read(&data.voxelSize, PredType::NATIVE_FLOAT);

    // Angle
    {
        DataSet ds = file.openDataSet("Angle");
        DataSpace space = ds.getSpace();

        hsize_t dims[1];
        space.getSimpleExtentDims(dims);

        data.angles.resize(dims[0]);
        ds.read(data.angles.data(), PredType::NATIVE_FLOAT);
    }

    // Projection
    {
        DataSet ds = file.openDataSet("Projection");
        DataSpace space = ds.getSpace();

        hsize_t dims[3];
        space.getSimpleExtentDims(dims);

        data.projection.resize(
            dims[0] * dims[1] * dims[2]
        );

        ds.read(data.projection.data(), PredType::NATIVE_FLOAT);
    }
    
    file.close();
    return data;
}

void saveVolumeData(const std::string& filename, const std::vector<float>& volume, int Volumen_num_xz, int Volumen_num_y)
{
    H5File file(filename, H5F_ACC_TRUNC);
    hsize_t dims[3] = {(hsize_t)Volumen_num_xz, (hsize_t)Volumen_num_xz, (hsize_t)Volumen_num_y};
    DataSpace dataspace(3, dims);
    DataSet dataset = file.createDataSet("ReconstructedVolume", PredType::NATIVE_FLOAT, dataspace);
    dataset.write(volume.data(), PredType::NATIVE_FLOAT);
    file.close();
}
