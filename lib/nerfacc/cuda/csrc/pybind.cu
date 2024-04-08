/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"
#include "include/helpers_math.h"


torch::Tensor unpack_info(
    const torch::Tensor packed_info, const int n_samples);

torch::Tensor unpack_info_to_mask(
    const torch::Tensor packed_info, const int n_samples);

std::vector<torch::Tensor> ray_resampling(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    torch::Tensor sdfs,
    const int steps);

std::vector<torch::Tensor> ray_resampling_merge(
    torch::Tensor packed_info,
    torch::Tensor vals,
    torch::Tensor is_left,
    torch::Tensor is_right,
    torch::Tensor weights,
    const int steps);

std::vector<torch::Tensor> ray_resampling_fine(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    const int steps);

std::vector<torch::Tensor> ray_resampling_sdf_fine(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor alphas,
    torch::Tensor sdfs,
    const int steps);

torch::Tensor unpack_data(
    torch::Tensor packed_info,
    torch::Tensor data,
    int n_samples_per_ray);

// cub implementations: parallel across samples
bool is_cub_available() {
    return (bool) CUB_SUPPORTS_SCAN_BY_KEY();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // importance sampling
    m.def("ray_resampling", &ray_resampling);
    m.def("ray_resampling_fine", &ray_resampling_fine);
    m.def("ray_resampling_merge", &ray_resampling_merge);
    m.def("ray_resampling_sdf_fine", &ray_resampling_sdf_fine);

    // pack & unpack
    m.def("unpack_data", &unpack_data);
    m.def("unpack_info", &unpack_info);
    m.def("unpack_info_to_mask", &unpack_info_to_mask);
}
