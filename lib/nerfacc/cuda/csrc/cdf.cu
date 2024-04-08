/*
 * Copyright (c) 2023 Shaofei Wang, ETH Zurich.
 */

#include "c10/core/TensorOptions.h"
#include "include/helpers_cuda.h"
#include <cstdint>

template <typename scalar_t>
__global__ void cdf_resampling_kernel(
    const uint32_t n_rays,
    const int *packed_info,  // input ray & point indices.
    const scalar_t *starts,  // input start t
    const scalar_t *ends,    // input end t
    const scalar_t *weights, // transmittance weights
    const scalar_t *sdfs,    // input sdf values
    const int *resample_packed_info,
    scalar_t *resample_ts,
    scalar_t *resample_offsets,
    int64_t *surface_idx,
    int64_t *resample_indices,
    int32_t *resample_fg_counts,
    int32_t *resample_bg_counts)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    weights += base;
    sdfs += base;
    resample_fg_counts += base;
    // resample_starts += resample_base;
    // resample_ends += resample_base;
    resample_ts += resample_base;
    resample_offsets += resample_base;
    resample_indices += resample_base;

    // Do not normalize weights, instead, add a new interval at the end
    // with weight = 1.0 - weights_sum
    scalar_t weights_sum = 0.0f;
    for (int j = 0; j < steps; j++)
        weights_sum += weights[j];
    weights_sum += fmaxf(1.0f - weights_sum, 0.0f);

    int num_bins = resample_steps;
    scalar_t cdf_step_size = (1.0f - 1.0 / num_bins) / (resample_steps - 1);

    int idx = 0, j = 0;
    scalar_t cdf_prev = 0.0f, cdf_next = weights[idx] / weights_sum;
    scalar_t cdf_u = 1.0 / (2 * num_bins);
    scalar_t sdf_prev = sdfs[0];
    // if (sdf_prev < 0) {
    //     printf("Resample error: %d %f\n", i, sdf_prev);
    // }
    scalar_t sdf_next = scalar_t(0);
    if (steps > 1)
        sdf_next = sdfs[1];
    bool found_surface = false;
    while (j < num_bins && idx < steps)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
            scalar_t offset = (cdf_u - cdf_prev) * scaling; 
            scalar_t t = offset + starts[idx];
            if (sdf_prev >= 0 && sdf_next < 0 && !found_surface) {
                // if the current interval crosses the iso-surface, we approximate
                // the sdf of current sample via linear interpolation between 
                // `sdf_prev` and `sdf_next`. If the approximation is negative, we
                // repeat `t` from previous sample for current sample, such that we
                // will not have a `t` that is located inside the shape.
                scalar_t sdf_approx = sdf_prev + (sdf_next - sdf_prev) * (offset / (ends[idx] - starts[idx]));
                // If j == 0 and the approximated sdf is negative, we simply take 
                // the start of the interval as `t`. This is a rather special case
                // and should almost never happen.
                resample_ts[j] = sdf_approx >= 0 ? t : (j > 0 ? resample_ts[j - 1] : starts[idx]);
                // resample_ts[j] = starts[idx];
            } else if (found_surface) {
                // If the current interval does not cross the iso-surface but a 
                // surface interval has been found previously, we repeat `t` from
                // previous sample for current sample, eventually all samples after
                // the first zero-crossing interval will use the `t` from the point
                // right before the first zero-crossing on the ray.
                resample_ts[j] = j > 0 ? resample_ts[j - 1] : starts[idx];
            } else {
                // If current interval is not crossing a surface, and a surface
                // has not been found yet, we do standard importance sampling
                resample_ts[j] = t;
            }
            resample_offsets[j] = offset;
            resample_indices[j] = idx + base;
            // Increasing the count of the interval via atomicAdd
            atomicAdd(&resample_fg_counts[idx], 1);
            // going further to next resample
            cdf_u += cdf_step_size;
            j += 1;
        }
        else if (idx < steps - 1)
        {
            // going to next interval
            idx += 1;
            if (sdf_prev >= 0 && sdf_next < 0 && !found_surface)
            {
                // record index of the surface point
                surface_idx[i] = idx - 1 + base;
                found_surface = true;
            }
            sdf_prev = sdfs[idx];
            sdf_next = idx < steps - 1 ? sdfs[idx + 1] : scalar_t(0);
            cdf_prev = cdf_next;
            cdf_next += weights[idx] / weights_sum;
        } else {
            break;
        }
    }
    // If we are out of the loop with j < num_bins, it means we have not sampled
    // enough points. In this case, the remaining points are sampled on the last
    // interval, i.e. the background.
    while (j < num_bins) {
        // no need to resample, just record fixed positions
        scalar_t offset = 10000.f; 
        scalar_t t = offset + ends[steps-1];
        resample_ts[j] = t;
        resample_offsets[j] = offset;
        resample_indices[j] = steps - 1 + base;
        // going further to next resample
        cdf_u += cdf_step_size;
        j += 1;
        // Increasing the count of the interval
        // Note that we do not need to use atomicAdd here, since we parallelize
        // over rays.
        resample_bg_counts[i] += 1;
    }
    if (j != num_bins)
    {
        printf("Error: %d %d %f\n", j, num_bins, weights_sum);
    }
    return;
}

std::vector<torch::Tensor> ray_resampling(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    torch::Tensor sdfs,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(weights);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = weights.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps;
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();
    torch::Tensor resample_ts = torch::empty({total_steps, 1}, starts.options());
    torch::Tensor resample_offsets = torch::empty({total_steps, 1}, starts.options());
    torch::Tensor resample_indices = torch::empty({total_steps}, starts.options().dtype(torch::kInt64));

    int total_samples = num_steps.sum().item<int>();
    torch::Tensor surface_idx = -torch::ones({n_rays}, starts.options().dtype(torch::kInt64));
    torch::Tensor resample_fg_counts = torch::zeros({total_samples}, starts.options().dtype(torch::kInt32));
    torch::Tensor resample_bg_counts = torch::zeros({n_rays}, starts.options().dtype(torch::kInt32));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(),
        "ray_resampling",
        ([&]
         { cdf_resampling_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               weights.data_ptr<scalar_t>(),
               sdfs.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_ts.data_ptr<scalar_t>(),
               resample_offsets.data_ptr<scalar_t>(),
               surface_idx.data_ptr<int64_t>(),
               resample_indices.data_ptr<int64_t>(),
               resample_fg_counts.data_ptr<int32_t>(),
               resample_bg_counts.data_ptr<int32_t>()); }));

    return {resample_packed_info, resample_ts, resample_offsets, resample_indices, resample_fg_counts, resample_bg_counts, surface_idx};
}

template <typename scalar_t>
__global__ void cdf_resampling_merge_kernel(
    const uint32_t n_rays,
    const int *packed_info,  // input ray & point indices (edges).
    const scalar_t *vals,    // input edge values
    const bool *is_left,    // input edge values
    const bool *is_right,    // input edge values
    const scalar_t *weights, // transmittance weights
    const int *resample_packed_info,
    scalar_t *resample_vals,
    scalar_t *resample_dists,
    bool *resample_is_left,
    bool *resample_is_right,
    bool *is_resample,
    bool *is_fg_sample)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1] - steps;
    if (steps == 0)
        return;

    vals += base;
    is_left += base;
    is_right += base;
    weights += base;
    is_fg_sample += resample_base;

    resample_vals += resample_base;
    resample_dists += resample_base;
    resample_is_left += resample_base;
    resample_is_right += resample_base;
    is_resample += resample_base;

    // Do not normalize weights, instead, add a new interval at the end
    // with weight = 1.0 - weights_sum
    scalar_t weights_sum = 0.0f;
    for (int j = 0; j < steps - 1; j++)
        weights_sum += (is_left[j] && is_right[j+1]) ? weights[j] : scalar_t(0);
    weights_sum += fmaxf(1.0f - weights_sum, 0.0f);

    int num_bins = resample_steps;
    scalar_t cdf_step_size = (1.0f - 1.0 / num_bins) / (resample_steps - 1);

    int idx = 0, j = 0;
    scalar_t start = 0.0f, end = 0.0f;
    scalar_t cdf_prev = 0.0f, cdf_next = weights[idx] / weights_sum;
    scalar_t cdf_u = 1.0 / (2 * num_bins);
    start = vals[0];
    end = vals[1];
    resample_vals[0] = start;
    is_fg_sample[0] = true;
    resample_is_left[0] = true;
    while (j < num_bins && idx < steps - 1)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (end - start) / (cdf_next - cdf_prev);
            scalar_t offset = (cdf_u - cdf_prev) * scaling; 
            scalar_t t = offset + start;
            // going further to next resample
            cdf_u += cdf_step_size;
            resample_dists[j + idx] = t - resample_vals[j + idx];
            j += 1;
            resample_vals[j + idx] = t;
            is_fg_sample[j + idx] = true;
            is_resample[j + idx] = true;
            resample_is_left[j + idx] = true;
            resample_is_right[j + idx] = true;
        } else {
            // going to next interval
            resample_dists[j + idx] = end - resample_vals[j + idx];
            idx += 1;
            resample_vals[j + idx] = end;
            is_fg_sample[j + idx] = true;
            resample_is_right[j + idx] = is_right[idx];
            if (idx >= steps - 1)
                break;
            start = vals[idx];
            end = vals[idx + 1];
            if (is_left[idx] && is_right[idx + 1]) {
                cdf_prev = cdf_next;
                cdf_next += weights[idx] / weights_sum;
                resample_is_left[j + idx] = true;
            }
        }
    }
    // If we are out of the loop with idx < steps - 1, it means we've sampled enough
    // points along the ray while there are still remaining input intervals not
    // traversed. We simply copy the remaining intervals to the end of the
    // output.
    while (idx < steps - 1) {
        // going to next interval
        resample_dists[j + idx] = end - resample_vals[j + idx];
        idx += 1;
        resample_vals[j + idx] = end;
        is_fg_sample[j + idx] = true;
        resample_is_right[j + idx] = is_right[idx];
        if (idx >= steps - 1)
            break;
        start = vals[idx];
        end = vals[idx + 1];
        if (is_left[idx] && is_right[idx + 1]) {
            resample_is_left[j + idx] = true;
        }
    }
    if (idx != steps - 1)
    {
        printf("Error: %d %d %f\n", j, num_bins, weights_sum);
    }
    return;
}

std::vector<torch::Tensor> ray_resampling_merge(
    torch::Tensor packed_info,
    torch::Tensor vals,
    torch::Tensor is_left,
    torch::Tensor is_right,
    torch::Tensor weights,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(vals);
    CHECK_INPUT(is_left);
    CHECK_INPUT(is_right);
    CHECK_INPUT(weights);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(vals.ndimension() == 1);
    TORCH_CHECK(is_left.ndimension() == 1);
    TORCH_CHECK(is_right.ndimension() == 1);
    TORCH_CHECK(weights.ndimension() == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = weights.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps + num_steps;
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();

    torch::Tensor resample_vals = torch::zeros({total_steps}, vals.options());
    torch::Tensor resample_dists = torch::zeros({total_steps}, vals.options());
    torch::Tensor resample_is_left = torch::zeros({total_steps}, is_left.options());
    torch::Tensor resample_is_right = torch::zeros({total_steps}, is_right.options());
    torch::Tensor is_fg_sample = torch::zeros({total_steps}, is_left.options());
    torch::Tensor is_resample = torch::zeros({total_steps}, is_left.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(),
        "ray_resampling_merge",
        ([&]
         { cdf_resampling_merge_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               vals.data_ptr<scalar_t>(),
               is_left.data_ptr<bool>(),
               is_right.data_ptr<bool>(),
               weights.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_vals.data_ptr<scalar_t>(),
               resample_dists.data_ptr<scalar_t>(),
               resample_is_left.data_ptr<bool>(),
               resample_is_right.data_ptr<bool>(),
               is_resample.data_ptr<bool>(),
               is_fg_sample.data_ptr<bool>()); }));

    return {resample_packed_info, resample_vals, resample_dists, resample_is_left, resample_is_right, is_resample, is_fg_sample};
}

template <typename scalar_t>
__global__ void cdf_resampling_fine_kernel(
    const uint32_t n_rays,
    const int *packed_info,  // input ray & point indices.
    const scalar_t *starts,  // input start t
    const scalar_t *ends,    // input end t
    const scalar_t *weights, // transmittance weights
    const int *resample_packed_info,
    scalar_t *resample_starts,
    scalar_t *resample_ends,
    bool *is_fg_sample)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    weights += base;
    resample_starts += resample_base;
    resample_ends += resample_base;
    is_fg_sample += resample_base;

    // Do not normalize weights, instead, add a new interval at the end
    // with weight = 1.0 - weights_sum
    scalar_t weights_sum = 0.0f;
    for (int j = 0; j < steps; j++)
        weights_sum += weights[j];
    weights_sum += fmaxf(1.0f - weights_sum, 0.0f);

    int num_bins = resample_steps + 1;
    scalar_t cdf_step_size = (1.0f - 1.0 / num_bins) / resample_steps;

    int idx = 0, j = 0;
    scalar_t cdf_prev = 0.0f, cdf_next = weights[idx] / weights_sum;
    scalar_t cdf_u = 1.0 / (2 * num_bins);
    while (j < num_bins && idx < steps)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
            scalar_t t = (cdf_u - cdf_prev) * scaling + starts[idx];
            if (j < num_bins - 1)
                resample_starts[j] = t;
            if (j > 0) {
                resample_ends[j - 1] = t;
                is_fg_sample[j - 1] = true;
            }
            // going further to next resample
            cdf_u += cdf_step_size;
            j += 1;
        }
        else
        {
            // going to next interval
            idx += 1;
            if (idx >= steps)
                break;
            cdf_prev = cdf_next;
            cdf_next += weights[idx] / weights_sum;
        }
    }
    if (j != num_bins && idx != steps)
    {
        printf("Error: %d %d %f\n", j, num_bins, weights_sum);
    }
    return;
}

std::vector<torch::Tensor> ray_resampling_fine(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(weights);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = weights.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps;
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();
    torch::Tensor resample_starts = torch::zeros({total_steps, 1}, starts.options());
    torch::Tensor resample_ends = torch::zeros({total_steps, 1}, ends.options());
    torch::Tensor is_fg_sample = torch::zeros({total_steps}, torch::dtype(torch::kBool).device(starts.device()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(),
        "ray_resampling",
        ([&]
         { cdf_resampling_fine_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               weights.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_starts.data_ptr<scalar_t>(),
               resample_ends.data_ptr<scalar_t>(),
               is_fg_sample.data_ptr<bool>()); }));

    return {resample_packed_info, resample_starts, resample_ends, is_fg_sample};
}

template <typename scalar_t>
__global__ void cdf_resampling_sdf_fine_kernel(
    const uint32_t n_rays,
    const int *packed_info,  // input ray & point indices.
    const scalar_t *starts,  // input start t
    const scalar_t *ends,    // input end t
    const scalar_t *alphas, // alpha values
    const scalar_t *sdfs,    // input sdf values
    const int *resample_packed_info,
    scalar_t *resample_starts,
    scalar_t *resample_ends,
    bool *is_fg_sample)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    alphas += base;
    sdfs += base;
    resample_starts += resample_base;
    resample_ends += resample_base;
    is_fg_sample += resample_base;

    // Search for zero-crossing point
    int idx = 0;
    scalar_t sdf_prev = sdfs[0];
    // if (sdf_prev < 0) {
    //     printf("Resample fine error: %d %f\n", i, sdf_prev);
    //     // return;
    // }
    bool found_surface = false;
    while (idx < steps) {
        // going to next interval
        idx += 1;
        if (idx >= steps)
            break;
        if (sdf_prev >= 0 && sdfs[idx] < 0 && !found_surface)
        {
            // record surface interval
            idx -= 1;
            found_surface = true;
            break;
        }
        sdf_prev = sdfs[idx];
    }

    if (!found_surface)
        return;

    // Starting from the surface interval, do the standard importance sampling
    int num_bins = resample_steps + 1;
    scalar_t cdf_step_size = (1.0f - 1.0 / num_bins) / resample_steps;

    int j = 0;
    scalar_t trans = 1.0f;
    scalar_t weight = alphas[idx];
    trans *= (1.0f - alphas[idx]);
    scalar_t cdf_prev = 0.0f, cdf_next = weight;
    scalar_t cdf_u = 1.0 / (2 * num_bins);
    while (j < num_bins && idx < steps)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
            scalar_t t = (cdf_u - cdf_prev) * scaling + starts[idx];
            if (j < num_bins - 1)
                resample_starts[j] = t;
            if (j > 0) {
                resample_ends[j - 1] = t;
                is_fg_sample[j - 1] = true;
            }
            // going further to next resample
            cdf_u += cdf_step_size;
            j += 1;
        }
        else
        {
            // going to next interval
            idx += 1;
            if (idx >= steps)
                break;
            weight = trans * alphas[idx];
            trans *= (1.0f - alphas[idx]);
            cdf_prev = cdf_next;
            cdf_next += weight;
        }
    }
    if (j != num_bins && idx != steps)
    {
        printf("Error: %d %d\n", j, num_bins);
    }
    return;
}

std::vector<torch::Tensor> ray_resampling_sdf_fine(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor alphas,
    torch::Tensor sdfs,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(alphas);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(alphas.ndimension() == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = alphas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps;
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();
    torch::Tensor resample_starts = torch::zeros({total_steps, 1}, starts.options());
    torch::Tensor resample_ends = torch::zeros({total_steps, 1}, ends.options());
    torch::Tensor is_fg_sample = torch::zeros({total_steps}, torch::dtype(torch::kBool).device(starts.device()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        alphas.scalar_type(),
        "ray_resampling",
        ([&]
         { cdf_resampling_sdf_fine_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               alphas.data_ptr<scalar_t>(),
               sdfs.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_starts.data_ptr<scalar_t>(),
               resample_ends.data_ptr<scalar_t>(),
               is_fg_sample.data_ptr<bool>()); }));

    return {resample_packed_info, resample_starts, resample_ends, is_fg_sample};
}
