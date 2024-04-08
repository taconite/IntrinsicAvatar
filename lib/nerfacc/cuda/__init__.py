"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


ray_resampling = _make_lazy_cuda_func("ray_resampling")
ray_resampling_merge = _make_lazy_cuda_func("ray_resampling_merge")
ray_resampling_fine = _make_lazy_cuda_func("ray_resampling_fine")
ray_resampling_sdf_fine = _make_lazy_cuda_func("ray_resampling_sdf_fine")


unpack_data = _make_lazy_cuda_func("unpack_data")
unpack_info = _make_lazy_cuda_func("unpack_info")
unpack_info_to_mask = _make_lazy_cuda_func("unpack_info_to_mask")
