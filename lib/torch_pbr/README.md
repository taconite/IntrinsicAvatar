# Torch PBR

Torch PBR is a light-weight library for differentiable PBR written purely in Python/PyTorch.

For example usage, please check [IntrinsicAvatar](https://github.com/taconite/IntrinsicAvatar).

If you find our code useful, please cite:

```bibtex
@inproceedings{WangCVPR2024,
  title   = {IntrinsicAvatar: Physically Based Inverse Rendering of Dynamic Humans from Monocular Videos via Explicit Ray Tracing},
  author  = {Shaofei Wang and Bo\v{z}idar Anti\'{c} and Andreas Geiger and Siyu Tang},
  booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2024}
}
```

## TODO
- [ ] Implement importance sampling and PDF evaluation for `EnvironmentLightSG` (https://arxiv.org/abs/2303.16617)

## Acknowledgement
The basic utility functions in `utils/nvdiffrecmc_util.py` are borrowed from [nvdiffrecmc](https://github.com/NVlabs/nvdiffrecmc). We thank authors of the paper for their wonderful works which greatly facilitates the development of this project.
