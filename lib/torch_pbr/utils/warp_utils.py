import torch
import numpy as np
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Coordinate system related functions
# ----------------------------------------------------------------------------
def to_local(v, a, b, c):
    """
    :param v: vector in world coordinate
    :param a: local x axis
    :param b: local y axis
    :param c: local z axis
    :return: vector in local coordinate
    """
    assert (
        len(v.shape) == 2 and v.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    assert (
        len(a.shape) == 2 and a.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    assert (
        len(b.shape) == 2 and b.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    assert (
        len(c.shape) == 2 and c.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    return torch.stack(
        [
            torch.sum(v * a, dim=-1),
            torch.sum(v * b, dim=-1),
            torch.sum(v * c, dim=-1),
        ],
        dim=-1,
    )


def to_world(v, a, b, c):
    """
    :param v: vector in local coordinate
    :param a: local x axis
    :param b: local y axis
    :param c: local z axis
    :return: vector in world coordinate
    """
    assert (
        len(v.shape) == 2 and v.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    assert (
        len(a.shape) == 2 and a.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    assert (
        len(b.shape) == 2 and b.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    assert (
        len(c.shape) == 2 and c.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"
    return F.normalize(v[:, 0:1] * a + v[:, 1:2] * b + v[:, 2:3] * c, dim=-1)


def coordinate_system(a):
    """Given normal direction n, generate a coordinate system (t, b, n)
    :param a: normal vectors
    :return: b, c - tangent vectors and bi-tangent vectors
    """
    assert (
        len(a.shape) == 2 and a.shape[1] == 3
    ), "Input tensor should have a shape of Nx3"

    condition = torch.abs(a[:, 0]) > torch.abs(a[:, 1])
    c = torch.tensor([[0, 0, 1]], dtype=a.dtype, device=a.device).repeat(a.shape[0], 1)

    # inv_len_1 = 1.0 / torch.sqrt(
    #     a[condition, 0] * a[condition, 0] + a[condition, 2] * a[condition, 2]
    # )
    # assert (len(a) > 0)
    if condition.sum() > 0:
        inv_len_1 = torch.reciprocal(torch.linalg.norm(a[condition, :][:, [0, 2]], dim=-1))
        c1 = torch.zeros_like(a[condition])
        c1[:, 0] = a[condition, 2] * inv_len_1
        c1[:, 2] = -a[condition, 0] * inv_len_1
        c.masked_scatter_(condition.unsqueeze(-1), c1)

    # inv_len_2 = 1.0 / torch.sqrt(
    #     a[~condition, 1] * a[~condition, 1] + a[~condition, 2] * a[~condition, 2]
    # )
    if (~condition).sum() > 0:
        inv_len_2 = torch.reciprocal(torch.linalg.norm(a[~condition, 1:], dim=-1))
        c2 = torch.zeros_like(a[~condition])
        c2[:, 1] = a[~condition, 2] * inv_len_2
        c2[:, 2] = -a[~condition, 1] * inv_len_2
        c.masked_scatter_(~condition.unsqueeze(-1), c2)

    # c = torch.zeros_like(a)
    # c[condition] = c1
    # c[~condition] = c2

    b = torch.cross(c, a)

    return b, c


# ----------------------------------------------------------------------------
# Basic sampling functions
# ----------------------------------------------------------------------------
@torch.no_grad()
def gen_stratified_samples(batch_size, n_rows, n_cols, device, is_training=True):
    """Generate stratified samples in [0, 1)^2
    :param n_rows: number of rows
    :param n_cols: number of columns
    :param device: device to store the samples
    :return: stratified samples
    """
    # Size of each stratum
    delta_x = 1.0 / n_cols
    delta_y = 1.0 / n_rows

    # Create an array of stratum indices
    i_indices = torch.arange(n_rows, device=device).view(1, -1, 1).float()
    j_indices = torch.arange(n_cols, device=device).view(1, 1, -1).float()

    # Random offsets within each stratum
    if is_training:
        random_offsets_x = torch.rand(batch_size, n_rows, n_cols, device=device) * delta_x
        random_offsets_y = torch.rand(batch_size, n_rows, n_cols, device=device) * delta_y
    else:
        random_offsets_x = torch.ones(batch_size, n_rows, n_cols, device=device) * 0.5 * delta_x
        random_offsets_y = torch.ones(batch_size, n_rows, n_cols, device=device) * 0.5 * delta_y

    # Compute x and y coordinates
    x_coords = (j_indices * delta_x + random_offsets_x).flatten()
    y_coords = (i_indices * delta_y + random_offsets_y).flatten()

    return torch.stack([x_coords, y_coords], dim=-1)


@torch.no_grad()
def sample_uniform_disk_concentric(sample):
    """Sample points on a unit disk
    :param sample: random samples in [0, 1)^2
    :return: random samples on a unit disk
    """
    # Concentric warping - PBR textbook Sec. 13.6.2
    # Point2f offset;
    # offset.x() = 2.f * sample.x() - 1.f;
    # offset.y() = 2.f * sample.y() - 1.f;
    # if (offset.x() == 0.f && offset.y() == 0.f)
    #     return Point2f(0.f, 0.f);
    offset = 2.0 * sample - 1.0

    # float theta, r;
    # if ( abs(offset.x()) > abs(offset.y()) ) {
    #     r = offset.x();
    #     theta = M_PI / 4.f * (offset.y() / offset.x());
    # } else {
    #     r = offset.y();
    #     theta = M_PI / 2.f -  M_PI / 4.f * (offset.x() / offset.y());
    # }
    abs_x = torch.abs(offset[:, 0])
    abs_y = torch.abs(offset[:, 1])
    r = torch.where(abs_x > abs_y, offset[:, 0], offset[:, 1])
    theta = torch.where(
        abs_x > abs_y,
        np.pi / 4.0 * (offset[:, 1] / offset[:, 0]),
        np.pi / 2.0 - np.pi / 4.0 * (offset[:, 0] / offset[:, 1]),
    )

    # return r * Point2f(cos(theta), sin(theta));
    return r.unsqueeze(-1) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


@torch.no_grad()
def sample_uniform_cylinder(sample):
    """Sample points uniformly in a unit cylinder
    :param sample: 2D samples for the cylinder sampling
    :return: sampled outgoing directions
    """
    z = sample[:, 0] * 2.0 - 1.0
    phi = 2.0 * np.pi * sample[:, 1]
    x = torch.cos(phi)
    y = torch.sin(phi)

    return torch.stack([x, y, z], dim=-1)


@torch.no_grad()
def sample_uniform_sphere(sample):
    """Sample points uniformly on a unit sphere
    :param sample: 2D samples for the sphere sampling
    :return: sampled outgoing directions
    """
    wo = sample_uniform_cylinder(sample)
    r = torch.sqrt(torch.clamp(1.0 - wo[:, 2]**2, min=0.0))
    wo[:, 0] *= r
    wo[:, 1] *= r

    return wo


def eval_uniform_sphere(wo):
    """Compute the PDF of the uniform sphere
    :param v: sampled outgoing directions
    :return: PDF of the uniform sphere
    """
    return torch.ones_like(wo[:, 0]) / (4.0 * np.pi)


@torch.no_grad()
def sample_uniform_hemisphere(sample, n):
    """Sample points uniformly on a hemisphere definede by normal n
    :param sample: 2D samples for the hemisphere sampling
    :param n: normal directions
    :return: sampled outgoing directions
    """
    t, b = coordinate_system(n)
    # Vector3f v = squareToUniformCylinder(sample);
    wo = sample_uniform_cylinder(sample)

    # v.z() = sample.x();
    wo[:, 2] = sample[:, 0]

    # float r = sqrt(std::max(0.f, 1.f - v.z() * v.z()));
    r = torch.sqrt(torch.clamp(1.0 - wo[:, 2] ** 2, min=0.0))

    # v.x() *= r;
    # v.y() *= r;
    wo[:, 0] *= r
    wo[:, 1] *= r

    wo = to_world(wo, t, b, n)

    return wo


def eval_uniform_hemisphere(wo, n):
    """Compute the PDF of the uniform hemisphere
    :param v: sampled outgoing directions
    :return: PDF of the uniform hemisphere
    """
    return torch.where((wo * n).sum(-1) >= 0.0, 0.5 / np.pi, 0.0)


# ----------------------------------------------------------------------------
# Microflake related functions
# ----------------------------------------------------------------------------
def _sample_SGGX_VNDF(sample, n, wi, alpha):
    """Sample the Visible Normal Distribution Function (VNDF) of SGGX
    Heitz et al. "The SGGX Microflake Distribution", SIGGRAPH 2015
    :param sample: random samples in [0, 1)^2
    :param n: normal directions
    :param wi: incoming directions (microflake -> camera)
    :param alpha: roughness parameter
    :return: sampled microflake normal
    """
    # Generate sample (u, v, w)
    r = torch.sqrt(sample[:, 0])
    phi = 2 * np.pi * sample[:, 1]
    u = r * torch.cos(phi)
    v = r * torch.sin(phi)
    w = torch.sqrt(1.0 - u**2 - v**2)

    # Build orthonormal basis w.r.t. the incident light direction
    wk, wj = coordinate_system(wi)
    # SGGX matrix S is defined by (t, b, n) and alpha
    t, b = coordinate_system(n)  # compute local shading frame

    # Project S onto the basis (w_k, w_j, w_i), denote as S^{kji}
    roughness = alpha * alpha
    dot_wk_t = (wk * t).sum(-1)
    dot_wk_b = (wk * b).sum(-1)
    dot_wk_n = (wk * n).sum(-1)

    dot_wj_t = (wj * t).sum(-1)
    dot_wj_b = (wj * b).sum(-1)
    dot_wj_n = (wj * n).sum(-1)

    dot_wi_t = (wi * t).sum(-1)
    dot_wi_b = (wi * b).sum(-1)
    dot_wi_n = (wi * n).sum(-1)

    S_kk = roughness * (dot_wk_t * dot_wk_t + dot_wk_b * dot_wk_b) + dot_wk_n * dot_wk_n
    S_kj = roughness * (dot_wk_t * dot_wj_t + dot_wk_b * dot_wj_b) + dot_wk_n * dot_wj_n
    S_ki = roughness * (dot_wk_t * dot_wi_t + dot_wk_b * dot_wi_b) + dot_wk_n * dot_wi_n

    S_jj = roughness * (dot_wj_t * dot_wj_t + dot_wj_b * dot_wj_b) + dot_wj_n * dot_wj_n
    S_ji = roughness * (dot_wj_t * dot_wi_t + dot_wj_b * dot_wi_b) + dot_wj_n * dot_wi_n

    S_ii = roughness * (dot_wi_t * dot_wi_t + dot_wi_b * dot_wi_b) + dot_wi_n * dot_wi_n

    # sample normal in the new orthonrmal basis defined by the incident light direction
    sqrtDetSkji = torch.sqrt(
        torch.abs(
            S_kk * S_jj * S_ii
            - S_kj * S_kj * S_ii
            - S_ki * S_ki * S_jj
            - S_ji * S_ji * S_kk
            + 2.0 * S_kj * S_ki * S_ji
        )
    )
    assert (S_ii >= 0.0).all()
    eps = 1e-6
    S_ii_sqrt = torch.sqrt(S_ii)
    inv_sqrtS_ii = torch.reciprocal(S_ii_sqrt + eps)
    assert (S_jj * S_ii - S_ji * S_ji >= 0.0).all()
    tmp = torch.sqrt(S_jj * S_ii - S_ji * S_ji)
    inv_tmp = torch.reciprocal(tmp + eps)
    # Vector3f Mk(sqrtDetSkji/tmp, 0.f, 0.f);
    Mk = torch.stack(
        [sqrtDetSkji * inv_tmp, torch.zeros_like(tmp), torch.zeros_like(tmp)], dim=-1
    )
    # Vector3f Mj(-inv_sqrtS_ii*( S_ki*S_ji - S_kj*S_ii ) / tmp, inv_sqrtS_ii*tmp, 0);
    Mj = torch.stack(
        [
            -inv_sqrtS_ii * (S_ki * S_ji - S_kj * S_ii) * inv_tmp,
            inv_sqrtS_ii * tmp,
            torch.zeros_like(tmp),
        ],
        dim=-1,
    )
    # Vector3f Mi(inv_sqrtS_ii*S_ki, inv_sqrtS_ii*S_ji, inv_sqrtS_ii*S_ii);
    Mi = torch.stack(
        [inv_sqrtS_ii * S_ki, inv_sqrtS_ii * S_ji, inv_sqrtS_ii * S_ii], dim=-1
    )
    # Vector3f wm_kji = (u*Mk + v*Mj + w*Mi).normalized();
    wm_kji = u[:, None] * Mk + v[:, None] * Mj + w[:, None] * Mi
    wm_kji = F.normalize(wm_kji, dim=-1)

    # rotate the sampled normal back into the local shading frame
    # return (wm_kji.x() * wk + wm_kji.y() * wj + wm_kji.z() * wi).normalized();
    ret = wm_kji[:, 0, None] * wk + wm_kji[:, 1, None] * wj + wm_kji[:, 2, None] * wi
    return F.normalize(ret + 1e-6, dim=-1)


def sample_SGGX_VNDF(sample, n, wi, alpha):
    with torch.no_grad():
        return _sample_SGGX_VNDF(sample, n, wi, alpha)


def sample_SGGX_VNDF_with_grad(sample, n, wi, alpha):
    with torch.enable_grad():
        return _sample_SGGX_VNDF(sample, n, wi, alpha)


def eval_SGGX_VNDF(wo, n, wi, alpha):
    """Evaluate the Visible Normal Distribution Function (VNDF) of the SGGX distribution
    :param wo: outgoing directions (microflake -> light)
    :param n: normal directions
    :param wi: incoming directions (microflake -> camera)
    :param alpha: roughness parameter
    :return: probability density of the visible normal distribution function
    """
    # TODO: this a simplified version for isotropic surface-like materials - should support other materials later
    roughness = alpha**2
    inv_roughness = torch.reciprocal(roughness)

    # determinant of a matrix is the product of its eigenvalues. Here, the eigenvalues are [alpha^2, alpha&2, 1], so the square root of  the determinant is the alpha^2.
    sqrtDet = roughness

    # SGGX matrix S is defined by (t, b, n) and alpha
    t, b = coordinate_system(n)  # compute local shading frame

    # Compute the intermediate quantaties
    dot_wo_t = (wo * t).sum(-1)
    dot_wo_b = (wo * b).sum(-1)
    dot_wo_n = (wo * n).sum(-1)

    dot_wi_t = (wi * t).sum(-1)
    dot_wi_b = (wi * b).sum(-1)
    dot_wi_n = (wi * n).sum(-1)

    wiSwi = torch.clamp(
        roughness * (dot_wi_t**2 + dot_wi_b**2) + dot_wi_n**2,
        min=0.0,
    )
    mSinvm = torch.clamp(
        inv_roughness * (dot_wo_t**2 + dot_wo_b**2) + dot_wo_n**2,
        min=0.0,
    )

    sigma = torch.sqrt(wiSwi)

    dot_wi_wo = torch.clamp((wi * wo).sum(-1), min=0.0, max=1.0)

    eps = 1e-6
    # PDF of normal distribution function (NDF)
    # D_wm = torch.zeros_like(mSinvm)
    # mask = mSinvm > 0.0
    # D_wm[mask] = (
    #     1.0
    #     / np.pi
    #     * torch.reciprocal(sqrtDet[mask])
    #     * torch.reciprocal(mSinvm[mask] ** 2)
    # )
    # sqrtDet > eps is ensured by material predictor
    mSinvm2 = mSinvm**2
    D_wm = torch.where(
        mSinvm2 > eps,
        1.0 / np.pi * torch.reciprocal(sqrtDet) * torch.reciprocal(mSinvm2 + eps),
        torch.zeros_like(mSinvm2),
    )

    # PDF of visible normal distribution function (VNDF)
    # D_wi_wm = torch.zeros_like(sigma)
    # mask = sigma > 0.0
    # D_wi_wm[mask] = dot_wi_wo[mask] * D_wm[mask] / sigma[mask]
    # # return D_wi_wm, D_wm
    D_wi_wm = torch.where(
        sigma > eps, dot_wi_wo * D_wm / (sigma + eps), torch.zeros_like(sigma)
    )
    return D_wi_wm


def eval_SGGX_NDF(wo, n, alpha):
    """Evaluate the Normal Distribution Function (NDF) of the SGGX distribution
    :param wo: outgoing directions (microflake -> light)
    :param n: normal direction
    :param alpha: roughness parameter
    :return: probability density of the normal distribution function
    """
    # TODO: this is for isotropic surface-like materials - should support other materials later
    roughness = alpha**2
    inv_roughness = torch.reciprocal(roughness)

    # determinant of a matrix is the product of its eigenvalues. Here, the eigenvalues are [alpha^2, alpha&2, 1], so the square root of  the determinant is the alpha^2.
    sqrtDet = roughness

    # SGGX matrix S is defined by (t, b, n) and alpha
    t, b = coordinate_system(n)  # compute local shading frame

    # Compute the intermediate quantaties
    dot_wo_t = (wo * t).sum(-1)
    dot_wo_b = (wo * b).sum(-1)
    dot_wo_n = (wo * n).sum(-1)

    eps = 1e-6
    mSinvm = inv_roughness * (dot_wo_t**2 + dot_wo_b**2) + dot_wo_n**2
    # D_wm = torch.zeros_like(mSinvm)
    # mask = mSinvm > 0.0
    # D_wm[mask] = (
    #     1.0
    #     / np.pi
    #     * torch.reciprocal(sqrtDet[mask])
    #     * torch.reciprocal(mSinvm[mask] ** 2)
    # )
    mSinvm2 = mSinvm**2
    D_wm = torch.where(
        mSinvm2 > eps,
        1.0 / np.pi * torch.reciprocal(sqrtDet) * torch.reciprocal(mSinvm2 + eps),
        torch.zeros_like(mSinvm2),
    )
    return D_wm


@torch.no_grad()
def sample_diffuse_SGGX(sample1, sample2, n, wi, alpha):
    """Sample outgoing directions using the diffuse SGGX distribution
    :param sample1: 2D samples for the VNDF sampling
    :param sample2: 2D samples for the disk sampling
    :param n: normal directions
    :param wi: incoming directions (microflake -> camera)
    :param alpha: square root of roughness
    :return: sampled outgoing directions
    """
    # Sample VNDF
    # Vector3f wm = squareToVND(sample1, n, wi, alpha);
    wm = sample_SGGX_VNDF(sample1, n, wi, alpha)

    # Sample diffuse reflectance
    # Vector3f w1, w2;
    # coordinateSystem(wm, w1, w2);
    w1, w2 = coordinate_system(wm)

    # Point2f diskPoint = squareToUniformDiskConcentric(sample2);
    diskPoint = sample_uniform_disk_concentric(sample2)

    # float x = diskPoint.x();
    # float y = diskPoint.y();
    # float z = sqrt(1.f - x * x - y * y);
    # // cout << sample2.toString() << endl;
    # return x * w1 + y * w2 + z * wm;
    x = diskPoint[:, 0, None]
    y = diskPoint[:, 1, None]
    z = torch.sqrt(1.0 - x**2 - y**2)
    return F.normalize(x * w1 + y * w2 + z * wm, dim=-1)


def eval_diffuse_SGGX(sample, wi, n, wo, alpha):
    """Compute the PDF of the diffuse SGGX distribution
    :param sample: 2D samples for the VNDF sampling
    :param wi: incoming directions (microflake -> camera)
    :param n: normal directions
    :param wo: sampled outgoing directions (microflake -> light)
    :param alpha: square root of roughness
    :return: PDF of the diffuse SGGX distribution
    """
    # Sample VNDF
    # Note that this function call is differtiable wrt. n, wi, and alpha - it is essentially a
    # reparameterization trick that reparameterizes wm as a function of n, wi, and alpha
    wm = sample_SGGX_VNDF_with_grad(sample, n, wi, alpha)

    # NOTE: this PDF is stochastic, and we estimate it using Monte Carlo integration with only one sample
    # Thus, the variance of the PDF is very high, and not suitable for Chi square test
    return 1.0 / np.pi * F.relu((wo * wm).sum(-1))


# class eval_diffuse_SGGX(Function):
#     @staticmethod
#     def forward(ctx, sample, wo, n, wi, alpha):
#         """Compute the PDF of the diffuse SGGX distribution
#         :param sample: 2D samples for the VNDF sampling
#         :param wo: sampled outgoing directions
#         :param n: normal directions
#         :param wi: incoming directions
#         :param alpha: square root of roughness
#         :return: PDF of the diffuse SGGX distribution
#         """
#         # Sample VNDF
#         wm = sample_SGGX_VNDF(sample, n, wi, alpha)
#
#         # NOTE: this PDF is stochastic, and we estimate it using Monte Carlo integration with only one sample
#         # Thus, the variance of the PDF is very high, and not suitable for Chi square test
#         pdf = 1.0 / np.pi * torch.clamp((wo * wm).sum(-1), min=0.0)
#         ctx.save_for_backward(wo, n, wi, alpha, pdf)
#
#         return pdf
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # NOTE: since this PDF is stochastic, we use the detached sampling strategy to estimate its gradients. See Zeltner et al, (2021) for more details.
#         wo, n, wi, alpha, pdf = ctx.saved_tensors
#         print("start!")
#
#         alpha.requires_grad_(True)
#         n.requires_grad_(True)
#         VNDF, _ = eval_SGGX_VNDF(wo, n, wi, alpha)
#         grad_n, grad_alpha = torch.autograd.grad(
#             VNDF, (n, alpha), grad_output, retain_graph=False
#         )
#         grad_n = pdf * grad_n * torch.reciprocal(VNDF)
#         grad_alpha = pdf * grad_alpha * torch.reciprocal(VNDF)
#
#         return None, None, grad_n, None, grad_alpha


@torch.no_grad()
def sample_specular_SGGX(sample, n, wi, alpha):
    """Sample outgoing directions using the specular SGGX distribution
    :param sample: 2D samples for the specular SGGX sampling
    :param n: normal directions
    :param wi: incoming directions (microflake -> camera)
    :param alpha: square root of roughness
    :return: sampled outgoing directions
    """
    # Sample visible normals
    wm = sample_SGGX_VNDF(sample, n, wi, alpha)
    # Get specular reflection
    wo = -wi + 2.0 * wm * (wm * wi).sum(-1, keepdim=True)
    return F.normalize(wo, dim=-1)


def eval_specular_SGGX(wi, n, wo, alpha):
    """Compute the PDF of the specular SGGX distribution
    :param wi: incoming directions (microflake -> camera)
    :param n: normal directions
    :param wo: outgoing directions (microflake -> light)
    :param alpha: square root of roughness
    :return: PDF of the specular SGGX distribution
    """
    roughness = alpha**2

    wh = wo + wi
    wh = F.normalize(wh, dim=-1)
    # SGGX matrix S is defined by (t, b, n) and alpha
    t, b = coordinate_system(n)

    dot_wi_t = (wi * t).sum(-1)
    dot_wi_b = (wi * b).sum(-1)
    dot_wi_n = (wi * n).sum(-1)

    wiSwi = roughness * (dot_wi_t**2 + dot_wi_b**2) + dot_wi_n**2
    sigma = torch.sqrt(torch.clamp(wiSwi, min=0.0))

    eps = 1e-6
    # ret = torch.zeros_like(sigma)
    # mask = sigma > 0.0
    # ret[mask] = 0.25 * eval_SGGX_NDF(wh[mask], n[mask], alpha[mask]) / sigma[mask]
    ret = torch.where(
        sigma > eps, 0.25 * eval_SGGX_NDF(wh, n, alpha) / (sigma + eps), torch.zeros_like(sigma)
    )

    return ret


# ------------------------------------------------------------------------------
# Lambertian surface related functions
# ------------------------------------------------------------------------------
@torch.no_grad()
def sample_Lambertian_surface(sample, n):
    """Sample outgoing directions with the Lambertian surface assumption
    :param sample: 2D samples for the VNDF sampling
    :param n: normal directions
    :return: sampled outgoing directions
    """
    # Sample diffuse reflectance
    t, b = coordinate_system(n)

    disk_point = sample_uniform_disk_concentric(sample)

    x = disk_point[:, 0]
    y = disk_point[:, 1]
    z = torch.sqrt((1.0 - x**2 - y**2).clamp(min=0.0))
    wo = torch.stack([x, y, z], dim=-1)
    wo = to_world(wo, t, b, n)

    return wo


def eval_Lambertian_surface(wo, n):
    """Compute the PDF of the Lambertian surface
    :param wo: sampled outgoing directions
    :param n: normal directions
    :return: PDF of the Lambertian surface
    """
    return F.relu((n * wo).sum(-1)) / np.pi


# ------------------------------------------------------------------------------
# GGX related functions
# ------------------------------------------------------------------------------
@torch.no_grad()
def sample_GGX_VNDF(sample, wi, alpha_x, alpha_y):
    """Sample the Visible Normal Distribution Function (VNDF) of the GGX distribution
    Eric Heitz, Sampling the GGX Distribution of Visible Normals, Journal of Computer Graphics Techniques Vol. 7, No. 4, 2018
    :param sample: 2D samples for the VNDF sampling
    :param wi: incoming directions
    :param alpha_x: roughness in x direction
    :param alpha_y: roughness in y direction
    :return: sampled visible normals
    """
    # t, b = coordinate_system(n)  # compute local shading frame
    # wi = to_local(wi, t, b, n)  # transform wi to local shading frame
    # Sampling the GGX distribution of visible normals
    # From Eric Heitz, Sampling the GGX Distribution of Visible Normals, Journal of Computer Graphics Techniques Vol. 7, No. 4, 2018
    # Section 3.2: transforming the view direction to the hemisphere configuration
    vh = torch.stack([alpha_x * wi[:, 0], alpha_y * wi[:, 1], wi[:, 2]], dim=-1)
    vh = F.normalize(vh, dim=-1)
    # Section 4.1: orthonormal basis (with special case if cross product is zero)
    lensq = vh[:, 0] * vh[:, 0] + vh[:, 1] * vh[:, 1]
    eps = 1e-6
    T1 = torch.where(
        lensq.unsqueeze(-1) > eps,
        torch.stack(
            [
                -vh[:, 1] / torch.sqrt(lensq + eps),
                vh[:, 0] / torch.sqrt(lensq + eps),
                torch.zeros_like(vh[:, 0]),
            ],
            dim=-1,
        ),
        torch.stack(
            [
                torch.ones_like(vh[:, 0]),
                torch.zeros_like(vh[:, 0]),
                torch.zeros_like(vh[:, 0]),
            ],
            dim=-1,
        ),
    )
    T2 = vh.cross(T1)
    # Section 4.2: parameterization of the projected area
    r = torch.sqrt(sample[:, 0])
    phi = 2.0 * np.pi * sample[:, 1]
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)
    s = 0.5 * (1.0 + vh[:, 2])
    t2 = (1.0 - s) * torch.sqrt(torch.clamp(1.0 - t1 * t1, min=0.0)) + s * t2
    # Section 4.3: reprojection onto hemisphere
    nh = (
        t1[:, None] * T1
        + t2[:, None] * T2
        + torch.sqrt(torch.clamp(1.0 - t1 * t1 - t2 * t2, min=0.0))[:, None] * vh
    )
    # Section 3.4: transforming the normal back to the ellipsoid configuration
    v = torch.stack(
        [alpha_x * nh[:, 0], alpha_y * nh[:, 1], torch.clamp(nh[:, 2], min=0.0)], dim=-1
    )
    v = F.normalize(v, dim=-1)

    return v


def fresnel_schlick(F0, F90, cos_theta):
    """Evaluate Schlick's approximation of the Fresnel term, and approximating the power term with spherical Gaussian
    Brian, Karis, Real Shading in Unreal Engine 4, SIGGRAPH 2013
    :param F0: reflectance at normal incidence
    :param F90: reflectance at grazing angle
    :param cos_theta: cosine of the angle between the normal and the view direction
    :return: Schlick's approximation of the Fresnel term
    """
    # return F0 + (F90 - F0) * (1.0 - cos_theta) ** 5
    return F0 + (F90 - F0) * 2 ** ((-5.55473 * cos_theta - 6.98316) * cos_theta)


def smith_GGX_G1_aniso(v, alpha_x, alpha_y):
    """Evaluate anisotropic Smith's shadowing-masking function G1 (anisotropic)
    :param v: sampled outgoing directions in local shading frame
    :param alpha_x: roughness in x direction
    :param alpha_y: roughness in y direction
    :return: Smith's shadowing-masking function G1
    """
    cos_theta = v[:, 2]
    cos2_theta = cos_theta * cos_theta

    eps = 1e-6
    delta = torch.where(
        cos2_theta > eps,
        -0.5
        + 0.5
        * torch.sqrt(
            1
            + ((v[:, 0] * alpha_x) ** 2 + (v[:, 1] * alpha_y) ** 2) / (cos2_theta + eps)
        ),
        torch.zeros_like(cos2_theta),
    )

    return 1 / (1 + delta)


def smith_GGX_G1_shclick(v, k):
    """Evaluate Schlick's approximation to isotropic Smith's shadowing-masking function G1
    :param v: sampled outgoing directions in local shading frame
    :param k: remapped roughness
    :return: Smith's shadowing-masking function G1
    Reference: https://learnopengl.com/PBR/Theory
    """
    nom = v[:, 2]
    denom = nom * (1.0 - k) + k

    eps = 1e-6
    ret = torch.where(
        denom > eps,
        nom / (denom + eps),
        torch.zeros_like(nom),
    )

    return ret


def eval_GGX_NDF(wh, alpha_x, alpha_y):
    if (alpha_x == alpha_y).all():
        return eval_GGX_NDF_isotropic(wh, alpha_x)
    else:
        return eval_GGX_NDF_anisotropic(wh, alpha_x, alpha_y)


def eval_GGX_NDF_anisotropic(wh, alpha_x, alpha_y, eps=1e-6):
    """Evaluate the anisotropic Normal Distribution Function (NDF) of the GGX distribution
    :param wh: sampled microfacet normal in local shading frame
    :param alpha_x: roughness in x direction
    :param alpha_y: roughness in y direction
    :return: GGX NDF
    """
    cos_theta = wh[:, 2]
    sin_theta_cos_phi = wh[:, 0]
    sin_theta_sin_phi = wh[:, 1]
    denom = (
        sin_theta_cos_phi * sin_theta_cos_phi / alpha_x / alpha_x
        + sin_theta_sin_phi * sin_theta_sin_phi / alpha_y / alpha_y
        + cos_theta * cos_theta
    )

    # return torch.where(
    #     denom > eps,
    #     torch.reciprocal(np.pi * alpha_x * alpha_y * (denom + eps) ** 2),
    #     torch.zeros_like(denom),
    # )

    return torch.reciprocal(np.pi * alpha_x * alpha_y * (denom + eps) ** 2)


def eval_GGX_NDF_isotropic(wh, alpha, eps=1e-6):
    """Evaluate the isotropic Normal Distribution Function (NDF) of the GGX distribution
    :param wh: sampled microfacet normal in local shading frame
    :param alpha: roughness
    :return: GGX NDF
    Reference: https://learnopengl.com/PBR/Theory
    """
    cos_theta = wh[:, 2]
    cos2_theta = cos_theta ** 2
    alpha2 = alpha ** 2
    denom = np.pi * (cos2_theta * (alpha2 - 1) + 1) ** 2

    return alpha2 * torch.reciprocal(denom + eps)


def eval_GGX_VNDF(wh, wi, alpha_x, alpha_y, eps=1e-6):
    """Evaluate the Visible Normal Distribution Function (VNDF) of the GGX distribution
    :param wh: sampled microfacet normal in local shading frame
    :param wi: sampled incoming directions in local shading frame
    :param alpha_x: roughness in x direction
    :param alpha_y: roughness in y direction
    :return: GGX VNDF
    """
    # TODO: check the deriviation of k
    k = (alpha_x ** 2 + 2 * alpha_x + 1) / 8.0
    return torch.where(
        torch.logical_and(wh[:, 2] > eps, wi[:, 2] > eps),
        # smith_GGX_G1_aniso(wi, alpha_x, alpha_y)
        smith_GGX_G1_shclick(wi, k)
        * torch.clamp((wh * wi).sum(-1), min=0.0)
        * eval_GGX_NDF(wh, alpha_x, alpha_y)
        / (wi[:, 2] + eps),
        torch.zeros_like(wh[:, 2]),
    )


# ------------------------------------------------------------------------------
# Other sampling functions
# ------------------------------------------------------------------------------
@torch.no_grad()
def sample_specular_mirror(sample, n, wi, alpha):
    """Sample outgoing directions using the specular SGGX distribution
    :param sample: 2D samples for the specular SGGX sampling
    :param n: normal directions
    :param wi: incoming directions
    :param alpha: square root of roughness
    :return: sampled outgoing directions
    """
    # Get specular reflection
    wo = -wi + 2.0 * n * (n * wi).sum(-1, keepdim=True)
    return wo
