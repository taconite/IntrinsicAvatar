import math
import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn

from pytorch_lightning.utilities.rank_zero import rank_zero_debug

from utils.misc import config_to_primitive, get_rank
from models.utils import get_activation, get_ml_array, sph_harm_coeff
from systems.utils import update_module_step


class VanillaFrequency(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.n_input_dims = in_channels
        self.x_scale = config.get('x_scale', 1.)
        self.x_offset = config.get('x_offset', 0.)
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get('n_masking_step', 0)
        self.start_step = config.get('start_step', 0)
        self.update_step(None, None)  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        x = x * self.x_scale + self.x_offset
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                if self.training:
                    out += [func(freq*x) * mask]
                else:
                    out += [func(freq*x) * mask.clone()]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            curr_step = max(global_step - self.start_step, 0)
            self.mask = (
                1.0
                - torch.cos(
                    math.pi
                    * (
                        curr_step / self.n_masking_step * self.N_freqs
                        - torch.arange(0, self.N_freqs)
                    ).clamp(0, 1)
                )
            ) / 2.0
            rank_zero_debug(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config['otype'] = 'HashGrid'
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=torch.float32)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config['n_levels']
        self.n_features_per_level = config['n_features_per_level']
        self.start_level, self.start_step, self.update_steps = config['start_level'], config['start_step'], config['update_steps']
        self.current_level = self.start_level
        self.mask = torch.zeros(self.n_level * self.n_features_per_level, dtype=torch.float32, device=get_rank())

        self.full_band_step = config.get('full_band_step', 5000)
        self.update_mode = config.get('update_mode', 'non_smooth')

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step):
        if self.update_mode == 'smooth':
            t = max(global_step - self.start_step, 0.0)
            N = self.full_band_step - self.start_step
            alpha = self.n_level * t / N

            for lvl in range(self.n_level):
                w = (1.0 - np.cos(np.pi * min(max(alpha - lvl, 0.0), 1.0))) / 2.0
                self.mask[
                    ...,
                    lvl
                    * self.n_features_per_level : (lvl + 1)
                    * self.n_features_per_level,
                ] = w
        else:
            current_level = min(self.start_level + max(global_step - self.start_step, 0) // self.update_steps, self.n_level)
            if current_level > self.current_level:
                rank_zero_debug(f'Update current level to {current_level}')
            self.current_level = current_level
            self.mask[:self.current_level * self.n_features_per_level] = 1.


class IntegratedDirectionalEncoding(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.n_input_dims = in_channels

        degree = config.get('degree', 5)
        ml_array = get_ml_array(degree)
        l_max = 2**(degree - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        mat = np.zeros((l_max + 1, ml_array.shape[1]), dtype=np.float32)
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = sph_harm_coeff(l, m, k)

        self.register_buffer('ml_array', torch.tensor(ml_array, dtype=torch.float32))
        self.register_buffer('mat', torch.from_numpy(mat))

        self.n_output_dims = len(self.ml_array[-1]) * 2

    def forward(self, xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).
        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.
        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(self.mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat([(x + 1j * y)**m for m in self.ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.einsum('pi,ij->pj', vmz, self.mat) # TODO: we assume input xyz is n_pts x 3, which does not consider batch size

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * self.ml_array[1, :] * (self.ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.cat([ide.real, ide.imag], axis=-1)


class CompositeEncoding(nn.Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        if hasattr(self.encoding, 'n_input_dims') and hasattr(self.encoding, 'n_output_dims'):
            self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims
        else:
            self.n_output_dims = 3  # Identity mapping of xyz

    def forward(self, x, *args):
        return self.encoding(x, *args) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self):
        if hasattr(self.encoding, 'regularizations'):
            return self.encoding.regularizations()
        else:
            return {}


def get_encoding(n_input_dims, config):
    # input suppose to be range [0, 1]
    if config.otype == 'Identity':
        encoding = nn.Identity()
    elif config.otype == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == 'ProgressiveBandHashGrid':
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    elif config.otype == 'IDE':
        encoding = IntegratedDirectionalEncoding(n_input_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config), dtype=torch.float32)
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=config.get("xyz_scale", 2.0),
        xyz_offset=config.get("xyz_offset", -1.0),
    )
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])

    def forward(self, x, *args):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()

        dim_cond = config.get("dim_cond", 0)
        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        # self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers
        self.n_neurons = config.get("n_neurons", 256)
        self.n_hidden_layers = config.get("n_hidden_layers", 8)
        self.sphere_init, self.weight_norm = config.get(
            "sphere_init", False
        ), config.get("weight_norm", False)
        self.last_layer_init = config.get("last_layer_init", True)
        self.last_layer_zeros = config.get("last_layer_zeros", False)
        self.sphere_init_radius = config.get("sphere_init_radius", 0.5)
        self.weight_norm = config.get("weight_norm", False)

        self.skip_in = config.get("skip_in", [])
        self.cond_in = config.get("cond_in", [])

        dims = (
            [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]
        )

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in self.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.sphere_init:
                # Geometric initialization: initialize the MLP the represent a sphere
                if self.last_layer_init and l == self.num_layers - 2:
                    # Last layer initialization
                    torch.nn.init.normal_(
                        lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, -self.sphere_init_radius)
                elif l == 0:
                    # First layer initialization (for positional encoding input)
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif l in self.skip_in:
                    # Skip layer initialization
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(
                        lin.weight[:, -(dims[0] - 3) :], 0.0
                    )
                elif l in self.cond_in:
                    # Conditional input layer initialization
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
                else:
                    # Other layers
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
            elif self.last_layer_zeros:
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -1e-5, 1e-5)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if self.sphere_init:
            self.activation = nn.Softplus(beta=100)
        else:
            self.activation = nn.ReLU()

    def forward(self, x, cond, *args):
        cond = cond.expand(x.shape[0], -1)

        coords_embedded = x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x


class LipshitzMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']

        self.layers = torch.nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.layers.append(torch.nn.Linear(dim_in, self.n_neurons))
            dim_in = self.n_neurons

        self.layers.append(torch.nn.Linear(dim_in, dim_out))

        # we make each weight separately because we want to add the normalize to it
        self.weights_per_layer = torch.nn.ParameterList()
        self.biases_per_layer = torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append(self.layers[i].weight)
            self.biases_per_layer.append(self.layers[i].bias)

        self.lipshitz_bound_per_layer = torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w = torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            # we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(torch.ones((1)) * max_w * 2)
            self.lipshitz_bound_per_layer.append(c)

        # self.weights_initialized = (
        #     True  # so that apply_weight_init_fn doesnt initialize anything
        # )

        self.activation = torch.nn.ReLU(inplace=True)
        self.output_activation = get_activation(config['output_activation'])

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci / absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:, None]

    def lipshitz_bound_full(self):
        lipshitz_full = 1
        for i in range(len(self.layers)):
            lipshitz_full = lipshitz_full * torch.nn.functional.softplus(
                self.lipshitz_bound_per_layer[i]
            )

        return lipshitz_full

    def forward(self, x, *args):
        for i in range(len(self.layers)):
            weight = self.weights_per_layer[i]
            bias = self.biases_per_layer[i]

            weight = self.normalization(
                weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])
            )

            x = torch.nn.functional.linear(x, weight, bias)

            is_last_layer = i == (len(self.layers) - 1)

            if is_last_layer:
                x = self.output_activation(x)
            else:
                x = self.activation(x)

        return x

    def regularizations(self):
        return {"lipshitz_bound": self.lipshitz_bound_full().mean()}


def sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network):
    rank_zero_debug('Initialize tcnn MLP to approximately represent a sphere.')
    """
    from https://github.com/NVlabs/tiny-cuda-nn/issues/96
    It's the weight matrices of each layer laid out in row-major order and then concatenated.
    Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
    The padded input dimensions get a constant value of 1.0,
    whereas the padded output dimensions are simply ignored,
    so the weights pertaining to those can have any value.
    """
    padto = 16 if config.otype == 'FullyFusedMLP' else 8
    n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
    n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    data = list(network.parameters())[0].data
    assert data.shape[0] == (n_input_dims + n_output_dims) * config.n_neurons + (config.n_hidden_layers - 1) * config.n_neurons**2
    new_data = []
    # first layer
    weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
    torch.nn.init.constant_(weight[:, 3:], 0.0)
    torch.nn.init.normal_(weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
    new_data.append(weight.flatten())
    # hidden layers
    for i in range(config.n_hidden_layers - 1):
        weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
        torch.nn.init.normal_(weight, 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
        new_data.append(weight.flatten())
    # last layer
    weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
    torch.nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(config.n_neurons), std=0.0001)
    new_data.append(weight.flatten())
    new_data = torch.cat(new_data)
    data.copy_(new_data)


def get_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'VanillaCondMLP':
        network = VanillaCondMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'LipshitzMLP':
        network = LipshitzMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'Identity':
        network = torch.nn.Identity()
    else:
        with torch.cuda.device(get_rank()):
            network = tcnn.Network(n_input_dims, n_output_dims, config_to_primitive(config))
            if config.get('sphere_init', False):
                sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    return network


class EncodingWithNetwork(nn.Module):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)

    def regularizations(self):
        if hasattr(self.encoding, 'regularizations'):
            return self.encoding.regularizations()
        else:
            return {}


def get_encoding_with_network(n_input_dims, n_output_dims, encoding_config, network_config):
    # input suppose to be range [0, 1]
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        encoding_with_network = EncodingWithNetwork(encoding, network)
    else:
        with torch.cuda.device(get_rank()):
            encoding_with_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config_to_primitive(encoding_config),
                network_config=config_to_primitive(network_config)
            )
    return encoding_with_network
