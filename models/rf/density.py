import torch
import torch.nn as nn
import models

from models.base import BaseModel


class Density(BaseModel):
    def setup(self):
        params_init = self.config.get('params_init', {})
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


@models.register('learned-laplace-density')
class LearnedLaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def setup(self):
        super().setup()
        self.beta_min = torch.tensor(self.config.get('beta_min', 0.0001)).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = torch.reciprocal(beta)
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


@models.register('scheduled-laplace-density')
class ScheduledLaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def setup(self):
        super().setup()
        self.beta0 = torch.tensor(self.config.get('beta0', 0.1)).cuda()
        self.beta1 = torch.tensor(self.config.get('beta1', 0.001)).cuda()
        self.t = 0.0

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = torch.reciprocal(beta)
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta0 / (
            1.0 + (self.beta0 - self.beta1) / self.beta1 * (self.t**0.8)
        )
        return beta

    def update_step(self, epoch, global_step):
        self.t = min(1.0, global_step / self.config.get("beta_schedule_steps", 10000))
