import models

from systems.utils import update_module_step
from models.base import BaseModel


class BaseDeformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def prepare(self, batch):
        pass


@models.register("dummy_deformer")
class DummyNonRigidDeformer(BaseDeformer):
    def setup(self):
        pass

    def forward(self, points, cond, geometry, *args, with_jac=False, eval_mode=False):
        return points

    def set_initialized(self, is_initialized):
        pass

    def get_rot_mats(self):
        return None

    def get_joints(self):
        return None


@models.register("snarf_deformer")
class SNARFDeformer(BaseDeformer):
    def setup(self):
        self.n_input_dims = 3
        self.n_output_dims = 3
        self.rigid_deformer = models.make(
            self.config.rigid_deformer.name, self.config.rigid_deformer
        )
        self.non_rigid_deformer = models.make(
            self.config.non_rigid_deformer.name, self.config.non_rigid_deformer
        )

    def prepare(self, batch):
        self.rigid_deformer.prepare_deformer(batch)
        self.non_rigid_deformer.prepare_bbox(self.rigid_deformer.bbox)

    def forward(self, points, cond, geometry_fn, *args, with_jac=False, eval_mode=False):
        def non_rigid_geometry_fn(x):
            x, J_inv = self.non_rigid_deformer(x, cond=cond, with_jac=with_jac)
            ret = geometry_fn(x)

            return ret, J_inv

        return self.rigid_deformer(
            points,
            non_rigid_geometry_fn,
            eval_mode=eval_mode,
        )

    def set_initialized(self, is_initialized):
        if hasattr(self.rigid_deformer, "initialized"):
            self.rigid_deformer.initialized = is_initialized
        if hasattr(self.non_rigid_deformer, "initialized"):
            self.non_rigid_deformer.initialized = is_initialized

    def get_rot_mats(self):
        return self.rigid_deformer.rot_mats

    def get_joints(self):
        return self.rigid_deformer.basic_joints

    def update_step(self, epoch, global_step):
        update_module_step(self.rigid_deformer, epoch, global_step)
        update_module_step(self.non_rigid_deformer, epoch, global_step)
