from models import register

# Need to manually register SNARFDeformer as it is not a subclass of BaseModel
from .snarf_deformer import SNARFDeformer
SNARFDeformer = register("fast-snarf")(SNARFDeformer)
