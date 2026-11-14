"""Geometries used by the solver examples and manuscript experiments."""
from ._frozen.manifolds import Euclidean, Sphere, ProductSpheres
from ._geometry.affine_spd import SPD
from ._geometry.additional_geometries import Grassmann, Hyperbolic
from ._geometry.author_stiefel import Stiefel
from ._geometry.report_problems import Product, Factor

__all__ = ["Euclidean", "Sphere", "ProductSpheres", "SPD", "Grassmann",
           "Hyperbolic", "Stiefel", "Product", "Factor"]
