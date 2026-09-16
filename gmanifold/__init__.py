"""gmanifold: learn G: R^m -> R^D from real states and sample ~uniformly on the learned manifold near the support."""
from .geometry import degenerate_mask, intrinsic_dimension, knn, outlier_mask
from .manifold import GlobalManifold, fit_many
from .tangent import TangentCharts
from .validate import KernelScore, check_sampling, support_report

__all__ = ["GlobalManifold", "fit_many", "TangentCharts", "KernelScore", "support_report", "check_sampling", "intrinsic_dimension", "knn", "degenerate_mask", "outlier_mask"]
