from .dataloaders import Dataloader  # noqa

# NeighborLoader requires mlx_cluster, whose 0.0.7 wheel is ABI-incompatible
# with mlx 0.31.2+. Import lazily so that environments without a working
# mlx_cluster can still use the rest of the package.
try:
    from .neighbor_loader import NeighborLoader  # noqa
except ImportError:
    NeighborLoader = None  # type: ignore[assignment, misc]
