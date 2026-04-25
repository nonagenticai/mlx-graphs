from typing import TYPE_CHECKING

from .dataloaders import Dataloader  # noqa

# NeighborLoader transitively imports mlx_cluster, whose 0.0.7 wheel is
# ABI-incompatible with mlx 0.31.2+ (missing
# mlx::core::metal::Device::get_command_encoder). Import lazily at
# runtime so environments without a working mlx_cluster can still use
# the rest of the package; tests exercising NeighborLoader gate with
# pytest.importorskip("mlx_cluster", exc_type=ImportError).
if TYPE_CHECKING:
    from .neighbor_loader import NeighborLoader  # noqa
else:
    try:
        from .neighbor_loader import NeighborLoader  # noqa
    except ImportError:
        NeighborLoader = None
