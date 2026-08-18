"""TANDEM-HRL mainline package.

The Gym task is registered only after the complete privileged-state
environment satisfies the architecture contract in ``contract.py``.
"""

from .contract import MAINLINE_CONTRACT
from .physics_contract import validate_physics_contract

__all__ = ["MAINLINE_CONTRACT", "validate_physics_contract"]
