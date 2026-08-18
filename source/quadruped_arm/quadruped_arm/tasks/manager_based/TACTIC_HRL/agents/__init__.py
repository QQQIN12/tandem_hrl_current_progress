"""TANDEM-HRL policy and runner configuration compatibility exports."""

from .tactic_actor_critic import TACTICActorCritic
from .tactic_ppo import TACTICPPO

__all__ = ["TACTICActorCritic", "TACTICPPO"]
