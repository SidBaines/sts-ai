"""A model-independent Slay the Spire environment."""
from sts_env.env import SlayTheSpireEnv, SimulatorError, SimulatorTimeout, ENVIRONMENT_VERSION

__version__ = ENVIRONMENT_VERSION
__all__ = ["SlayTheSpireEnv", "SimulatorError", "SimulatorTimeout"]
