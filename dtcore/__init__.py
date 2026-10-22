"""dtcore: the thin skeleton every twin in this project runs on (see docs/architecture.md).

A system plugs in by providing two objects:
  Source  the physical side: observe() -> next observation (dict) or None when finished,
          act(action) to receive a decision, advance() to move to the next sample.
  Twin    the digital side: update(obs) -> estimate (dict), predict(horizon, **scenario),
          decide() -> action or None, and `capabilities` saying which of these it supports.
run() connects them: observe -> update -> decide -> act -> record -> advance.
Nothing here knows about temperatures, batteries or drones.
"""
from .contract import Source, Twin
from .runtime import run
from .recorder import Recorder

__all__ = ["Source", "Twin", "run", "Recorder"]
__version__ = "0.4.0"
