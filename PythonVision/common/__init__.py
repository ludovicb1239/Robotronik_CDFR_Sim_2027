"""Shared library for the PythonVision algorithms.

Everything in here is used by at least two of the algorithms; anything that only
one of them needs stays in that algorithm's own module. The split, briefly:

* `paths`    - where each algorithm's input and output folders are
* `geometry` - the fixed camera mounting, field outline and ground-plane warp
* `warp`     - `GroundWarper`, a capture rectified flat onto the ground
* `imageio`  - reading and writing images, step-by-step stage dumps
* `pose`     - the true pose read out of a capture's filename
* `viz`      - drawing poses and quads
"""

from __future__ import annotations

from . import geometry, imageio, paths, pose, viz, warp

__all__ = ["geometry", "imageio", "paths", "pose", "viz", "warp"]
