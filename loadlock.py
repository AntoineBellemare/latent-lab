"""One lock for every model the lab loads, whichever part of it asks.

transformers and diffusers build a model with empty weights first and fill them from the checkpoint,
and the trick that builds it empty patches torch for the whole process. Two models loading at the
same moment on two threads (a mask's SAM 2 while the tokens tab loads RAE's encoder) can then leave
one of them with weights that were never filled, which fails as "Cannot copy out of meta tensor".
So the engine's loading lock is this one, and every helper that loads its own model takes it too.
"""

import threading

LOAD = threading.RLock()
