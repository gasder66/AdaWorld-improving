"""AdaWorld model modules.

Active models (V10, V11):
  - LatentActionModelV10: Shared VAE + RGB reconstruction
  - LatentActionModelV11: Frame-Diff LAM (iVideoGPT-style)

Archived (V3/V8/V9) — available via lam.modules._archived.*:
  - LatentActionModel (V3): Original slot competition LAM
  - LatentActionModelV8: MOT-Guided Slot-Time LAM (bbox-based)
  - LatentActionModelV9: V8 + recon decoder variants
  - MotionTokenEncoder, FlowDecoder, ProbeDecoder
"""

# Active models
from lam.modules.v10_model import LatentActionModelV10
from lam.modules.v11_model import LatentActionModelV11

# Building blocks
from lam.modules.blocks import (
    CrossAttention, ObjectReconHead,
    MaskedPool, ObjectSpatioTemporalAttention, compute_subject_positions,
)

# Archived — backward compatibility
try:
    from lam.modules._archived.lam import LatentActionModel
except ImportError:
    LatentActionModel = None

try:
    from lam.modules._archived.slot_time_lam import LatentActionModelV8
except ImportError:
    LatentActionModelV8 = None

try:
    from lam.modules._archived.v9_model import LatentActionModelV9
except ImportError:
    LatentActionModelV9 = None

try:
    from lam.modules._archived.motion_token_encoder import MotionTokenEncoder
except ImportError:
    MotionTokenEncoder = None

try:
    from lam.modules._archived.flow_decoder import FlowDecoder, warp_with_flow
except ImportError:
    FlowDecoder = None
    warp_with_flow = None

try:
    from lam.modules._archived.probe_decoder import ProbeDecoder
except ImportError:
    ProbeDecoder = None
