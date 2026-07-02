"""Archived modules — kept for reference and backward compatibility.

V3:  lam.py              — Original slot competition LAM
V8:  slot_time_lam.py    — MOT-Guided Slot-Time LAM (bbox-based, NO RECONSTRUCTION)
     motion_token_encoder.py — V8 encoder (has _crop_resize bug)
V9:  v9_model.py         — V8 + recon decoder variants (A/B/C/D)
     v9_decoder.py       — FiLM U-Net recon decoder
     flow_decoder.py     — V9-D FlowDecoder + warp_with_flow
     probe_decoder.py    — z_actor → RGB reconstruction probe

Note: V8/V9 line was deprecated because:
  1. Loss was bbox-based (L_motion), not pixel reconstruction
  2. _crop_resize had a scale-inversion bug inflating PSNR by 10-15 dB
  3. z_actor (global 16-dim) cannot support per-pixel reconstruction

Active development is on V10/V11 (top-level modules).
"""
