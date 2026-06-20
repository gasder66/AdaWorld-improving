import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8,
    'text.color': '#1a1a1a',
})

# === Layout constants ===
FW = 9.5   # figure width (inches)
FH = 10.2  # figure height
GAP_S = 0.08  # small gap
GAP_M = 0.18  # medium gap

# Main pipeline box dimensions
MW = 5.5     # module width
MH = 1.08    # module height
SW = 0.48    # sub-box width ratio relative to MW
SH = 0.42    # sub-box height

# Center X for main column
CX = FW / 2

# Color palette (muted, academic, grayscale-printable)
C_ST1 = '#D6E8F0'  # pale blue – encoder
C_ST2 = '#D5E8D4'  # pale green – pooling
C_ST3 = '#FDE8D0'  # pale orange – attention
C_ST4 = '#E8D5F5'  # pale purple – VAE / latent
C_ST5 = '#F5D5D5'  # pale pink – decoder
C_IN   = '#F0F0F0'  # light gray – input/output
C_BG   = '#FFF8E1'  # pale yellow – background slot highlight
C_LOSS = '#FAFAFA'  # white-ish – loss
C_SUB  = '#FFFFFF'  # white – sub-components
C_BN   = '#333333'  # border normal
C_BL   = '#555555'  # border light
C_ARROW = '#444444' # arrows
C_ACC1 = '#1565C0'  # accent blue
C_ACC2 = '#C62828'  # accent red
C_ACC3 = '#6A1B9A'  # accent purple

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# === Helper functions ===
def rbox(x, y, w, h, fc='#fff', ec=C_BN, lw=1.0, z=2, rad=0.08):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch((x - w/2, y), w, h,
                         boxstyle=mpatches.BoxStyle(f"round,pad={rad}"),
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(box)
    return box

def text(x, y, s, **kw):
    """Centered text with defaults."""
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    kw.setdefault('zorder', 5)
    ax.text(x, y, s, **kw)

def arrow(x1, y1, x2, y2, c=C_ARROW, lw=1.3, z=1, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                connectionstyle='arc3,rad=0'),
                zorder=z)

# === y-coordinate manager ===
class Y:
    def __init__(self, start):
        self.v = start
    def step(self, dh):
        self.v -= dh
        return self.v

Y = Y(FH - 0.5)  # top margin

# =====================================================================
# TITLE
# =====================================================================
text(CX, Y.v, 'AdaWorld V6c — Multi-Agent Latent Action Model', fontsize=12,
     fontweight='bold', color='#111')
Y.step(0.55)

# =====================================================================
# INPUT
# =====================================================================
m_in = rbox(CX, Y.v, MW - 0.6, 0.48, fc=C_IN, ec=C_BL)
text(CX, Y.v + 0.24, 'Input: videos (B,T,H,W,C)  +  actor masks (B,T,A,H,W)',
     fontsize=8.5)
Y.step(0.48 + GAP_M)

arrow(CX, Y.v + GAP_M, CX, Y.v, c=C_ARROW, lw=1.2)

# =====================================================================
# STAGE 1: SpatioTemporalTransformer
# =====================================================================
y1 = Y.v
arrow_head_y = y1 + GAP_M
y1 = Y.v
rbox(CX, y1, MW, MH, fc=C_ST1, ec=C_BN, lw=1.3)
text(CX - MW/2 + 0.7, y1 + MH - 0.14, 'Stage 1: SpatioTemporal Transformer',
     fontsize=9, fontweight='bold', ha='left')

# -- sub-box: Patchify --
sx1 = CX - MW/2 + 0.45
sy1 = y1 + 0.22
rbox(sx1, sy1, 1.3, SH, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx1 + 0.65, sy1 + SH/2, 'Patchify\n(p=16, d=768)', fontsize=7)

# -- sub-box: 4× ST Blocks --
sx2 = sx1 + 1.6
rbox(sx2, sy1, 2.1, SH, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx2 + 1.05, sy1 + SH/2 + 0.06, '4× SpatioTemporal Block', fontsize=7,
     fontweight='bold')
text(sx2 + 1.05, sy1 + SH/2 - 0.13, 'Spatial SA → Temporal SA (RoPE) → FFN', fontsize=6.5,
     color='#666')

# -- sub-box: Output --
sx3 = sx2 + 2.4
rbox(sx3, sy1, 1.1, SH, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx3 + 0.55, sy1 + SH/2, 'Linear\n256→256', fontsize=7)

# sub-arrows
arrow(sx1 + 1.3, sy1 + SH/2, sx2, sy1 + SH/2, c=C_BL, lw=1.0)
arrow(sx2 + 2.1, sy1 + SH/2, sx3, sy1 + SH/2, c=C_BL, lw=1.0)

# output annotation
text(CX + MW/2 + 0.55, y1 + MH/2, 'encoded\n(B,T,256,256)', fontsize=6.5,
     color='#555', ha='left')

Y.step(MH + GAP_M)
arrow(CX, y1 - 0.03, CX, Y.v + GAP_M, c=C_ARROW, lw=1.2)

# =====================================================================
# STAGE 2: MaskedPool
# =====================================================================
y2 = Y.v
rbox(CX, y2, MW, MH - 0.15, fc=C_ST2, ec=C_BN, lw=1.3)
text(CX - MW/2 + 0.7, y2 + MH - 0.14 - 0.075, 'Stage 2: MaskedPool (per-subject spatial aggregation)',
     fontsize=9, fontweight='bold', ha='left')

sx = CX - MW/2 + 0.45
sy = y2 + 0.15
# sub 1
rbox(sx, sy, 1.4, SH - 0.05, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx + 0.7, sy + (SH-0.05)/2, 'masks → patch grid\nAdaptiveAvgPool2d', fontsize=6.8)

# sub 2
sx2 = sx + 1.65
rbox(sx2, sy, 1.4, SH - 0.05, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx2 + 0.7, sy + (SH-0.05)/2, 'masks_sum > 0.5\n→ valid_mask (bool)', fontsize=6.8)

# sub 3
sx3 = sx2 + 1.65
rbox(sx3, sy, 1.7, SH - 0.05, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx3 + 0.85, sy + (SH-0.05)/2, 'weighted avg: Σ(feat × mask)\n→ obj_feats (B,T,K+1,256)', fontsize=6.8)

arrow(sx + 1.4, sy + (SH-0.05)/2, sx2, sy + (SH-0.05)/2, c=C_BL, lw=1.0)
arrow(sx2 + 1.4, sy + (SH-0.05)/2, sx3, sy + (SH-0.05)/2, c=C_BL, lw=1.0)

# bg always valid
text(sx3 + 0.85, sy - 0.03, 'Slot 0 (BG) always valid', fontsize=6, color=C_ACC1, fontstyle='italic')

Y.step(MH - 0.15 + GAP_M)
arrow(CX, y2 - 0.03, CX, Y.v + GAP_M, c=C_ARROW, lw=1.2)

# =====================================================================
# STAGE 3: ObjectSpatioTemporalAttention
# =====================================================================
y3 = Y.v
rbox(CX, y3, MW, MH - 0.15, fc=C_ST3, ec=C_BN, lw=1.3)
text(CX - MW/2 + 0.7, y3 + MH - 0.14 - 0.075, 'Stage 3: Object SpatioTemporal Attention',
     fontsize=9, fontweight='bold', ha='left')

sx = CX - MW/2 + 0.45
sy = y3 + 0.18
rbox(sx, sy, 2.6, SH - 0.1, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx + 1.3, sy + (SH-0.1)/2 + 0.06, '2× TransformerEncoder (8 heads)', fontsize=7,
     fontweight='bold')
text(sx + 1.3, sy + (SH-0.1)/2 - 0.13, 'full self-attn over T×(K+1) tokens\ncross-frame × cross-subject', fontsize=6.5,
     color='#666')

sx2 = sx + 2.85
rbox(sx2, sy, 2.1, SH - 0.1, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx2 + 1.05, sy + (SH-0.1)/2, 'reshape back\n→ obj_feats (B,T,K+1,256)', fontsize=7)

arrow(sx + 2.6, sy + (SH-0.1)/2, sx2, sy + (SH-0.1)/2, c=C_BL, lw=1.0)

Y.step(MH - 0.15 + GAP_M)
arrow(CX, y3 - 0.03, CX, Y.v + GAP_M, c=C_ARROW, lw=1.2)

# =====================================================================
# STAGE 4: Per-Object VAE  [innovation — emphasized]
# =====================================================================
y4 = Y.v
rbox(CX, y4, MW, MH + 0.08, fc=C_ST4, ec='#7B1FA2', lw=1.8)  # thicker border = emphasis
text(CX - MW/2 + 0.7, y4 + MH + 0.08 - 0.14, 'Stage 4: Per-Object VAE  ★  core innovation',
     fontsize=9, fontweight='bold', ha='left', color='#5C007A')

sx = CX - MW/2 + 0.45
sy = y4 + 0.22

# fc_0 (BG slot) - highlighted
rbox(sx, sy, 1.15, SH - 0.02, fc=C_BG, ec='#F9A825', lw=1.2)
text(sx + 0.575, sy + (SH-0.02)/2 + 0.05, 'fc₀ (Slot 0)', fontsize=7, fontweight='bold', color='#E65100')
text(sx + 0.575, sy + (SH-0.02)/2 - 0.12, '→ z₀ : background\ncamera / global motion', fontsize=6, color='#888')

# fc_1..K
sx2 = sx + 1.45
rbox(sx2, sy, 2.3, SH - 0.02, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx2 + 1.15, sy + (SH-0.02)/2 + 0.05, 'fc₁ .. fc_K (Slot 1..K)', fontsize=7,
     fontweight='bold')
text(sx2 + 1.15, sy + (SH-0.02)/2 - 0.12, '→ z_k : per-actor independent latent action', fontsize=6.5,
     color='#666')

# reparameterization
sx3 = sx2 + 2.55
rbox(sx3, sy, 1.1, SH - 0.02, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx3 + 0.55, sy + (SH-0.02)/2 + 0.05, 'Reparam.', fontsize=7, fontweight='bold')
text(sx3 + 0.55, sy + (SH-0.02)/2 - 0.12, 'z = μ + ε·σ\n32-dim latent', fontsize=6, color='#888')

arrow(sx + 1.15, sy + (SH-0.02)/2, sx2, sy + (SH-0.02)/2, c=C_BL, lw=1.0)
arrow(sx2 + 2.3, sy + (SH-0.02)/2, sx3, sy + (SH-0.02)/2, c=C_BL, lw=1.0)

Y.step(MH + 0.08 + GAP_M)
arrow(CX, y4 - 0.03, CX, Y.v + GAP_M, c=C_ARROW, lw=1.2)

# =====================================================================
# STAGE 5: Decoder
# =====================================================================
y5 = Y.v
rbox(CX, y5, MW, MH + 0.05, fc=C_ST5, ec=C_BN, lw=1.3)
text(CX - MW/2 + 0.7, y5 + MH + 0.05 - 0.14, 'Stage 5: Cross-Attention Decoder',
     fontsize=8.5, fontweight='bold', ha='left')

sx = CX - MW/2 + 0.45
sy = y5 + 0.22
rbox(sx, sy, 1.7, SH - 0.05, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx + 0.85, sy + (SH-0.05)/2 + 0.05, 'CrossAttention', fontsize=7, fontweight='bold')
text(sx + 0.85, sy + (SH-0.05)/2 - 0.13, 'q=video_patches\nkv=action_embed(z)', fontsize=6.5, color='#666')

sx2 = sx + 1.95
rbox(sx2, sy, 1.3, SH - 0.05, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx2 + 0.65, sy + (SH-0.05)/2 + 0.05, 'Residual +', fontsize=7, fontweight='bold')
text(sx2 + 0.65, sy + (SH-0.05)/2 - 0.13, 'fused + patches', fontsize=6.5, color='#666')

sx3 = sx2 + 1.55
rbox(sx3, sy, 1.65, SH - 0.05, fc=C_SUB, ec=C_BL, lw=0.8)
text(sx3 + 0.825, sy + (SH-0.05)/2 + 0.05, 'SpatioTransformer', fontsize=7, fontweight='bold')
text(sx3 + 0.825, sy + (SH-0.05)/2 - 0.13, '4× Spatial Block\n→ unpatchify', fontsize=6.5, color='#666')

arrow(sx + 1.7, sy + (SH-0.05)/2, sx2, sy + (SH-0.05)/2, c=C_BL, lw=1.0)
arrow(sx2 + 1.3, sy + (SH-0.05)/2, sx3, sy + (SH-0.05)/2, c=C_BL, lw=1.0)

Y.step(MH + 0.05 + GAP_M)
arrow(CX, y5 - 0.03, CX, Y.v + GAP_M, c=C_ARROW, lw=1.2)

# =====================================================================
# OUTPUT
# =====================================================================
y_out = Y.v
rbox(CX, y_out, MW - 1.0, 0.45, fc=C_IN, ec=C_BL, lw=1.2)
text(CX, y_out + 0.225, 'Output: predicted next frame  recon (B,T−1,H,W,C)', fontsize=8.5,
     fontweight='bold')

# =====================================================================
# LOSS FUNCTIONS (right side, branching from Stage 4)
# =====================================================================
LX = CX + MW/2 + 0.35   # start of loss boxes
LW = 1.6                 # loss box width
LH = 0.36                # loss box height

loss_y_start = y4 + (MH + 0.08) * 0.55

# L_recon
ly1 = loss_y_start
rbox(LX, ly1, LW, LH, fc=C_LOSS, ec=C_ACC2, lw=1.2)
text(LX + LW/2, ly1 + LH/2, 'L_recon = MSE(recon, tgt)', fontsize=7, fontweight='bold',
     color=C_ACC2)
arrow(CX + MW/2, y4 + MH * 0.65, LX, ly1 + LH/2, c=C_ACC2, lw=1.0)

# FreeBits KL
ly2 = ly1 - LH - 0.06
rbox(LX, ly2, LW, LH, fc=C_LOSS, ec=C_ACC1, lw=1.2)
text(LX + LW/2, ly2 + LH/2, 'FreeBits KL = clamp(KL, λ=0.1)', fontsize=7, fontweight='bold',
     color=C_ACC1)
arrow(CX + MW/2, y4 + MH * 0.35, LX, ly2 + LH/2, c=C_ACC1, lw=1.0)

# L_obj
ly3 = ly2 - LH - 0.06
rbox(LX, ly3, LW, LH, fc=C_LOSS, ec=C_ACC3, lw=1.2)
text(LX + LW/2, ly3 + LH/2, 'L_obj = MSE(MLP(z), obj_Δ)', fontsize=7, fontweight='bold',
     color=C_ACC3)
arrow(CX + MW/2, y4 + MH * 0.15, LX, ly3 + LH/2, c=C_ACC3, lw=1.0)

# L_total
ly4 = ly3 - LH - 0.1
rbox(LX, ly4, LW, 0.42, fc='#FFF3E0', ec='#E65100', lw=1.5)
text(LX + LW/2, ly4 + 0.21, 'L_total = L_recon\n+ β·KL + λ·L_obj', fontsize=7,
     fontweight='bold', color='#E65100')

# =====================================================================
# LEGEND (bottom)
# =====================================================================
leg_y = 0.25
leg_items = [
    ('Encoder', C_ST1),
    ('Pooling', C_ST2),
    ('Attention / Interaction', C_ST3),
    ('Latent / VAE', C_ST4),
    ('Decoder', C_ST5),
]
for i, (lab, col) in enumerate(leg_items):
    lx = 0.8 + i * 1.42
    rbox(lx + 0.18, leg_y, 0.3, 0.18, fc=col, ec=C_BN, lw=0.7, rad=0.04, z=2)
    text(lx + 0.33, leg_y + 0.09, lab, fontsize=6.5, ha='left', va='center')
    lx += 0.35

# =====================================================================
# SAVE
# =====================================================================
out_dir = os.path.dirname(os.path.abspath(__file__))
for fmt, ext in [('pdf', '.pdf'), ('png', '.png')]:
    path = os.path.join(out_dir, f'architecture_v6c{ext}')
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    print(f'Saved: {path}')
