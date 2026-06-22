"""AdaWorld V6c — Multi-Agent Latent Action Model.
Publication-quality architecture figure (PDF + PNG).

Layout: horizontal left-to-right main pipeline with the Per-Object VAE
(core innovation) emphasized in the centre; loss branches below.
Targets a double-column full-width figure (scales cleanly to ~7.16 in).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8,
    'text.color': '#1a1a1a',
})

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
FW = 11.6   # figure width  (inches)
FH = 6.0    # figure height (inches)

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# ---------------------------------------------------------------------------
# Palette (muted, color-blind friendly, grayscale-printable)
# ---------------------------------------------------------------------------
C_ENC  = '#D6E8F0'; B_ENC  = '#1565C0'   # encoder  – blue
C_POOL = '#D5E8D4'; B_POOL = '#2E7D32'   # pooling  – green
C_ATT  = '#FDE8D0'; B_ATT  = '#E65100'   # attention– orange
C_VAE  = '#E8D5F5'; B_VAE  = '#6A1B9A'   # latent   – purple
C_DEC  = '#F5D5D5'; B_DEC  = '#C62828'   # decoder  – red
C_IN   = '#F0F0F0'; B_IN   = '#555555'   # in/out   – gray
C_BG   = '#FFF8E1'; B_BG   = '#F9A825'   # bg slot  – amber
C_SUB  = '#FFFFFF'; B_SUB  = '#888888'   # sub-box  – light
C_LOSS = '#FFFFFF'                        # loss fill
C_ARROW = '#333333'
C_DASH  = '#1976D2'                        # skip / residual
TXT_GRAY = '#555555'

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
MY = 3.55          # main-row vertical centre
BW = 1.78          # stage block width
BH = 1.78          # stage block height
SUB_W = 1.56       # sub-box width
SUB_H = 0.40       # sub-box height

# stage centre x-coordinates (left -> right)
SX = {
    's1': 1.55,
    's2': 3.55,
    's3': 5.55,
    's4': 7.75,   # core innovation – extra gap
    's5': 9.95,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rbox(cx, cy, w, h, fc='#fff', ec=B_SUB, lw=1.0, z=2, rad=0.06):
    """Rounded rectangle centred at (cx, cy)."""
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle=mpatches.BoxStyle(f"round,pad={rad}"),
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(box)
    return box

def text(cx, cy, s, **kw):
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    kw.setdefault('zorder', 5)
    ax.text(cx, cy, s, **kw)

def arrow(x1, y1, x2, y2, c=C_ARROW, lw=1.3, z=1, style='->', ls='-', rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=z)

def stage(cx, title, fill, border, subs, emph=False, title_color=None):
    """Draw a stage block with title + stacked sub-boxes.
    subs: list of (label, detail) tuples (detail may be None)."""
    lw = 1.9 if emph else 1.2
    rbox(cx, MY, BW, BH, fc=fill, ec=border, lw=lw, z=2)
    tc = title_color or border
    text(cx, MY + BH / 2 - 0.16, title, fontsize=8.5, fontweight='bold',
         color=tc)
    n = len(subs)
    top = MY + BH / 2 - 0.34
    gap = 0.05
    sy = top - SUB_H / 2
    for i, (lab, det) in enumerate(subs):
        fc = C_BG if (emph and i == 0) else C_SUB
        ec = B_BG if (emph and i == 0) else B_SUB
        ew = 1.2 if (emph and i == 0) else 0.8
        rbox(cx, sy, SUB_W, SUB_H, fc=fc, ec=ec, lw=ew, z=3)
        if det:
            text(cx, sy + 0.07, lab, fontsize=6.8, fontweight='bold',
                 color='#222')
            text(cx, sy - 0.10, det, fontsize=6.0, color=TXT_GRAY)
        else:
            text(cx, sy, lab, fontsize=6.8, color='#222')
        sy -= (SUB_H + gap)
    return cx

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
text(FW / 2, FH - 0.28,
     'AdaWorld V6c — Multi-Agent Latent Action Model',
     fontsize=12, fontweight='bold', color='#111')
text(FW / 2, FH - 0.55,
     'Per-subject latent actions with a dedicated background slot, '
     'trained fully self-supervised via next-frame prediction',
     fontsize=8, color=TXT_GRAY, fontstyle='italic')

# ---------------------------------------------------------------------------
# Input / Output labels
# ---------------------------------------------------------------------------
IN_X = 0.42
OUT_X = FW - 0.42

text(IN_X, MY + 0.30, 'videos', fontsize=7.5, fontweight='bold')
text(IN_X, MY + 0.12, '(B,T,H,W,C)', fontsize=6.2, color=TXT_GRAY,
     fontstyle='italic')
text(IN_X, MY - 0.18, 'actor masks', fontsize=7.5, fontweight='bold')
text(IN_X, MY - 0.36, '(B,T,A,H,W)', fontsize=6.2, color=TXT_GRAY,
     fontstyle='italic')

text(OUT_X, MY + 0.18, 'predicted', fontsize=7.5, fontweight='bold')
text(OUT_X, MY + 0.00, 'next frame', fontsize=7.5, fontweight='bold')
text(OUT_X, MY - 0.22, '(B,T−1,H,W,C)', fontsize=6.2, color=TXT_GRAY,
     fontstyle='italic')

# input -> s1
arrow(IN_X + 0.30, MY, SX['s1'] - BW / 2, MY, c=C_ARROW, lw=1.4)
# s5 -> output
arrow(SX['s5'] + BW / 2, MY, OUT_X - 0.30, MY, c=C_ARROW, lw=1.4)

# ---------------------------------------------------------------------------
# Stage 1 — SpatioTemporal Encoder
# ---------------------------------------------------------------------------
stage(SX['s1'], 'SpatioTemporal Encoder', C_ENC, B_ENC, [
    ('Patchify', 'p=16, d=768'),
    ('4× ST Block', 'Spatial SA → Temporal SA (RoPE) → FFN'),
    ('Linear 256→256', None),
])
text(SX['s1'], MY - BH / 2 - 0.16, 'encoded (B,T,256,256)',
     fontsize=6.2, color=TXT_GRAY, fontstyle='italic')

# ---------------------------------------------------------------------------
# Stage 2 — MaskedPool
# ---------------------------------------------------------------------------
stage(SX['s2'], 'MaskedPool', C_POOL, B_POOL, [
    ('masks → patch grid', 'AdaptiveAvgPool2d'),
    ('valid_mask', 'masks_sum > 0.5'),
    ('weighted avg', '→ obj_feats (B,T,K+1,256)'),
])
text(SX['s2'], MY - BH / 2 - 0.16, '+ background slot (always valid)',
     fontsize=6.0, color=B_ENC, fontstyle='italic')

# ---------------------------------------------------------------------------
# Stage 3 — Object SpatioTemporal Attention
# ---------------------------------------------------------------------------
stage(SX['s3'], 'Object ST Attention', C_ATT, B_ATT, [
    ('2× TransformerEnc.', '8 heads, norm_first'),
    ('full self-attn', 'over T×(K+1) tokens'),
    ('cross-frame × cross-subject', None),
])

# ---------------------------------------------------------------------------
# Stage 4 — Per-Object VAE  (★ core innovation)
# ---------------------------------------------------------------------------
stage(SX['s4'], 'Per-Object VAE  ★', C_VAE, B_VAE, [
    ('fc₀ → z₀  (Slot 0)', 'background: camera / global'),
    ('fc₁..fc_K → z_k', 'per-actor latent action'),
    ('Reparam. z=μ+εσ', '32-dim, Free-Bits KL'),
], emph=True, title_color='#4A148C')

# ---------------------------------------------------------------------------
# Stage 5 — Cross-Attention Decoder
# ---------------------------------------------------------------------------
stage(SX['s5'], 'Cross-Attn Decoder', C_DEC, B_DEC, [
    ('CrossAttention', 'q=video_patches, kv=z'),
    ('Residual + 4× Spatio', 'fused + patches'),
    ('unpatchify + sigmoid', None),
])

# ---------------------------------------------------------------------------
# Main-flow arrows between stages (with shape labels above)
# ---------------------------------------------------------------------------
def flow(a, b, label=None, lab_color=TXT_GRAY):
    ax_x1 = SX[a] + BW / 2
    ax_x2 = SX[b] - BW / 2
    arrow(ax_x1, MY, ax_x2, MY, c=C_ARROW, lw=1.5)
    if label:
        text((ax_x1 + ax_x2) / 2, MY + 0.16, label, fontsize=6.0,
             color=lab_color, fontstyle='italic')

flow('s1', 's2', 'encoded')
flow('s2', 's3', 'obj_feats')
flow('s3', 's4', 'obj_feats')
flow('s4', 's5', 'z_rep')

# ---------------------------------------------------------------------------
# Skip connection: previous-frame patches -> decoder (residual "where")
# ---------------------------------------------------------------------------
SK_Y = MY + BH / 2 + 0.45
# start above s1, route over the top to s5
arrow(SX['s1'], MY + BH / 2, SX['s1'], SK_Y, c=C_DASH, lw=1.1, ls='--')
arrow(SX['s1'], SK_Y, SX['s5'], SK_Y, c=C_DASH, lw=1.1, ls='--')
arrow(SX['s5'], SK_Y, SX['s5'], MY + BH / 2, c=C_DASH, lw=1.1, ls='--')
text((SX['s1'] + SX['s5']) / 2, SK_Y + 0.14,
     'prev-frame patches  (where)  — residual to decoder',
     fontsize=6.2, color=C_DASH, fontstyle='italic')

# ---------------------------------------------------------------------------
# Loss branches (below the main row)
# ---------------------------------------------------------------------------
LY = 1.55          # loss-row centre
LW = 1.80          # loss box width
LH = 0.46

def loss_box(cx, cy, title, formula, ec, weight):
    rbox(cx, cy, LW, LH, fc=C_LOSS, ec=ec, lw=1.3, z=3)
    text(cx, cy + 0.10, title, fontsize=7.2, fontweight='bold', color=ec)
    text(cx, cy - 0.10, f'{formula}   (w={weight})', fontsize=6.0,
         color='#333')

# L_recon  <- decoder output
lx_recon = SX['s5']
loss_box(lx_recon, LY, 'L_recon', 'MSE(recon, target)', B_DEC, '1.0')
arrow(SX['s5'], MY - BH / 2, lx_recon, LY + LH / 2, c=B_DEC, lw=1.1,
      rad=-0.15)

# FreeBits KL  <- VAE
lx_kl = SX['s4'] + 0.05
loss_box(lx_kl, LY, 'FreeBits KL', 'clamp(KL_dim, λ=0.1)', B_VAE, '2e-4')
arrow(SX['s4'], MY - BH / 2, lx_kl, LY + LH / 2, c=B_VAE, lw=1.1, rad=0.15)

# L_obj_recon  <- VAE predicts obj delta
lx_obj = SX['s3'] + 0.20
loss_box(lx_obj, LY, 'L_obj_recon', 'MSE(MLP(z), obj_Δ)', B_ATT, '0.01')
arrow(SX['s4'] - 0.15, MY - BH / 2 - 0.05, lx_obj + 0.15, LY + LH / 2,
      c=B_ATT, lw=1.1, rad=0.2)

# L_total
lx_tot = SX['s2']
rbox(lx_tot, LY, LW, LH + 0.06, fc='#FFF3E0', ec='#E65100', lw=1.6, z=3)
text(lx_tot, LY + 0.12, 'L_total', fontsize=7.4, fontweight='bold',
     color='#E65100')
text(lx_tot, LY - 0.12, '= L_recon + β·KL + λ·L_obj', fontsize=6.2,
     color='#333')

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
LEG_Y = 0.30
items = [
    ('Encoder', C_ENC, B_ENC),
    ('Pooling', C_POOL, B_POOL),
    ('Attention', C_ATT, B_ATT),
    ('Latent / VAE', C_VAE, B_VAE),
    ('Decoder', C_DEC, B_DEC),
    ('Background slot', C_BG, B_BG),
]
n = len(items)
span = FW - 1.0
step = span / n
x0 = 0.5
for i, (lab, fc, ec) in enumerate(items):
    cx = x0 + step * (i + 0.5)
    rbox(cx - 0.55, LEG_Y, 0.30, 0.18, fc=fc, ec=ec, lw=0.9, rad=0.04, z=2)
    text(cx - 0.36, LEG_Y, lab, fontsize=6.6, ha='left', va='center')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ('.pdf', '.png'):
    path = os.path.join(out_dir, f'architecture_v6c{ext}')
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    print(f'Saved: {path}')
