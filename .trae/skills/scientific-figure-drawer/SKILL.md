---
name: "scientific-figure-drawer"
description: "Generates publication-quality scientific/architecture figures (PDF+PNG) via matplotlib. Invoke when user asks to draw architecture diagrams, model flowcharts, or paper figures for top-tier conferences (CVPR/NeurIPS/ICLR/ICML)."
---

# Scientific Figure Drawer

This skill produces **publication-quality** figures suitable for top-tier ML/CV
conferences (CVPR, NeurIPS, ICLR, ICML, ACL, etc.). It renders vector PDF +
high-DPI PNG via matplotlib (no LaTeX toolchain required, works headless).

## When to Invoke

- User asks to "draw / 绘制 an architecture diagram / 架构图"
- User asks for a "paper figure / 论文图 / 顶会图"
- User wants to visualize a model pipeline, dataflow, or module stack
- User mentions "科研绘图 SKILL" or "scientific figure"

## Output Contract

Always produce **both**:
1. A `.pdf` (vector, primary deliverable for paper submission)
2. A `.png` at 300+ DPI (preview / review)

Default output directory: project `figures/`. Reuse an existing script there if
one exists (edit, don't create duplicates) unless the user asks for a new style.

## Design Principles (MUST follow)

### 1. Layout & Composition
- **Single-column** fig width ≤ 3.5 in (≈8.9 cm); **double-column** ≤ 7.16 in
  (≈18.2 cm). Pick based on content density; default double-column for
  architecture diagrams.
- Use a **coordinate-managed layout** (a `Y` cursor or a grid) so boxes never
  overlap and arrows have consistent gaps.
- Left-to-right dataflow is preferred for pipelines; top-to-bottom only when
  the pipeline is long and narrow.
- Group related modules with a faint background panel + a group label to show
  hierarchy (e.g. "Encoder", "Latent", "Decoder").

### 2. Typography
- Font: a clean sans-serif (`DejaVu Sans` / `Helvetica` / `Arial`). Set via
  `rcParams`. **Never** use the default matplotlib serif for paper figures.
- Base font 7–8 pt for double-column, 8–9 pt for single-column. Titles 9–11 pt
  bold. Annotations 6–6.5 pt. Nothing below 6 pt.
- Use **bold** for module names, regular for descriptions, *italic* for
  tensor shapes / notes.
- Tensor shapes in monospace or italic gray, e.g. `(B,T,256,256)`.

### 3. Color Palette (academic, color-blind friendly, grayscale-printable)
- Use **muted** pastel fills with darker borders. Recommended base set:
  - Encoder  `#D6E8F0` / border `#1565C0`
  - Pooling  `#D5E8D4` / border `#2E7D32`
  - Attn     `#FDE8D0` / border `#E65100`
  - Latent   `#E8D5F5` / border `#6A1B9A`
  - Decoder  `#F5D5D5` / border `#C62828`
  - Input/Out `#F0F0F0` / border `#555555`
  - BG highlight `#FFF8E1`
- Highlight the **core innovation** module with a thicker border (1.6–1.8 pt)
  and a star ★ or accent label.
- Loss boxes: white fill, colored border matching the branch they attach to.
- Verify the figure is still legible when printed grayscale (fills must differ
  in lightness, not just hue).

### 4. Boxes, Arrows & Connectors
- Rounded rectangles (`FancyBboxPatch` with `round,pad=...`) for all modules.
- Sub-components: smaller white boxes inside a module, thin (0.8 pt) borders.
- Arrows: `ax.annotate` with `arrowstyle='->'`, lw 1.0–1.3 for main flow,
  0.8–1.0 for sub-flow. Use a slightly curved `connectionstyle` only to avoid
  overlaps; straight otherwise.
- **Skip-connections / residual paths**: dashed arrows in a distinct color,
  with a short label (`+`, `residual`).
- Loss branches: thin colored arrows from the module that produces the signal
  to the loss box, placed in the margin.

### 5. Information Density
- Show tensor shapes on key edges (input, after encoder, latent z, output).
- Annotate the **innovation** with a callout (★ + one-line "why it matters").
- Keep the legend compact (color → stage mapping) at the bottom or in a corner.
- Do NOT cram every implementation detail — the caption carries the rest.

### 6. Rendering Quality
- `plt.savefig(..., dpi=300, bbox_inches='tight', pad_inches=0.15,
  facecolor='white')` for PNG; same (dpi ignored) for PDF.
- `matplotlib.use('Agg')` at top so it runs headless on servers.
- Vector PDF is the source of truth; PNG is only for quick review.

## Implementation Pattern

A reusable script structure (edit the existing one if present):

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8, 'text.color': '#1a1a1a',
})

# --- palette + layout constants ---
# --- helpers: rbox(), text(), arrow() ---
# --- Y-cursor for vertical stacking ---
# --- draw stages, sub-boxes, arrows, losses, legend ---
# --- save pdf + png ---
```

## Workflow

1. Read the source spec (report/markdown) the user points to.
2. Identify: inputs, pipeline stages, the core innovation, latent variables,
   losses, outputs.
3. Decide single- vs double-column and left-to-right vs top-to-bottom.
4. Edit or create the script under `figures/`.
5. Run it; confirm both PDF and PNG are written.
6. Show the PNG path so the user can preview.

## Anti-patterns to Avoid

- Rainbow palettes / pure primary colors (look unprofessional).
- Boxes touching or arrows passing through text.
- Font < 6 pt (illegible at print size).
- Raster-only output (always ship the PDF).
- Recreating a new script when an editable one already exists — extend it.
- Over-decorating with shadows / 3D / gradients — flat is the paper norm.
