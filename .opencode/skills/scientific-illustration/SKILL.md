---
name: scientific-illustration
description: Create publication-quality scientific figures, architecture diagrams, and research paper illustrations using matplotlib, TikZ/LaTeX, or other tools. Use for drawing neural network architectures, flowchart models, data pipelines, and experimental result visualizations.
license: MIT
compatibility: opencode
metadata:
  audience: researchers
  workflow: scientific-visualization
---

## What I Do
- Create publication-quality architecture diagrams for neural networks, ML pipelines, and system designs
- Generate scientific plots (line charts, scatter plots, bar charts, heatmaps, etc.) with proper formatting
- Produce flowchart-style illustrations for research papers
- Apply academic figure conventions (font sizes, dpi, color palettes, labeling)

## When to Use Me
Use this skill when the user asks for:
- Drawing architecture diagrams for research papers
- Creating scientific illustrations or figures
- Plotting experimental results for publication
- Visualizing model architectures, data flows, or system designs

## Tools & Dependencies
- **Python + matplotlib**: For architecture diagrams, flowcharts, and general scientific figures
- **Python + numpy**: For data handling
- **TikZ/LaTeX**: For highest-quality vector diagrams (optional, requires TeXLive)

Install requirements when needed:
```bash
pip install matplotlib numpy
```

## Figure Design Guidelines

### Architecture Diagrams
- Use rounded rectangles for functional blocks with clear borders
- Color-code different stages/modules (consistent palette)
- Use arrows (→ ↓) to show data flow direction
- Add concise labels for tensor shapes (e.g., `(B,T,H,W,C)`)
- Group related components with background boxes
- Keep text readable: fontsize 10-12pt for labels, 14-16pt for titles
- Output format: PDF (vector) for papers, PNG for quick preview
- DPI: 300 for publication

### Color Palette (Publication-friendly)
```python
COLORS = {
    'input':      '#E8F5E9',  # light green
    'encoder':    '#E3F2FD',  # light blue  
    'latent':     '#FFF3E0',  # light orange
    'decoder':    '#FCE4EC',  # light pink
    'loss':       '#F3E5F5',  # light purple
    'attention':  '#E0F7FA',  # light cyan
    'pooling':    '#F9FBE7',  # light lime
    'border':     '#333333',  # dark gray
    'arrow':      '#555555',  # medium gray
}
```

### Figure Size Guidelines
- Single column: width ~3.5 inches (IEEE/ACM), ~8.5cm (Elsevier)
- Double column: width ~7 inches, ~17cm
- Aspect ratio: golden ratio (1.618) or balanced (4:3, 3:2)

## Workflow
1. Read the user's report/code to understand the architecture or data
2. Determine figure type: architecture diagram, flowchart, or data plot
3. Write a Python script using matplotlib
4. Execute the script and verify output
5. Save as both PDF (vector) and PNG (preview) in `figures/` directory
6. Display the PNG preview to the user
