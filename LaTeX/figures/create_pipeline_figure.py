"""
Creates a methodology pipeline figure for the Master's thesis.
Based on the original draft structure.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up the figure (reduced height)
fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
ax.set_xlim(0, 12)
ax.set_ylim(0.8, 5)
ax.axis('off')

# Neutral color scheme
colors = {
    'data': '#2C3E50',        # Dark blue-gray
    'process': '#5D6D7E',     # Medium gray-blue
    'analysis': '#6C3483',    # Purple
    'text': '#2F2F2F',
    'box_edge': '#1C2833',
}

def draw_box(ax, x, y, width, height, text, color, fontsize=9, text_color='white'):
    """Draw a rounded rectangle box with text."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor=color, edgecolor=colors['box_edge'],
                          linewidth=1.5, alpha=0.95)
    ax.add_patch(box)
    lines = text.split('\n')
    line_height = 0.22
    start_y = y + (len(lines) - 1) * line_height / 2
    for i, line in enumerate(lines):
        ax.text(x, start_y - i * line_height, line, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold')

def draw_arrow(ax, start, end, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, connectionstyle=connectionstyle,
                            arrowstyle='-|>', mutation_scale=15, lw=2,
                            color='#34495E')
    ax.add_patch(arrow)

# Define positions - horizontal flow like draft
box_h = 0.75

# Row positions (more spacing for arrows)
top_y = 4.2
mid_y = 1.8
bot_y = 0.5

# ============ TOP ROW: Data flow ============
# WVS Data
draw_box(ax, 1.5, top_y, 2.5, box_h, 'World Values Survey', colors['data'])
ax.text(1.5, top_y - 0.5, 'USA · Indonesia · Netherlands · Mexico',
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# Data Preparation
draw_box(ax, 4.5, top_y, 2.5, box_h, 'Data Preparation', colors['process'])
ax.text(4.5, top_y - 0.5, '36 segments (4 countries × 3 income × 3 health)',
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# Prompt Design
draw_box(ax, 7.8, top_y, 2.2, box_h, 'Prompt Design', colors['process'])
ax.text(7.8, top_y - 0.5, 'Persona · Task · Questionnaire',
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# Arrows top row
draw_arrow(ax, (2.75, top_y), (3.25, top_y))
draw_arrow(ax, (5.75, top_y), (6.7, top_y))

# ============ MIDDLE ROW: Generation ============
# Synthetic Data Generation
draw_box(ax, 7.8, mid_y, 2.8, box_h, 'Synthetic Data Generation', colors['process'], fontsize=8)
ax.text(7.8, mid_y - 0.5, '3 LLMs · 5 Scales · 16,200 responses',
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# Arrow from Prompt to Synthetic (start below subtext)
draw_arrow(ax, (7.8, top_y - box_h/2 - 0.38), (7.8, mid_y + box_h/2 + 0.1))

# ============ MIDDLE ROW: Comparison (same level as Synthetic) ============
# Distributional Comparison - under WVS, same level as Synthetic
draw_box(ax, 1.5, mid_y, 2.8, box_h, 'Distributional Comparison', colors['analysis'])
ax.text(1.5, mid_y - 0.5, 'Wasserstein distance · KS statistic',
        ha='center', va='center', fontsize=9, color=colors['text'], style='italic')

# Arrow from WVS to Comparison (real data, start below subtext)
draw_arrow(ax, (1.5, top_y - box_h/2 - 0.38), (1.5, mid_y + box_h/2 + 0.1))

# Arrow from Synthetic to Comparison
draw_arrow(ax, (6.4, mid_y), (2.9, mid_y))

plt.tight_layout()
plt.savefig('/Users/milana/Documents/UniMannheim/MasterThesis/figures/fig_methodology_pipeline.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/milana/Documents/UniMannheim/MasterThesis/figures/fig_methodology_pipeline.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Pipeline figure saved.")
