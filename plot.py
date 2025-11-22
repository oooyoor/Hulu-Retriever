#!/usr/bin/env python3
"""Generate a diagram of the ANN search pipeline that produces Top-K results."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, center, width, height, text_lines, fontsize=12, **patch_kwargs):
  """Draw a rounded rectangle annotated with multi-line text."""
  x, y = center
  patch = FancyBboxPatch(
      (x - width / 2, y - height / 2),
      width,
      height,
      boxstyle="round,pad=0.3",
      linewidth=1.5,
      facecolor=patch_kwargs.get("facecolor", "#ffffff"),
      edgecolor=patch_kwargs.get("edgecolor", "#111111"),
  )
  ax.add_patch(patch)
  ax.text(
      x,
      y,
      "\n".join(text_lines),
      ha="center",
      va="center",
      fontsize=fontsize,
  )


def add_arrow(ax, start, end, text=None, text_offset=(0, 0.2)):
  """Draw an arrow between two points and optionally annotate it."""
  arrow = FancyArrowPatch(
      start,
      end,
      arrowstyle="Simple,tail_width=0.7,head_width=10,head_length=10",
      linewidth=1.2,
      color="#111111",
      shrinkA=6,
      shrinkB=6,
  )
  ax.add_patch(arrow)
  if text:
    mx = (start[0] + end[0]) / 2 + text_offset[0]
    my = (start[1] + end[1]) / 2 + text_offset[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=11)


def draw_diagram(output_path: pathlib.Path):
  fig, ax = plt.subplots(figsize=(7.5, 7.5))
  ax.axis("off")
  ax.set_xlim(-5, 5)
  ax.set_ylim(-3.5, 6)

  add_box(
      ax,
      center=(0, 5),
      width=4.3,
      height=1.8,
      text_lines=["Enhanced ANN Index", "vector", "label", "offset"],
      fontsize=14,
  )
  ax.text(0, 3.9, "I/O read", ha="center", va="center", fontsize=12)
  add_arrow(ax, (0, 4.1), (0, 3.4))

  ax.text(-3.3, 3.2, "Traditional (Two Layers)", fontsize=12, fontweight="bold")
  add_box(
      ax,
      center=(-3, 2),
      width=3.2,
      height=1.4,
      text_lines=["ANN Index", "vectors → labels"],
  )
  add_box(
      ax,
      center=(-3, 0.4),
      width=3.2,
      height=1.4,
      text_lines=["FS Index", "labels → payload"],
  )
  add_arrow(ax, (-3, 1.3), (-3, 0.9), text="vector lookup", text_offset=(-0.1, 0.25))
  add_arrow(
      ax,
      (-3, -0.3),
      (0, -1.3),
      text="re-ranked distances",
      text_offset=(-0.7, -0.1),
  )

  ax.text(2.5, 3.2, "Unified Single-Layer Indexing", fontsize=12, fontweight="bold")
  add_box(
      ax,
      center=(2.5, 1.2),
      width=3.7,
      height=2.5,
      text_lines=["Enhanced ANN Index", "vector + label", "direct payload access"],
  )
  add_arrow(
      ax,
      (2.5, -0.1),
      (0.2, -1.4),
      text="direct read",
      text_offset=(0.8, -0.2),
  )

  add_box(
      ax,
      center=(0, -2.3),
      width=4.4,
      height=1.4,
      text_lines=["Top-K Results", "labels + metadata returned to caller"],
      fontsize=13,
  )

  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"Wrote diagram to {output_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Plot the ANN bottom-layer search that returns Top-K results."
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      default=pathlib.Path("ann_topk_diagram.png"),
      help="Path of the image file to write (PNG).",
  )
  parser.add_argument(
      "--show",
      action="store_true",
      help="Display the figure interactively in addition to saving it.",
  )
  args = parser.parse_args()

  draw_diagram(args.output)
  if args.show:
    plt.show()


if __name__ == "__main__":
  main()
