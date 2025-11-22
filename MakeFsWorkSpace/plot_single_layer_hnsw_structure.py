#!/usr/bin/env python3
"""
Generate a single-layer HNSW-style search graph that highlights Top-K targets.

This script builds a dense random geometric graph to mimic the bottom layer of
an HNSW index. The nodes with the highest degree are marked as the Top-K
targets in red, while optional layout tweaks (spring + repulsion) keep the
highlights visible. A "plane" rendering mode draws the layer like a 3D slab and
projects Top-K nodes upward with red dashed lines, similar to HNSW diagrams in
papers and decks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path as PathlibPath
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import networkx as nx


def build_single_layer_graph(
    num_nodes: int, radius: float, seed: int
) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
  """Create a connected random geometric graph that mimics HNSW's base layer."""
  graph = nx.random_geometric_graph(num_nodes, radius, seed=seed)

  if not nx.is_connected(graph):
    components = list(nx.connected_components(graph))
    for idx in range(len(components) - 1):
      u = next(iter(components[idx]))
      v = next(iter(components[idx + 1]))
      graph.add_edge(u, v)

  return graph, nx.get_node_attributes(graph, "pos")


def select_topk_nodes(graph: nx.Graph, topk: int) -> Iterable[int]:
  """Choose the nodes with the highest degree to represent Top-K results."""
  return [
      node for node, _ in sorted(graph.degree, key=lambda pair: pair[1], reverse=True)[:topk]
  ]


def compute_layout(
    graph: nx.Graph,
    base_positions: Dict[int, Tuple[float, float]],
    layout: str,
    spacing: float,
    iterations: int,
    seed: int,
) -> Dict[int, Tuple[float, float]]:
  """Derive node positions while preserving the geometric feel."""
  if layout == "spring":
    return nx.spring_layout(
        graph,
        pos=base_positions,
        seed=seed,
        k=spacing,
        iterations=iterations,
    )
  return base_positions


def repel_nodes(
    positions: Dict[int, Tuple[float, float]],
    min_distance: float,
    iterations: int,
    step_scale: float,
) -> Dict[int, Tuple[float, float]]:
  """Push nodes apart to avoid overlap when highlighting Top-K targets."""
  for _ in range(iterations):
    moved = False
    for a, (ax, ay) in positions.items():
      dx, dy = 0.0, 0.0
      for b, (bx, by) in positions.items():
        if a == b:
          continue
        diff_x = ax - bx
        diff_y = ay - by
        dist = math.hypot(diff_x, diff_y)
        if dist < 1e-6:
          continue
        if dist < min_distance:
          strength = (min_distance - dist) / dist
          dx += diff_x * strength
          dy += diff_y * strength
      if dx or dy:
        moved = True
        positions[a] = (ax + dx * step_scale, ay + dy * step_scale)
    if not moved:
      break
  return positions


def plot_graph_style(
    graph: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    topk_nodes: Iterable[int],
    output_path: PathlibPath,
    figsize: Tuple[float, float],
) -> None:
  """Render a flat graph visualization."""
  plt.style.use("seaborn-v0_8-white")
  fig, ax = plt.subplots(figsize=figsize)

  nx.draw_networkx_edges(
      graph,
      positions,
      ax=ax,
      width=1.2,
      edge_color="#b3b3b3",
      alpha=0.7,
  )

  non_top_nodes = [node for node in graph.nodes if node not in topk_nodes]

  nx.draw_networkx_nodes(
      graph,
      positions,
      nodelist=non_top_nodes,
      node_size=190,
      node_color="#d9dadb",
      edgecolors="#7f8c8d",
      linewidths=0.4,
      ax=ax,
  )

  nx.draw_networkx_nodes(
      graph,
      positions,
      nodelist=topk_nodes,
      node_size=320,
      node_color="#f45a54",
      edgecolors="#9c1b1b",
      linewidths=0.8,
      ax=ax,
  )

  ax.set_title("HNSW Bottom-Layer Search with Top-K Targets", fontsize=14)
  ax.set_axis_off()
  fig.tight_layout()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=320, bbox_inches="tight")
  plt.close(fig)


def normalize_positions(
    positions: Dict[int, Tuple[float, float]]
) -> Dict[int, Tuple[float, float]]:
  xs = [pos[0] for pos in positions.values()]
  ys = [pos[1] for pos in positions.values()]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span_x = max(max_x - min_x, 1e-6)
  span_y = max(max_y - min_y, 1e-6)
  return {
      node: ((x - min_x) / span_x, (y - min_y) / span_y)
      for node, (x, y) in positions.items()
  }


def draw_plane_background(ax) -> Dict[str, Tuple[float, float]]:
  """Draw a stylized plane patch and return its corners."""
  corners = {
      "bl": (0.3, 0.1),
      "br": (5.0, 0.7),
      "tr": (5.7, 3.0),
      "tl": (0.9, 2.4),
  }
  path = MplPath(
      [
          corners["bl"],
          corners["br"],
          corners["tr"],
          corners["tl"],
          corners["bl"],
      ],
      [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY],
  )
  patch = PathPatch(
      path,
      facecolor="#fdfefe",
      edgecolor="#2c3034",
      linewidth=1.4,
      joinstyle="round",
  )
  ax.add_patch(patch)
  return corners


def interpolate_point(
    a: Tuple[float, float], b: Tuple[float, float], t: float
) -> Tuple[float, float]:
  return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def map_to_plane(
    normalized_positions: Dict[int, Tuple[float, float]],
    corners: Dict[str, Tuple[float, float]],
    padding: float = 0.08,
) -> Dict[int, Tuple[float, float]]:
  mapped = {}
  for node, (x, y) in normalized_positions.items():
    x = padding + (1 - 2 * padding) * x
    y = padding + (1 - 2 * padding) * y
    bottom = interpolate_point(corners["bl"], corners["br"], x)
    top = interpolate_point(corners["tl"], corners["tr"], x)
    mapped[node] = interpolate_point(bottom, top, y)
  return mapped


def plot_plane_style(
    graph: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    topk_nodes: Iterable[int],
    output_path: PathlibPath,
    figsize: Tuple[float, float],
) -> None:
  """Render the graph on a skewed plane with highlighted Top-K projections."""
  plt.style.use("seaborn-v0_8-white")
  fig, ax = plt.subplots(figsize=figsize)
  ax.set_xlim(-0.2, 6.2)
  ax.set_ylim(-0.1, 3.9)

  corners = draw_plane_background(ax)
  normalized = normalize_positions(positions)
  plane_positions = map_to_plane(normalized, corners)

  nx.draw_networkx_edges(
      graph,
      plane_positions,
      ax=ax,
      width=1.0,
      edge_color="#8f9aa3",
      alpha=0.8,
  )

  non_top_nodes = [node for node in graph.nodes if node not in topk_nodes]
  nx.draw_networkx_nodes(
      graph,
      plane_positions,
      nodelist=non_top_nodes,
      node_size=150,
      node_color="#d2d7dc",
      edgecolors="#5e656b",
      linewidths=0.4,
      ax=ax,
  )

  nx.draw_networkx_nodes(
      graph,
      plane_positions,
      nodelist=topk_nodes,
      node_size=240,
      node_color="#f25748",
      edgecolors="#8b2419",
      linewidths=0.9,
      ax=ax,
  )

  for node in topk_nodes:
    x, y = plane_positions[node]
    ax.plot(
        [x, x],
        [y, y + 0.9],
        linestyle=(0, (3, 3)),
        color="#f25748",
        linewidth=1.0,
    )

  ax.set_axis_off()
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=320, bbox_inches="tight")
  plt.close(fig)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generate a single-layer HNSW-style graph visualization."
  )
  parser.add_argument("--num-nodes", type=int, default=40)
  parser.add_argument("--radius", type=float, default=0.30)
  parser.add_argument("--topk", type=int, default=6)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--layout",
      choices=("spring", "geo"),
      default="spring",
      help="spring spreads nodes, geo keeps the random geometric positions",
  )
  parser.add_argument("--spring-spacing", type=float, default=0.35)
  parser.add_argument("--spring-iters", type=int, default=200)
  parser.add_argument(
      "--min-distance",
      type=float,
      default=0.08,
      help="Minimum distance enforced between nodes after layout.",
  )
  parser.add_argument("--repel-iters", type=int, default=8)
  parser.add_argument("--repel-step", type=float, default=0.08)
  parser.add_argument(
      "--style",
      choices=("graph", "plane"),
      default="plane",
      help="Render as a simple graph or as a plane-like slab visualization.",
  )
  parser.add_argument(
      "--figsize",
      type=float,
      nargs=2,
      default=(6.5, 6.5),
      metavar=("WIDTH", "HEIGHT"),
  )
  parser.add_argument(
      "--output",
      type=PathlibPath,
      default=PathlibPath("MakeFsWorkSpace/results/single_layer_hnsw_structure.png"),
      help="Path to save the generated figure.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  graph, positions = build_single_layer_graph(
      num_nodes=args.num_nodes, radius=args.radius, seed=args.seed
  )
  positions = compute_layout(
      graph=graph,
      base_positions=positions,
      layout=args.layout,
      spacing=args.spring_spacing,
      iterations=args.spring_iters,
      seed=args.seed,
  )
  positions = repel_nodes(
      positions=positions,
      min_distance=args.min_distance,
      iterations=args.repel_iters,
      step_scale=args.repel_step,
  )
  topk_nodes = select_topk_nodes(graph, args.topk)
  if args.style == "plane":
    plot_plane_style(
        graph=graph,
        positions=positions,
        topk_nodes=topk_nodes,
        output_path=args.output,
        figsize=tuple(args.figsize),
    )
  else:
    plot_graph_style(
        graph=graph,
        positions=positions,
        topk_nodes=topk_nodes,
        output_path=args.output,
        figsize=tuple(args.figsize),
    )
  print(f"Figure saved to {args.output.resolve()}")


if __name__ == "__main__":
  main()
