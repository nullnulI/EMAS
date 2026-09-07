from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle as pkl
from pathlib import Path
from typing import Any

import numpy as np

VALID_RELATIONS = {"a on b", "b on a", "a in b", "b in a"}
RELATION_COLORS = {
    "a on b": "#2563eb",
    "b on a": "#1d4ed8",
    "a in b": "#059669",
    "b in a": "#047857",
}

CANVAS_W = 1600.0
CANVAS_H = 960.0
PADDING = 120.0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an interactive HTML scene graph from ConceptGraph cache outputs."
    )
    parser.add_argument("--cachedir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    return parser


def load_scene_map_nodes(cachedir: Path) -> list[dict[str, Any]]:
    from conceptgraph.slam.slam_classes import MapObjectList

    scene_map = MapObjectList()
    map_path = cachedir / "map" / "scene_map_cfslam_pruned.pkl.gz"
    with gzip.open(map_path, "rb") as f:
        scene_map.load_serializable(pkl.load(f))

    nodes = []
    for idx, segment in enumerate(scene_map):
        caption = segment.get("caption_dict", {})
        response = caption.get("response", {})
        center = np.round(segment["bbox"].center, 3).tolist()
        extent = np.round(segment["bbox"].extent, 3).tolist()
        nodes.append(
            {
                "id": idx,
                "label": response.get("object_tag", f"node {idx}"),
                "summary": response.get("summary", ""),
                "possible_tags": response.get("possible_tags", []),
                "bbox_center": center,
                "bbox_extent": extent,
                "captions": caption.get("captions", []),
            }
        )
    return nodes


def load_edges(cachedir: Path) -> list[dict[str, Any]]:
    relations_path = cachedir / "cfslam_object_relations.json"
    with open(relations_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    edges = []
    for relation in relations:
        relation_type = relation.get("object_relation", "FAIL")
        if relation_type not in VALID_RELATIONS:
            continue
        edges.append(
            {
                "source": relation["object1"]["id"],
                "target": relation["object2"]["id"],
                "relation": relation_type,
                "reason": relation.get("reason", ""),
                "object1": relation["object1"],
                "object2": relation["object2"],
                "color": RELATION_COLORS[relation_type],
            }
        )
    return edges


def project_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not nodes:
        return nodes

    xs = [node["bbox_center"][0] for node in nodes]
    zs = [node["bbox_center"][2] for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    width = max(max_x - min_x, 1e-6)
    height = max(max_z - min_z, 1e-6)
    usable_w = CANVAS_W - 2 * PADDING
    usable_h = CANVAS_H - 2 * PADDING

    for idx, node in enumerate(nodes):
        x = node["bbox_center"][0]
        z = node["bbox_center"][2]
        px = PADDING + ((x - min_x) / width) * usable_w if width > 1e-6 else CANVAS_W / 2
        py = PADDING + ((z - min_z) / height) * usable_h if height > 1e-6 else CANVAS_H / 2
        radius = round(12 + 2.4 * math.log1p(max(node["bbox_extent"])), 2)
        node["px"] = round(px, 2)
        node["py"] = round(CANVAS_H - py, 2)
        node["x"] = node["px"]
        node["y"] = node["py"]
        node["radius"] = radius
        node["color"] = f"hsl({(idx * 41) % 360} 58% 50%)"
    return nodes


def build_html(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], title: str) -> str:
    node_json = json.dumps(nodes, ensure_ascii=False)
    edge_json = json.dumps(edges, ensure_ascii=False)
    title_json = json.dumps(title, ensure_ascii=False)
    relation_colors_json = json.dumps(RELATION_COLORS, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scene Graph</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f4ef;
      --panel: rgba(255,255,255,0.92);
      --ink: #1f2937;
      --muted: #6b7280;
      --line: rgba(31,41,55,0.22);
      --edge: #334155;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #fffdf6 0%, var(--bg) 70%);
      color: var(--ink);
    }}
    .page {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(260px, 320px) 1fr;
      gap: 16px;
      padding: 16px;
    }}
    .sidebar {{
      background: var(--panel);
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 0;
    }}
    .sidebar h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      padding: 10px;
      border-radius: 8px;
      background: rgba(15, 118, 110, 0.08);
    }}
    .stat .k {{
      font-size: 12px;
      color: var(--muted);
    }}
    .stat .v {{
      font-size: 18px;
      font-weight: 600;
      margin-top: 4px;
    }}
    .legend {{
      display: grid;
      gap: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .controls button {{
      border: 1px solid rgba(31,41,55,0.12);
      background: white;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    .details {{
      border-top: 1px solid rgba(31,41,55,0.08);
      padding-top: 12px;
      min-height: 0;
    }}
    .details h2 {{
      margin: 0 0 10px 0;
      font-size: 14px;
    }}
    .details-card {{
      font-size: 13px;
      line-height: 1.5;
      color: var(--ink);
      display: grid;
      gap: 8px;
      max-height: 52vh;
      overflow: auto;
      padding-right: 4px;
    }}
    .graph-wrap {{
      position: relative;
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 8px;
      overflow: hidden;
      min-height: 70vh;
    }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .edge-hit {{ fill: none; stroke: transparent; stroke-width: 16; cursor: pointer; }}
    .edge-line {{ fill: none; stroke-width: 3.2; stroke-linecap: round; opacity: 0.9; }}
    .edge-label {{ font-size: 11px; fill: var(--muted); pointer-events: none; }}
    .node {{ cursor: grab; }}
    .node.dragging {{ cursor: grabbing; }}
    .node circle {{ stroke: white; stroke-width: 2.5; }}
    .node text {{ font-size: 12px; font-weight: 600; fill: var(--ink); paint-order: stroke; stroke: rgba(255,255,255,0.92); stroke-width: 4px; stroke-linejoin: round; }}
    .tooltip {{
      position: absolute;
      z-index: 5;
      pointer-events: none;
      max-width: 360px;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(17, 24, 39, 0.94);
      color: #f9fafb;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 20px 40px rgba(17, 24, 39, 0.18);
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 120ms ease, transform 120ms ease;
      white-space: normal;
    }}
    .tooltip.visible {{ opacity: 1; transform: translateY(0); }}
    @media (max-width: 960px) {{
      .page {{ grid-template-columns: 1fr; }}
      .graph-wrap {{ min-height: 62vh; }}
      .details-card {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <div>
        <h1 id="title"></h1>
      </div>
      <div class="meta">
        <div class="stat"><div class="k">Nodes</div><div class="v" id="node-count"></div></div>
        <div class="stat"><div class="k">Edges</div><div class="v" id="edge-count"></div></div>
      </div>
      <div>
        <div style="font-size:14px;font-weight:600;margin-bottom:8px;">Relation Legend</div>
        <div class="legend" id="legend"></div>
      </div>
      <div class="hint">Hover nodes or edges for details. Drag nodes to inspect crowded areas. The layout starts from projected 3D positions, then runs a light collision-aware relaxation pass to separate overlaps.</div>
      <div class="controls">
        <button id="relax-btn" type="button">Relax Layout</button>
        <button id="reset-btn" type="button">Reset Positions</button>
      </div>
      <div class="details">
        <h2>Selection</h2>
        <div class="details-card" id="details-card">Nothing selected yet.</div>
      </div>
    </aside>
    <main class="graph-wrap" id="graph-wrap">
      <svg id="graph" viewBox="0 0 {int(CANVAS_W)} {int(CANVAS_H)}" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="tooltip" id="tooltip"></div>
    </main>
  </div>
  <script>
    const nodes = {node_json};
    const edges = {edge_json};
    const title = {title_json};
    const relationColors = {relation_colors_json};
    const canvas = {{ width: {CANVAS_W}, height: {CANVAS_H}, pad: {PADDING} }};

    const titleEl = document.getElementById('title');
    const nodeCountEl = document.getElementById('node-count');
    const edgeCountEl = document.getElementById('edge-count');
    const legendEl = document.getElementById('legend');
    const detailsCard = document.getElementById('details-card');
    const svg = document.getElementById('graph');
    const tooltip = document.getElementById('tooltip');
    const graphWrap = document.getElementById('graph-wrap');
    const resetBtn = document.getElementById('reset-btn');
    const relaxBtn = document.getElementById('relax-btn');

    document.title = title;
    titleEl.textContent = title;
    nodeCountEl.textContent = String(nodes.length);
    edgeCountEl.textContent = String(edges.length);

    Object.entries(relationColors).forEach(([label, color]) => {{
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<span class="swatch" style="background:${{color}}"></span><span>${{label}}</span>`;
      legendEl.appendChild(item);
    }});

    const nodeMap = new Map(nodes.map((node) => [node.id, node]));
    nodes.forEach((node) => {{
      node.baseX = node.px;
      node.baseY = node.py;
      node.vx = 0;
      node.vy = 0;
      node.fx = null;
      node.fy = null;
    }});

    function escapeHtml(text) {{
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    function shortLabel(label) {{
      return label.length > 24 ? label.slice(0, 21) + '...' : label;
    }}

    function nodeDetails(node) {{
      return [
        `<strong>${{escapeHtml(node.label)}}</strong>`,
        `id: ${{node.id}}`,
        `center: [${{node.bbox_center.join(', ')}}]`,
        `extent: [${{node.bbox_extent.join(', ')}}]`,
        `summary: ${{escapeHtml(node.summary || '')}}`,
        `possible_tags: ${{escapeHtml((node.possible_tags || []).join(', '))}}`,
        `captions: ${{escapeHtml((node.captions || []).slice(0, 3).join(' | '))}}`
      ].join('<br>');
    }}

    function edgeDetails(edge) {{
      const a = nodeMap.get(edge.source);
      const b = nodeMap.get(edge.target);
      return [
        `<strong>${{escapeHtml(edge.relation)}}</strong>`,
        `source: ${{a ? `${{a.id}} - ${{a.label}}` : edge.source}}`,
        `target: ${{b ? `${{b.id}} - ${{b.label}}` : edge.target}}`,
        `reason: ${{escapeHtml(edge.reason || '')}}`
      ].join('<br>');
    }}

    function setDetails(html) {{
      detailsCard.innerHTML = html;
    }}

    function showTooltip(event, html) {{
      tooltip.innerHTML = html;
      tooltip.classList.add('visible');
      const bounds = graphWrap.getBoundingClientRect();
      const x = event.clientX - bounds.left + 12;
      const y = event.clientY - bounds.top + 12;
      tooltip.style.left = `${{Math.min(x, bounds.width - 280)}}px`;
      tooltip.style.top = `${{Math.min(y, bounds.height - 160)}}px`;
    }}

    function hideTooltip() {{
      tooltip.classList.remove('visible');
    }}

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}

    function norm(dx, dy) {{
      const d = Math.hypot(dx, dy) || 1e-6;
      return [dx / d, dy / d, d];
    }}

    function runRelaxation(iterations = 180) {{
      for (let step = 0; step < iterations; step += 1) {{
        for (const node of nodes) {{
          if (node.fx != null && node.fy != null) {{
            node.x = node.fx;
            node.y = node.fy;
            node.vx = 0;
            node.vy = 0;
            continue;
          }}
          const pull = 0.018;
          node.vx += (node.baseX - node.x) * pull;
          node.vy += (node.baseY - node.y) * pull;
          node.vx *= 0.86;
          node.vy *= 0.86;
        }}

        for (let i = 0; i < nodes.length; i += 1) {{
          for (let j = i + 1; j < nodes.length; j += 1) {{
            const a = nodes[i];
            const b = nodes[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            if (Math.abs(dx) < 1e-3 && Math.abs(dy) < 1e-3) {{
              dx = (Math.random() - 0.5) * 0.1;
              dy = (Math.random() - 0.5) * 0.1;
            }}
            const [ux, uy, dist] = norm(dx, dy);
            const minDist = a.radius + b.radius + 22;
            if (dist < minDist) {{
              const push = (minDist - dist) * 0.42;
              if (a.fx == null) {{
                a.vx -= ux * push;
                a.vy -= uy * push;
              }}
              if (b.fx == null) {{
                b.vx += ux * push;
                b.vy += uy * push;
              }}
            }} else if (dist < 180) {{
              const repel = (180 - dist) * 0.0016;
              if (a.fx == null) {{
                a.vx -= ux * repel;
                a.vy -= uy * repel;
              }}
              if (b.fx == null) {{
                b.vx += ux * repel;
                b.vy += uy * repel;
              }}
            }}
          }}
        }}

        for (const edge of edges) {{
          const a = nodeMap.get(edge.source);
          const b = nodeMap.get(edge.target);
          if (!a || !b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const [ux, uy, dist] = norm(dx, dy);
          const target = 120 + (a.radius + b.radius) * 1.6;
          const spring = (dist - target) * 0.012;
          if (a.fx == null) {{
            a.vx += ux * spring;
            a.vy += uy * spring;
          }}
          if (b.fx == null) {{
            b.vx -= ux * spring;
            b.vy -= uy * spring;
          }}
        }}

        for (const node of nodes) {{
          if (node.fx != null && node.fy != null) {{
            node.x = node.fx;
            node.y = node.fy;
            continue;
          }}
          node.x = clamp(node.x + node.vx, canvas.pad, canvas.width - canvas.pad);
          node.y = clamp(node.y + node.vy, canvas.pad, canvas.height - canvas.pad);
        }}
      }}
    }}

    function curvedPath(source, target, edgeIndex) {{
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const [ux, uy, dist] = norm(dx, dy);
      const startX = source.x + ux * (source.radius + 3);
      const startY = source.y + uy * (source.radius + 3);
      const endX = target.x - ux * (target.radius + 3);
      const endY = target.y - uy * (target.radius + 3);
      const perpX = -uy;
      const perpY = ux;
      const bend = Math.min(36, Math.max(16, dist * 0.08)) * (edgeIndex % 2 === 0 ? 1 : -1);
      const cx = (startX + endX) / 2 + perpX * bend;
      const cy = (startY + endY) / 2 + perpY * bend;
      return {{
        d: `M ${{startX}} ${{startY}} Q ${{cx}} ${{cy}} ${{endX}} ${{endY}}`,
        labelX: cx,
        labelY: cy - 4,
      }};
    }}

    const edgeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    const nodeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    edges.forEach((edge, edgeIndex) => {{
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('class', 'edge-line');
      path.setAttribute('stroke', edge.color);
      edgeLayer.appendChild(path);
      edge.pathEl = path;

      const hit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      hit.setAttribute('class', 'edge-hit');
      hit.addEventListener('mouseenter', (event) => showTooltip(event, edgeDetails(edge)));
      hit.addEventListener('mousemove', (event) => showTooltip(event, edgeDetails(edge)));
      hit.addEventListener('mouseleave', hideTooltip);
      hit.addEventListener('click', () => setDetails(edgeDetails(edge)));
      edgeLayer.appendChild(hit);
      edge.hitEl = hit;

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('class', 'edge-label');
      label.textContent = edge.relation;
      edgeLayer.appendChild(label);
      edge.labelEl = label;
      edge.edgeIndex = edgeIndex;
    }});

    function attachDrag(node, group) {{
      let dragging = false;
      function onMove(event) {{
        if (!dragging) return;
        const bounds = svg.getBoundingClientRect();
        const sx = canvas.width / bounds.width;
        const sy = canvas.height / bounds.height;
        node.fx = clamp((event.clientX - bounds.left) * sx, canvas.pad, canvas.width - canvas.pad);
        node.fy = clamp((event.clientY - bounds.top) * sy, canvas.pad, canvas.height - canvas.pad);
        node.x = node.fx;
        node.y = node.fy;
        render();
      }}
      function onUp() {{
        if (!dragging) return;
        dragging = false;
        group.classList.remove('dragging');
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      }}
      group.addEventListener('pointerdown', (event) => {{
        dragging = true;
        group.classList.add('dragging');
        group.setPointerCapture?.(event.pointerId);
        onMove(event);
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
      }});
    }}

    nodes.forEach((node) => {{
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'node');
      g.addEventListener('mouseenter', (event) => showTooltip(event, nodeDetails(node)));
      g.addEventListener('mousemove', (event) => showTooltip(event, nodeDetails(node)));
      g.addEventListener('mouseleave', hideTooltip);
      g.addEventListener('click', () => setDetails(nodeDetails(node)));

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('r', node.radius);
      circle.setAttribute('fill', node.color);
      g.appendChild(circle);
      node.circleEl = circle;

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('text-anchor', 'middle');
      text.textContent = shortLabel(node.label);
      g.appendChild(text);
      node.textEl = text;

      node.groupEl = g;
      nodeLayer.appendChild(g);
      attachDrag(node, g);
    }});

    function render() {{
      for (const edge of edges) {{
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) continue;
        const curve = curvedPath(source, target, edge.edgeIndex);
        edge.pathEl.setAttribute('d', curve.d);
        edge.hitEl.setAttribute('d', curve.d);
        edge.labelEl.setAttribute('x', curve.labelX);
        edge.labelEl.setAttribute('y', curve.labelY);
      }}

      for (const node of nodes) {{
        node.groupEl.setAttribute('transform', `translate(${{node.x}}, ${{node.y}})`);
        node.textEl.setAttribute('y', -node.radius - 10);
      }}
    }}

    function resetPositions() {{
      for (const node of nodes) {{
        node.x = node.baseX;
        node.y = node.baseY;
        node.vx = 0;
        node.vy = 0;
        node.fx = null;
        node.fy = null;
      }}
      runRelaxation(220);
      render();
    }}

    resetBtn.addEventListener('click', resetPositions);
    relaxBtn.addEventListener('click', () => {{
      runRelaxation(120);
      render();
    }});

    runRelaxation(260);
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = get_parser().parse_args()
    cachedir = Path(args.cachedir)
    output = Path(args.output) if args.output is not None else cachedir / "scenegraph_interactive.html"
    title = args.title or f"Scene Graph: {cachedir.name}"

    nodes = project_nodes(load_scene_map_nodes(cachedir))
    edges = load_edges(cachedir)
    output.write_text(build_html(nodes, edges, title), encoding="utf-8")
    print(f"Wrote interactive scenegraph HTML to {output}")


if __name__ == "__main__":
    main()
