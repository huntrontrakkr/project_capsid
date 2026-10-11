"""Write out/boards.html: a page of generated boards to look at.

    uv run python scripts/board_gallery.py [--radius 4] [--count 12] [--seed 0]
"""
import argparse
from dataclasses import replace
from pathlib import Path

from capsid.board import GenParams, generate
from capsid.render import board_svg

ap = argparse.ArgumentParser()
ap.add_argument("--radius", type=int, default=4)
ap.add_argument("--count", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gap", type=float, default=GenParams.gap)
args = ap.parse_args()

params = replace(GenParams(), radius=args.radius, gap=args.gap)
size = 22 if args.radius <= 4 else 16
cards = []
for seed in range(args.seed, args.seed + args.count):
    b = generate(seed, params)
    i = b.info
    title = (f"seed {seed}: {i['cells']} cells, start path {i['start_path']}, "
             f"{i['chokepoints']} chokepoints, {i['entrances'][0]} ways into each start")
    cards.append(board_svg(b, size=size, title=title, show_disc=args.radius))
out = Path(__file__).resolve().parents[1] / "out" / "boards.html"
out.parent.mkdir(exist_ok=True)
out.write_text('<!doctype html><meta charset="utf-8"><title>Generated boards</title>'
               '<body style="margin:12px;background:#faf9f5;display:flex;flex-wrap:wrap;gap:14px">'
               '<p style="width:100%;font:13px system-ui;color:#5f5e5a;margin:0">Start cells are outlined in blue (player 1) '
               'and orange (player 2). Dashed outlines are gaps. Dots on each inner ring are the shell: filled nodes are '
               'occupied. Small diamonds are beacon positions.</p>' + "".join(cards) + "</body>")
print(out)
