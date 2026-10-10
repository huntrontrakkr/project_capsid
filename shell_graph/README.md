# Shell state graph

The 64 states of a six-node hexagonal shell, joined by one-unit changes (the 6-cube, 192 edges) and grouped by rotation into 14 necklaces. Developed on 2026-10-10.

- `shell_graph.py`: graph definitions, every layout algorithm, the layout searches, and cheapest-path search. `python shell_graph.py` rebuilds `layouts.json`; `--search rows|arcs|compact|codex` reruns a search.
- `layouts.json`: states, necklaces, edges, and coordinates plus crossing counts for every layout.
- `pages/`: one standalone page per graph (open `pages/index.html`). `scripts/build_pages.py` regenerates pages 07 to 10.
- `scripts/exploration/`: the original scripts from the session, kept as written.

| Layout | Crossings |
| ------ | --------- |
| orbit_rows | 1,518 |
| twelve_gon | 1,002 |
| layered | 906 |
| necklace_rows | 1,516 |
| necklace_columns | 1,767 |
| stratified | 990 |
| stratified_arcs | 1,076 |
| arc_rings | 1,164 |
| compact_rings | 988 |
| codex (bowed edges) | 1,072 |
