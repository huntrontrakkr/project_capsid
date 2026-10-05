# Open Questions

Items left undecided in the `adapted-rules` revision of `game_rules.md`. Each has the context from the April 2026 simulation work and the options on the table.

## 1. Does modifying a shell require an adjacent owned beacon?

**Original spec:** Modify was only available at a cell adjacent to a beacon the active player owned.

**April work:** Modification was set at 1 unit and bundled with rotation as shell ops. The beacon requirement was never decided in either direction.

**Options:**
- Keep the requirement. Beacons then gate shell control, which gives them a second use beyond scoring.
- Drop it and gate modification by dominance in the cell, matching rotation.
- Keep it as a game-mode setting.

**Why it matters:** Free rotation already makes shells central (57% of activity). The modify gate decides how much of that depends on holding beacons, and beacons are already hard to build.

## 2. Baseline beacon build cost: 1 or 2 units per adjacent store?

**April work:** The agreed rules say 1 to 2 per adjacent store, with capture at 1 per store. The 6-player test used a build cost of 3 per store.

**Findings that bear on it:**
- In the 6-player greedy runs, beacons were built about 0.8 times per game and captured about 0.1 times per game.
- The main bottleneck was holding dominance in all three cells around a vertex at once. Cost was secondary, and a cost of 1 per cell would still leave beacons hard to build.

**Options:**
- Set the baseline at 1 and rely on the dominance gate for difficulty.
- Set it at 2 so that building stays clearly pricier than capturing (capture is 1).
- Leave it as a per-game setting and pick the default after human playtests or self-play.

## 3. Turn-order tie-breaks

**Current text:** Turn order each round goes from least to most units in the coffer. Ties go to the player who was earlier in the prior round. The setup bid uses a different tie-break (least experienced player goes last, then random).

**Gap:** The text does not say what happens when the tie spans players who were all tied the prior round, or how this works in round one after setup.

**Options:**
- Fall back to the setup tie-break (least experience last, then random).
- Fall back to a fixed seating order.
- Use a random draw each time a tie is unresolved.

## Suggested way to close these

Questions 2 and 3 are balance and bookkeeping choices that self-play or a few human playtests can settle. Question 1 changes rules structure, so it is worth deciding first.
