Development Plan: Rules Engine and Self-Play
===

This plan covers the work between the current rule set (`game_rules.md`, `open_questions.md`) and anything physical or visual. The goal is to use self-play agents to stress the rules, find balanced defaults, and discover whether distinct classes of strategy emerge, before committing to a board, a map builder, or a client.

The agents are playtesters, not the product. A rule set is only worth keeping if it makes a game that is interesting for people: real choices on every turn, enough depth that skill keeps paying off, several ways to win, and a game that fits in about an hour. Every experiment reports against the design scorecard below, and the output of the whole plan is a shortlist of rule sets for human playtests.

The first pass is 2-player. Three to six players come after the 2-player game is settled.

Target scope for the experiments:

| Parameter | Range |
| --------- | ----- |
| Players | 2 first, then 3 to 6 |
| Rounds | 6, with beacons and every rule in `game_rules.md` |
| Human game length | about 60 minutes for 6 rounds, so about 10 minutes per round |
| Board size | up to 256 cells, from templates or the map generator |
| Boards | generated boards with balanced start areas (see Board generator), plus hand-picked templates |

# Design scorecard

Every rule variant is trained with the same budget and scored on the same card. Variants that score badly are dropped early, and more compute goes to the promising ones.

| Question | Measure |
| -------- | ------- |
| Are the choices real? | Effective number of good options per decision: the exponential of the entropy of a strong agent's search policy. Near 1 means every move is obvious; very large means good and bad moves are hard to tell apart. Reported per micro-action and per whole turn, since a person plans a whole turn. |
| Do the choices matter? | Win probability lost by taking the second- or third-best option instead of the best. |
| Depth | Length of the skill ladder: agents trained in sequence, each beating the last about 75% of the time, counted until improvement stalls. Also whether more search per decision keeps raising the win rate. |
| Is a simple strategy enough? | Win rate of the random and greedy bots against the strongest agent. |
| Does every rule earn its place? | Restricted play: the win rate an agent loses when it may not use one mechanic (destroy, pay, shell ops, beacons). Near zero means a dead rule; near total means a dominant one. |
| Is it tense? | Lead changes in tally and predicted outcome, how often the player behind at round 3 wins, and the round by which the outcome becomes predictable. |
| Is it fair? | Win rate by seat and draw rate. Every generated board is played twice with the seats swapped, so board asymmetry cancels out of the seat measure. |
| Can a person play it? | Micro-actions per turn, turns per round, and estimated minutes per round. The estimate converts action counts with an assumed time per action, calibrated against the first human playtest. Target: about 10 minutes per round. |
| Are there several ways to win? | Distinct playing styles among the strong agents (Phase 4). Several competitive styles is the goal; one style that beats all others is a warning. |

# Phase 0: Lock the rules

The engine needs an answer (or a config flag) for every item below before any training run.

| # | Item | Proposed default | Config flag |
| - | ---- | ---------------- | ----------- |
| 1 | **Inactivity rule.** Harvest income goes into cell stores, and coffers only fill when a rival pays in (Move 5). As written, any player nobody paid has an empty coffer, places nothing during harvest, and becomes inactive. | **Decided 2026-10-10 (provisional):** a player is inactive only when they hold no units anywhere on the board after the harvest phase. Their beacons and tally stay. | `inactivity_rule` |
| 2 | **Beacon gate on shell modification.** Greedy 6-player runs built about 0.8 beacons per game, so a beacon gate would make modify nearly unavailable. | **Decided 2026-10-10:** no beacon gate. Anyone with units in the cell may modify for 1 unit. | `modify_requires_beacon` |
| 3 | **Beacon build cost** (open question 2). | Leave configurable, settle with Phase 2 sweeps. Capture stays at 1 per adjacent store. | `beacon_build_cost`, `beacon_capture_cost` |
| 4 | **Turn-order tie-breaks** (open question 3). | Fall back to the setup tie-break, then a seeded random draw. | `tiebreak_mode` |
| 5 | **Shift semantics.** The rules allow one shift move to rotate a shell any number of positions. | One shift move rotates by any k in 1 to 5, free for the dominant player, 1 unit otherwise. | `shift_cost_nondominant` |
| 6 | **Setup bid.** The bid is secret and simultaneous, which is hidden information that AlphaZero-style search does not handle. | Training v1 skips the bid and uses templated or seeded random setups. The bid returns in a later phase. | `setup_mode` |
| 7 | **Active phase length.** The phase ends once every player has consecutively had a cost-free turn or passed. That terminates, but a training run still needs a safety cap. | Hard cap on turns per round. Games that hit it are logged as anomalies. | `max_turns_per_round` |
| 8 | **Chained transfers** (open question 1). As written, units can move more than once per turn. | Allowed for now. A flag limits each unit to one crossing per turn for comparison. | `multi_transfer` |
| 9 | **Early end** (open question 5). Inactive players keep scoring, so ending the game when one active player remains can cut their chances. | Play every round; compare tallies at the end. | `early_end` |

# Phase 1: Rules engine

One deterministic engine serves the bots, the training loop, and later the game client, so balance findings carry straight into the shipped game.

**Language.** JAX is the recommended choice for the training engine. Fixed-shape, padded arrays let thousands of games run in parallel on a GPU with `jit` and `vmap`, and the engine plugs into DeepMind's `mctx` search library. A Rust or TypeScript port for the client can be generated later and checked against the JAX engine with shared test vectors.

**Padded state (per game).**

| Field | Shape | Notes |
| ----- | ----- | ----- |
| Cell stores | `[MAX_CELLS, MAX_PLAYERS]` int | `MAX_CELLS = 256`, `MAX_PLAYERS = 6` |
| Shells | `[MAX_CELLS]` uint8 | 6-bit ring, bit i is the node at the corner between edges i and i+1 |
| Adjacency | `[MAX_CELLS, 6]` int | neighbor index per edge, -1 for none |
| Beacon positions | `[MAX_VERTICES, 3]` int | the 2 or 3 cells touching each shared vertex |
| Beacon owner | `[MAX_VERTICES]` int | -1 for empty |
| Coffers, tally | `[MAX_PLAYERS]` int | |
| Turn state | scalars | round, phase, current player, move types used this turn, consecutive cost-free count |

**Crossing costs.** The cost `p:m` across a shared edge depends only on the two shells and the edge direction, so it is precomputed once as a lookup table of 64 x 64 x 6 entries using the Hamming/GCD rule in `game_rules.md`.

**Action encoding.** A turn becomes a sequence of micro-actions followed by `end_turn` (or `pass`), and the legal-move mask enforces the two-move-type constraint, costs, and dominance:

- `transfer(cell, direction, amount_bucket)`
- `destroy(cell, target_player)`
- `shift(cell, k)` and `modify(cell, node)`
- `beacon(vertex)` (build or capture, resolved by the vertex state)
- `pay(cell, target_player, amount_bucket)`
- `end_turn`, `pass`

Amount buckets (for example 1, half, all) keep the action space near a few thousand entries at 256 cells.

**Board generator.** Plain hex boards by radius, generated boards, and a template format for hand-built maps. Generated boards follow these rules:

- **Not a plain hexagon.** The field can have gaps, holes, narrow passages, and walls, but it is always one connected group of cells.
- **Balanced start areas.** For 2 players, the two start areas are mirror images of each other (point symmetry through the board's center, with each mirrored shell turned 180 degrees so it plays the same). Cells, gaps, and shells near the starts are mirrored exactly.
- **Randomness grows with distance from the starts.** The chance that a cell, gap, or shell is mirrored falls off with its distance from the nearer start, so the middle of the field is the most varied.
- **Fairness checks.** A board is rejected if the two starts are too close or too far apart by path, or if one player is much closer to more of the board than the other. Settings control how sheltered a start area is (how many ways in), the share of gaps, and the number of narrow passages.
- **Shells.** Start-area shells are mirrored; the rest are drawn at random. Templates can set every shell by hand.

Every board is checked against the rules: one contiguous group, every cell touching at least one beacon position.

**Tests.** Unit tests for every worked example in `game_rules.md`, property tests (unit conservation through every move, tally only changes in harvest, crossing cost symmetry), and replay tests comparing the batched engine against a slow reference implementation.

# Phase 2: Baseline bots and balance sweeps

This phase runs on CPU and needs no GPU.

**Bots.**

1. Random legal play, as a smoke test.
2. The greedy agent from the April 2026 simulations, ported to the new engine.
3. A heuristic agent that uses the shell state graph (see the design note below) to route each shell toward the crossing costs it wants.

**Metrics logged per game.** Shell engagement (share of moves that are shift or modify), beacons built and captured, game length in turns and micro-actions, win rate by seat, units per cell over time, coffer payments, and how often the turn cap triggers. Every decision also logs its legal-move count, the agent's policy, the value of each option, and the predicted outcome, which feed the design scorecard.

**Sweeps.** Beacon build cost (1, 2, 3), the modify gate on and off, board size, and player count. These settle items 2 and 3 from Phase 0 before any network training starts.

# Phase 3: Self-play training (GPU)

**Network.** A graph neural network over two node types: cells (store per player, shell bits, dominance) and beacon positions (owner), with edges carrying the precomputed crossing cost features. The policy head scores actions per node and per edge, so one network handles any board shape up to 256 cells. The value head outputs one predicted final tally share per player.

**Search.** Gumbel AlphaZero through `mctx`, which stays strong with 16 to 64 simulations per decision. With more than two players, each node backs up a vector of values, one per player (max^n style).

**Curriculum.**

| Stage | Cells | Players |
| ----- | ----- | ------- |
| 1 | 19 | 2 |
| 2 | 37 | 2 to 3 |
| 3 | 61 | 3 to 4 |
| 4 | evaluate at 127 to 256, fine-tune where play degrades | 4 to 6 |

Larger boards and six players are evaluated first rather than trained from scratch, since every added player raises both cost and variance.

**Compute estimates.** These are order-of-magnitude figures to be replaced with measurements once the engine exists.

- About 1,000 decisions per 4-player game (6 rounds, several turns per round, several micro-actions per turn).
- At 16 to 64 simulations per decision, roughly 16k to 64k network evaluations per game.
- Self-play typically needs tens of thousands of games or more per stage.
- On a 2-core CPU with no GPU, that is weeks to months even on small boards. On one modern GPU with the batched engine, small-board stages are a matter of days.

# Phase 4: Strategy classes

- Train a population of agents (league style) with a diversity bonus, so the run does not collapse onto a single style.
- Describe each agent by its behavior metrics from Phase 2 and cluster them. Distinct clusters are candidate strategy classes.
- Keep a catalog of boards where the strongest agents still split their results, or where one class dominates. Those become the templates and the "known ideal" configurations for the map builder.
- Watch for multiplayer behavior the greedy agents never showed: spoiling, focusing the leader, shared occupation, coffer payments as bribes.

# Design note: the shell state graph

Code, layouts, and standalone pages for every graph are in `shell_graph/`.

Worked out on 2026-10-10 while exploring the video game UI.

- A hexagonal shell has 64 states. Single add or remove moves connect them as a 6-dimensional cube (192 edges, 6 per state).
- Rotation groups the states into 14 necklaces (orbit sizes 1, 6, 6, 6, 3, 6, 6, 6, 2, 6, 6, 3, 6, 1 by fill level).
- Complementing a shell (swapping open and filled) maps necklaces onto each other. The three-in-a-row and alternating necklaces map onto themselves, and the two mirror-image three-unit necklaces swap.
- For a dominant player, rotation is free, so routing between shell states collapses to the 14-necklace quotient, and the unit cost between two states is the minimum Hamming distance over rotations.
- In play, a player usually wants target crossing costs on particular edges more than a particular shell state. The set of states meeting that target is a set of nodes in this graph, and the cheapest route to the nearest one is a Dijkstra search with configurable rotation and unit costs. The same search serves the client's auto-routing and the Phase 2 heuristic bot.

# First steps on the training machine

1. Check the hardware: `nvidia-smi`, `nproc`, `free -h`, and the installed CUDA version.
2. Record the Phase 0 decisions in `game_rules.md` and close the matching items in `open_questions.md`.
3. Scaffold the JAX engine with the padded state, the crossing cost table, and the tests for the worked examples.
4. Port the greedy agent and run the first Phase 2 sweep at 37 and 61 cells.

# Milestones

- [x] Phase 0 decisions recorded in `game_rules.md` (open items are settings; see `open_questions.md`)
- [ ] Board generator with balanced start areas, gaps, and passages
- [ ] Engine passes rule tests and property tests
- [ ] Batched engine matches the reference engine on replayed games
- [ ] Random, greedy, and heuristic bots run on boards up to 256 cells
- [ ] Balance sweep report for beacon cost and the modify gate
- [ ] Self-play agent beats the greedy agent on 19 and 37 cells
- [ ] Curriculum reaches 61 cells with 3 to 4 players
- [ ] Evaluation at 256 cells with 6 players
- [ ] Strategy class clustering report
- [ ] Design scorecard for each surviving rule variant, and a shortlist for human playtests
