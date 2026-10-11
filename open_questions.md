# Open Questions

Items still undecided in `game_rules.md`, each with its context and the options on the table. Decisions reached so far are listed at the end.

## 1. Can units move more than once per turn?

**Current text:** A turn may include any number of transfers, at any cells and in any order. Nothing stops the same units from crossing again, so a group can travel A → B → C in one turn, paying at each crossing.

**Provisional:** allowed, as written.

**Why it matters:** with free crossings, a large group could sweep across the board in a single turn, which makes fronts very fluid and makes free crossings (H = 0) much more valuable than other costs.

**Options:**
- Allow it, as written.
- Limit each unit to one crossing per turn.
- Allow a fixed number of crossings per unit per turn.

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

## 4. Starting setup without the bid

The first training runs skip the secret bid (see `development_plan.md`, Phase 0 item 6), so they need a fixed starting position.

**To decide:** units per player, how many starting cells each player gets and how they are chosen, which board sizes to use for which player counts, and the number of rounds (the plan assumes 6).

## 5. Does the game still end early when one active player remains?

**Current text:** "The game also ends early if one player is the last one remaining."

**Gap:** Inactive players now keep their beacons and keep scoring every round, so an inactive player can still win on tally. Ending the game as soon as one active player remains cuts that short.

**Options:**
- Play every round and compare tallies at the end.
- End early and compare tallies at that point.
- End early and award the win to the last active player.

## 6. Inactivity rule (provisional)

**Decided for now:** a player is inactive only when they hold no units in any cell store after the harvest phase. Their beacons stay, keep scoring, and can be captured, and their tally stands. To be revisited after playtests.

## Decided on 2026-10-10

- Transfer cost is a rate: moving _n_ units costs ceil(_n_ / _m_) × _p_.
- A turn uses at most two move groups, one after the other with no going back. The groups are units (transfer and destroy), shells (shift and modify), beacons, and payments. The grouping is provisional and may change; the engine has it as a setting.
- Shell nodes point at the cell's corners (the beacon positions), and pair across a shared edge by reflection: node _d+k_ of A with node _d+2−k_ of B.
- Any player with units in a cell may modify its shell for 1 unit, with no dominance or beacon requirement. (Was open question 1.)
- Where shell units come from does not affect play: the unit paid is discarded and a neutral unit from an unlimited supply fills the node.
- Starting shells are part of the map: set by a template or drawn at random.
- Dominance is checked at the moment of each move.
- A beacon is a marker and is not paid for; the cost is the units from the adjacent stores.
- Paying into another player's coffer has no extra fee, but the units count as expended.
- Harvest income is computed from the board at the start of harvest and applied at once.
- The active phase ends after one cost-free turn or pass by each active player in a row; any costly turn restarts the count.

## Suggested way to close these

Questions 2 to 4 are balance and bookkeeping choices that self-play or a few human playtests can settle. Questions 1 and 5 change how the game plays, so they are worth deciding first.
