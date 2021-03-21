Game Design Documentation - Mechanisms
===

The following is a design spec and preliminary rules for a untitled game currently under design. These are far from the final rules and may be obtuse but can be reasonably used to develop the game further. The reader will notice that the rules are flexible as well as is the general topology of the game. Truthfully, this will remain as open as possible to offer as much flexibility for game modes as well as to offer opportunities for optimization on templated and standard play modes.

# Objective

The objective of the game is to have had held historically the largest number of beacons by the end of each round once a certain number of rounds have passed or if you are the remaining player.

# Layout

## Units

Each player has associated units that act as both resources and area representation within the game, Each with an associated color.
There shall also be uncolored unassociated units that will appear in the shell within the cells. (see Shell).

## Game Grid

The grid is represented as a 2 dimensional grid usually of regular tiled polygons known as cells (triangles, squares, and hexagons for Euclidean spaces). Any sized cells can be used, however, with the limitation that all cell polygons be of _2ⁿk_ sides, where _k_ is the smallest used cell polygon side size and n ≥ 1. The curvature of the 2D plane may be variable as well as the topology. For the original game mode, these cells are Hexagons on a standard euclidean 2d plane.
The grid itself is amorphous but must consist of a singular contiguous group of adjacent cells. At every shared vertex of a cell, there shall be a beacon position. The layout of the game board can be set based on a template or randomly, with the only constraint being that every cell be connected to at least one beacon, and that the number of players is taken into consideration for the number of available cells. (design constraints.)

## Cells

Each cell shall consist of a store and a shell.

### Store

The cell store can contain units belonging to any player that collectively total to no more than some limit _L_.

### Shell

The shell shall consist of an n-ary ring where n is the maximum number of polygon sides of any of the given cells in the game (ex: for a game consisting of triangular [3 sides] and hexagonal [6 sides] cells, the size of the given ring will be 6 nodes). The shell itself shall be physically represented as a ring or wall surrounding the store. Each position of the shell (known as a node) an unassigned unit, nothing (empty), or can contain a player's unit (for some game modes).
The shell itself shall be positioned such that a node is centered on a corresponding edge of the cell, and shall be associated with the adjacent edge of the cell. The shell can be rotated clockwise and counterclockwise, but positions shall be discretized such that every node shall be positioned in the same spot of the corresponding note's position; i.e., rotations shall be conformationally represented by a bit shift operation on the positions of the nodes.

### Coffer

Each player shall have a Coffer/store from which any other may pay units into.

### Tally

A running score tally shall be kept for all players.

## Beacons

At every shared vertex of a cell, there shall be a beacon position. This position can be occupied by a player's unit or empty.

# Concepts

## Payment

As mentioned, the cell store represents both area of control and local spending power for the given cell. To pay, unit(s) are removed from a cell store and discarded from the game / returned to storage.

## Adjacency

One cell is considered adjacent to another if they share a common edge/side.

## Occupation

A player occupies any cell which cell's store contains a players units.

## Dominance

A player is considered dominant in any cell in which the cell's store contains a majority of their units. If a tie exist for the highest number of units, neither player is dominant of that cell

## Occupation/Vacancies in shell nodes.

If a unit of any type, whether player or unassigned, exist in a particular node, that node is considered _occupied_. Otherwise, if it is empty, it is considered _vacant_.

## Paying from the Cell Store

The player's capabilities are localized to each of the cells, and actions taken place from a cell are generally payed from the store of the cell unless otherwise noted.

## Cost and Cost-Free Moves

Most moves are at cost; i.e., to perform that move, one must expend and remove units from the board. There are two moves, however, that are at no cost to the player: passing and, on rare occasions, moving units between cells. A turn in which no units are expended for a player is considered a cost-free turn.

## Overburdened Cell Stores

There is a known limit on the amount any cell store can hold per game for all player units collectively. During a players turn, a cell may become overburdened, or exist in a state were the collective store count exceeds the limit placed on the cell. This will, however be corrected by the end of the turn by reducing the last player's count such that the store total is below the limit.

# Setup

Once a game board has been decided upon and the variables for the rules have been chosen, a bidding round takes place. Each player is given x units to distribute to themselves and other players' coffers. In secret, one shall select what amount each player, including oneself, receive. Once all players have made their choice, the totals for each player's coffer will be tallied and the players shall be ordered from least to most number of units. If any ties exist, the player with the least experience in the game shall go last. Failing that, the order between ties shall be determined randomly. An additional 1 (or more, depending on established rules) units will be placed into each players' coffer.

Starting with the first player in the list, one shall place any number of units from their coffers onto any cell's store on the board, provided that the cell store does not contain another player's units and the store maximum has not been reached for that cell, up to the number of cells divided by the number of players rounded down.

> max_cells_allowed = floor(total_cells / player_count)

Any remaining units in any coffers shall be discarded.

NB: Alternatively, set positions and amounts could be determined on a per template basis with similar rules for turn orders/selection.

# Gameplay

The game shall consist of two phases per round, beginning with active phase and ending with a harvest phase.

## Active Phase

During the active phase, the players take turns based on the current order. If it is the first round, the order follows that of the order decided during the initial bid. during the player's turn, they may commit as many moves with any cells in any order of only 2 types of the 6 following move types as well as passing or ending their turn:

1. Transfer units between cells
2. Destroy units
3. Shift shell
4. Modify shell
5. Create a beacon
6. Pay into another player's coffer

(See Active Phase Moves.) 

A player may temporarily exceed any cell's max limit during one's turn and overburden that store. At the end of the turn, however, a player must reduce any of ones pieces such that all stores are no longer overburdened.

The active phase continues up until every player has consecutively either had a cost-free turn or has passed their turn completely.


## Harvest Phase

 Each player shall be rewarded a number of points for their tally based on the number of active beacons held for themselfs.

 During this phase, every cell is evaluated and each player is rewarded units based on the following rules:

  1. If the player is dominant in that cell (has the most units in the cell), that player receives one (1) additional unit if the store limit has not been reached.

  2. Additionally, if the player is dominant in that cell, that player receives one (1) unit per each adjacency which the player occupies, if and up until the maximum limit of that store has been achieved.


 At this point, the coffer totals will be tallied once again. The orders for the next round shall be determined from least to highest, with ties won by who was first from a prior round.

 Any units from a players' coffer may be transferred to any cell store which one occupies already, up to the store limit.

 Any remaining units are discarded from the coffers. If a player failed to move any units onto the board during this phase, they are considered inactive and must remove the remaining pieces from any cell store.

 If a player has been made inactive at this point, the setup may allow for a buy in at this juncture, where a player is given n units to place in any cell store as they see fit, up to the limit of a cell store and regardless if the cell store is already occupied.

 

## Active Phase Moves

### 1. Transfer Units Between Cells

The player may be able to move between adjacent cells at a possible cost. This cost/move, marked in the notation _p:m_, (where _p_ is the amount of units to pay and _m_ is the limit of the units that can be moved) is dependant and calculated for a given adjacency through the comparison of the cell shells (to be discussed later). The cost _p_ must be paid in full from the player's units within the store of the cell which one wishes to move from before any the units are moved, up to the amount _m_ (ex: a player wishes to move from cell A to cell B with a given cost of 2:5. This means that a player must pay two units before moving __up to 5__ units. If the player currently has 5 units in Cell A's store, and wants to move as many to Cell B as possible, the player would pay the cost of 2 from the store first before moving the remaining 3.) 

If the cost for movement is 0:n, than any number of units may move freely to the adjacent cell. _This is considered a cost free move._ If the cost is n:0, than movement is blocked between one cell and the other, and no units may move at any cost.

#### Calculating Costs

```
NB: this is here for explanation purposes only. The actual calculation within the game app will be handled by the game engine. Additionally, some ideas are underway to create a simple circuit based method for any physical copies of this game. in the prototype of the game, this is tracked through use of a tracker that lied on all of the edges of the adjacent cells and was somewhat cumbersome, but workable.
```

The costs associated with adjacent moves between cells the are calculated by comparing across the shared edge between the cells the individual nodes of each of the shells in a reflective manner. Using this method, each node of the shell Cell A is paired up with a node of Cell B and a Hamming distance _H_ is calculated on vacancies vs. occupancies. That is: one by one the pairs of nodes are compared on similarity (whether occupied or not) and all the dissimilar pairs are counted. This, combined with the inverse Hamming distance _~H_ (Hamming distance - total shell size, or the number of similarities between pairs) is used to make a fraction of _H:~H_ which is then reduced by the greatest common denominator (i.e. a value _2:4_ is reduced to _1:2_). Thus:

```

H  = hamming(index_for_A_on_B >> shellA, index_for_B_on_A >> shellB) where >> is a circular shift
~H = Total - H
p  = H / gcd(H, ~H)
m  = ~H / gcd(H, ~H)

p:m
```

// TODO: Add lots of visualizations and examples to aid in understanding this. This may need a whole section for itself as it's core to the game.

### 2. Destroy Units

This action occurs locally to a store in question. A player may destroy one unit belonging to another player in the same store as a player's own units at a cost of one of their units in the store, or one for one. (Ex: player a has 5 of their units in a store with 3 units of another player's units. Player A removes a unit one for one twice from Player B's units, leaving 2 units for Player A and 1 unit for Player B)

### 3. Shift Shell

At a cost of one unit from the cell store in question (or more, depending on the game settings), the shell position may be rotated left or right (if the shell were represented as a vector, this would effectively be the same as a circular shift to the left or right.)

### 4. Modify a Shell

If a beacon occupied by an active player is adjacent to a cell, this cell may perform this action. For the cost of 1 unit from the corresponding cell store (or some value greater determined by the game setup) a player may either:

 1. Remove and discard a unit from any occupied node in the cell shell, or
 2. Add a unit to the cell shell.

### 5. Create A Beacon

A player may place one of their units in the position of a beacon if and only if they are the dominant player in all cells encompassing/adjacent to the beacon, regardless if it is occupied or not, at a cost of 3 units (or other) __from each adjacent store__.

### 6. Pay Into Another Player's Coffer

A player may place into any _other_ player's coffer from any store cell any amount of units. This shall be considered an at-cost move.

# End of Game

The game continues through as many rounds as decided. Once the final harvest has taken place, the winner is the player who has the highest tally/score. Ties may be determined by who has the highest amount of dominant cells.

