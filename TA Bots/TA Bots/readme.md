# TA Agent Overview


## George

**Strategy:** Greedy — no minimax, no lookahead.

George makes decisions by checking a fixed priority list each turn: roll a carpet if one is available, search the rat if the expected value is high enough, prime in the best direction, walk toward the best priming opportunity, or search/move as a fallback. There is no tree search, every decision is a single-step greedy evaluation of the current board state.

**Rat tracking:** Bayesian belief over all 64 cells, updated with noise and distance sensors each turn. A search hit resets to the stationary prior; a miss zeroes only the searched cell (rather than a full reset). Note that the rat moves on every turn, but your agent only receives information on their turn. 

**Carpet logic:** George tries to extend short runs before rolling, checks whether the opponent is on the carpet, and switches to immediate roll in the late game (≤4 turns left). If search EV beats the carpet payout, it searches instead.

---

Both Alberts and Carrie have an average search depth of around 12-13 on the server. Albert is likely able to search deep thanks to its simple heuristic. Carrie's heuristic is slower, but its other optimizations allow it to search to around the same depth. 

## Albert Lite

**Strategy:** Pure alpha-beta minimax — no rat searching.

Albert Lite runs iterative-deepening alpha-beta minimax (depths 1–30, 4-second budget) with a simple point-difference heuristic and basic move ordering. 

**Heuristic:** `player_points − opponent_points`.

**Move ordering:** Long carpets first (longest roll first), then primes, then plain moves, single-cell carpets last.

---

## Albert

**Strategy:** Iterative-deepening alpha-beta minimax with rat search evaluated inside the tree.

Albert extends Albert Lite by integrating the rat search decision into minimax. When search EV > 0, the best rat cell is evaluated as a chance node at the minimax root alongside all legal game moves: the expected value is `p_hit * value(hit) + (1 − p_hit) * value(miss)`. value(hit) and value(miss) are determined via minimax off of a simulated board where our agent has 1 less turn (to account for the search action), and their points have been updated accordingly for search hit/miss. Minimax picks whichever option, move or search, produces the highest value at that depth.

We only evalute search actions at the root, since our beliefs will have changed significantly as we search deeper into the tree. This also reduces our branching factor which is nice.

---

## Carrie

**Strategy:** Albert's minimax with chess-engine optimizations for deeper search and a more advanced heuristic.

Carrie uses the same Bayesian tracker, the same search chance-node integration, and the same board-position heuristic structure as Albert, but adds several techniques to search significantly deeper within the same time budget:

- **Transposition table** — caches exact/lower/upper bounds keyed on board state; depth-preferred replacement means a shallower re-search never evicts a deeper result.
- **Killer move heuristic** — stores the two quiet moves per depth-from-root that most recently caused a beta cutoff; these are tried early in sibling nodes.
- **History heuristic** — quiet moves that cause cutoffs accumulate a score of `depth²`; used for move ordering across all nodes.
- **Quiescence search** — at depth 0, continues searching long carpet moves (roll ≥ 3) for up to 3 extra plies with stand-pat pruning, avoiding horizon-effect misjudgments on big rolls.
- **Null move pruning** — at depth ≥ 3, tries skipping the player's turn (R=2 reduction); if the result still exceeds beta, the subtree is pruned without a full search.

**Heuristic:**  Uses BFS from both workers to compute a board-control score: each traversable cell contributes a carpet-potential value weighted by `turns_left / (distance + 1)`. The player's and opponent's control scores are differenced and added to the raw point difference scaled by turns remaining.

The intuition for this is that a squares "potential" should be weighted by how far it is from the bot. Farther squares should have less potential. To take into account how many turns are also left in the game, we multiply by turns_left. Naturally, any squares that are further away than there are turns remaining, will have a scale factor <1. We multiply the points difference by turns_left to keep it in the same scale as potential. We treated points already scored as having the maximum possible weight (when distance is 0)
