"""Policy and value network over the cells of a hex-disc board.

Each cell starts from its own features plus the global ones, then exchanges information with its
six neighbors for a few rounds (keeping track of which edge each neighbor is on, since costs and
shells depend on direction), with a pooled summary of the whole board mixed back in each round.
Per-cell heads score the cell's transfer, destroy, shift and modify actions; each beacon position
is scored from the corner-specific features of the cells around it; a pooled head scores ending
the turn and predicts the value for the player to move.
"""
import flax.linen as nn
import jax.numpy as jnp

CELL_ACTIONS = 18 + 2 + 5 + 6   # transfer (6 edges x 3 amounts), destroy, shift, modify


class Net(nn.Module):
    hidden: int = 128
    layers: int = 6
    dtype: object = jnp.float32     # computation dtype; parameters stay float32

    @nn.compact
    def __call__(self, cells, glob, nbr, vcells, vcorner, mask):
        n = cells.shape[0]
        D = lambda k: nn.Dense(k, dtype=self.dtype)
        LN = lambda: nn.LayerNorm(dtype=self.dtype)
        m = mask[:, None].astype(self.dtype)
        cells, glob = cells.astype(self.dtype), glob.astype(self.dtype)
        count = jnp.maximum(m.sum(), 1.0)
        x = nn.relu(D(self.hidden)(cells) + D(self.hidden)(glob)[None, :]) * m
        for _ in range(self.layers):
            nb = jnp.where((nbr >= 0)[..., None], x[jnp.maximum(nbr, 0)], 0.0)          # [N, 6, H], edge order kept
            pooled = (x * m).sum(0) / count
            h = jnp.concatenate([x, nb.reshape(n, -1), jnp.broadcast_to(pooled, x.shape)], -1)
            h = D(self.hidden)(nn.relu(D(self.hidden)(LN()(h))))
            x = (x + h) * m
        x = LN()(x) * m
        per_cell = D(CELL_ACTIONS)(x)
        corners = D(6 * 32)(x).reshape(n, 6, 32)                                        # a feature per corner
        vx = jnp.where((vcells >= 0)[..., None], corners[jnp.maximum(vcells, 0), vcorner], 0.0).sum(1)
        beacon = D(1)(nn.relu(D(64)(vx)))[:, 0]
        pooled = jnp.concatenate([(x * m).sum(0) / count, jnp.where(m > 0, x, -1e9).max(0), glob])
        g = nn.relu(D(self.hidden)(pooled))
        end = D(1)(g)
        value = jnp.tanh(D(1)(nn.relu(D(self.hidden)(g))).astype(jnp.float32))[0]
        logits = jnp.concatenate([per_cell[:, :18].ravel(), per_cell[:, 18:20].ravel(), per_cell[:, 20:25].ravel(),
                                  per_cell[:, 25:].ravel(), beacon, end])
        return logits.astype(jnp.float32), value
