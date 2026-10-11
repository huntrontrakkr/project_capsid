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

    @nn.compact
    def __call__(self, cells, glob, nbr, vcells, vcorner, mask):
        n = cells.shape[0]
        m = mask[:, None].astype(jnp.float32)
        count = jnp.maximum(m.sum(), 1.0)
        x = nn.relu(nn.Dense(self.hidden)(cells) + nn.Dense(self.hidden)(glob)[None, :]) * m
        for _ in range(self.layers):
            nb = jnp.where((nbr >= 0)[..., None], x[jnp.maximum(nbr, 0)], 0.0)          # [N, 6, H], edge order kept
            pooled = (x * m).sum(0) / count
            h = jnp.concatenate([x, nb.reshape(n, -1), jnp.broadcast_to(pooled, x.shape)], -1)
            h = nn.Dense(self.hidden)(nn.relu(nn.Dense(self.hidden)(nn.LayerNorm()(h))))
            x = (x + h) * m
        x = nn.LayerNorm()(x) * m
        per_cell = nn.Dense(CELL_ACTIONS)(x)
        corners = nn.Dense(6 * 32)(x).reshape(n, 6, 32)                                 # a feature per corner
        vx = jnp.where((vcells >= 0)[..., None], corners[jnp.maximum(vcells, 0), vcorner], 0.0).sum(1)
        beacon = nn.Dense(1)(nn.relu(nn.Dense(64)(vx)))[:, 0]
        pooled = jnp.concatenate([(x * m).sum(0) / count, jnp.where(m > 0, x, -1e9).max(0), glob])
        g = nn.relu(nn.Dense(self.hidden)(pooled))
        end = nn.Dense(1)(g)
        value = jnp.tanh(nn.Dense(1)(nn.relu(nn.Dense(self.hidden)(g))))[0]
        logits = jnp.concatenate([per_cell[:, :18].ravel(), per_cell[:, 18:20].ravel(), per_cell[:, 20:25].ravel(),
                                  per_cell[:, 25:].ravel(), beacon, end])
        return logits, value
