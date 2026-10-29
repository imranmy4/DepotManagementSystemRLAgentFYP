import numpy as np


class MinMaxReshuffler:
    """
    MinMax reshuffle heuristic (Caserta et al., 2009 adapted).

    Given a container being moved, choose a destination stack such that:
      1. Prefer stacks where no container is more urgent than the one being moved.
         Among those, pick the one whose most urgent container is highest dwell
         (leaves emptier/cleaner stacks free).
      2. If no such stacks exist, pick the stack whose most urgent container
         has the lowest dwell (least harm).

    Search scope: same block first, then adjacent blocks (up/down/left/right on the
    block grid) if no candidate found.
    """

    def __init__(self, env):
        self.env = env

    # ─── Public API ─────────────────────────────────────────────────

    def relocate_above(self, bl: int, ba: int, r: int, target_h: int) -> list:
        """
        Move all containers above target_h in stack (bl, ba, r) elsewhere.
        Returns a list of tuples (block_dist, src_bl, src_ba, src_r, src_h,
        dest_bl, dest_ba, dest_r, dest_h) — one per relocated container.
        block_dist = Manhattan distance between source and destination blocks on
        the block grid (0 same, 1 adjacent, >=2 farther).
        The number of relocations is simply len(result).
        """
        moves = []
        # Process top-down: move highest tier first
        h = self.env._stack_height(bl, ba, r) - 1
        while h > target_h:
            dwell = int(self.env.grid[bl, ba, r, h])
            dest = self._find_destination(bl, ba, r, dwell)
            dest_h = self.env._stack_height(dest[0], dest[1], dest[2])
            self._move_container(bl, ba, r, h, dest, dwell)
            moves.append((int(self.env.block_dist(bl, dest[0])), bl, ba, r, h, dest[0], dest[1], dest[2], dest_h))
            h -= 1
        return moves

    # ─── Destination Selection ──────────────────────────────────────

    def _find_destination(self, src_bl, src_ba, src_r, dwell):
        # Per-stack summaries for the whole yard, computed once per relocation
        # (vectorised; the grid changes after every move so they can't be cached longer).
        open_, smax = self._stack_summaries(src_bl, src_ba, src_r)

        # Step 1: same block
        dest = self._search_block(src_bl, open_, smax, dwell)
        if dest is not None:
            return dest

        # Step 2: adjacent blocks
        for adj_bl in self._adjacent_blocks(src_bl):
            dest = self._search_block(adj_bl, open_, smax, dwell)
            if dest is not None:
                return dest

        # Step 3: fallback
        return self._least_harmful_stack(open_, smax)

    def _stack_summaries(self, src_bl, src_ba, src_r):
        """
        open_: (B, Ba, R) bool — stack has a free tier and is not the source stack.
        smax:  (B, Ba, R) float — max dwell in the stack, -inf if empty.
        """
        grid = self.env.grid
        heights = (grid >= 0).sum(axis=3)
        open_ = heights < self.env.H
        open_[src_bl, src_ba, src_r] = False
        # Empty slots are -1 and dwell >= 0, so the raw max is the oldest container.
        smax = grid.max(axis=3).astype(np.float64)
        smax[heights == 0] = -np.inf
        return open_, smax

    def _search_block(self, bl, open_, smax, dwell):
        """
        Search within block `bl` for a valid destination: an open stack with
        max_dwell < dwell. Among those, pick the highest max_dwell (first in
        (ba, r) order on ties). Returns None if no candidate.
        """
        block_smax = smax[bl].ravel()
        idx = np.flatnonzero(open_[bl].ravel() & (block_smax < dwell))
        if idx.size == 0:
            return None
        best = int(idx[np.argmax(block_smax[idx])])
        ba, r = divmod(best, self.env.R)
        return (bl, ba, r)

    def _least_harmful_stack(self, open_, smax):
        """
        Fallback when no valid candidate exists anywhere.
        Pick open stack with lowest max_dwell across whole yard (first in
        (bl, ba, r) order on ties).
        """
        idx = np.flatnonzero(open_.ravel())
        if idx.size == 0:
            raise RuntimeError("No valid reshuffle destination — yard is full")
        best = int(idx[np.argmin(smax.ravel()[idx])])
        bl, rem = divmod(best, self.env.Ba * self.env.R)
        ba, r = divmod(rem, self.env.R)
        return (bl, ba, r)

    # ─── Helpers ────────────────────────────────────────────────────

    def _adjacent_blocks(self, bl):
        """Return block indices sharing a side with bl on the block grid
        (up, down, left, right — whichever exist)."""
        rows, cols = self.env.config.block_rows, self.env.block_cols
        row, col = self.env.block_coords(bl)
        adj = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                adj.append(nr * cols + nc)
        return adj

    def _move_container(self, src_bl, src_ba, src_r, src_h, dest, dwell):
        """Physically move container in the grid."""
        dest_bl, dest_ba, dest_r = dest
        dest_h = self.env._stack_height(dest_bl, dest_ba, dest_r)
        assert dest_h < self.env.H, f"Destination {dest} is full"

        self.env.grid[src_bl, src_ba, src_r, src_h] = -1
        self.env.grid[dest_bl, dest_ba, dest_r, dest_h] = dwell