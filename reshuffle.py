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

    Search scope: same block first, then adjacent blocks if no candidate found.
    """

    def __init__(self, env):
        self.env = env

    # ─── Public API ─────────────────────────────────────────────────

    def relocate_above(self, bl: int, ba: int, r: int, target_h: int) -> list:
        """
        Move all containers above target_h in stack (bl, ba, r) elsewhere.
        Returns the per-relocation block distance |dest_block - src_block|
        (0 = same block, 1 = adjacent, >=2 = farther) for scope-scaled penalties.
        The number of relocations is simply len(result).
        """
        block_dists = []
        # Process top-down: move highest tier first
        h = self.env._stack_height(bl, ba, r) - 1
        while h > target_h:
            dwell = int(self.env.grid[bl, ba, r, h])
            dest = self._find_destination(bl, ba, r, dwell)
            self._move_container(bl, ba, r, h, dest, dwell)
            block_dists.append(abs(dest[0] - bl))
            h -= 1
        return block_dists

    # ─── Destination Selection ──────────────────────────────────────

    def _find_destination(self, src_bl, src_ba, src_r, dwell):
        # Step 1: same block
        dest = self._search_block(src_bl, src_bl, src_ba, src_r, dwell)
        if dest is not None:
            return dest

        # Step 2: adjacent blocks
        for adj_bl in self._adjacent_blocks(src_bl):
            dest = self._search_block(adj_bl, src_bl, src_ba, src_r, dwell)
            if dest is not None:
                return dest

        # Step 3: fallback
        return self._least_harmful_stack(src_bl, src_ba, src_r)

    def _search_block(self, bl, src_bl, src_ba, src_r, dwell):
        """
        Search within block `bl` for a valid destination.
        Skips the source stack (src_bl, src_ba, src_r) if it falls within this block.
        """
        candidates = []
        for ba in range(self.env.Ba):
            for r in range(self.env.R):
                # Skip the source stack
                if bl == src_bl and ba == src_ba and r == src_r:
                    continue

                h = self.env._stack_height(bl, ba, r)
                if h >= self.env.H:
                    continue

                max_dwell = self._stack_max_dwell(bl, ba, r)
                if max_dwell < dwell:
                    candidates.append(((bl, ba, r), max_dwell))

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x[1])
        return best[0]

    def _least_harmful_stack(self, src_bl, src_ba, src_r):
        """
        Fallback when no valid candidate exists anywhere.
        Pick stack with lowest max_dwell across whole yard.
        """
        best_stack = None
        best_max = float("inf")
        for bl in range(self.env.B):
            for ba in range(self.env.Ba):
                for r in range(self.env.R):
                    if bl == src_bl and ba == src_ba and r == src_r:
                        continue
                    h = self.env._stack_height(bl, ba, r)
                    if h >= self.env.H:
                        continue
                    max_dwell = self._stack_max_dwell(bl, ba, r)
                    if max_dwell < best_max:
                        best_max = max_dwell
                        best_stack = (bl, ba, r)

        if best_stack is None:
            raise RuntimeError("No valid reshuffle destination — yard is full")
        return best_stack

    # ─── Helpers ────────────────────────────────────────────────────

    def _stack_max_dwell(self, bl, ba, r):
        """Max dwell time in a stack, or -inf if empty."""
        stack = self.env.grid[bl, ba, r, :]
        occupied = stack[stack >= 0]
        if len(occupied) == 0:
            return float("-inf")
        return int(occupied.max())

    def _adjacent_blocks(self, bl):
        """Return block indices adjacent to bl (bl-1 and bl+1 if valid)."""
        adj = []
        if bl > 0:
            adj.append(bl - 1)
        if bl < self.env.B - 1:
            adj.append(bl + 1)
        return adj

    def _move_container(self, src_bl, src_ba, src_r, src_h, dest, dwell):
        """Physically move container in the grid."""
        dest_bl, dest_ba, dest_r = dest
        dest_h = self.env._stack_height(dest_bl, dest_ba, dest_r)
        assert dest_h < self.env.H, f"Destination {dest} is full"

        self.env.grid[src_bl, src_ba, src_r, src_h] = -1
        self.env.grid[dest_bl, dest_ba, dest_r, dest_h] = dwell