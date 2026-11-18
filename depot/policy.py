"""Per-slot scoring policy for MaskablePPO.

Every action gets its logit from one small network shared by all positions: a placement is scored
from its stack's features, a retrieval from its slot's features, both with a summary of the whole
yard as context. Learning "the oldest accessible container near the crane" is then the same
comparison at every slot, instead of separate output weights per slot as in a flat MLP.

Works on the raw (unnormalised) observation of DepotEnv; the yard dimensions come from
policy_kwargs (`yard=(B, Ba, R, H)`, `max_dist`). With `max_age` (the release deadline in days)
each slot and stack also gets an overdue flag and its distance to the deadline: on the log-age
scale alone the deadline is a ~1 % difference in the input.
"""
import torch as th
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

AGE_SCALE = 6.0          # log(1 + 365 days) ~ 5.9


def _mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def _age(x):
    return th.log1p(th.clamp(x, min=0.0)) / AGE_SCALE


class SlotNetwork(nn.Module):
    """Actor: per-action scores (logits). Critic: value features from a yard summary."""

    N_GLOBAL = 9

    def __init__(self, yard, max_dist, hidden=64, context=64, value_hidden=256, max_age=None):
        super().__init__()
        self.B, self.Ba, self.R, self.H = yard
        self.max_dist = float(max_dist)
        self.max_age = None if max_age is None else float(max_age)
        deadline = 2 if self.max_age else 0
        self.N_STACK = 10 + deadline        # per-stack features
        self.N_SLOT = 4 + deadline          # per-slot features (added to the stack's)
        self.n_cells = self.B * self.Ba * self.R * self.H
        self.n_stacks = self.B * self.Ba * self.R
        self.context = nn.Sequential(nn.Linear(self.N_GLOBAL + 2 * self.N_STACK, context), nn.ReLU())
        self.place_score = _mlp(self.N_STACK + context, hidden, 1)
        self.retrieve_score = _mlp(self.N_STACK + self.N_SLOT + context, hidden, 1)
        self.value_net = nn.Sequential(nn.Linear(context + 2 * self.N_STACK, value_hidden), nn.ReLU(),
                                       nn.Linear(value_hidden, value_hidden), nn.ReLU())
        self.latent_dim_pi = self.n_stacks + self.n_cells
        self.latent_dim_vf = value_hidden

    def _parse(self, obs):
        """Split the flat observation into yard tensors (layout of DepotEnv._get_obs)."""
        n, B, Ba, R, H = obs.shape[0], self.B, self.Ba, self.R, self.H
        c, s = self.n_cells, self.n_stacks
        occ = obs[:, :c].view(n, B, Ba, R, H)
        dwell = obs[:, c:2 * c].view(n, B, Ba, R, H)
        o = 2 * c
        height, top, smax, dist = (obs[:, o + i * s:o + (i + 1) * s].view(n, B, Ba, R) for i in range(4))
        sc = obs[:, o + 4 * s:o + 4 * s + 8]             # mode, tick, crane(3), occ_ratio, mean, std
        block_max = obs[:, o + 4 * s + 8:o + 4 * s + 8 + B]
        return occ, dwell, height, top, smax, dist, sc, block_max

    def _deadline(self, age, present):
        """[overdue flag, (age - deadline) / deadline in [-1, 1]] where a container is present;
        nothing without a deadline."""
        if not self.max_age:
            return []
        p = present.float()
        return [(age > self.max_age).float() * p,
                th.clamp((age - self.max_age) / self.max_age, -1.0, 1.0) * p]

    def _features(self, obs):
        occ, dwell, height, top, smax, dist, sc, block_max = self._parse(obs)
        n, B, Ba, R, H = obs.shape[0], self.B, self.Ba, self.R, self.H
        h = height / H
        # In-block neighbour heights (up/down/left/right); -1 marks no neighbour.
        pad = nn.functional.pad(h, (1, 1, 1, 1), value=-1.0)
        nb = th.stack([pad[:, :, :-2, 1:-1], pad[:, :, 2:, 1:-1],
                       pad[:, :, 1:-1, :-2], pad[:, :, 1:-1, 2:]], dim=-1)
        nb_real = (nb >= 0).float()
        nb_mean = (nb * nb_real).sum(-1) / nb_real.sum(-1).clamp(min=1)
        nb_max = th.where(nb_real > 0, nb, th.zeros_like(nb)).amax(-1)
        stack = th.stack([
            h, (height >= H).float(), (height == 0).float(), _age(top), _age(smax),
            dist / self.max_dist, nb_mean, nb_max,
            _age(block_max).view(n, B, 1, 1).expand(n, B, Ba, R),
            h - nb_mean,
            *self._deadline(smax, height > 0),
        ], dim=-1)                                                       # (n, B, Ba, R, N_STACK)

        tier = th.arange(H, device=obs.device, dtype=obs.dtype).view(1, 1, 1, 1, H)
        above = (height.unsqueeze(-1) - 1 - tier).clamp(min=0) * occ     # containers above the slot
        slot = th.stack([occ, _age(dwell), above / H, tier.expand_as(occ) / H,
                         *self._deadline(dwell, occ > 0)], dim=-1)

        yard = th.cat([stack.mean(dim=(1, 2, 3)), stack.amax(dim=(1, 2, 3))], dim=-1)
        glob = th.stack([sc[:, 0], sc[:, 1] / 30.0, sc[:, 2] / max(B - 1, 1), sc[:, 3] / max(Ba - 1, 1),
                         sc[:, 4] / max(R - 1, 1), sc[:, 5], _age(sc[:, 6]), _age(sc[:, 7]),
                         _age(block_max).mean(-1)], dim=-1)
        return stack, slot, yard, glob

    def _actor(self, stack, slot, ctx):
        n = stack.shape[0]
        ctx_s = ctx.view(n, 1, 1, 1, -1).expand(*stack.shape[:4], -1)
        stack_ctx = th.cat([stack, ctx_s], dim=-1)
        place = self.place_score(stack_ctx).view(n, -1)
        # The retrieval head's first layer is linear in [slot, stack, context], so the stack and
        # context part is computed once per stack and shared by its slots (same function as
        # applying the full layer per slot, without repeating that work for every tier).
        first, rest = self.retrieve_score[0], self.retrieve_score[1:]
        w_slot, w_stack = first.weight[:, :self.N_SLOT], first.weight[:, self.N_SLOT:]
        per_stack = nn.functional.linear(stack_ctx, w_stack, first.bias).unsqueeze(4)
        hidden = nn.functional.linear(slot, w_slot) + per_stack
        retrieve = rest(hidden).view(n, -1)
        return th.cat([place, retrieve], dim=1)          # action order of DepotEnv

    def _critic(self, ctx, yard):
        return self.value_net(th.cat([ctx, yard], dim=-1))

    def forward_actor(self, obs):
        stack, slot, yard, glob = self._features(obs)
        return self._actor(stack, slot, self.context(th.cat([glob, yard], dim=-1)))

    def forward_critic(self, obs):
        _, _, yard, glob = self._features(obs)
        return self._critic(self.context(th.cat([glob, yard], dim=-1)), yard)

    def forward(self, obs):
        stack, slot, yard, glob = self._features(obs)
        ctx = self.context(th.cat([glob, yard], dim=-1))
        return self._actor(stack, slot, ctx), self._critic(ctx, yard)


class SlotPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose logits are the SlotNetwork scores."""

    def __init__(self, *args, yard=None, max_dist=None, slot_hidden=64, max_age=None, **kwargs):
        self.yard, self.max_dist, self.slot_hidden = tuple(yard), max_dist, slot_hidden
        self.max_age = max_age
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SlotNetwork(self.yard, self.max_dist, hidden=self.slot_hidden,
                                         max_age=self.max_age)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # The scores already are the logits; drop the default linear head and rebuild the
        # optimizer so it only holds parameters that are used.
        self.action_net = nn.Identity()
        for head in (self.mlp_extractor.place_score, self.mlp_extractor.retrieve_score):
            nn.init.orthogonal_(head[-1].weight, gain=0.01)   # near-uniform initial policy
            nn.init.zeros_(head[-1].bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(yard=self.yard, max_dist=self.max_dist, slot_hidden=self.slot_hidden,
                    max_age=self.max_age)
        return data
