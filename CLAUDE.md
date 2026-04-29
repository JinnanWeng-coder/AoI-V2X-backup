# Project context for Claude — AoI-V2X RL research

This file is auto-loaded at session start. Read it first before doing anything.

## TL;DR

C-V2X resource allocation with MARL, extending Parvini et al. IEEE TVT 2023.
PhD student is building a paper around adding **Jain's fairness** to the global reward as a third metric (AoI + PRR + Jain).

**Current state (2026-04-29):** A `.clone().detach()` gradient bug was identified and fixed in two ways (direction-A, direction-B). **Both fixes empirically *hurt* performance** vs the original buggy code. The "fast-fading is the noise source" hypothesis was tested with a fading-off ablation (seed=3, main vs dir-B) and **was falsified**: rg_std barely changed (1.00× / 0.92×) and dir-B AoI got *worse*, not better (5.90 → 8.44). Fast fading is NOT the dominant noise source in `reward_global` — and turning it off actually degrades training (likely removes useful channel diversity). Option A of the paper plan (fading-averaged reward) is dead. Next step: diagnostic noise-source attribution on a frozen policy before any more training runs.

**User's stance:** "Bit-identical" finding (main+Jain = main no-Jain) is trivially true and NOT a paper hook — don't repackage it as a contribution. Real research question is *why* the fix degrades performance.

---

## Directory layout

```
X:\AoI-V2X-IEEE-TVT-2023-main-back\
├── 1-ModifiedMADDPGwithTDec\   ← active codebase (Algorithm 1)
│   ├── Main.py                 ← training loop
│   ├── global_critic.py        ← Global Critic (TD3 twin)
│   ├── local_critic.py         ← Per-agent local critics + actor
│   ├── Classes\
│   │   ├── Environment_Platoon.py   ← env, reward functions
│   │   ├── networks.py              ← Actor + local Critic NN
│   │   ├── G_network.py             ← Global critic NN
│   │   └── buffer.py                ← replay buffer
│   └── model\marl_model\       ← latest run output (overwritten each run)
├── 2-ModifiedMADDPG\, 3-MADDPGFDec\, 4-DDPG\   ← other algorithms (NOT the focus)
├── data\
│   ├── ngioe\                          ← lab machine experiment archive
│   │   ├── baseline-main\seedN          (no-Jain, has detach bug)
│   │   ├── direction-a\seedN            (no-Jain, fix variant A)
│   │   ├── direction-b\seedN            (no-Jain, fix variant B)
│   │   └── baseline-fixed_random_policy\
│   ├── ngioe-Jain\                     ← experiments after adding Jain to reward
│   │   ├── baseline-main\Add-Jain-AoI\seedN              (main + Jain, has detach bug)
│   │   ├── baseline-main\04271640seed3                   (main + Jain, fading ON, seed3)
│   │   ├── baseline-main\withoutfastfading\04282101seed3 (main + Jain, fading OFF, seed3)
│   │   ├── direction-b\Add-Jain-AoI\seedN                (B + Jain, fix variant B)
│   │   ├── direction-b\04252211seed3                     (dir-B + Jain, fading ON, seed3)
│   │   └── direction-b\withoutfastfading\04291341seed3   (dir-B + Jain, fading OFF, seed3)
│   └── 1-ModifiedMADDPGwithTDec\, 2-ModifiedMADDPG\   ← old archives (not relevant)
└── AoI-Aware_Resource_Allocation...pdf  ← original Parvini 2023 paper
```

## Git state

- Branches: `main`, `direction-a`, `direction-b`
- Currently checked out: `main` (verify with `git branch`)
- Recent commits relate to adding Jain reward + logging to .mat file

Key recent commits:
- `8b0b70c` — plot_results updated to show Jain
- `40a513b` — save Jain.mat + tensorboard logging
- `266a5b8` — add Jain AoI to reward formula

## Code modifications from original Parvini repo

Three modifications stack:

### 1. `.clone().detach()` bug fix (direction-A and direction-B branches only; main has the bug)

In original code (`global_critic.py`):
```python
actor_global_loss = -global_critic1(states, actions_)
for i in range(N):
    actor_global_loss_ = actor_global_loss.clone().detach()  # severs gradient
    agents[i].local_learn(actor_global_loss_, ...)
```

`.detach()` makes the global-critic loss a constant w.r.t. actor params → its gradient contribution to actor update is exactly zero. Global critic was effectively decorative.

**Direction-A fix:** centralize global actor-gradient computation inside `global_learn()`, single backward distributes grads to all 5 actors before any optimizer.step(). Adds `update_actor` flag to `local_learn`. Mathematically clean (Jacobi-style synchronous update).

**Direction-B fix:** keep cross-file structure; rebuild global loss per-agent with current agent's actor live and others detached. Sequential (Gauss-Seidel-style) update. Empirically more stable than A.

### 2. Logging fix (all branches)

`record_reward_global_[i_episode]` originally stored only the LAST step's value. Changed to per-episode mean to match `reward_t1` / `reward_t2` logging convention.

### 3. Jain's fairness in global reward (all branches; latest commits)

In `Environment_Platoon.py`:
```python
self.LAMBDA_JAIN = 0.3   # in __init__

def compute_jain_aoi(self):
    n = int(self.n_Veh / self.size_platoon)
    x = np.asarray(self.AoI, dtype=np.float64)
    return (np.sum(x))**2 / (n * np.sum(x**2) + 1e-12)

# In act_for_training:
global_reward = -np.mean((self.Interference_all + 60) / 60) \
                + self.LAMBDA_JAIN * self.compute_jain_aoi()
```

`Main.py` adds `Jain_total[n_episode]` buffer, `record_Jain[step]`, and saves `Jain.mat`.

## Scenario parameters (from Main.py)

- `n_platoon = 5`, `size_platoon = 4` → 5 agents, each with platoon leader + 3 followers
- `n_RB = 3`, `n_S = 2`, `max_power = 30 dBm`
- `n_episode = 500`, `n_step_per_episode = 100`
- Actor: 2-layer MLP (1024 → 512), tanh output (3D action: RB, mode, power)
- Local critic τ1, τ2: 2-layer MLP (512 → 256)
- Global critic: 3-layer MLP (1024 → 512 → 256), input = 5 × n_input + 5 × 3
- TD3 twin critics on global side, target-policy smoothing noise N(0, 0.2²)
- Optimizer: Adam, lr_actor = 1e-4, lr_critic = 1e-3, weight_decay 0.01 on critics
- Replay buffer: standard, batch_size 64
- `RENEW_POS_EVERY = 20` (every 20 episodes new platoon positions)
- `env.renew_channels_fastfading()` called every step in step loop (Main.py L244) — was the suspected noise source, **ruled out by 2026-04-29 ablation**

## Reward structure

Per agent (local):
- `task_1_r`: V2V demand completion penalty + power log penalty (depending on mode)
- `task_2_r`: V2I rate revenue (Revenue_function on C_rate vs V2I_min) + AoI penalty (-AoI/20)

Global (shared):
- `interference_term`: `-mean((Interference_all + 60) / 60)`, where `Interference_all` is per-platoon dBm
- `jain_term`: Jain's fairness over per-platoon AoI, in (0, 1]
- `global_reward = interference_term + 0.3 * jain_term` (current latest commit)

## Experiment results — key findings (the actual numerical truth)

All last-100 episode means, averaged across same 3 seeds (2, 3, 4):

| Group | reward_global | AoI mean | AoI p90 | Jain (per-ep) |
|---|---|---|---|---|
| **main no-Jain** (has detach bug) | 0.71 | **5.02** | **7.98** | **0.857** |
| main + Jain (has detach bug) | 0.91 | 5.02 | 7.98 | 0.857 |
| **direction-B no-Jain** (fixed) | 0.69 | **6.15** | 10.36 | 0.837 |
| **direction-B + Jain** (fixed + λ=0.3) | 0.81 | **10.52** | **20.34** | 0.697 |

Random-policy baseline (env-only test, no learning): reward ≈ 0.59, AoI ≈ 8
Fixed-policy baseline (RB=1 for all agents): reward ≈ 0.36, AoI ≈ 77 (collapse)

### Three core findings

1. **main + Jain is bit-identical to main no-Jain** in actions, AoI, V2I/V2V rates, demand, power, AoI_evolution — only `reward_global.mat` differs. This is *trivially explained* by the detach bug (Jain reward never reaches actor) and **is NOT a paper contribution per the user's explicit feedback**.

2. **Fix degrades performance** vs the buggy original. direction-B no-Jain has AoI 6.15 vs main no-Jain 5.02 (+22% degradation). This is the real anomaly to investigate.

3. **Adding Jain after fix degrades further**. direction-B + Jain has AoI 10.52, p90 20.34, and Jain *decreased* from 0.837 → 0.697 (the "subjective" pursuit is worse than the "passive" baseline). P2 becomes a persistent loser at AoI = 15.7.

### Working hypothesis (PARTIALLY FALSIFIED 2026-04-29)

Local critic τ2 (containing V2I rate term) implicitly proxies interference reduction — V2I rate and interference are strongly negatively correlated. Therefore actor was already being pushed in roughly the right direction without global critic. The global reward signal was *hypothesized* to be dominated by:
- ~~Rayleigh fast fading (renewed every step, ~10 dB std)~~ — **FALSIFIED, see below**
- Multi-agent exploration noise (Gaussian σ=0.1 per actor) — still plausible, untested
- Geometry refresh shocks (every 20 episodes) — still plausible, untested
- Discrete RB/mode action switching → Interference_all jumps (untested, newly suspected)

User correctly identified the central question: **"为什么修复后 global reward 表现反而不如不修复？"**

### Fading-off ablation result (2026-04-29) — hypothesis FALSIFIED

Setup: seed=3, 4 runs comparing fading ON vs OFF for both main and direction-B (last-100 episode means).

| run                  | rg_mean | rg_std | AoI_mean | AoI_p90 | Jain   |
|----------------------|---------|--------|----------|---------|--------|
| main, fading ON      | 0.642   | 0.0421 | 4.73     | 7.64    | 0.843  |
| dir-B, fading ON     | 0.648   | 0.0464 | 5.90     | 9.28    | 0.862  |
| main, fading OFF     | 0.620   | 0.0419 | 5.69     | 9.20    | 0.836  |
| dir-B, fading OFF    | 0.668   | 0.0429 | **8.44** | 15.31   | 0.797  |

Both predictions failed:
- **rg_std nearly unchanged** (main 1.00×, dir-B 0.92×). Predicted 8× drop. Fast fading is NOT the dominant variance source in `reward_global`.
- **dir-B AoI got worse, not better** (5.90 → 8.44, +43%). main also degraded (4.73 → 5.69). Gap widened from +1.16 to +2.75. Fading appears to be a *useful* training-signal source (channel diversity → robust policy).
- per-platoon failure mode for dir-B fading-OFF: P0/P1/P2 all collapse to 9-10, P3 thrives at 4.40 ("winner-take-all" pattern, consistent with fix re-coupling actors via gradient flow).

**Implication for paper:** Option A (fading-averaged reward as the cure) is no longer viable. Need to re-attribute rg variance before proposing any reward redesign.

## Branches and what's where

- `main`: original Parvini code + logging fix + Jain reward addition. **Has detach bug.**
- `direction-a`: main + Direction-A fix. Empirically unstable (2/7 seeds catastrophic failure historically).
- `direction-b`: main + Direction-B fix. More stable than A, slightly worse than main no-Jain.

User's plan: focus on direction-B as primary, retire direction-A.

## Pending experiments (next session priority)

### COMPLETED 2026-04-29: fading-off ablation — hypothesis falsified

See "Fading-off ablation result" section above. Fast fading is not the dominant noise source in `reward_global`, and disabling it makes training worse, not better.

### NEW P0: rg variance attribution on a frozen policy (no training)

Don't burn more lab-machine GPU on speculative full training runs. Instead, load a trained checkpoint and decompose `reward_global` variance by ablating sources one at a time over a fixed evaluation episode set:

1. **baseline**: full env, deterministic actor (exploration noise OFF), N steps → measure rg_std_baseline
2. **+ exploration noise ON, fading OFF, fixed geometry** → isolate exploration noise contribution
3. **+ fading ON, no exploration noise, fixed geometry** → isolate per-step fading contribution (should be ~0 based on training-time finding above)
4. **+ geometry renew ON, no exploration noise, no fading** → isolate geometry-shock contribution
5. **+ random RB/mode switching only** → isolate discrete-action interference jumps

Decision rule: whichever source contributes the most to rg_std is the next target for reward redesign. Cheap (no training) and high-information.

### Speculative follow-ups (only after P0 attribution)

- If exploration noise dominates: try smaller σ on actors, or anneal σ → 0 over training
- If geometry shocks dominate: increase RENEW_POS_EVERY, or smooth the transition
- If discrete-action jumps dominate: revise interference reward to use a smoothed/averaged form
- Sweep LAMBDA_JAIN ∈ {0.05, 0.1, 0.5} on direction-B (lower priority now that fix-degrades story is in question)
- IS_TEST generalization evaluation (100 unseen geometry episodes)

## Paper narrative — current options

User explicitly rejected "bit-identical as contribution". The paper must be built around something else.

**Option A (DEAD as of 2026-04-29):** ~~"Why fixing the detach bug hurts: noisy global reward in V2X MARL" with fading-averaged reward as solution.~~ Fading-off ablation falsified the premise — fading is not the dominant noise source, and removing it hurts training. Could be revived only if P0 attribution finds a different dominant noise source AND a corresponding reward redesign that rescues the fix.

**Option B (negative result paper):**
If no reward redesign saves the fix, write up the negative finding.
"When CTDE Hurts: an empirical study of global critic gradient flow in C-V2X MARL"
Suitable for ML workshops; harder for IEEE TVT. Stronger now that the fading hypothesis has been ruled out — story becomes "we tried the obvious noise fix and it didn't work either".

**Option C (abandon fix story, narrow to Jain):**
Write paper on "Jain's fairness in V2X resource allocation" using main branch (with bug) + Jain treated as evaluation metric (not reward). Skip the detach-bug discussion entirely. Simplest paper to finish but weakest contribution. Currently the safest fallback if P0 attribution doesn't yield a clean story.

## Conventions for this session

- User writes Chinese; respond in Chinese for analysis, code/prompts in English for clarity.
- User often says 先不要改代码 — respect this. Discuss before editing.
- User prefers prompts they can hand to other agents over inline code edits.
- Don't run full 500-episode training experiments — user runs them on a lab machine.
- Small validation/sanity scripts (≤10 episodes) are OK if needed.
- Always verify file paths against this doc — directory naming has shifted (e.g., `1-Modified MADDPG with TDec` vs `1-ModifiedMADDPGwithTDec`).

## Code-quality issues known but not yet fixed

- `compute_prr()` in `Environment_Platoon.py` is NOT 3GPP standard PRR. It returns V2V delivery progress (1 - demand/demand_size). Will need replacement before paper submission.
- `done = True` set on last step of episode in `memory.store_transition`. This is technically wrong for a continuing task — should be `False`. Low priority unless it causes problems.
- `record_reward_global_[i_episode]` was previously logging only last-step value (now fixed in latest commits to log episode mean).

## Useful commands

Quick comparison of last-100 metrics across runs:
```python
# In repo root
import os, numpy as np
from scipy.io import loadmat
def summary(p):
    rg = loadmat(p+'/reward_global.mat')['reward_global'].squeeze()
    aoi = loadmat(p+'/AoI.mat')['AoI']
    return dict(
        rg=float(np.mean(rg[-100:])),
        aoi=float(np.mean(aoi[:,-100:])),
        p90=float(np.percentile(aoi[:,-100:], 90)),
    )
```

Verify two runs are identical (control experiment for detach bug):
```python
import numpy as np
from scipy.io import loadmat
a1 = loadmat('run1/AoI.mat')['AoI']
a2 = loadmat('run2/AoI.mat')['AoI']
print(np.array_equal(a1, a2))   # True if policy unchanged
```
