#!/usr/bin/env python3
"""
Minimal finite-horizon splitting sampler for one classical bit,
now using MCMC over failing histories.

Model:
    - One classical bit starts in state 0.
    - Each time step has an independent bit-flip fault with probability p.
    - A history is a binary vector h of length t_max.
    - h[t] = True means a fault occurred at time t.
    - Failure event F: at least one fault occurs before t_max.

Goal:
    Estimate P_p(F) for small p using splitting from a larger p_max.

Exact answer:
    P_p(F) = 1 - (1 - p)**t_max

This script samples from P_p(history | F) by MCMC.
"""

import numpy as np


def sample_history(p, t_max, rng):
    """Sample h ~ P_p(h)."""
    return rng.random(t_max) < p


def failed(history):
    """Finite-horizon failure event."""
    return np.any(history)


def first_failure_time(history):
    """Return 1-based first failure time, or None if no failure."""
    idx = np.flatnonzero(history)
    if len(idx) == 0:
        return None
    return int(idx[0] + 1)


def process_weight(history):
    """Number of faults in the history."""
    return int(np.sum(history))


def log_prob(history, p):
    """Log P_p(history) for iid Bernoulli(p) faults."""
    w = process_weight(history)
    n = len(history)
    return w * np.log(p) + (n - w) * np.log1p(-p)


def exact_failure_probability(p, t_max):
    """Exact P_p(F)."""
    return 1.0 - (1.0 - p) ** t_max


def direct_failure_probability(p, t_max, shots, rng):
    """Direct MC estimate at p, also returning one failing seed."""
    failures = 0
    seed = None

    for _ in range(shots):
        h = sample_history(p, t_max, rng)
        if failed(h):
            failures += 1
            if seed is None:
                seed = h.copy()

    if seed is None:
        raise RuntimeError("No failing seed found. Increase p_max or direct_shots.")

    return failures / shots, seed


def propose_relocation(history, rng):
    """
    Move one existing fault to an empty location.

    This preserves process weight and always preserves failure when w > 0.
    For homogeneous noise, it is accepted with probability 1.
    """
    h = history.copy()
    ones = np.flatnonzero(h)
    zeros = np.flatnonzero(~h)

    if len(ones) == 0 or len(zeros) == 0:
        return h, 0.0, True

    old_pos = rng.choice(ones)
    new_pos = rng.choice(zeros)

    h[old_pos] = False
    h[new_pos] = True

    # Proposal is symmetric: Q(h->h') = Q(h'->h).
    log_q_ratio = 0.0
    return h, log_q_ratio, True


def propose_birth(history, rng):
    """
    Add one fault at a currently empty location.

    Proposal probability:
        Q_birth(h -> h') = p_birth * 1 / n_zero(h)

    Reverse death proposal:
        Q_death(h' -> h) = p_death * 1 / n_one(h')

    The move-type probabilities are handled in mcmc_step.
    Here we return the log proposal ratio without move-type factors.
    """
    h = history.copy()
    zeros = np.flatnonzero(~h)

    if len(zeros) == 0:
        return h, 0.0, False

    n_zero_old = len(zeros)

    pos = rng.choice(zeros)
    h[pos] = True

    n_one_new = process_weight(h)

    # log Q_reverse / Q_forward, excluding move-type probabilities.
    log_q_ratio = np.log(n_zero_old) - np.log(n_one_new)

    return h, log_q_ratio, True


def propose_death(history, rng):
    """
    Remove one existing fault.

    If this removes the only fault, the proposal leaves F and is rejected later.

    Proposal probability:
        Q_death(h -> h') = p_death * 1 / n_one(h)

    Reverse birth proposal:
        Q_birth(h' -> h) = p_birth * 1 / n_zero(h')

    The move-type probabilities are handled in mcmc_step.
    Here we return the log proposal ratio without move-type factors.
    """
    h = history.copy()
    ones = np.flatnonzero(h)

    if len(ones) == 0:
        return h, 0.0, False

    n_one_old = len(ones)

    pos = rng.choice(ones)
    h[pos] = False

    n_zero_new = len(h) - process_weight(h)

    # log Q_reverse / Q_forward, excluding move-type probabilities.
    log_q_ratio = np.log(n_one_old) - np.log(n_zero_new)

    return h, log_q_ratio, True


def mcmc_step(history, p, rng, move_probs=(0.60, 0.20, 0.20)):
    """
    One Metropolis-Hastings step targeting P_p(history | F).

    Move types:
        relocation: preserves weight
        birth:      w -> w + 1
        death:      w -> w - 1

    The chain rejects proposals outside the failing set F.
    """
    p_reloc, p_birth, p_death = move_probs
    u = rng.random()

    if u < p_reloc:
        proposal, log_q_ratio, valid = propose_relocation(history, rng)
        move_type_ratio = 0.0

    elif u < p_reloc + p_birth:
        proposal, log_q_ratio, valid = propose_birth(history, rng)

        # Reverse of birth is death.
        move_type_ratio = np.log(p_death) - np.log(p_birth)

    else:
        proposal, log_q_ratio, valid = propose_death(history, rng)

        # Reverse of death is birth.
        move_type_ratio = np.log(p_birth) - np.log(p_death)

    if not valid:
        return history, False

    if not failed(proposal):
        return history, False

    log_a = (
        log_prob(proposal, p)
        - log_prob(history, p)
        + log_q_ratio
        + move_type_ratio
    )

    if np.log(rng.random()) < min(0.0, log_a):
        return proposal, True

    return history, False


def mcmc_conditioned_samples(
    seed_history,
    p,
    n_samples,
    burn_in,
    thin,
    rng,
    move_probs=(0.60, 0.20, 0.20),
):
    """
    Generate approximate samples from P_p(history | F).
    """
    if not failed(seed_history):
        raise ValueError("MCMC seed must be a failing history.")

    h = seed_history.copy()

    accepted_burn = 0
    for _ in range(burn_in):
        h, accepted = mcmc_step(h, p, rng, move_probs=move_probs)
        accepted_burn += int(accepted)

    samples = []
    accepted_main = 0
    steps_main = 0

    for _ in range(n_samples):
        for _ in range(thin):
            h, accepted = mcmc_step(h, p, rng, move_probs=move_probs)
            accepted_main += int(accepted)
            steps_main += 1
        samples.append(h.copy())

    diagnostics = {
        "accept_burn": accepted_burn / max(1, burn_in),
        "accept_main": accepted_main / max(1, steps_main),
        "final_weight": process_weight(h),
        "final_first_T": first_failure_time(h),
    }

    return samples, h, diagnostics


def estimate_ratio_by_mcmc(
    seed_history,
    p_curr,
    p_next,
    t_max,
    n_samples,
    burn_in,
    thin,
    rng,
    move_probs=(0.60, 0.20, 0.20),
):
    """
    Estimate

        P_{p_next}(F) / P_{p_curr}(F)

    using MCMC samples h ~ P_{p_curr}(h | F):

        E[ P_{p_next}(h) / P_{p_curr}(h) ].
    """
    samples, last_history, mcmc_diag = mcmc_conditioned_samples(
        seed_history=seed_history,
        p=p_curr,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        rng=rng,
        move_probs=move_probs,
    )

    weights = np.empty(n_samples, dtype=float)
    ws = np.empty(n_samples, dtype=int)
    ts = np.empty(n_samples, dtype=int)

    for i, h in enumerate(samples):
        weights[i] = np.exp(log_prob(h, p_next) - log_prob(h, p_curr))
        ws[i] = process_weight(h)
        ts[i] = first_failure_time(h)

    ratio = float(np.mean(weights))
    ratio_se_naive = float(np.std(weights, ddof=1) / np.sqrt(n_samples))

    ess = float((np.sum(weights) ** 2) / np.sum(weights**2))

    diagnostics = {
        "mean_weight": float(np.mean(ws)),
        "mean_first_failure_time": float(np.mean(ts)),
        "ess": ess,
        **mcmc_diag,
    }

    return ratio, ratio_se_naive, last_history, diagnostics


def splitting_estimate_mcmc(
    p_min=1e-4,
    p_max=0.05,
    t_max=500,
    n_levels=25,
    direct_shots=10_000,
    samples_per_level=20_000,
    burn_in=5_000,
    thin=10,
    seed=123,
    move_probs=(0.60, 0.20, 0.20),
    verbose=True,
):
    """
    Estimate P_{p_min}(F) by splitting with MCMC conditioned sampling.
    """
    rng = np.random.default_rng(seed)

    pF, history = direct_failure_probability(
        p=p_max,
        t_max=t_max,
        shots=direct_shots,
        rng=rng,
    )

    exact_pmax = exact_failure_probability(p_max, t_max)

    if verbose:
        print("initialization")
        print(f"  p_max direct est. = {pF:.8e}")
        print(f"  p_max exact       = {exact_pmax:.8e}")
        print(f"  seed weight       = {process_weight(history)}")
        print(f"  seed first T      = {first_failure_time(history)}")
        print()

    ps = np.geomspace(p_max, p_min, n_levels + 1)
    ratios = []

    for level, (p_curr, p_next) in enumerate(zip(ps[:-1], ps[1:]), start=1):
        ratio, ratio_se, history, diag = estimate_ratio_by_mcmc(
            seed_history=history,
            p_curr=p_curr,
            p_next=p_next,
            t_max=t_max,
            n_samples=samples_per_level,
            burn_in=burn_in,
            thin=thin,
            rng=rng,
            move_probs=move_probs,
        )

        pF *= ratio
        ratios.append(ratio)

        exact_ratio = (
            exact_failure_probability(p_next, t_max)
            / exact_failure_probability(p_curr, t_max)
        )

        if verbose:
            print(
                f"level {level:02d}: "
                f"{p_curr:.3e} -> {p_next:.3e} | "
                f"ratio est {ratio:.8f} +/- {ratio_se:.2e} | "
                f"exact {exact_ratio:.8f} | "
                f"rel {(ratio / exact_ratio - 1):+.2%} | "
                f"mean w {diag['mean_weight']:.3f} | "
                f"ESS {diag['ess']:.0f}/{samples_per_level} | "
                f"acc {diag['accept_main']:.2%}"
            )

    return pF, ps, ratios


if __name__ == "__main__":
    p_min = 1e-4
    p_max = 0.05
    t_max = 500

    estimate, ps, ratios = splitting_estimate_mcmc(
        p_min=p_min,
        p_max=p_max,
        t_max=t_max,
        n_levels=25,
        direct_shots=10_000,
        samples_per_level=20_000,
        burn_in=5_000,
        thin=10,
        seed=123,
        move_probs=(0.60, 0.20, 0.20),
        verbose=True,
    )

    exact = exact_failure_probability(p_min, t_max)

    print()
    print("final result")
    print(f"p_min          = {p_min}")
    print(f"p_max          = {p_max}")
    print(f"t_max          = {t_max}")
    print()
    print(f"splitting est. = {estimate:.8e}")
    print(f"exact          = {exact:.8e}")
    print(f"relative error = {(estimate / exact - 1):+.3%}")
