"""Deterministic, diversity-first candidate execution programs."""

import random

from .storage import digest


def _single_edits(depth, max_block, max_length):
    base = tuple(range(depth))
    paths = []
    for size in range(1, max_block + 1):
        for start in range(depth - size + 1):
            end = start + size
            paths.append(base[:start] + base[end:])
            repeated = base[:end] + base[start:end] + base[end:]
            if len(repeated) <= max_length:
                paths.append(repeated)
    return paths


def _random_edit(path, rng, max_block, max_length):
    if not path:
        return path
    size = rng.randint(1, min(max_block, len(path)))
    start = rng.randint(0, len(path) - size)
    end = start + size
    block = path[start:end]
    actions = ["skip"]
    if len(path) + size <= max_length:
        actions.append("repeat")
    if rng.choice(actions) == "skip":
        candidate = path[:start] + path[end:]
        return candidate if candidate else path
    return path[:end] + block + path[end:]


def _interleave(local_paths, exploratory_paths):
    """Use three local edits per broader edit to favor early useful labels."""
    result = []
    local_index = exploratory_index = 0
    while local_index < len(local_paths) or exploratory_index < len(exploratory_paths):
        for _ in range(3):
            if local_index < len(local_paths):
                result.append(local_paths[local_index])
                local_index += 1
        if exploratory_index < len(exploratory_paths):
            result.append(exploratory_paths[exploratory_index])
            exploratory_index += 1
    return result


def generate_candidate_pool(*, depth, maximum, seed, max_block, max_length):
    """Return exactly ``maximum`` unique non-empty paths when feasible.

    The front of the pool mixes one-edit paths with broader 2--6 edit paths.
    This is intentionally a bounded diagnostic search, not a claim of faithful
    reproduction of the paper's underspecified MCTS implementation.
    """
    if depth <= 0 or maximum <= 0 or max_block <= 0 or max_length < depth:
        raise ValueError("Invalid candidate-pool configuration")
    base = tuple(range(depth))
    rng = random.Random(seed)
    local = list(dict.fromkeys(_single_edits(depth, max_block, max_length)))
    rng.shuffle(local)

    seen = {base, *local}
    exploratory = []
    attempts = 0
    while len(seen) < maximum and attempts < maximum * 1000:
        attempts += 1
        path = base
        edit_count = 2 + (attempts % 5)
        for _ in range(edit_count):
            path = _random_edit(path, rng, max_block, max_length)
        if path and len(path) <= max_length and path not in seen:
            seen.add(path)
            exploratory.append(path)

    ordered = [base] + _interleave(local, exploratory)
    if len(ordered) < maximum:
        raise RuntimeError(
            f"Could create only {len(ordered)} unique paths; requested {maximum}"
        )
    ordered = ordered[:maximum]
    return [{"candidate_id": f"path_{index:04d}", "path": list(path),
             "length": len(path), "path_digest": digest(path)}
            for index, path in enumerate(ordered)]

