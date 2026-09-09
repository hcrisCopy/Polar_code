"""Appendix B: complete-program edit MCTS, not next-layer token prediction."""

from dataclasses import dataclass, field
import math
import random

from tqdm import tqdm


@dataclass
class Node:
    path: tuple
    visits: int = 0
    total_reward: float = 0.0
    children: list = field(default_factory=list)
    unexpanded: list | None = None


def edits(path, *, max_block, max_repeats, max_length):
    """Edit contiguous positions in the current program (Appendix B.2).

    Repeat count denotes additional copies. No representability filter here:
    the diagnostic search space is broader than the released predictor grammar.
    """
    candidates = set()
    for start in range(len(path)):
        for size in range(1, min(max_block, len(path) - start) + 1):
            end = start + size
            shortened = path[:start] + path[end:]
            if shortened:
                candidates.add(shortened)
            for count in range(1, max_repeats + 1):
                if len(path) + size * count > max_length:
                    break
                candidates.add(path[:end] + path[start:end] * count + path[end:])
    candidates.discard(path)
    return sorted(candidates)


def search(evaluate, *, depth, simulations, exploration, length_penalty,
           max_block, max_repeats, max_length, seed, rank, on_evaluation):
    """Evaluate each unique program once; backpropagate binary reward only.

    Revisited states use their actual recorded reward. Ancestor states are
    excluded to prevent edit cycles; states reached by other branches may share
    the reward cache but retain local tree visit statistics.
    """
    rng = random.Random(seed)
    root = Node(tuple(range(depth)))
    scores = {}
    cache_hits = 0

    def score(path):
        nonlocal cache_hits
        if path in scores:
            cache_hits += 1
            return scores[path]
        result = float(evaluate(list(path)))
        if result not in (0.0, 1.0):
            raise ValueError(f"Expected binary reward, received {result}")
        scores[path] = result
        on_evaluation(path, result)
        return result

    initial = score(root.path)
    for _ in tqdm(range(simulations), desc=f"rank {rank} MCTS", position=2 * rank + 1,
                  leave=False, mininterval=2):
        node = root
        chain = [node]
        ancestors = {node.path}
        while True:
            if node.unexpanded is None:
                node.unexpanded = [p for p in edits(
                    node.path, max_block=max_block, max_repeats=max_repeats,
                    max_length=max_length) if p not in ancestors]
                rng.shuffle(node.unexpanded)
            if node.unexpanded:
                node = Node(node.unexpanded.pop())
                chain[-1].children.append(node)
                chain.append(node)
                break
            if not node.children:
                break
            # V is the total simulations so far, as defined in Appendix B.3.
            def ucb(child):
                return (child.total_reward / child.visits
                        + exploration * math.sqrt(math.log(max(1, root.visits)) / child.visits)
                        - length_penalty * len(child.path) / depth)
            values = [ucb(child) for child in node.children]
            best = max(values)
            node = rng.choice([child for child, value in zip(node.children, values) if value == best])
            chain.append(node)
            ancestors.add(node.path)
        reward = score(node.path)
        for visited in chain:
            visited.visits += 1
            visited.total_reward += reward

    return {"initial_score": initial, "simulations_completed": root.visits,
            "unique_evaluations": len(scores), "cache_hits": cache_hits,
            "root_cumulative_reward": root.total_reward}
