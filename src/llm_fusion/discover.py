"""Automated strategy discovery: parameter tuning, composition, and evolution."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_fusion.fusion import Fuser, softmax_top_k

ALL_STRATEGIES = (
    "average", "product", "min-entropy", "min-perplexity", "cascade",
    "dynamic", "adaptive", "confidence", "hybrid", "slerp", "simple",
    "sqrt-product", "min", "log-sum", "norm-product",
)

PARAMETER_FREE = (
    "product", "min-entropy", "min-perplexity", "adaptive", "confidence",
    "simple", "sqrt-product", "min", "log-sum", "norm-product",
)

PARAMETERIZED = {
    "average": {"ouro_weight": (0.0, 1.0)},
    "slerp": {"ouro_weight": (0.0, 1.0)},
    "cascade": {"cascade_threshold": (0.0, 1.0)},
    "dynamic": {
        "dynamic_initial_weight": (0.0, 1.0),
        "dynamic_final_weight": (0.0, 1.0),
        "dynamic_total_steps": (10, 200),
    },
    "hybrid": {
        "dynamic_initial_weight": (0.0, 1.0),
        "dynamic_final_weight": (0.0, 1.0),
        "dynamic_total_steps": (10, 200),
    },
}


@dataclass
class Candidate:
    name: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    composed: list[tuple[str, float]] | None = None
    tree: dict | None = None
    gain: float = 0.0
    win_rate: float = 0.0
    regret: float = 0.0
    ppl: float = 0.0


@dataclass
class DiscoveryResult:
    pareto_front: list[Candidate]
    all_candidates: list[Candidate]
    tier: int
    prompt_count: int
    search_budget: int
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Pareto ranking
# ---------------------------------------------------------------------------

def dominates(a: Candidate, b: Candidate) -> bool:
    better_or_equal = (a.gain >= b.gain and a.win_rate >= b.win_rate and a.regret <= b.regret)
    strictly_better = (a.gain > b.gain or a.win_rate > b.win_rate or a.regret < b.regret)
    if better_or_equal and strictly_better:
        return True
    if better_or_equal and strictly_better is False:
        return a.ppl < b.ppl
    return False


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    front = []
    for c in candidates:
        if not any(dominates(other, c) for other in candidates if other is not c):
            front.append(c)
    front.sort(key=lambda c: (-c.gain, -c.win_rate, c.regret))
    return front


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def _make_config(candidate: Candidate) -> dict[str, Any]:
    cfg: dict[str, Any] = {"model": "fused"}
    if candidate.composed is not None:
        cfg["strategy"] = "average"
        cfg["_composed"] = candidate.composed
    elif candidate.tree is not None:
        cfg["strategy"] = "average"
        cfg["_tree"] = candidate.tree
    else:
        cfg["strategy"] = candidate.strategy
        cfg.update(candidate.params)
    return cfg


def evaluate_candidate(
    candidate: Candidate,
    battery: list[dict[str, str]],
    loaded: Any,
    max_new_tokens: int = 30,
    top_k: int = 30,
) -> Candidate:
    from llm_fusion.benchmark import run_benchmark

    total_gain = 0.0
    total_win = 0.0
    total_regret = 0.0
    total_ppl = 0.0
    n = 0

    for entry in battery:
        prompt = entry["prompt"]
        if not prompt:
            continue
        cfg = _make_config(candidate)

        if cfg.get("_composed") is not None:
            result = _eval_composed(prompt, cfg["_composed"], loaded, max_new_tokens, top_k)
        elif cfg.get("_tree") is not None:
            result = _eval_tree(prompt, cfg["_tree"], loaded, max_new_tokens, top_k)
        else:
            results = run_benchmark(
                text=prompt, max_new_tokens=max_new_tokens,
                top_k=top_k, temperature=0.0, local=True,
                configs=[cfg], loaded=loaded,
            )
            if not results:
                continue
            result = results[0]

        total_gain += result.avg_fusion_gain
        total_win += result.fusion_win_rate
        total_regret += result.avg_fusion_regret
        total_ppl += result.fused_ppl
        n += 1

    if n > 0:
        candidate.gain = total_gain / n
        candidate.win_rate = total_win / n
        candidate.regret = total_regret / n
        candidate.ppl = total_ppl / n
    return candidate


# ---------------------------------------------------------------------------
# Composed strategy evaluation
# ---------------------------------------------------------------------------

def _fuse_composed(
    ouro_logits: list[float],
    hrm_logits: list[float],
    fusers: list[Fuser],
    weights: list[float],
) -> list[tuple[int, float, str]]:
    merged: dict[int, float] = {}
    for fuser, w in zip(fusers, weights):
        for tid, prob, _ in fuser.fuse_logits(ouro_logits, hrm_logits):
            merged[tid] = merged.get(tid, 0.0) + prob * w
    items = sorted(merged.items(), key=lambda x: -x[1])
    return [(tid, p, fusers[0].hrm_tok.decode([tid])) for tid, p in items]


def _eval_composed(
    text: str,
    composed: list[tuple[str, float]],
    loaded: Any,
    max_new_tokens: int,
    top_k: int,
) -> Any:
    from llm_fusion.benchmark import BenchmarkResult, format_hrm_prompt
    from llm_fusion.generate import HRM_EOS_ID, OURO_EOS_ID

    matcher = loaded.matcher
    ouro_tok = loaded.ouro_tok
    hrm_tok = loaded.hrm_tok
    ouro_model = loaded.ouro_model
    hrm_model = loaded.hrm_model
    device = loaded.device

    fusers = [
        Fuser(matcher, ouro_tok, hrm_tok, 0.5, top_k, s) for s, _ in composed
    ]
    weights = [w for _, w in composed]

    r = BenchmarkResult(model="fused", strategy="composed")
    ouro_prompt_ids = ouro_tok.encode(text).ids
    prompt = format_hrm_prompt(text, "direct")
    hrm_ids = hrm_tok.encode(prompt).ids
    r.prompt_tokens = len(ouro_prompt_ids)

    if len(hrm_ids) < 2:
        return r

    import torch

    generated_text = ""
    ouro_gen_ids: set[int] = set()
    hrm_gen_ids: set[int] = set()
    ouro_ids = list(ouro_prompt_ids)
    hrm_ids_list = list(hrm_ids)
    total_gain = 0.0
    fusion_wins = 0
    n_steps = 0

    for step in range(min(max_new_tokens, 30)):
        with torch.no_grad():
            ouro_out = ouro_model(input_ids=torch.tensor([ouro_ids], device=device, dtype=torch.long))
            hrm_out = hrm_model(
                input_ids=torch.tensor([hrm_ids_list], device=device, dtype=torch.long),
                token_type_ids=torch.ones(len(hrm_ids_list), dtype=torch.long, device=device).unsqueeze(0),
            )
        ouro_logits = ouro_out.logits[0, -1, :].tolist()
        hrm_logits = hrm_out.logits[0, -1, :].tolist()

        candidates = _fuse_composed(ouro_logits, hrm_logits, fusers, weights)
        if not candidates:
            break

        tid, prob, token_str = candidates[0][0], candidates[0][1], candidates[0][2]

        from llm_fusion.metrics import fusion_gain, parent_prob_for_token
        ouro_prob = parent_prob_for_token(ouro_logits, tid, top_k)
        hrm_prob = parent_prob_for_token(hrm_logits, tid, top_k)
        gain = fusion_gain(prob, ouro_prob, hrm_prob)
        total_gain += gain
        if prob > max(ouro_prob, hrm_prob):
            fusion_wins += 1

        hrm_gen_ids.add(tid)
        hrm_ids_list = [tid]
        generated_text += token_str
        ouro_ids = ouro_tok.encode(generated_text).ids or [OURO_EOS_ID]
        ouro_gen_ids.update(ouro_ids)
        n_steps += 1

        if tid in (HRM_EOS_ID, OURO_EOS_ID):
            break

    n = max(n_steps, 1)
    r.avg_fusion_gain = total_gain / n
    r.fusion_win_rate = fusion_wins / n
    r.tokens_generated = n_steps
    return r


# ---------------------------------------------------------------------------
# Tree evaluation (Tier 3 DSL)
# ---------------------------------------------------------------------------

def _eval_tree_node(
    node: dict,
    ouro_probs: dict[int, float],
    hrm_probs: dict[int, float],
    fusers: list[Fuser],
    ouro_logits: list[float],
    hrm_logits: list[float],
) -> dict[int, float]:
    op = node.get("op", "Leaf")

    if op == "Leaf":
        idx = node.get("value", 0)
        fuser = fusers[idx % len(fusers)]
        result = {}
        for tid, p, _ in fuser.fuse_logits(ouro_logits, hrm_logits):
            result[tid] = p
        return result

    if op == "W":
        w = node.get("value", 1.0)
        child = _eval_tree_node(node["children"][0], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        return {tid: p * w for tid, p in child.items()}

    if op == "Prod":
        a = _eval_tree_node(node["children"][0], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        b = _eval_tree_node(node["children"][1], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        all_tids = set(a) | set(b)
        return {tid: a.get(tid, 0.0) * b.get(tid, 0.0) for tid in all_tids}

    if op == "Min":
        a = _eval_tree_node(node["children"][0], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        b = _eval_tree_node(node["children"][1], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        all_tids = set(a) | set(b)
        return {tid: min(a.get(tid, 0.0), b.get(tid, 0.0)) for tid in all_tids}

    if op == "Sum":
        a = _eval_tree_node(node["children"][0], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        b = _eval_tree_node(node["children"][1], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        all_tids = set(a) | set(b)
        return {tid: a.get(tid, 0.0) + b.get(tid, 0.0) for tid in all_tids}

    if op == "Normalize":
        child = _eval_tree_node(node["children"][0], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        total = sum(child.values())
        if total < 1e-10:
            return child
        return {tid: p / total for tid, p in child.items()}

    if op == "SigmoidGate":
        a = _eval_tree_node(node["children"][0], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        b = _eval_tree_node(node["children"][1], ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        ouro_top = softmax_top_k(ouro_logits, 1)
        hrm_top = softmax_top_k(hrm_logits, 1)
        ouro_conf = ouro_top[1][0] if ouro_top[1] else 0.0
        hrm_conf = hrm_top[1][0] if hrm_top[1] else 0.0
        total = ouro_conf + hrm_conf
        alpha = ouro_conf / total if total > 1e-10 else 0.5
        all_tids = set(a) | set(b)
        return {tid: alpha * a.get(tid, 0.0) + (1 - alpha) * b.get(tid, 0.0) for tid in all_tids}

    return ouro_probs


def _eval_tree(
    text: str,
    tree: dict,
    loaded: Any,
    max_new_tokens: int,
    top_k: int,
) -> Any:
    from llm_fusion.benchmark import BenchmarkResult, format_hrm_prompt
    from llm_fusion.generate import HRM_EOS_ID, OURO_EOS_ID

    matcher = loaded.matcher
    ouro_tok = loaded.ouro_tok
    hrm_tok = loaded.hrm_tok
    ouro_model = loaded.ouro_model
    hrm_model = loaded.hrm_model
    device = loaded.device

    leaf_strategies = _collect_leaf_strategies(tree)
    if not leaf_strategies:
        leaf_strategies = ["average", "product"]
    fusers = [Fuser(matcher, ouro_tok, hrm_tok, 0.5, top_k, s) for s in leaf_strategies]

    r = BenchmarkResult(model="fused", strategy="evolved")
    ouro_prompt_ids = ouro_tok.encode(text).ids
    prompt = format_hrm_prompt(text, "direct")
    hrm_ids = hrm_tok.encode(prompt).ids
    r.prompt_tokens = len(ouro_prompt_ids)

    if len(hrm_ids) < 2:
        return r

    import torch

    generated_text = ""
    ouro_ids = list(ouro_prompt_ids)
    hrm_ids_list = list(hrm_ids)
    total_gain = 0.0
    fusion_wins = 0
    n_steps = 0

    for step in range(min(max_new_tokens, 30)):
        with torch.no_grad():
            ouro_out = ouro_model(input_ids=torch.tensor([ouro_ids], device=device, dtype=torch.long))
            hrm_out = hrm_model(
                input_ids=torch.tensor([hrm_ids_list], device=device, dtype=torch.long),
                token_type_ids=torch.ones(len(hrm_ids_list), dtype=torch.long, device=device).unsqueeze(0),
            )
        ouro_logits = ouro_out.logits[0, -1, :].tolist()
        hrm_logits = hrm_out.logits[0, -1, :].tolist()

        ouro_top_ids, ouro_probs_list = softmax_top_k(ouro_logits, top_k)
        hrm_top_ids, hrm_probs_list = softmax_top_k(hrm_logits, top_k)
        ouro_probs = dict(zip(ouro_top_ids, ouro_probs_list))
        hrm_probs = dict(zip(hrm_top_ids, hrm_probs_list))

        fused = _eval_tree_node(tree, ouro_probs, hrm_probs, fusers, ouro_logits, hrm_logits)
        if not fused:
            break

        tid = max(fused, key=fused.get)
        prob = fused[tid]

        from llm_fusion.metrics import fusion_gain, parent_prob_for_token
        ouro_prob = parent_prob_for_token(ouro_logits, tid, top_k)
        hrm_prob = parent_prob_for_token(hrm_logits, tid, top_k)
        gain = fusion_gain(prob, ouro_prob, hrm_prob)
        total_gain += gain
        if prob > max(ouro_prob, hrm_prob):
            fusion_wins += 1

        hrm_ids_list = [tid]
        generated_text += hrm_tok.decode([tid])
        ouro_ids = ouro_tok.encode(generated_text).ids or [OURO_EOS_ID]
        n_steps += 1

        if tid in (HRM_EOS_ID, OURO_EOS_ID):
            break

    n = max(n_steps, 1)
    r.avg_fusion_gain = total_gain / n
    r.fusion_win_rate = fusion_wins / n
    r.tokens_generated = n_steps
    return r


def _collect_leaf_strategies(node: dict) -> list[str]:
    if node.get("op") == "Leaf":
        return [ALL_STRATEGIES[node.get("value", 0) % len(ALL_STRATEGIES)]]
    result = []
    for child in node.get("children", []):
        result.extend(_collect_leaf_strategies(child))
    return result


# ---------------------------------------------------------------------------
# Tier 1: Parameter search
# ---------------------------------------------------------------------------

def search_tier1(
    battery: list[dict[str, str]],
    loaded: Any,
    budget: int = 200,
    seed: int | None = None,
    max_new_tokens: int = 30,
    top_k: int = 30,
) -> DiscoveryResult:
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    t0 = time.time()

    for name in PARAMETER_FREE:
        c = Candidate(name=name, strategy=name, params={})
        evaluate_candidate(c, battery, loaded, max_new_tokens, top_k)
        candidates.append(c)
        _progress(len(candidates), budget, c.name, c.gain)

    n_param = budget - len(PARAMETER_FREE)
    strat_names = list(PARAMETERIZED.keys())
    per_strat = n_param // len(strat_names) if strat_names else 0

    for sname in strat_names:
        space = PARAMETERIZED[sname]
        for i in range(per_strat):
            params = {}
            for k, (lo, hi) in space.items():
                if isinstance(lo, int):
                    params[k] = rng.randint(lo, hi)
                else:
                    params[k] = round(rng.uniform(lo, hi), 4)
            name = f"{sname}:{_params_tag(params)}"
            c = Candidate(name=name, strategy=sname, params=params)
            evaluate_candidate(c, battery, loaded, max_new_tokens, top_k)
            candidates.append(c)
            _progress(len(candidates), budget, c.name, c.gain)

    front = pareto_front(candidates)
    return DiscoveryResult(
        pareto_front=front, all_candidates=candidates,
        tier=1, prompt_count=len(battery),
        search_budget=len(candidates), elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Tier 2: Composition search
# ---------------------------------------------------------------------------

def search_tier2(
    battery: list[dict[str, str]],
    loaded: Any,
    budget: int = 300,
    seed: int | None = None,
    max_new_tokens: int = 30,
    top_k: int = 30,
    top_strategies: list[str] | None = None,
) -> DiscoveryResult:
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    t0 = time.time()

    pool = top_strategies or list(ALL_STRATEGIES)

    for i in range(budget):
        n = rng.randint(2, 4)
        strats = rng.sample(pool, min(n, len(pool)))
        weights = [rng.random() for _ in strats]
        total = sum(weights)
        weights = [round(w / total, 4) for w in weights]
        composed = list(zip(strats, weights))
        tag = "+".join(f"{s}{w:.2f}" for s, w in composed)
        c = Candidate(name=f"composed:{tag}", strategy="composed", composed=composed)
        evaluate_candidate(c, battery, loaded, max_new_tokens, top_k)
        candidates.append(c)
        _progress(len(candidates), budget, c.name, c.gain)

    front = pareto_front(candidates)
    return DiscoveryResult(
        pareto_front=front, all_candidates=candidates,
        tier=2, prompt_count=len(battery),
        search_budget=len(candidates), elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Tier 3: Evolutionary search
# ---------------------------------------------------------------------------

def _random_tree(rng: random.Random, depth: int = 0, max_depth: int = 3) -> dict:
    if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
        return {"op": "Leaf", "value": rng.randint(0, len(ALL_STRATEGIES) - 1)}
    op = rng.choice(["W", "Prod", "Min", "Sum", "Normalize", "SigmoidGate"])
    if op == "W":
        return {"op": "W", "value": round(rng.uniform(0.1, 2.0), 3), "children": [_random_tree(rng, depth + 1, max_depth)]}
    if op == "Normalize":
        return {"op": "Normalize", "children": [_random_tree(rng, depth + 1, max_depth)]}
    return {"op": op, "children": [_random_tree(rng, depth + 1, max_depth), _random_tree(rng, depth + 1, max_depth)]}


def _tree_size(node: dict) -> int:
    return 1 + sum(_tree_size(c) for c in node.get("children", []))


def _random_subtree(node: dict, rng: random.Random, depth: int = 0) -> tuple[dict, int]:
    children = node.get("children", [])
    if not children or (depth > 0 and rng.random() < 0.3):
        return node, 0
    idx = rng.randint(0, len(children) - 1)
    return _random_subtree(children[idx], rng, depth + 1)


def _replace_subtree(node: dict, target_depth: int, replacement: dict, rng: random.Random, depth: int = 0) -> dict:
    if depth == target_depth:
        return replacement
    children = node.get("children", [])
    if not children:
        return node
    idx = rng.randint(0, len(children) - 1)
    new_children = list(children)
    new_children[idx] = _replace_subtree(children[idx], target_depth, replacement, rng, depth + 1)
    return {**node, "children": new_children}


def _mutate(tree: dict, rng: random.Random) -> dict:
    mutation = rng.choice(["grow", "prune", "weight", "swap_strategy"])
    if mutation == "grow":
        return _random_tree(rng, max_depth=2)
    if mutation == "prune":
        return {"op": "Leaf", "value": rng.randint(0, len(ALL_STRATEGIES) - 1)}
    if mutation == "weight" and tree.get("op") == "W":
        w = tree.get("value", 1.0) + rng.uniform(-0.2, 0.2)
        return {**tree, "value": round(max(0.1, min(2.0, w)), 3)}
    if mutation == "swap_strategy" and tree.get("op") == "Leaf":
        return {**tree, "value": rng.randint(0, len(ALL_STRATEGIES) - 1)}
    return tree


def _crossover(a: dict, b: dict, rng: random.Random) -> dict:
    if rng.random() < 0.5:
        return a
    _, depth = _random_subtree(a, rng)
    sub, _ = _random_subtree(b, rng)
    return _replace_subtree(a, depth, sub, rng)


def search_tier3(
    battery: list[dict[str, str]],
    loaded: Any,
    budget: int = 600,
    seed: int | None = None,
    max_new_tokens: int = 30,
    top_k: int = 30,
) -> DiscoveryResult:
    rng = random.Random(seed)
    t0 = time.time()
    pop_size = min(30, budget // 2)
    n_generations = budget // pop_size

    population: list[tuple[dict, Candidate]] = []
    for _ in range(pop_size):
        tree = _random_tree(rng)
        c = Candidate(name=f"evo:{_tree_summary(tree)}", strategy="evolved", tree=tree)
        evaluate_candidate(c, battery, loaded, max_new_tokens, top_k)
        population.append((tree, c))

    all_candidates = [c for _, c in population]
    _progress(len(all_candidates), budget, "init", max(c.gain for c in all_candidates))

    for gen in range(n_generations - 1):
        ranked = sorted(population, key=lambda tc: (-tc[1].gain, -tc[1].win_rate, tc[1].regret))
        elites = ranked[:max(2, pop_size // 5)]

        new_pop = list(elites)
        while len(new_pop) < pop_size:
            parents = rng.sample(elites, min(2, len(elites)))
            child_tree = _crossover(parents[0][0], parents[1][0], rng)
            if rng.random() < 0.3:
                child_tree = _mutate(child_tree, rng)
            c = Candidate(name=f"evo:gen{gen}:{_tree_summary(child_tree)}", strategy="evolved", tree=child_tree)
            evaluate_candidate(c, battery, loaded, max_new_tokens, top_k)
            new_pop.append((child_tree, c))
            all_candidates.append(c)

        population = new_pop[:pop_size]
        best = max(population, key=lambda tc: tc[1].gain)
        _progress(len(all_candidates), budget, f"gen{gen}", best[1].gain)

    front = pareto_front(all_candidates)
    return DiscoveryResult(
        pareto_front=front, all_candidates=all_candidates,
        tier=3, prompt_count=len(battery),
        search_budget=len(all_candidates), elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_tag(params: dict[str, Any]) -> str:
    parts = []
    for k, v in sorted(params.items()):
        short = k.replace("dynamic_", "d").replace("cascade_", "c").replace("initial_weight", "i").replace("final_weight", "f").replace("total_steps", "t")
        parts.append(f"{short}={v}")
    return ",".join(parts)


def _tree_summary(node: dict) -> str:
    op = node.get("op", "?")
    if op == "Leaf":
        return ALL_STRATEGIES[node.get("value", 0) % len(ALL_STRATEGIES)]
    children = [_tree_summary(c) for c in node.get("children", [])]
    if op == "W":
        return f"{op}{node.get('value', 1.0):.2f}({children[0]})"
    return f"{op}({','.join(children)})"


def _progress(done: int, total: int, name: str, gain: float) -> None:
    pct = done / max(total, 1) * 100
    bar_len = 30
    filled = int(bar_len * done / max(total, 1))
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  gain={gain:+.4f}  {name[:40]:<40}", end="", flush=True)
    if done >= total:
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_discover_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Discover optimal fusion strategies")
    parser.add_argument("--tier", type=str, default="1", choices=["1", "2", "3", "all"],
                        help="Discovery tier (1=param, 2=compose, 3=evolve, all=sequential)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Search budget per tier (default: T1=200, T2=300, T3=600)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompts", type=int, default=48, help="Number of battery prompts")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Tokens per evaluation")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k for sampling")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--local", action="store_true", default=True, help="Load models locally")
    return parser


def main() -> None:
    parser = build_discover_parser()
    args = parser.parse_args()

    from llm_fusion.benchmark import ROBUSTNESS_BATTERY, load_models

    battery = ROBUSTNESS_BATTERY[:args.prompts]
    print("Loading models...", file=sys.stderr)
    loaded = load_models(local=args.local, device=args.device)

    tiers = [1, 2, 3] if args.tier == "all" else [int(args.tier)]
    budgets = {1: 200, 2: 300, 3: 600}
    all_results: list[DiscoveryResult] = []
    top_strategies: list[str] | None = None

    for tier in tiers:
        budget = args.budget or budgets[tier]
        print(f"\n{'='*60}")
        print(f"  Discovery Tier {tier}  |  {len(battery)} prompts  |  budget={budget}")
        print(f"{'='*60}")

        if tier == 1:
            result = search_tier1(battery, loaded, budget, args.seed, args.max_new_tokens, args.top_k)
        elif tier == 2:
            if top_strategies is None:
                top_strategies = list(ALL_STRATEGIES)
            result = search_tier2(battery, loaded, budget, args.seed, args.max_new_tokens, args.top_k, top_strategies)
        else:
            result = search_tier3(battery, loaded, budget, args.seed, args.max_new_tokens, args.top_k)

        all_results.append(result)

        print(f"\n  Pareto front ({len(result.pareto_front)} solutions):")
        print(f"  {'#':>3}  {'Strategy':<45}  {'Gain':>8}  {'WinRate':>8}  {'Regret':>8}  {'PPL':>8}")
        print(f"  {'-'*3}  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for i, c in enumerate(result.pareto_front[:10], 1):
            print(f"  {i:>3}  {c.name:<45}  {c.gain:+.4f}  {c.win_rate:.1%}  {c.regret:.6f}  {c.ppl:.2f}")

        print(f"\n  Elapsed: {result.elapsed_s:.1f}s  |  Candidates: {result.search_budget}")

        if tier == 1 and result.pareto_front:
            top_strategies = [c.strategy for c in result.pareto_front[:5]]

    out_path = args.output
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        tier_tag = "all" if len(tiers) > 1 else str(tiers[0])
        out_path = f"results/discover_tier{tier_tag}_{ts}.json"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output = []
    for r in all_results:
        output.append({
            "tier": r.tier,
            "prompt_count": r.prompt_count,
            "search_budget": r.search_budget,
            "elapsed_s": round(r.elapsed_s, 1),
            "pareto_front": [asdict(c) for c in r.pareto_front],
            "all_candidates": [asdict(c) for c in r.all_candidates],
        })
    Path(out_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
