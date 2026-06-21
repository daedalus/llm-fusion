"""Tests for the strategy discovery system."""

from __future__ import annotations

import random

from llm_fusion.discover import (
    ALL_STRATEGIES,
    Candidate,
    DiscoveryResult,
    _collect_leaf_strategies,
    _crossover,
    _mutate,
    _params_tag,
    _random_tree,
    _tree_size,
    _tree_summary,
    dominates,
    pareto_front,
)


class TestCandidate:
    def test_defaults(self):
        c = Candidate(name="test", strategy="dynamic")
        assert c.gain == 0.0
        assert c.win_rate == 0.0
        assert c.regret == 0.0
        assert c.ppl == 0.0
        assert c.params == {}
        assert c.composed is None
        assert c.tree is None


class TestParetoDominance:
    def test_dominates_strict(self):
        a = Candidate(name="a", strategy="x", gain=0.1, win_rate=0.8, regret=0.001)
        b = Candidate(name="b", strategy="y", gain=0.05, win_rate=0.7, regret=0.002)
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_equal_not_dominating(self):
        a = Candidate(name="a", strategy="x", gain=0.1, win_rate=0.8, regret=0.001)
        b = Candidate(name="b", strategy="y", gain=0.1, win_rate=0.8, regret=0.001)
        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_tiebreak_ppl(self):
        a = Candidate(name="a", strategy="x", gain=0.1, win_rate=0.8, regret=0.001, ppl=10.0)
        b = Candidate(name="b", strategy="y", gain=0.1, win_rate=0.8, regret=0.001, ppl=20.0)
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_tradeoff_no_dominance(self):
        a = Candidate(name="a", strategy="x", gain=0.2, win_rate=0.6, regret=0.003)
        b = Candidate(name="b", strategy="y", gain=0.1, win_rate=0.9, regret=0.001)
        assert not dominates(a, b)
        assert not dominates(b, a)


class TestParetoFront:
    def test_front_of_three(self):
        a = Candidate(name="a", strategy="x", gain=0.2, win_rate=0.8, regret=0.001)
        b = Candidate(name="b", strategy="y", gain=0.1, win_rate=0.9, regret=0.001)
        c = Candidate(name="c", strategy="z", gain=0.05, win_rate=0.7, regret=0.005)
        front = pareto_front([a, b, c])
        names = [c.name for c in front]
        assert "a" in names
        assert "b" in names
        assert "c" not in names

    def test_front_single(self):
        a = Candidate(name="a", strategy="x", gain=0.1, win_rate=0.8, regret=0.001)
        front = pareto_front([a])
        assert len(front) == 1

    def test_front_empty(self):
        front = pareto_front([])
        assert len(front) == 0


class TestTreeGeneration:
    def test_random_tree_produces_valid_structure(self):
        rng = random.Random(42)
        for _ in range(20):
            tree = _random_tree(rng, max_depth=3)
            assert "op" in tree

    def test_random_tree_size(self):
        rng = random.Random(42)
        for _ in range(10):
            tree = _random_tree(rng, max_depth=2)
            size = _tree_size(tree)
            assert size >= 1

    def test_collect_leaf_strategies(self):
        tree = {"op": "Prod", "children": [
            {"op": "Leaf", "value": 0},
            {"op": "Leaf", "value": 3},
        ]}
        leaves = _collect_leaf_strategies(tree)
        assert len(leaves) == 2
        assert leaves[0] == ALL_STRATEGIES[0]
        assert leaves[1] == ALL_STRATEGIES[3]


class TestTreeMutation:
    def test_mutate_preserves_validity(self):
        rng = random.Random(42)
        tree = _random_tree(rng, max_depth=2)
        mutated = _mutate(tree, rng)
        assert "op" in mutated

    def test_crossover_produces_valid_tree(self):
        rng = random.Random(42)
        a = _random_tree(rng, max_depth=2)
        b = _random_tree(rng, max_depth=2)
        child = _crossover(a, b, rng)
        assert "op" in child


class TestTreeSummary:
    def test_leaf_summary(self):
        tree = {"op": "Leaf", "value": 0}
        assert _tree_summary(tree) == "average"

    def test_nested_summary(self):
        tree = {"op": "Prod", "children": [
            {"op": "Leaf", "value": 1},
            {"op": "Leaf", "value": 2},
        ]}
        s = _tree_summary(tree)
        assert "Prod" in s
        assert "product" in s


class TestParamsTag:
    def test_dynamic_params(self):
        params = {"dynamic_initial_weight": 0.8, "dynamic_final_weight": 0.2, "dynamic_total_steps": 50}
        tag = _params_tag(params)
        assert "i=0.8" in tag
        assert "f=0.2" in tag
        assert "t=50" in tag

    def test_cascade_params(self):
        params = {"cascade_threshold": 0.7}
        tag = _params_tag(params)
        assert "threshold=0.7" in tag


class TestDiscoveryResult:
    def test_creation(self):
        r = DiscoveryResult(
            pareto_front=[], all_candidates=[], tier=1, prompt_count=10, search_budget=50
        )
        assert r.tier == 1
        assert r.prompt_count == 10
        assert r.elapsed_s == 0.0
