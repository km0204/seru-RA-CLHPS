#!/usr/bin/env python3
"""Reproducible experiments for adaptive hierarchical seru scheduling.

The implementation intentionally depends only on Python and NumPy.  It builds
paired synthetic instances, learns interpretable hierarchical scoring policies
with the cross-entropy method, and evaluates fixed and regime-adaptive variants
on exactly the same instances.  Every number written by this program is the
result of an executed simulation; no manuscript values are embedded here.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


TASK_TYPES = 5
WORKERS = 12
SERUS = 4
ORDERS = 60
WORKERS_PER_SERU = WORKERS // SERUS
BASE_TIMES = np.array([8.0, 9.5, 11.0, 12.5, 14.0])
LEARNING_RATES = np.array([0.020, 0.025, 0.030, 0.035, 0.040])
PROFICIENCY_EFFECT = 1.2
SKILL_THRESHOLD = 0.70


@dataclass(frozen=True)
class OrderSpec:
    arrival: float
    tasks: Tuple[int, ...]


@dataclass(frozen=True)
class Instance:
    seed: int
    regime: str
    orders: Tuple[OrderSpec, ...]
    proficiency: np.ndarray


@dataclass
class OrderState:
    spec: OrderSpec
    seru: int = -1
    next_task: int = 0
    processing: bool = False
    completion: float = math.nan


def arrival_rate(regime: str, fraction: float) -> float:
    """Piecewise-constant order arrival intensity."""
    if regime == "steady":
        return 0.19
    if regime == "surge":
        return 0.17 if fraction < 0.35 else (0.33 if fraction < 0.70 else 0.17)
    if regime == "recovery":
        return 0.31 if fraction < 0.30 else (0.20 if fraction < 0.65 else 0.13)
    raise ValueError(f"unknown regime: {regime}")


def make_instance(seed: int, regime: str, orders: int = ORDERS) -> Instance:
    rng = np.random.default_rng(seed)
    proficiency = rng.uniform(0.30, 0.80, size=(WORKERS, TASK_TYPES))
    # Guarantee feasibility within every seru without eliminating heterogeneity.
    for m in range(SERUS):
        ws = slice(m * WORKERS_PER_SERU, (m + 1) * WORKERS_PER_SERU)
        for k in range(TASK_TYPES):
            if proficiency[ws, k].max() < 0.34:
                proficiency[m * WORKERS_PER_SERU, k] = 0.34

    now = 0.0
    specs: List[OrderSpec] = []
    for i in range(orders):
        rate = arrival_rate(regime, i / max(1, orders - 1))
        now += float(rng.exponential(1.0 / rate))
        n_tasks = int(rng.integers(3, 6))
        tasks = tuple(int(x) for x in rng.integers(0, TASK_TYPES, size=n_tasks))
        specs.append(OrderSpec(now, tasks))
    return Instance(seed, regime, tuple(specs), proficiency)


class Policy:
    name = "policy"

    def reset(self) -> None:
        self.component_evals = 0
        self.full_component_evals = 0
        self.decision_count = 0

    def choose_seru(self, sim: "Simulator", order_id: int, now: float) -> int:
        raise NotImplementedError

    def choose_worker(
        self, sim: "Simulator", order_id: int, candidates: Sequence[int], now: float
    ) -> int:
        raise NotImplementedError


class FIFO(Policy):
    name = "FIFO"

    def choose_seru(self, sim: "Simulator", order_id: int, now: float) -> int:
        return min(range(SERUS), key=lambda m: (sim.seru_order_count(m), m))

    def choose_worker(self, sim: "Simulator", order_id: int, candidates: Sequence[int], now: float) -> int:
        return min(candidates)


class SkillMatch(Policy):
    name = "SMH"

    def choose_seru(self, sim: "Simulator", order_id: int, now: float) -> int:
        task = sim.orders[order_id].spec.tasks[0]
        return max(range(SERUS), key=lambda m: (sim.max_seru_skill(m, task, now), -m))

    def choose_worker(self, sim: "Simulator", order_id: int, candidates: Sequence[int], now: float) -> int:
        task = sim.current_task(order_id)
        return max(candidates, key=lambda w: (sim.skill(w, task, now), -w))


class LoadBalance(Policy):
    name = "LBH"

    def choose_seru(self, sim: "Simulator", order_id: int, now: float) -> int:
        return min(range(SERUS), key=lambda m: (sim.seru_work[m], m))

    def choose_worker(self, sim: "Simulator", order_id: int, candidates: Sequence[int], now: float) -> int:
        return min(candidates, key=lambda w: (sim.worker_work[w], w))


class ScoringPolicy(Policy):
    """Hierarchical policy with optional state-dependent objective weights."""

    def __init__(self, params: Sequence[float], adaptive: bool, name: str):
        self.params = np.asarray(params, dtype=float)
        self.adaptive = adaptive
        self.name = name
        needed = 14 if adaptive else 10
        if self.params.size != needed:
            raise ValueError(f"{name} needs {needed} parameters")
        self.base = np.exp(np.clip(self.params[:10], -3.0, 3.0))
        # Softplus permits an adaptive gain arbitrarily close to zero, so the
        # fixed policy is nested within the adaptive policy class.
        self.gains = np.log1p(np.exp(np.clip(self.params[10:14], -10.0, 6.0))) if adaptive else np.zeros(4)
        self.reset()

    def pressures(self, sim: "Simulator", now: float) -> Tuple[float, float, float]:
        if not self.adaptive:
            return 1.0, 1.0, 1.0
        demand = sim.demand_pressure()
        active = sum(1 for o in sim.orders if o.seru >= 0 and math.isnan(o.completion))
        congestion = min(2.5, max(0.0, demand - 1.0) + active / (2.0 * WORKERS))
        mean_work = float(np.mean(sim.worker_work)) + 1e-9
        dispersion = min(2.5, float(np.std(sim.worker_work)) / mean_work)
        coverage_gap = 1.0 - float(np.mean(sim.proficiency >= SKILL_THRESHOLD))
        eff = float(np.clip(1.0 + self.gains[0] * congestion, 0.25, 6.0))
        bal = float(np.clip(1.0 + self.gains[1] * dispersion, 0.25, 6.0))
        dev = float(np.clip(1.0 + self.gains[2] * coverage_gap - self.gains[3] * congestion, 0.25, 6.0))
        return eff, bal, dev

    def choose_seru(self, sim: "Simulator", order_id: int, now: float) -> int:
        task = sim.orders[order_id].spec.tasks[0]
        eff, bal, dev = self.pressures(sim, now)
        self.decision_count += 1
        self.full_component_evals += 5 * SERUS
        self.component_evals += 5 * SERUS
        scores = [sum(self.seru_components(sim, m, task, now, eff, bal, dev)) for m in range(SERUS)]
        return min(range(SERUS), key=lambda m: (scores[m], m))

    def choose_worker(self, sim: "Simulator", order_id: int, candidates: Sequence[int], now: float) -> int:
        task = sim.current_task(order_id)
        eff, bal, dev = self.pressures(sim, now)
        self.decision_count += 1
        self.full_component_evals += 5 * len(candidates)
        self.component_evals += 5 * len(candidates)
        scored = [(sum(self.worker_components(sim, w, task, now, eff, bal, dev)), w) for w in candidates]
        return min(scored)[1]

    def seru_components(
        self, sim: "Simulator", m: int, task: int, now: float,
        eff: float, bal: float, dev: float,
    ) -> Tuple[float, float, float, float, float]:
        mean_seru = float(np.mean(sim.seru_work)) + BASE_TIMES.mean()
        skill = sim.max_seru_skill(m, task, now)
        projected = sim.projected_seru_work(m, now) / mean_seru
        count = sim.seru_order_count(m) / max(1.0, len(sim.orders) / SERUS)
        skill_gap = 1.0 - skill
        imbalance = abs((sim.seru_work[m] + BASE_TIMES[task]) - mean_seru) / mean_seru
        learning = max(0.0, SKILL_THRESHOLD - skill)
        a = self.base[:5]
        return (
            -dev * a[4] * learning,
            eff * a[1] * count,
            bal * a[3] * imbalance,
            eff * a[2] * skill_gap,
            eff * a[0] * projected,
        )

    def worker_components(
        self, sim: "Simulator", w: int, task: int, now: float,
        eff: float, bal: float, dev: float,
    ) -> Tuple[float, float, float, float, float]:
        mean_load = float(np.mean(sim.worker_work)) + BASE_TIMES[task]
        a = self.base[5:]
        p = sim.skill(w, task, now)
        proc_time = sim.processing_time(task, p)
        proc = proc_time / BASE_TIMES[task]
        load = sim.worker_work[w] / mean_load
        skill_gap = 1.0 - p
        imbalance = abs((sim.worker_work[w] + proc_time) - mean_load) / mean_load
        learning = (1.0 - p) * (1.0 if p < SKILL_THRESHOLD else 0.35)
        return (
            -dev * a[4] * learning,
            eff * a[1] * load,
            bal * a[3] * imbalance,
            eff * a[2] * skill_gap,
            eff * a[0] * proc,
        )


class CertifiedLazyScoringPolicy(ScoringPolicy):
    """Exact action selection with lower-bound certified early stopping.

    Each score is one non-positive development term plus four nonnegative cost
    terms.  After any prefix of the nonnegative terms, the partial sum is a
    valid lower bound.  A candidate is discarded only when this lower bound
    cannot improve the incumbent, so the selected action is identical to full
    evaluation (including deterministic tie breaking).
    """

    def __init__(self, params: Sequence[float], adaptive: bool, name: str, bound_order: bool = True):
        super().__init__(params, adaptive, name)
        self.bound_order = bound_order

    @staticmethod
    def _can_prune(partial: float, candidate: int, best: float, best_id: int) -> bool:
        return partial > best + 1e-12 or (abs(partial - best) <= 1e-12 and candidate > best_id)

    def _lazy_min(self, component_map: Dict[int, Tuple[float, ...]]) -> int:
        ids = list(component_map)
        if self.bound_order:
            ids.sort(key=lambda j: (component_map[j][0], j))
        else:
            ids.sort()
        best_id = ids[0]
        first = component_map[best_id]
        best = sum(first)
        self.component_evals += len(first)
        for candidate in ids[1:]:
            comps = component_map[candidate]
            partial = comps[0]
            self.component_evals += 1
            pruned = self._can_prune(partial, candidate, best, best_id)
            if not pruned:
                for value in comps[1:]:
                    partial += value
                    self.component_evals += 1
                    if self._can_prune(partial, candidate, best, best_id):
                        pruned = True
                        break
            if not pruned and (partial < best - 1e-12 or (abs(partial - best) <= 1e-12 and candidate < best_id)):
                best, best_id = partial, candidate
        return best_id

    def _lazy_min_dynamic(
        self,
        ids: Sequence[int],
        lower_bound,
        positive_term,
    ) -> int:
        ordered = list(ids)
        bounds = {j: float(lower_bound(j)) for j in ordered}
        self.component_evals += len(ordered)
        if self.bound_order:
            ordered.sort(key=lambda j: (bounds[j], j))
        else:
            ordered.sort()
        best_id = ordered[0]
        best = bounds[best_id]
        for term_idx in range(4):
            best += float(positive_term(best_id, term_idx))
            self.component_evals += 1
        for candidate in ordered[1:]:
            partial = bounds[candidate]
            pruned = self._can_prune(partial, candidate, best, best_id)
            if not pruned:
                for term_idx in range(4):
                    partial += float(positive_term(candidate, term_idx))
                    self.component_evals += 1
                    if self._can_prune(partial, candidate, best, best_id):
                        pruned = True
                        break
            if not pruned and (partial < best - 1e-12 or (abs(partial - best) <= 1e-12 and candidate < best_id)):
                best, best_id = partial, candidate
        return best_id

    def choose_seru(self, sim: "Simulator", order_id: int, now: float) -> int:
        task = sim.orders[order_id].spec.tasks[0]
        eff, bal, dev = self.pressures(sim, now)
        self.decision_count += 1
        self.full_component_evals += 5 * SERUS
        mean_seru = float(np.mean(sim.seru_work)) + BASE_TIMES.mean()
        a = self.base[:5]
        skills: Dict[int, float] = {}
        def skill(m: int) -> float:
            if m not in skills:
                skills[m] = sim.max_seru_skill(m, task, now)
            return skills[m]
        def lower(m: int) -> float:
            return -dev * a[4] * max(0.0, SKILL_THRESHOLD - skill(m))
        def term(m: int, idx: int) -> float:
            if idx == 0:
                count = sim.seru_order_count(m) / max(1.0, len(sim.orders) / SERUS)
                return eff * a[1] * count
            if idx == 1:
                imbalance = abs((sim.seru_work[m] + BASE_TIMES[task]) - mean_seru) / mean_seru
                return bal * a[3] * imbalance
            if idx == 2:
                return eff * a[2] * (1.0 - skill(m))
            projected = sim.projected_seru_work(m, now) / mean_seru
            return eff * a[0] * projected
        return self._lazy_min_dynamic(range(SERUS), lower, term)

    def choose_worker(self, sim: "Simulator", order_id: int, candidates: Sequence[int], now: float) -> int:
        task = sim.current_task(order_id)
        eff, bal, dev = self.pressures(sim, now)
        self.decision_count += 1
        self.full_component_evals += 5 * len(candidates)
        mean_load = float(np.mean(sim.worker_work)) + BASE_TIMES[task]
        a = self.base[5:]
        skills = {w: sim.skill(w, task, now) for w in candidates}
        proc_times: Dict[int, float] = {}
        def proc(w: int) -> float:
            if w not in proc_times:
                proc_times[w] = sim.processing_time(task, skills[w])
            return proc_times[w]
        def lower(w: int) -> float:
            p = skills[w]
            learning = (1.0 - p) * (1.0 if p < SKILL_THRESHOLD else 0.35)
            return -dev * a[4] * learning
        def term(w: int, idx: int) -> float:
            if idx == 0:
                return eff * a[1] * sim.worker_work[w] / mean_load
            if idx == 1:
                imbalance = abs((sim.worker_work[w] + proc(w)) - mean_load) / mean_load
                return bal * a[3] * imbalance
            if idx == 2:
                return eff * a[2] * (1.0 - skills[w])
            return eff * a[0] * proc(w) / BASE_TIMES[task]
        return self._lazy_min_dynamic(candidates, lower, term)


class Simulator:
    def __init__(self, instance: Instance, policy: Policy):
        self.instance = instance
        self.policy = policy
        self.orders = [OrderState(x) for x in instance.orders]
        self.proficiency = instance.proficiency.copy()
        self.initial_proficiency = instance.proficiency.copy()
        self.worker_seru = np.repeat(np.arange(SERUS), WORKERS_PER_SERU)
        self.worker_busy_until = np.zeros(WORKERS)
        self.worker_work = np.zeros(WORKERS)
        self.seru_work = np.zeros(SERUS)
        self.recent_arrivals: List[float] = []
        self.events: List[Tuple[float, int, int, int]] = []
        self.serial = 0
        self.projection_calls = 0
        self.projection_task_scans = 0
        self.action_trace: List[Tuple[str, float, int, int]] = []

    @staticmethod
    def processing_time(task: int, p: float) -> float:
        return float(BASE_TIMES[task] / (1.0 + PROFICIENCY_EFFECT * p))

    def skill(self, worker: int, task: int, now: float) -> float:
        return float(self.proficiency[worker, task])

    def max_seru_skill(self, seru: int, task: int, now: float) -> float:
        ws = np.where(self.worker_seru == seru)[0]
        return float(np.max(self.proficiency[ws, task]))

    def current_task(self, order_id: int) -> int:
        order = self.orders[order_id]
        return order.spec.tasks[order.next_task]

    def seru_order_count(self, seru: int) -> int:
        return sum(1 for o in self.orders if o.seru == seru and math.isnan(o.completion))

    def projected_seru_work(self, seru: int, now: float) -> float:
        self.projection_calls += 1
        remaining = sum(max(0.0, self.worker_busy_until[w] - now) for w in np.where(self.worker_seru == seru)[0])
        for order in self.orders:
            if order.seru != seru or not math.isnan(order.completion):
                continue
            start = order.next_task + (1 if order.processing else 0)
            for task in order.spec.tasks[start:]:
                self.projection_task_scans += 1
                p = self.max_seru_skill(seru, task, now)
                remaining += self.processing_time(task, p)
        return remaining / WORKERS_PER_SERU

    def demand_pressure(self) -> float:
        if len(self.recent_arrivals) < 4:
            return 1.0
        gaps = np.diff(self.recent_arrivals[-8:])
        observed = 1.0 / max(1e-6, float(np.mean(gaps)))
        return observed / 0.19

    def push(self, when: float, kind: int, a: int, b: int = -1) -> None:
        self.serial += 1
        heapq.heappush(self.events, (when, self.serial, kind, a if b < 0 else (a * 1000 + b)))

    def run(self) -> Dict[str, float]:
        self.policy.reset()
        wall_start = time.perf_counter()
        for i, order in enumerate(self.orders):
            self.push(order.spec.arrival, 0, i)

        while self.events:
            now = self.events[0][0]
            batch = []
            while self.events and abs(self.events[0][0] - now) < 1e-10:
                batch.append(heapq.heappop(self.events))
            # Completions are applied before arrivals and new assignments.
            for _, _, kind, payload in sorted(batch, key=lambda x: -x[2]):
                if kind == 1:
                    worker, order_id = payload // 1000, payload % 1000
                    order = self.orders[order_id]
                    task = order.spec.tasks[order.next_task]
                    p = self.proficiency[worker, task]
                    self.proficiency[worker, task] = min(1.0, p + LEARNING_RATES[task] * (1.0 - p))
                    order.next_task += 1
                    order.processing = False
                    self.worker_busy_until[worker] = now
                    if order.next_task == len(order.spec.tasks):
                        order.completion = now
                else:
                    order_id = payload
                    self.recent_arrivals.append(now)
                    seru = self.policy.choose_seru(self, order_id, now)
                    self.orders[order_id].seru = seru
                    self.action_trace.append(("dispatch", round(now, 12), order_id, seru))
            self.schedule_ready(now)

        completions = np.array([o.completion for o in self.orders])
        arrivals = np.array([o.spec.arrival for o in self.orders])
        makespan = float(completions.max() - arrivals.min())
        mean_flow = float(np.mean(completions - arrivals))
        worker_cv2 = float(np.var(self.worker_work) / (np.mean(self.worker_work) ** 2 + 1e-9))
        seru_cv2 = float(np.var(self.seru_work) / (np.mean(self.seru_work) ** 2 + 1e-9))
        gain = float(np.mean(self.proficiency - self.initial_proficiency))
        coverage = float(np.mean(self.proficiency >= SKILL_THRESHOLD))
        composite = makespan / 600.0 + mean_flow / 100.0 + 0.35 * worker_cv2 + 0.25 * seru_cv2
        composite -= 2.0 * gain + 0.30 * coverage
        wall_time_ms = 1000.0 * (time.perf_counter() - wall_start)
        full_evals = float(getattr(self.policy, "full_component_evals", 0))
        actual_evals = float(getattr(self.policy, "component_evals", 0))
        pruning_rate = 0.0 if full_evals <= 0 else 1.0 - actual_evals / full_evals
        return {
            "makespan": makespan,
            "mean_flow": mean_flow,
            "worker_cv2": worker_cv2,
            "seru_cv2": seru_cv2,
            "skill_gain": gain,
            "coverage": coverage,
            "composite": composite,
            "component_evals": actual_evals,
            "full_component_evals": full_evals,
            "pruning_rate": pruning_rate,
            "decision_count": float(getattr(self.policy, "decision_count", 0)),
            "wall_time_ms": wall_time_ms,
            "projection_calls": float(self.projection_calls),
            "projection_task_scans": float(self.projection_task_scans),
        }

    def schedule_ready(self, now: float) -> None:
        while True:
            choices = []
            for order_id, order in enumerate(self.orders):
                if order.seru < 0 or order.processing or not math.isnan(order.completion):
                    continue
                task = self.current_task(order_id)
                candidates = [
                    int(w)
                    for w in np.where(self.worker_seru == order.seru)[0]
                    if self.worker_busy_until[w] <= now + 1e-10 and self.proficiency[w, task] >= 0.30
                ]
                if candidates:
                    choices.append((order.spec.arrival, order_id, candidates))
            if not choices:
                return
            _, order_id, candidates = min(choices, key=lambda x: (x[0], x[1]))
            worker = self.policy.choose_worker(self, order_id, candidates, now)
            self.action_trace.append(("assign", round(now, 12), order_id, worker))
            order = self.orders[order_id]
            task = self.current_task(order_id)
            duration = self.processing_time(task, self.proficiency[worker, task])
            finish = now + duration
            order.processing = True
            self.worker_busy_until[worker] = finish
            self.worker_work[worker] += duration
            self.seru_work[order.seru] += duration
            self.push(finish, 1, worker, order_id)


def training_instances() -> List[Instance]:
    regimes = ("steady", "surge", "recovery")
    return [make_instance(1000 + i, regimes[i % 3], orders=42) for i in range(9)]


def objective(params: np.ndarray, adaptive: bool, instances: Sequence[Instance]) -> float:
    name = "adaptive" if adaptive else "fixed"
    policy = ScoringPolicy(params, adaptive, name)
    vals = [Simulator(x, policy).run()["composite"] for x in instances]
    return float(np.mean(vals))


def train_cem(
    adaptive: bool,
    seed: int,
    iterations: int = 12,
    population: int = 28,
    fixed_base: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    if adaptive and fixed_base is not None:
        dim = 4
        mean = np.ones(dim) * -2.0
        sd = np.ones(dim) * 1.2
        def expand(x: np.ndarray) -> np.ndarray:
            return np.concatenate([fixed_base, x])
    else:
        dim = 14 if adaptive else 10
        mean = np.zeros(dim)
        sd = np.ones(dim) * 0.8
        def expand(x: np.ndarray) -> np.ndarray:
            return x
    instances = training_instances()
    history = []
    elite_n = max(4, population // 5)
    best_p = mean.copy()
    if adaptive and fixed_base is not None:
        best_p = np.ones(dim) * -9.0
        best_v = objective(expand(best_p), True, instances)
    else:
        best_v = math.inf
    for _ in range(iterations):
        population_params = rng.normal(mean, sd, size=(population, dim))
        values = np.array([objective(expand(p), adaptive, instances) for p in population_params])
        elite_idx = np.argsort(values)[:elite_n]
        elite = population_params[elite_idx]
        mean = 0.70 * mean + 0.30 * elite.mean(axis=0)
        sd = np.maximum(0.08, 0.70 * sd + 0.30 * elite.std(axis=0))
        if values[elite_idx[0]] < best_v:
            best_v = float(values[elite_idx[0]])
            best_p = population_params[elite_idx[0]].copy()
        history.append(best_v)
    return expand(best_p), history


def sign_test_p(differences: Sequence[float]) -> float:
    nonzero = [x for x in differences if abs(x) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(sum(x > 0 for x in nonzero), sum(x < 0 for x in nonzero))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = [
        "makespan", "mean_flow", "worker_cv2", "seru_cv2", "skill_gain",
        "coverage", "composite", "component_evals", "full_component_evals",
        "pruning_rate", "decision_count", "wall_time_ms",
        "projection_calls", "projection_task_scans",
    ]
    result = []
    methods = sorted({str(r["method"]) for r in rows})
    regimes = ["all", "steady", "surge", "recovery"]
    for regime in regimes:
        for method in methods:
            subset = [r for r in rows if r["method"] == method and (regime == "all" or r["regime"] == regime)]
            out: Dict[str, object] = {"regime": regime, "method": method, "n": len(subset)}
            for metric in metrics:
                values = [float(r[metric]) for r in subset]
                mean = statistics.fmean(values)
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                half = 1.96 * sd / math.sqrt(len(values)) if values else math.nan
                out[f"{metric}_mean"] = mean
                out[f"{metric}_sd"] = sd
                out[f"{metric}_ci_low"] = mean - half
                out[f"{metric}_ci_high"] = mean + half
            result.append(out)
    return result


def make_svg(path: Path, summary: Sequence[Dict[str, object]]) -> None:
    data = [r for r in summary if r["regime"] == "all"]
    methods = [str(r["method"]) for r in data]
    values = [float(r["composite_mean"]) for r in data]
    width, height = 860, 460
    left, top, bottom = 90, 50, 90
    plot_h = height - top - bottom
    max_v, min_v = max(values) * 1.08, min(values) * 0.92
    bar_w = 80
    gap = (width - left - 40 - bar_w * len(values)) / max(1, len(values) - 1)
    def y(v: float) -> float:
        return top + (max_v - v) / (max_v - min_v) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.label{font-size:15px}.title{font-size:21px;font-weight:bold}.value{font-size:14px}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">Paired test performance (lower composite score is better)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{width-35}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        v = min_v + (max_v - min_v) * i / 5
        yy = y(v)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-35}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" class="label">{v:.2f}</text>')
    colors = {"Adaptive-HPS": "#0072B2", "Fixed-HPS": "#E69F00", "FIFO": "#999999", "LBH": "#56B4E9", "SMH": "#009E73"}
    for i, (method, value) in enumerate(zip(methods, values)):
        x = left + 22 + i * (bar_w + gap)
        yy = y(value)
        parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w}" height="{top+plot_h-yy:.1f}" fill="{colors.get(method, "#777")}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{yy-7:.1f}" text-anchor="middle" class="value">{value:.3f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="label">{method}</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" class="label">Composite score</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def latex_table(summary: Sequence[Dict[str, object]]) -> str:
    rows = [r for r in summary if r["regime"] == "all"]
    rows.sort(key=lambda r: float(r["composite_mean"]))
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Method & Makespan & Mean flow & Worker CV$^2$ & Seru CV$^2$ & $\Delta P$ & Coverage \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['method']} & {float(r['makespan_mean']):.2f} & {float(r['mean_flow_mean']):.2f} & "
            f"{float(r['worker_cv2_mean']):.3f} & {float(r['seru_cv2_mean']):.3f} & "
            f"{float(r['skill_gain_mean']):.4f} & {float(r['coverage_mean']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--test-seeds", type=int, default=60, help="total paired test instances")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--parameter-file", type=Path, default=None, help="reuse fixed/adaptive parameters from metadata.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.parameter_file is not None:
        saved = json.loads(args.parameter_file.read_text(encoding="utf-8"))
        fixed = np.asarray(saved["fixed_parameters"], dtype=float)
        adaptive = np.asarray(saved["adaptive_parameters"], dtype=float)
        fixed_history = list(saved.get("fixed_training_best", []))
        adaptive_history = list(saved.get("adaptive_training_best", []))
        print("reused saved parameters", flush=True)
    else:
        fixed, fixed_history = train_cem(False, 20261106, args.iterations, args.population)
        print("trained fixed policy", flush=True)
        adaptive, adaptive_history = train_cem(
            True, 20261107, args.iterations, args.population, fixed_base=fixed
        )
        print("trained adaptive policy", flush=True)
    policies: List[Policy] = [
        FIFO(), SkillMatch(), LoadBalance(),
        ScoringPolicy(fixed, False, "Fixed-HPS"),
        ScoringPolicy(adaptive, True, "Adaptive-HPS"),
        CertifiedLazyScoringPolicy(adaptive, True, "Adaptive-CLHPS", bound_order=False),
        CertifiedLazyScoringPolicy(adaptive, True, "Adaptive-Lazy-BoundOrder", bound_order=True),
    ]

    regimes = ("steady", "surge", "recovery")
    rows: List[Dict[str, object]] = []
    for i in range(args.test_seeds):
        regime = regimes[i % 3]
        instance = make_instance(50000 + i, regime)
        instance_rows: Dict[str, Dict[str, object]] = {}
        instance_traces: Dict[str, List[Tuple[str, float, int, int]]] = {}
        for policy in policies:
            simulator = Simulator(instance, policy)
            metrics = simulator.run()
            row = {"seed": instance.seed, "regime": regime, "method": policy.name, **metrics}
            rows.append(row)
            instance_rows[policy.name] = row
            instance_traces[policy.name] = simulator.action_trace
        # Certified lazy evaluation must reproduce the fully evaluated action
        # sequence and therefore every terminal quality metric exactly.
        reference = instance_rows["Adaptive-HPS"]
        for lazy_name in ("Adaptive-CLHPS", "Adaptive-Lazy-BoundOrder"):
            if instance_traces[lazy_name] != instance_traces["Adaptive-HPS"]:
                raise AssertionError(f"certified pruning changed the action trace on seed {instance.seed}")
            for metric in ("makespan", "mean_flow", "worker_cv2", "seru_cv2", "skill_gain", "coverage", "composite"):
                if not math.isclose(float(reference[metric]), float(instance_rows[lazy_name][metric]), rel_tol=0.0, abs_tol=1e-10):
                    raise AssertionError(f"certified pruning changed {metric} on seed {instance.seed}")

    raw_fields = [
        "seed", "regime", "method", "makespan", "mean_flow", "worker_cv2",
        "seru_cv2", "skill_gain", "coverage", "composite", "component_evals",
        "full_component_evals", "pruning_rate", "decision_count", "wall_time_ms",
        "projection_calls", "projection_task_scans",
    ]
    write_csv(args.output / "raw_results.csv", rows, raw_fields)
    summary = summarize(rows)
    summary_fields = list(summary[0].keys())
    write_csv(args.output / "summary.csv", summary, summary_fields)

    paired = []
    metrics = ["makespan", "mean_flow", "worker_cv2", "seru_cv2", "skill_gain", "coverage", "composite"]
    for metric in metrics:
        a = [float(r[metric]) for r in rows if r["method"] == "Adaptive-HPS"]
        f = [float(r[metric]) for r in rows if r["method"] == "Fixed-HPS"]
        diffs = [x - y for x, y in zip(a, f)]
        paired.append({
            "metric": metric,
            "adaptive_minus_fixed_mean": statistics.fmean(diffs),
            "adaptive_better_fraction": statistics.fmean(
                (d < 0) if metric not in ("skill_gain", "coverage") else (d > 0) for d in diffs
            ),
            "two_sided_sign_test_p": sign_test_p(diffs),
            "n": len(diffs),
        })
    write_csv(args.output / "paired_tests.csv", paired, list(paired[0].keys()))

    # Scalability audit: policy parameters remain fixed; only episode size grows.
    scale_rows: List[Dict[str, object]] = []
    for order_count in (40, 60, 100):
        for j in range(15):
            regime = regimes[j % 3]
            instance = make_instance(70000 + order_count * 100 + j, regime, orders=order_count)
            scale_traces: Dict[str, List[Tuple[str, float, int, int]]] = {}
            for policy in (
                ScoringPolicy(adaptive, True, "Adaptive-HPS"),
                CertifiedLazyScoringPolicy(adaptive, True, "Adaptive-CLHPS", bound_order=False),
            ):
                simulator = Simulator(instance, policy)
                metrics_out = simulator.run()
                scale_rows.append({"orders": order_count, "seed": instance.seed, "regime": regime, "method": policy.name, **metrics_out})
                scale_traces[policy.name] = simulator.action_trace
            if scale_traces["Adaptive-CLHPS"] != scale_traces["Adaptive-HPS"]:
                raise AssertionError(
                    f"certified pruning changed the scale-test action trace on seed {instance.seed}"
                )
    write_csv(args.output / "scale_results.csv", scale_rows, ["orders", *raw_fields])
    scale_summary: List[Dict[str, object]] = []
    for order_count in (40, 60, 100):
        for method in ("Adaptive-HPS", "Adaptive-CLHPS"):
            subset = [r for r in scale_rows if r["orders"] == order_count and r["method"] == method]
            scale_summary.append({
                "orders": order_count,
                "method": method,
                "n": len(subset),
                "component_evals_mean": statistics.fmean(float(r["component_evals"]) for r in subset),
                "pruning_rate_mean": statistics.fmean(float(r["pruning_rate"]) for r in subset),
                "wall_time_ms_mean": statistics.fmean(float(r["wall_time_ms"]) for r in subset),
                "projection_calls_mean": statistics.fmean(float(r["projection_calls"]) for r in subset),
                "projection_task_scans_mean": statistics.fmean(float(r["projection_task_scans"]) for r in subset),
                "composite_mean": statistics.fmean(float(r["composite"]) for r in subset),
            })
    write_csv(args.output / "scale_summary.csv", scale_summary, list(scale_summary[0].keys()))
    (args.output / "table_overall.tex").write_text(latex_table(summary), encoding="utf-8")
    make_svg(args.output / "policy_comparison.svg", summary)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "test_instances": args.test_seeds,
        "paired_design": True,
        "parameter_source": "trained_in_run" if args.parameter_file is None else str(args.parameter_file),
        "training_instance_seeds": [1000 + i for i in range(9)],
        "test_instance_seeds": [50000 + i for i in range(args.test_seeds)],
        "fixed_parameters": fixed.tolist(),
        "adaptive_parameters": adaptive.tolist(),
        "fixed_training_best": fixed_history,
        "adaptive_training_best": adaptive_history,
        "notes": [
            "Synthetic instances only; no empirical shop-floor data were supplied.",
            "Sign tests are exact two-sided paired tests and require no SciPy dependency.",
            "The original manuscript tables were not used as input to this experiment.",
            "Certified lazy variants were asserted to match full evaluation on every action and terminal quality metric.",
            "Component-evaluation counts, not wall-clock time alone, are the implementation-independent computation metric.",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runtime_seconds": metadata["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
