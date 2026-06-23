"""
ARC Prize 2026 — Evaluation Script
====================================
Compares beam-only vs hybrid (beam + GA) on a subset of training tasks.
"""

import json
import time
from arc_solver import beam_search_solve
from hybrid_solver import hybrid_solve
from heuristics import ALL_SOLVERS

# Load training data
with open('arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json', 'r') as f:
    training_challenges = json.load(f)

print(f"Loaded {len(training_challenges)} training challenges.\n")

# Evaluate on a subset
subset_keys = list(training_challenges.keys())[:20]

# ── Beam-Only Baseline ────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Beam Search Only (baseline)")
print("=" * 60)

beam_solved = 0
beam_scores = {}
t0 = time.time()

for key in subset_keys:
    task = training_challenges[key]
    predictions, score = beam_search_solve(task, ALL_SOLVERS, beam_width=10)
    beam_scores[key] = score
    tag = "SOLVED" if score == 1.0 else f"  score={score:.3f}"
    print(f"  {key}: {tag}")
    if score == 1.0:
        beam_solved += 1

beam_time = time.time() - t0
print(f"\nBeam Search: {beam_solved}/{len(subset_keys)} solved in {beam_time:.1f}s\n")

# ── Hybrid (Beam + GA) ───────────────────────────────────────────────
print("=" * 60)
print("PHASE 2: Hybrid (Beam Search + Genetic Algorithm)")
print("=" * 60)

hybrid_solved = 0
hybrid_scores = {}
t0 = time.time()

for key in subset_keys:
    task = training_challenges[key]
    predictions, score = hybrid_solve(
        task, ALL_SOLVERS,
        beam_width=10,
        population_size=100,
        max_generations=30,
        ga_verbose=False,
    )
    hybrid_scores[key] = score
    tag = "SOLVED" if score == 1.0 else f"  score={score:.3f}"

    # Highlight improvements over beam
    if score > beam_scores[key]:
        delta = score - beam_scores[key]
        tag += f"  [+{delta:.3f} vs beam]"

    print(f"  {key}: {tag}")
    if score == 1.0:
        hybrid_solved += 1

hybrid_time = time.time() - t0
print(f"\nHybrid: {hybrid_solved}/{len(subset_keys)} solved in {hybrid_time:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Beam Search:  {beam_solved}/{len(subset_keys)} perfectly solved  ({beam_time:.1f}s)")
print(f"  Hybrid (B+GA): {hybrid_solved}/{len(subset_keys)} perfectly solved ({hybrid_time:.1f}s)")

if hybrid_solved > beam_solved:
    print(f"\n  >> GA improved: +{hybrid_solved - beam_solved} additional tasks solved!")
elif hybrid_solved == beam_solved:
    print(f"\n  == Same solve count (GA may have improved partial scores)")
else:
    print(f"\n  !! Regression detected - investigate")

# Show tasks where GA improved scores
improved = [(k, beam_scores[k], hybrid_scores[k])
            for k in subset_keys if hybrid_scores[k] > beam_scores[k]]
if improved:
    print(f"\n  Tasks with improved scores ({len(improved)}):")
    for k, bs, hs in improved:
        print(f"    {k}: {bs:.3f} -> {hs:.3f}  (+{hs-bs:.3f})")
