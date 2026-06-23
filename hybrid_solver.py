"""
ARC Prize 2026 — Hybrid Solver v2 (Beam Search + Island-Model GA)
==================================================================
Cascade strategy:
  1. Beam Search runs first  (fast deterministic exploitation)
     - Tries all 15 heuristic solvers + depth-2 compositions
     - Returns immediately if a perfect score is found
  2. Island-Model GA kicks in (global stochastic exploration)
     - Seeded with both beam-search survivors AND diff-based task seeds
     - 4 specialised sub-populations with migration
     - Schema-preserving crossover + local hill-climbing
  3. Merge & Rank
     - Beam candidates and GA programs are re-scored on raw accuracy
     - Top-2 returned as attempt_1 / attempt_2
"""

from arc_solver import (
    Grid, beam_search_solve_extended, grid_similarity,
)
from genetic_solver import GeneticSolver, extract_seed_programs


def hybrid_solve(task, solvers,
                 beam_width=10,
                 population_size=120,
                 max_generations=50,
                 max_program_length=8,
                 ga_verbose=False):
    """
    Hybrid Beam Search + Island-Model GA solver.

    Args:
        task:               dict with 'train' and 'test' lists
        solvers:            list of heuristic solver functions (for beam search)
        beam_width:         beam search width (top-k candidates kept)
        population_size:    total GA population (split across 4 islands)
        max_generations:    GA generation limit
        max_program_length: maximum gene sequence length for GA programs
        ga_verbose:         print GA progress per generation

    Returns:
        (predictions, best_score)
          predictions: list of {'attempt_1': ..., 'attempt_2': ...}
          best_score:  float in [0, 1] — best training accuracy found
    """
    train_pairs = task['train']
    test_inputs = task['test']

    # ── Phase 1: Beam Search (fast exploitation) ──────────────────────────
    beam_predictions, beam_score, beam_candidates = beam_search_solve_extended(
        task, solvers, beam_width=beam_width
    )

    # Perfect solution → return immediately, no GA needed
    if beam_score >= 1.0:
        return beam_predictions, beam_score

    # ── Phase 2: Island-Model GA (global exploration) ─────────────────────
    # Seed GA with the beam search's best symbolic programs
    seed_programs = extract_seed_programs(beam_candidates)

    ga = GeneticSolver(
        population_size=population_size,
        max_generations=max_generations,
        crossover_rate=0.70,
        mutation_rate=0.30,
        elite_fraction=0.10,
        max_program_length=max_program_length,
        tournament_size=5,
        migration_interval=5,
        migration_size=3,
        local_search_fraction=0.20,
        verbose=ga_verbose,
    )

    ga_population = ga.solve(task, seed_programs=seed_programs)

    # ── Phase 3: Merge & Rank ─────────────────────────────────────────────
    # Re-score everything on pure cell-accuracy (no simplicity penalty)
    # so final ranking is fair across beam and GA candidates.

    all_candidates = []

    # Beam candidates
    for candidate in beam_candidates:
        all_candidates.append({
            'name':  candidate.name,
            'score': candidate.score,
            'apply': candidate.apply,
        })

    # GA candidates — re-score on raw accuracy
    n = len(train_pairs)
    for program in ga_population[:beam_width * 2]:
        accuracy = 0.0
        for pair in train_pairs:
            inp = Grid(pair['input'])
            expected = Grid(pair['output'])
            try:
                predicted = program.apply(inp)
                if predicted is not None:
                    accuracy += grid_similarity(predicted, expected)
            except Exception:
                pass
        accuracy /= n if n > 0 else 1

        all_candidates.append({
            'name':  f"GA[{' → '.join(program.genes)}]",
            'score': accuracy,
            'apply': program.apply,
        })

    # Sort by accuracy descending
    all_candidates.sort(key=lambda c: c['score'], reverse=True)

    # ── Phase 4: Generate predictions from top-2 candidates ───────────────
    best_score = all_candidates[0]['score'] if all_candidates else 0.0

    predictions = []
    for test in test_inputs:
        inp = Grid(test['input'])

        # Attempt 1: best candidate
        try:
            attempt_1 = all_candidates[0]['apply'](inp)
            if attempt_1 is None:
                attempt_1 = inp
        except Exception:
            attempt_1 = inp

        # Attempt 2: second-best (different from attempt_1 if possible)
        attempt_2 = attempt_1
        for cand in all_candidates[1:]:
            try:
                a2 = cand['apply'](inp)
                if a2 is not None and a2 != attempt_1:
                    attempt_2 = a2
                    break
            except Exception:
                continue

        predictions.append({
            'attempt_1': attempt_1.to_list(),
            'attempt_2': attempt_2.to_list(),
        })

    return predictions, best_score
