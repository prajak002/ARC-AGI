# ARC-AGI Hybrid Solver

This repository contains a hybrid symbolic solver for the ARC Prize 2026 / ARC-AGI-2 task format. The goal is to infer a transformation rule from a small number of training input-output grid pairs, apply that rule to hidden test inputs, and write a Kaggle-compatible `submission.json`.

ARC tasks are not ordinary numeric matrix problems. A grid is a small colored scene. Each integer from `0` to `9` represents a color, and the solver must reason about objects, shapes, symmetry, repetition, cropping, movement, color replacement, and other visual transformations.

## Problem Setting

Each task contains:

- `train`: a few examples of `input -> output`
- `test`: one or more input grids whose outputs must be predicted

For every test input, Kaggle expects exactly two predictions:

```json
{
  "attempt_1": [[0, 1], [2, 3]],
  "attempt_2": [[0, 1], [2, 3]]
}
```

The final `submission.json` must contain every task id from `arc-agi_test_challenges.json`.

## Exact Grid Matching Rule

The official scoring rule is strict exact matching.

A prediction receives score `1` only if:

- the predicted grid has the same height as the ground truth
- the predicted grid has the same width as the ground truth
- every cell has the exact same color value in the exact same position

If even one cell is wrong, the official score for that test output is `0`.

This is not matrix subtraction. Matrix subtraction can help debug differences, but final matching is element-wise equality:

```python
np.array_equal(predicted, expected)
```

Example:

```text
expected:   8 points correct
predicted:  5 points correct, 3 points wrongly placed

official score: 0
internal partial similarity: 5 / 8 = 0.625
```

The solver uses partial similarity internally to rank candidate rules, but Kaggle only rewards exact full-grid matches.

## Motivation

ARC tasks are designed to resist memorization. A useful solver must generalize from very few examples. Pure brute force is too large, and pure hand-written heuristics are too narrow.

This project uses a hybrid approach:

1. Fast symbolic heuristics search for common ARC transformations.
2. Beam search ranks the best candidate rules.
3. A genetic algorithm explores compositions of primitive transformations.
4. The final system returns the top two predictions for each test grid.

The motivation is pragmatic: many ARC tasks are solved by simple rules, but the correct simple rule is not known in advance. The solver therefore generates many plausible symbolic rules, scores them on the training pairs, and applies the best rules to the test inputs.

## Core Intuition

An ARC grid is treated as a structured visual object, not as a plain matrix.

The solver asks questions like:

- Is the output a rotation or mirror of the input?
- Is the output a crop of the meaningful non-background content?
- Did a color mapping happen?
- Were objects moved by gravity?
- Was a pattern tiled or scaled?
- Was a missing symmetry completed?
- Were connected components extracted, removed, or recolored?
- Can several small transformations be composed into the final rule?

The central intuition is:

> If a candidate transformation explains all training examples well, it is more likely to be the hidden rule for the test inputs.

Because exact matching is hard, the solver first uses partial cell similarity to guide search, then tries to produce exact predictions.

## Repository Structure

```text
ARC_Hybrid_Solver_Final.ipynb    Self-contained Kaggle notebook
arc_solver.py                    Core grid representation, primitives, beam search
heuristics.py                    Hand-written symbolic candidate generators
genetic_solver.py                Island-model genetic search over primitive programs
hybrid_solver.py                 Beam + genetic solver orchestrator
solve_and_submit.py              Script that writes submission.json
evaluate.py                      Local evaluation helper
visualize_demo.py                Visualization helper
build_notebook.py                Builds the self-contained notebook
arc-prize-2026-arc-agi-2/        ARC-AGI-2 challenge and solution JSON files
submission.json                  Generated Kaggle submission file
```

## How The Hybrid Solver Works

### 1. Grid Representation

`arc_solver.py` defines the `Grid` class. It wraps a 2D array of integers and provides helper methods for shape, color counts, background color estimation, and conversion back to Python lists for JSON output.

### 2. Object Extraction

The solver uses BFS connected components to identify objects in the grid. An object is a connected group of non-background cells. For each object, the solver tracks coordinates, bounding box, area, colors, primary color, centroid, and extracted subgrid.

This supports object-level reasoning such as largest object extraction, smallest object extraction, object cropping, and spatial relationship analysis.

### 3. Primitive Transformations

The solver defines many primitive operations, including:

- identity
- rotate 90 / 180 / 270
- horizontal and vertical mirror
- transpose
- crop to content
- scale up and downscale
- tile
- color swap and color map
- flood fill
- overlay
- gravity in multiple directions
- border add/remove
- symmetry enforcement
- object outline
- keep/remove/fill specific colors
- boolean half-grid operations such as OR, AND, XOR

These primitives are the alphabet of the solver.

### 4. Heuristic Candidate Generation

`heuristics.py` contains specialized solvers. Each heuristic looks at the training pairs and proposes candidate transformation rules.

Examples:

- `solver_geometric` proposes rotations and mirrors.
- `solver_tiling` proposes repeated patterns.
- `solver_scaling` proposes integer upscaling or downscaling.
- `solver_color_mapping` proposes color remaps.
- `solver_crop` proposes cropping to content.
- `solver_gravity` proposes object movement under gravity.
- `solver_symmetry_completion` proposes completing missing symmetry.
- `solver_overlay` proposes logical combinations of grid halves.

These heuristics are fast and give the search a strong starting point.

### 5. Beam Search

Beam search scores every candidate rule against all training pairs.

For each candidate:

1. Apply the rule to every training input.
2. Compare the result with the training output.
3. Compute average similarity.
4. Keep the best candidates.

If no candidate is good enough, beam search also tries depth-2 compositions, such as:

```text
crop -> rotate90
mirror_h -> color_map
extract_largest -> scale_2x
```

Beam search is useful because it quickly finds simple transformations.

### 6. Genetic Algorithm

`genetic_solver.py` adds a broader search layer.

A program is a list of primitive names:

```python
["crop", "mirror_h", "fill_enclosed"]
```

The genetic algorithm evolves these programs using task-aware seed programs, random generation, tournament selection, crossover, mutation, local hill climbing, and island-model populations.

The island model keeps separate sub-populations with different behavior:

- Explorer: more mutation, broader search
- Exploiter: stronger selection around good candidates
- ColorSpec: color-heavy primitives
- GeoSpec: geometry-heavy primitives

This helps avoid getting stuck too early in one weak local solution.

### 7. Hybrid Orchestration

`hybrid_solver.py` combines the two search strategies:

1. Run beam search first.
2. If beam search finds a perfect training solution, return it immediately.
3. Otherwise, seed the genetic algorithm with beam survivors.
4. Run genetic search.
5. Merge beam and GA candidates.
6. Re-score candidates on raw training accuracy.
7. Use the top two distinct candidates as `attempt_1` and `attempt_2`.

This gives the solver both speed and exploration.

## Why Two Attempts Matter

Kaggle allows two attempts per test output.

The solver uses:

- `attempt_1`: best-scoring candidate
- `attempt_2`: best alternative candidate that differs from attempt 1 when possible

This is useful when the training examples are ambiguous. Two different rules may fit the training data equally well, but only one may generalize to the hidden test output.

## Important Corollaries

### Corollary 1: Partial correctness is not official correctness

If 5 cells are correct and 3 are wrong, the internal similarity may be useful, but the official score is still `0`.

### Corollary 2: Shape must be correct before color accuracy matters

If the predicted grid has the wrong height or width, it cannot be an exact match.

### Corollary 3: A simple rule is often better than a complex rule

ARC tasks usually prefer clean abstractions. A short transformation that explains all examples is often more reliable than a long accidental program.

### Corollary 4: Training accuracy is necessary but not sufficient

A rule can fit the examples and still fail the hidden test. This is why the solver keeps two attempts and prefers general symbolic operations over memorized outputs.

### Corollary 5: ARC grids are visual structures

Treating the data only as numeric matrices misses important concepts like objects, holes, symmetry, borders, and relative placement.

## Running Locally

Generate a submission from the test challenges:

```bash
python solve_and_submit.py
```

This writes:

```text
submission.json
```

Run a small evaluation helper:

```bash
python evaluate.py
```

Rebuild the self-contained notebook:

```bash
python build_notebook.py
```

## Kaggle Submission Contract

The generated `submission.json` must look like:

```json
{
  "00576224": [
    {
      "attempt_1": [[0, 0], [0, 0]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ]
}
```

Rules:

- every task id from the test challenges file must be present
- each task value must be a list
- the list length must match the number of test inputs
- every prediction must contain `attempt_1` and `attempt_2`
- each attempt must be a rectangular 2D list
- all cell values must be integers from `0` to `9`

## Limitations

This solver is symbolic and heuristic. It does not understand every possible ARC abstraction.

Known limitations:

- It may overfit small training sets.
- It may miss transformations not included in the primitive registry.
- Genetic search can be slow on difficult tasks.
- Partial similarity can rank a visually wrong candidate above a structurally better one.
- Exact hidden test performance depends on whether the true rule is expressible by the available primitives.

## Final Conclusion

The hybrid solver is built around a simple idea:

> Generate many plausible symbolic transformations, score them on the examples, compose and evolve better programs, then submit the two most likely outputs.

Beam search gives speed. Heuristics give domain knowledge. The genetic algorithm gives broader exploration. Exact grid matching defines the final objective.

The project therefore combines practical ARC reasoning with a Kaggle-ready submission pipeline: load challenges, infer rules, generate two attempts per test grid, validate the JSON structure, and write `submission.json`.
