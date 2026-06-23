import re
import os
import json

def new_notebook():
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def new_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def new_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


nb = new_notebook()

# ---------------------------------------------------------------------------
# 1. Notebook Title & Intro
# ---------------------------------------------------------------------------
markdown_intro = """# ARC Prize 2026 - Hybrid Solver (Beam Search + Island-Model GA)

This notebook is a self-contained, singleton solution designed for the Kaggle environment. 
It breaks down the solver architecture into distinct, readable code cells and concludes with an **Exploratory Data Analysis (EDA)** section that runs the solver on sample data and visually plots the predictions.

### Architecture Overview
1. **Core Solver Components**: Grid representation, Object extraction, and Primitive transformations (75+ operations).
2. **Heuristic Solvers**: Fast deterministic algorithms for common ARC patterns.
3. **Genetic Algorithm**: Island-model GA with diff-based seeding and schema-preserving crossover for complex tasks.
4. **Hybrid Orchestrator**: Combines Beam Search and GA into a unified prediction pipeline.
5. **Exploratory Analysis & Visualization**: Output-centric testing on evaluation datasets.
"""
nb["cells"].append(new_markdown_cell(markdown_intro))

# ---------------------------------------------------------------------------
# Helper to clean local imports
# ---------------------------------------------------------------------------
def clean_imports(code):
    """Remove local module imports after inlining files into the notebook."""
    local_modules = {"arc_solver", "genetic_solver", "heuristics", "hybrid_solver"}
    cleaned = []
    lines = code.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        match = re.match(r"from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\b", stripped)
        if match and match.group(1) in local_modules:
            # Skip single-line imports and parenthesized multi-line imports.
            if "(" in stripped and ")" not in stripped:
                i += 1
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
                if i < len(lines):
                    i += 1
            else:
                i += 1
            continue
        cleaned.append(lines[i])
        i += 1
    return "\n".join(cleaned).lstrip()

# ---------------------------------------------------------------------------
# Helper to split file into smaller cells by major section markers
# ---------------------------------------------------------------------------
def split_and_add_cells(filename, section_title):
    nb["cells"].append(new_markdown_cell(f"## {section_title}"))
    
    if not os.path.exists(filename):
        return
        
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
    
    code = clean_imports(code)
    
    # Split by the large comment blocks # ========
    parts = re.split(r'\n# ={70,}\n', code)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        # Try to extract the section name if it has one (e.g. # Section 1 - ...)
        lines = part.split('\n')
        header = ""
        if lines[0].startswith('# Section'):
            header = lines[0].replace('# ', '')
            part = '\n'.join(lines[1:]).strip()
            
        if header:
            nb["cells"].append(new_markdown_cell(f"### {header}"))
            
        if part:
            nb["cells"].append(new_code_cell(part))


# ---------------------------------------------------------------------------
# 2. Add the source files
# ---------------------------------------------------------------------------
# We add a global imports cell first
global_imports = """import numpy as np
import collections
import random
import time
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
"""
nb["cells"].append(new_code_cell(global_imports))

split_and_add_cells("arc_solver.py", "Core Solver Components")
split_and_add_cells("heuristics.py", "Heuristic Solvers")
split_and_add_cells("genetic_solver.py", "Genetic Algorithm Engine")
split_and_add_cells("hybrid_solver.py", "Hybrid Orchestrator")


# ---------------------------------------------------------------------------
# 3. Add Exploratory Data Analysis & Visualization Section
# ---------------------------------------------------------------------------
eda_markdown = """## Exploratory Data Analysis (EDA) & Evaluation

The following cells load sample tasks from the ARC-AGI evaluation set (or fallback dummy tasks), run the `hybrid_solve` pipeline, and visually plot the predictions. This output-centric approach allows you to inspect the model's logic directly within the Kaggle environment.
"""
nb["cells"].append(new_markdown_cell(eda_markdown))

eda_code = """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import time
import os

# ARC Color Mapping
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap = ListedColormap(ARC_COLORS)


def plot_grid(ax, grid_data, title):
    if hasattr(grid_data, 'data'):
        grid_data = grid_data.data
    grid_data = np.array(grid_data)
    ax.imshow(grid_data, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=10, pad=5)
    h, w = grid_data.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

def visualize_prediction(task_id, test_idx, inp, expected, pred1, pred2):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    fig.suptitle(f"Task: {task_id} - Test Pair {test_idx}", fontsize=12)
    
    plot_grid(axes[0], inp, "Input")
    if expected is not None:
        plot_grid(axes[1], expected, "Expected Output")
    else:
        axes[1].text(0.5, 0.5, 'Hidden', ha='center', va='center')
        axes[1].axis('off')
        
    plot_grid(axes[2], pred1, "Attempt 1 (Best)")
    plot_grid(axes[3], pred2, "Attempt 2 (Alt)")
    
    plt.tight_layout()
    plt.show()

# Helper to load Kaggle data or use fallback
def first_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_eda():
    eval_path = first_existing_path([
        '/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json',
        'arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json',
    ])
    sol_path = first_existing_path([
        '/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json',
        'arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json',
    ])
    
    tasks = {}
    solutions = {}
    
    if eval_path:
        print(f"Loading tasks from {eval_path}")
        with open(eval_path, 'r') as f:
            tasks = json.load(f)
        if sol_path:
            with open(sol_path, 'r') as f:
                solutions = json.load(f)
    else:
        print("Dataset not found. Using a fallback synthetic task for EDA.")
        tasks = {
            "synthetic_task_1": {
                "train": [
                    {"input": [[1,1,0],[0,1,0],[0,0,0]], "output": [[2,2,0],[0,2,0],[0,0,0]]},
                    {"input": [[0,3,3],[0,3,0],[0,0,0]], "output": [[0,2,2],[0,2,0],[0,0,0]]}
                ],
                "test": [{"input": [[0,0,0],[4,4,0],[4,0,0]]}]
            }
        }
        solutions = {
            "synthetic_task_1": [[[0,0,0],[2,2,0],[2,0,0]]]
        }

    # Run solver on first 3 tasks
    task_ids = list(tasks.keys())[:3]
    
    for tid in task_ids:
        print("-" * 60)
        print(f"Evaluating Task: {tid}")
        task = tasks[tid]
        
        t0 = time.time()
        # Fast config for EDA
        preds, score = hybrid_solve(task, ALL_SOLVERS, beam_width=5, population_size=40, max_generations=10)
        elapsed = time.time() - t0
        
        print(f"Training Score: {score:.4f} | Solved in {elapsed:.2f}s")
        
        # Visualize test pairs
        for i, test_pair in enumerate(task['test']):
            inp = test_pair['input']
            expected = solutions.get(tid, [])[i] if tid in solutions and i < len(solutions[tid]) else None
            
            p1 = preds[i]['attempt_1']
            p2 = preds[i]['attempt_2']
            
            visualize_prediction(tid, i, inp, expected, p1, p2)

# Run EDA
run_eda()
"""
nb["cells"].append(new_code_cell(eda_code))

# ---------------------------------------------------------------------------
# 4. Add Kaggle submission generation and validation
# ---------------------------------------------------------------------------
submission_markdown = """## Submission Generation

This final cell solves every task in the ARC-AGI-2 test challenges file and writes a Kaggle-compatible `submission.json` with exactly two attempts per test input.
"""
nb["cells"].append(new_markdown_cell(submission_markdown))

submission_code = """
import json
import os
import time


def load_test_challenges():
    candidates = [
        '/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json',
        'arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading test challenges from {path}")
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("Could not find arc-agi_test_challenges.json")


def zero_like_input(test_pair):
    inp = test_pair['input']
    h = len(inp)
    w = len(inp[0]) if h else 1
    return [[0 for _ in range(w)] for _ in range(max(h, 1))]


def validate_submission(submission, challenges):
    missing = set(challenges) - set(submission)
    extra = set(submission) - set(challenges)
    if missing or extra:
        raise ValueError(f"Task id mismatch: missing={len(missing)}, extra={len(extra)}")

    for task_id, task in challenges.items():
        preds = submission[task_id]
        if not isinstance(preds, list) or len(preds) != len(task['test']):
            raise ValueError(f"{task_id}: expected {len(task['test'])} prediction entries")
        for idx, pred in enumerate(preds):
            if set(pred.keys()) != {'attempt_1', 'attempt_2'}:
                raise ValueError(f"{task_id}[{idx}]: must contain attempt_1 and attempt_2 only")
            for key in ('attempt_1', 'attempt_2'):
                grid = pred[key]
                if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
                    raise ValueError(f"{task_id}[{idx}].{key}: prediction must be a non-empty 2D list")
                width = len(grid[0])
                if width == 0 or any(len(row) != width for row in grid):
                    raise ValueError(f"{task_id}[{idx}].{key}: rows must be rectangular")
                for row in grid:
                    for value in row:
                        if not isinstance(value, int) or value < 0 or value > 9:
                            raise ValueError(f"{task_id}[{idx}].{key}: values must be integers 0..9")


def build_submission(output_file='submission.json',
                     beam_width=8,
                     population_size=48,
                     max_generations=12):
    challenges = load_test_challenges()
    submission = {}
    started = time.time()

    for index, (task_id, task) in enumerate(challenges.items(), start=1):
        task_start = time.time()
        try:
            preds, score = hybrid_solve(
                task,
                ALL_SOLVERS,
                beam_width=beam_width,
                population_size=population_size,
                max_generations=max_generations,
                ga_verbose=False,
            )
        except Exception as exc:
            print(f"{index:03d}/{len(challenges)} {task_id}: solver error, using zero fallback: {exc}")
            preds = []
            for test_pair in task['test']:
                fallback = zero_like_input(test_pair)
                preds.append({'attempt_1': fallback, 'attempt_2': fallback})
            score = 0.0

        submission[task_id] = preds
        print(f"{index:03d}/{len(challenges)} {task_id}: train_score={score:.4f}, elapsed={time.time() - task_start:.2f}s")

    validate_submission(submission, challenges)
    with open(output_file, 'w') as f:
        json.dump(submission, f)

    print(f"Saved {output_file} with {len(submission)} tasks in {time.time() - started:.1f}s")
    return submission


submission = build_submission()
"""
nb["cells"].append(new_code_cell(submission_code))


with open("ARC_Hybrid_Solver_Final.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Successfully generated ARC_Hybrid_Solver_Final.ipynb")
