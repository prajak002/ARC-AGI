import json
import os
from hybrid_solver import hybrid_solve
from heuristics import ALL_SOLVERS

def _first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _zero_like_input(test_pair):
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
                raise ValueError(f"{task_id}[{idx}]: must contain attempt_1 and attempt_2")
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


def solve_and_submit(test_file=None, output_file='submission.json'):
    if test_file is None:
        test_file = _first_existing_path([
            '/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json',
            'arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json',
        ])
    if test_file is None:
        raise FileNotFoundError("Could not find arc-agi_test_challenges.json")

    with open(test_file, 'r') as f:
        test_challenges = json.load(f)

    print(f"Loaded {len(test_challenges)} test challenges.")
    
    submission = {}
    
    # Process each task
    for task_id, task in test_challenges.items():
        print(f"Solving task {task_id}...")
        
        # We need to make sure the structure matches the Kaggle submission format
        # The submission should be a dict where keys are task_ids and values are lists of dicts
        # (one dict per test input, containing 'attempt_1' and 'attempt_2')
        
        try:
            predictions, score = hybrid_solve(task, ALL_SOLVERS,
                                               beam_width=10,
                                               population_size=100,
                                               max_generations=30)
            submission[task_id] = predictions
        except Exception as e:
            print(f"Error solving {task_id}: {e}")
            preds = []
            for test_pair in task['test']:
                fallback_pred = _zero_like_input(test_pair)
                preds.append({
                    "attempt_1": fallback_pred,
                    "attempt_2": fallback_pred
                })
            submission[task_id] = preds

    validate_submission(submission, test_challenges)
    with open(output_file, 'w') as f:
        json.dump(submission, f)
        
    print(f"Saved submission to {output_file}")

if __name__ == "__main__":
    solve_and_submit()
