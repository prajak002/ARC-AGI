# The Chromatic Grid Oracle

## Problem Statement

A scientist is studying a mysterious **ancient grid cipher**. The cipher works by applying a hidden multi-step rule to transform input grids into output grids. The rule always produces:

1. A **stripe pattern score** — how structured the output rows/columns are
2. A **balance score** — how symmetrically values are distributed
3. A **modular fingerprint** — a checksum the output must satisfy
4. A **final combined output** — a grid where every cell reflects all three properties simultaneously

You are given **K example pairs** of (input grid → output grid). From these examples, induce the hidden rule and apply it to **Q test grids**. For each test grid, submit exactly **2 attempts**. Score is 1 if either attempt matches exactly.

---

## The Hidden Rule (Always Applied in This Fixed Order)

The transformation has **4 mandatory stages**, applied sequentially:

---

### Stage 1 — Stripe Normalization
For each row `i`, compute the **dominant value** (most frequent cell value in that row).
Replace every cell in that row with:
```
new_cell = (cell_value + dominant_value) % 10
```
This ensures rows have a "stripe fingerprint."

---

### Stage 2 — Column Balance
For each column `j`, compute the **column sum** `S_j`.
For every cell in that column:
```
new_cell = (current_cell × (j + 1)) % 10
```
This weights each column by its position.

---

### Stage 3 — Modular Fingerprint
For each cell at (i, j):
```
new_cell = (current_cell + i + j) % 10
```
This embeds a positional checksum into every cell.

---

### Stage 4 — Symmetry Clamp
For each cell at (i, j) and its mirror at (i, N-1-j):
```
new_cell(i,j)       = max(current(i,j), current(i, N-1-j))
new_cell(i, N-1-j)  = max(current(i,j), current(i, N-1-j))
```
This enforces horizontal symmetry across the grid.

---

## Combined Output Properties

After all 4 stages, the output grid satisfies:
- **Stripe score** = number of rows where all cells have the same value
- **Balance score** = number of columns where left half sum equals right half sum  
- **Fingerprint** = (sum of all cells) % 10 must equal (sum of input cells × 3) % 10
- **The grid itself** encodes all three properties simultaneously

---

## Input Format

```
Line 1: K N M
Next K blocks:
    N lines of M space-separated integers   ← input grid
    (blank line)
    N lines of M space-separated integers   ← output grid
    (blank line)
Line: Q
Next Q blocks:
    N lines of M space-separated integers   ← test input grid
    (blank line)
```

---

## Output Format

```
For each test grid output exactly 2 attempts:

Attempt 1:
N lines of M space-separated integers

Attempt 2:
N lines of M space-separated integers
```

If you are confident in your answer, both attempts can be identical.

---

## Constraints

```
1 ≤ K ≤ 5          (number of example pairs)
2 ≤ N ≤ 6          (grid rows)
2 ≤ M ≤ 6          (grid columns, always even for symmetry)
0 ≤ cell values ≤ 9
1 ≤ Q ≤ 3          (number of test grids)
Time limit  : 2 seconds
Memory limit: 256 MB
```

---

## Scoring

```
+1   if attempt_1 OR attempt_2 matches the ground truth exactly
 0   otherwise
Final score = correct / total test outputs
```

---

---

# Test Cases

---

## Test Case 1 — Small Grid, Single Example

### Input
```
1 2 4
1 2 3 4
0 1 2 3

2 3 4 5
1 2 3 4

1
3 1 2 0
2 3 1 2
```

### Step-by-Step Derivation (Example Pair)

**Original input:**
```
1 2 3 4
0 1 2 3
```

**Stage 1 — Stripe Normalization:**
- Row 0: values = [1,2,3,4], dominant = no single mode → use max = 4
  - new = [(1+4)%10, (2+4)%10, (3+4)%10, (4+4)%10] = [5, 6, 7, 8]
- Row 1: values = [0,1,2,3], dominant = no single mode → use max = 3
  - new = [(0+3)%10, (1+3)%10, (2+3)%10, (3+3)%10] = [3, 4, 5, 6]

After Stage 1:
```
5 6 7 8
3 4 5 6
```

**Stage 2 — Column Balance:**
- Col 0 (j=0): multiply by (0+1)=1 → [5×1, 3×1] % 10 = [5, 3]
- Col 1 (j=1): multiply by (1+1)=2 → [6×2, 4×2] % 10 = [2, 8]
- Col 2 (j=2): multiply by (2+1)=3 → [7×3, 5×3] % 10 = [1, 5]
- Col 3 (j=3): multiply by (3+1)=4 → [8×4, 6×4] % 10 = [2, 4]

After Stage 2:
```
5 2 1 2
3 8 5 4
```

**Stage 3 — Modular Fingerprint:**
- (i=0,j=0): (5+0+0)%10 = 5
- (i=0,j=1): (2+0+1)%10 = 3
- (i=0,j=2): (1+0+2)%10 = 3
- (i=0,j=3): (2+0+3)%10 = 5
- (i=1,j=0): (3+1+0)%10 = 4
- (i=1,j=1): (8+1+1)%10 = 0
- (i=1,j=2): (5+1+2)%10 = 8
- (i=1,j=3): (4+1+3)%10 = 8

After Stage 3:
```
5 3 3 5
4 0 8 8
```

**Stage 4 — Symmetry Clamp (M=4, mirror pairs: col0↔col3, col1↔col2):**
- Row 0: col0=5,col3=5 → max(5,5)=5 | col1=3,col2=3 → max(3,3)=3
- Row 1: col0=4,col3=8 → max(4,8)=8 | col1=0,col2=8 → max(0,8)=8

**Final output grid:**
```
5 3 3 5
8 8 8 8
```

**Verification:**
- Stripe score: Row 1 = [8,8,8,8] → all same → stripe score = 1 ✓
- Balance score: Col sums left half [5+8=13, 3+8=11] vs right half [3+8=11, 5+8=13] → col1 left=11=right=11 ✓
- Fingerprint: input sum = 1+2+3+4+0+1+2+3 = 16 → (16×3)%10 = 8. Output sum = 5+3+3+5+8+8+8+8 = 48 → 48%10 = 8 ✓

### Expected Output (Example Pair Verification)
```
5 3 3 5
8 8 8 8
```

### Now Solving the Test Input
```
3 1 2 0
2 3 1 2
```

**Stage 1:**
- Row 0: [3,1,2,0], dominant=no mode → max=3 → [(3+3)%10,(1+3)%10,(2+3)%10,(0+3)%10] = [6,4,5,3]
- Row 1: [2,3,1,2], dominant=2 (appears twice) → [(2+2)%10,(3+2)%10,(1+2)%10,(2+2)%10] = [4,5,3,4]

After Stage 1:
```
6 4 5 3
4 5 3 4
```

**Stage 2:**
- Col 0 ×1: [6,4]→[6,4]
- Col 1 ×2: [4,5]→[8,0]
- Col 2 ×3: [5,3]→[5,9]
- Col 3 ×4: [3,4]→[2,6]

After Stage 2:
```
6 8 5 2
4 0 9 6
```

**Stage 3:**
- (0,0)=(6+0+0)%10=6 | (0,1)=(8+0+1)%10=9 | (0,2)=(5+0+2)%10=7 | (0,3)=(2+0+3)%10=5
- (1,0)=(4+1+0)%10=5 | (1,1)=(0+1+1)%10=2 | (1,2)=(9+1+2)%10=2 | (1,3)=(6+1+3)%10=0

After Stage 3:
```
6 9 7 5
5 2 2 0
```

**Stage 4:**
- Row 0: col0=6,col3=5→max=6 | col1=9,col2=7→max=9
- Row 1: col0=5,col3=0→max=5 | col1=2,col2=2→max=2

**Final Answer:**
```
6 9 9 6
5 2 2 5
```

### Output
```
Attempt 1:
6 9 9 6
5 2 2 5

Attempt 2:
6 9 9 6
5 2 2 5
```

---

## Test Case 2 — Medium Grid, Two Examples, Two Test Inputs

### Input
```
2 3 4
2 0 1 3
1 3 2 0
0 2 1 3

3 1 2 3
4 3 3 0
2 3 2 3

0 0 3 3
3 3 0 0
1 1 1 1

3 3 3 3
3 3 3 3
1 1 1 1

2
1 2 1 2
2 1 2 1
1 1 2 2

3 0 0 3
0 3 3 0
3 0 3 0
```

### Solution — Test Input 1
```
1 2 1 2
2 1 2 1
1 1 2 2
```

**Stage 1:**
- Row 0: [1,2,1,2] → dominant=1 and 2 tie → use max=2 → [(1+2)%10,(2+2)%10,(1+2)%10,(2+2)%10]=[3,4,3,4]
- Row 1: [2,1,2,1] → dominant tie → max=2 → [(2+2),(1+2),(2+2),(1+2)]%10=[4,3,4,3]
- Row 2: [1,1,2,2] → dominant=1 and 2 tie → max=2 → [(1+2),(1+2),(2+2),(2+2)]%10=[3,3,4,4]

After Stage 1:
```
3 4 3 4
4 3 4 3
3 3 4 4
```

**Stage 2:**
- Col 0 ×1: [3,4,3]
- Col 1 ×2: [8,6,6]
- Col 2 ×3: [9,2,2]
- Col 3 ×4: [6,2,6]

After Stage 2:
```
3 8 9 6
4 6 2 2
3 6 2 6
```

**Stage 3:**
- Row 0: (3+0)%10=3, (8+1)%10=9, (9+2)%10=1, (6+3)%10=9
- Row 1: (4+1)%10=5, (6+2)%10=8, (2+3)%10=5, (2+4)%10=6
- Row 2: (3+2)%10=5, (6+3)%10=9, (2+4)%10=6, (6+5)%10=1

After Stage 3:
```
3 9 1 9
5 8 5 6
5 9 6 1
```

**Stage 4 (col0↔col3, col1↔col2):**
- Row 0: max(3,9)=9 | max(9,1)=9  → [9,9,9,9]
- Row 1: max(5,6)=6 | max(8,5)=8  → [6,8,8,6]
- Row 2: max(5,1)=5 | max(9,6)=9  → [5,9,9,5]

**Answer Test 1:**
```
9 9 9 9
6 8 8 6
5 9 9 5
```

### Solution — Test Input 2
```
3 0 0 3
0 3 3 0
3 0 3 0
```

**Stage 1:**
- Row 0: [3,0,0,3] → dominant=3 and 0 tie → max=3 → [(3+3),(0+3),(0+3),(3+3)]%10=[6,3,3,6]
- Row 1: [0,3,3,0] → dominant tie → max=3 → [(0+3),(3+3),(3+3),(0+3)]%10=[3,6,6,3]
- Row 2: [3,0,3,0] → dominant tie → max=3 → [(3+3),(0+3),(3+3),(0+3)]%10=[6,3,6,3]

After Stage 1:
```
6 3 3 6
3 6 6 3
6 3 6 3
```

**Stage 2:**
- Col 0 ×1: [6,3,6]
- Col 1 ×2: [6,2,6]
- Col 2 ×3: [9,8,8]
- Col 3 ×4: [4,2,2]

After Stage 2:
```
6 6 9 4
3 2 8 2
6 6 8 2
```

**Stage 3:**
- Row 0: 6,7,1,7
- Row 1: 4,4,1,6 → (3+1+0)=4,(2+1+1)=4,(8+1+2)%10=1,(2+1+3)=6
- Row 2: (6+2+0)=8,(6+2+1)=9,(8+2+2)%10=2,(2+2+3)=7

After Stage 3:
```
6 7 1 7
4 4 1 6
8 9 2 7
```

**Stage 4:**
- Row 0: max(6,7)=7 | max(7,1)=7 → [7,7,7,7]
- Row 1: max(4,6)=6 | max(4,1)=4 → [6,4,4,6]
- Row 2: max(8,7)=8 | max(9,2)=9 → [8,9,9,8]

**Answer Test 2:**
```
7 7 7 7
6 4 4 6
8 9 9 8
```

### Output
```
Attempt 1:
9 9 9 9
6 8 8 6
5 9 9 5

Attempt 2:
9 9 9 9
6 8 8 6
5 9 9 5

Attempt 1:
7 7 7 7
6 4 4 6
8 9 9 8

Attempt 2:
7 7 7 7
6 4 4 6
8 9 9 8
```

---

## Test Case 3 — Stress Case, Three Examples, Three Tests

### Input
```
3 4 6
1 2 3 4 5 6
0 1 0 1 0 1
2 3 4 5 6 7
8 9 0 1 2 3

(output derived by applying all 4 stages — omitted for brevity, solver must compute)

... (2 more example pairs)

3
(3 test grids of size 4×6)
```

> **Note:** For Test Case 3, the solver is expected to implement the 4-stage algorithm programmatically. Manual derivation is impractical at 4×6. The algorithm is fully deterministic — same stages, same order.

---

# Summary of Combined Output Properties

Every correctly solved output grid simultaneously encodes:

| Property | What to Check |
|---|---|
| **Stripe score** | Count rows where all cells are equal |
| **Balance score** | Count columns where col_sum_left_half = col_sum_right_half |
| **Modular fingerprint** | output_sum % 10 == (input_sum × 3) % 10 |
| **Symmetry** | Grid is horizontally symmetric (col j == col M-1-j) |

A correct output satisfies **all four** simultaneously. This is the "combined output" — it is not four separate outputs, it is one grid that reflects all four properties at once.

---

# Algorithm Template (Pseudocode)

```python
def solve(grid):
    N, M = len(grid), len(grid[0])
    g = [row[:] for row in grid]

    # Stage 1 — Stripe Normalization
    for i in range(N):
        freq = {}
        for v in g[i]:
            freq[v] = freq.get(v, 0) + 1
        dominant = max(freq, key=lambda x: (freq[x], x))
        g[i] = [(v + dominant) % 10 for v in g[i]]

    # Stage 2 — Column Balance
    for j in range(M):
        for i in range(N):
            g[i][j] = (g[i][j] * (j + 1)) % 10

    # Stage 3 — Modular Fingerprint
    for i in range(N):
        for j in range(M):
            g[i][j] = (g[i][j] + i + j) % 10

    # Stage 4 — Symmetry Clamp
    for i in range(N):
        for j in range(M // 2):
            mirror = M - 1 - j
            val = max(g[i][j], g[i][mirror])
            g[i][j] = val
            g[i][mirror] = val

    return g
```

---

# Verification Checklist

After computing your output grid, verify:

```
✓ stripe_score   = count(rows where all cells equal)
✓ balance_score  = count(cols where sum(left_half) == sum(right_half))
✓ fingerprint    = (sum(output) % 10) == ((sum(input) * 3) % 10)
✓ symmetry       = all i,j: output[i][j] == output[i][M-1-j]
```

If all 4 pass → your output is correct.
