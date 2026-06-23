"""
ARC Prize 2026 — Heuristic Solvers
====================================
Each solver analyzes training pairs and generates TransformationRule candidates
for the beam search. Covers the most common ARC task patterns.
"""

import numpy as np
import collections
from arc_solver import (
    Grid, TransformationRule, ArcObject,
    find_objects, find_objects_by_color, BlockGraph,
    rotate90, rotate180, rotate270,
    mirror_horizontal, mirror_vertical, transpose,
    crop_to_content, scale_up, tile_grid,
    color_swap, color_map, flood_fill, overlay,
    gravity_down, extract_subgrid, pad_grid,
    detect_symmetry, detect_scaling, detect_tiling,
    detect_color_mapping, compute_diff, grid_similarity,
)


# =============================================================================
# Helper: Wrap a lambda with captured params into a proper callable
# =============================================================================

def _make_fn(fn):
    """Ensure function is picklable and safely wrapped."""
    return fn


# =============================================================================
# Solver 1: Identity / Direct Copy
# =============================================================================

def solver_identity(train_pairs):
    """Check if output == input (identity transform)."""
    return [TransformationRule("identity", lambda g: g)]


# =============================================================================
# Solver 2: Simple Geometric Transforms
# =============================================================================

def solver_geometric(train_pairs):
    """Try all 8 rigid geometric transforms (rotations + mirrors)."""
    transforms = [
        ("rotate90", rotate90),
        ("rotate180", rotate180),
        ("rotate270", rotate270),
        ("mirror_h", mirror_horizontal),
        ("mirror_v", mirror_vertical),
        ("transpose", transpose),
        ("mirror_h+rotate90", lambda g: rotate90(mirror_horizontal(g))),
        ("mirror_v+rotate90", lambda g: rotate90(mirror_vertical(g))),
    ]
    return [TransformationRule(name, fn) for name, fn in transforms]


# =============================================================================
# Solver 3: Tiling
# =============================================================================

def solver_tiling(train_pairs):
    """Detect if output is input tiled N×M times, possibly with modifications."""
    candidates = []
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])
    ih, iw = inp.shape
    oh, ow = out.shape

    if ih == 0 or iw == 0:
        return candidates

    # Simple tiling
    for tr in range(1, min(oh // ih + 1, 8)):
        for tc in range(1, min(ow // iw + 1, 8)):
            if tr * ih == oh and tc * iw == ow:
                r, c = tr, tc  # capture
                candidates.append(TransformationRule(
                    f"tile_{r}x{c}",
                    lambda g, _r=r, _c=c: tile_grid(g, _r, _c)
                ))

    # Tiling with alternating mirror rows/cols
    if oh % ih == 0 and ow % iw == 0:
        tr, tc = oh // ih, ow // iw

        def _tile_mirror_rows(g, _tr=tr, _tc=tc):
            rows_of_tiles = []
            for ri in range(_tr):
                row_g = g if ri % 2 == 0 else mirror_vertical(g)
                row_of_cols = []
                for ci in range(_tc):
                    cell_g = row_g if ci % 2 == 0 else mirror_horizontal(row_g)
                    row_of_cols.append(cell_g.data)
                rows_of_tiles.append(np.concatenate(row_of_cols, axis=1))
            return Grid(np.concatenate(rows_of_tiles, axis=0))

        candidates.append(TransformationRule(
            f"tile_mirror_{tr}x{tc}", _tile_mirror_rows
        ))

    return candidates


# =============================================================================
# Solver 4: Scaling (Upscale / Downscale)
# =============================================================================

def solver_scaling(train_pairs):
    """Detect integer scaling between input and output."""
    candidates = []
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])
    ih, iw = inp.shape
    oh, ow = out.shape

    # Upscale
    if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
        sr, sc = oh // ih, ow // iw
        if sr == sc and sr > 1:
            f = sr
            candidates.append(TransformationRule(
                f"scale_up_{f}x", lambda g, _f=f: scale_up(g, _f)
            ))

    # Downscale (output smaller than input)
    if oh > 0 and ow > 0 and ih % oh == 0 and iw % ow == 0:
        sr, sc = ih // oh, iw // ow
        if sr == sc and sr > 1:
            f = sr
            def _downscale(g, _f=f):
                return Grid(g.data[::_f, ::_f])
            candidates.append(TransformationRule(f"downscale_{f}x", _downscale))

    return candidates


# =============================================================================
# Solver 5: Color Mapping
# =============================================================================

def solver_color_mapping(train_pairs):
    """Detect consistent color remapping across all training pairs."""
    candidates = []

    # Attempt to find a single consistent mapping
    global_mapping = None
    for pair in train_pairs:
        inp, out = Grid(pair['input']), Grid(pair['output'])
        m = detect_color_mapping(inp, out)
        if m is None:
            global_mapping = None
            break
        if global_mapping is None:
            global_mapping = m
        else:
            if global_mapping != m:
                global_mapping = None
                break

    if global_mapping is not None:
        # Check it's not just identity
        if any(k != v for k, v in global_mapping.items()):
            mapping = dict(global_mapping)
            candidates.append(TransformationRule(
                f"color_map_{mapping}",
                lambda g, _m=mapping: color_map(g, _m)
            ))

    # Try single color swaps
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])
    if inp.shape == out.shape:
        in_colors = inp.unique_colors()
        out_colors = out.unique_colors()
        for c1 in in_colors:
            for c2 in range(10):
                if c1 != c2:
                    candidates.append(TransformationRule(
                        f"swap_{c1}→{c2}",
                        lambda g, _c1=c1, _c2=c2: color_swap(g, _c1, _c2)
                    ))

    return candidates


# =============================================================================
# Solver 6: Crop to Content
# =============================================================================

def solver_crop(train_pairs):
    """Detect if output is a crop of non-background content."""
    candidates = []
    for bg in range(10):
        candidates.append(TransformationRule(
            f"crop_bg={bg}",
            lambda g, _bg=bg: crop_to_content(g, bg=_bg)
        ))
    return candidates


# =============================================================================
# Solver 7: Largest / Smallest Object Extraction
# =============================================================================

def solver_object_extraction(train_pairs):
    """Extract specific objects from the grid."""
    candidates = []

    for bg in [0]:
        # Extract largest object
        def _extract_largest(g, _bg=bg):
            objs = find_objects(g, bg=_bg)
            if not objs:
                return g
            largest = max(objs, key=lambda o: o.area)
            return largest.subgrid(bg=_bg)

        candidates.append(TransformationRule(f"largest_object_bg={bg}", _extract_largest))

        # Extract smallest object
        def _extract_smallest(g, _bg=bg):
            objs = find_objects(g, bg=_bg)
            if not objs:
                return g
            smallest = min(objs, key=lambda o: o.area)
            return smallest.subgrid(bg=_bg)

        candidates.append(TransformationRule(f"smallest_object_bg={bg}", _extract_smallest))

        # Extract most common color object
        def _extract_by_most_common_color(g, _bg=bg):
            objs = find_objects(g, bg=_bg)
            if not objs:
                return g
            color_counts = collections.Counter()
            for o in objs:
                color_counts[o.primary_color] += 1
            most_common = color_counts.most_common(1)[0][0]
            target = [o for o in objs if o.primary_color == most_common]
            if target:
                return target[0].subgrid(bg=_bg)
            return g

        candidates.append(TransformationRule(
            f"extract_most_common_color_bg={bg}", _extract_by_most_common_color
        ))

    return candidates


# =============================================================================
# Solver 8: Fill Enclosed Regions
# =============================================================================

def solver_fill(train_pairs):
    """Fill enclosed background regions with a specific color."""
    candidates = []
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])

    if inp.shape != out.shape:
        return candidates

    # Detect which cells changed
    diff = compute_diff(inp, out)
    if diff is None or len(diff) == 0:
        return candidates

    # Find the fill color (most common new color in changed cells)
    new_colors = collections.Counter(nc for _, _, _, nc in diff)
    fill_color = new_colors.most_common(1)[0][0]

    # Strategy: fill all enclosed bg regions
    def _fill_enclosed(g, _fc=fill_color):
        h, w = g.height, g.width
        bg = g.background_color()
        result = g.data.copy()

        # Mark cells reachable from edges (not enclosed)
        reachable = np.zeros((h, w), dtype=bool)
        queue = collections.deque()
        for r in range(h):
            for c in [0, w - 1]:
                if int(result[r, c]) == bg and not reachable[r, c]:
                    reachable[r, c] = True
                    queue.append((r, c))
        for c in range(w):
            for r in [0, h - 1]:
                if int(result[r, c]) == bg and not reachable[r, c]:
                    reachable[r, c] = True
                    queue.append((r, c))

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and int(result[nr, nc]) == bg:
                    reachable[nr, nc] = True
                    queue.append((nr, nc))

        # Fill all unreachable bg cells
        for r in range(h):
            for c in range(w):
                if int(result[r, c]) == bg and not reachable[r, c]:
                    result[r, c] = _fc
        return Grid(result)

    candidates.append(TransformationRule(f"fill_enclosed_{fill_color}", _fill_enclosed))

    return candidates


# =============================================================================
# Solver 9: Gravity (Drop cells down)
# =============================================================================

def solver_gravity(train_pairs):
    """Apply gravity in 4 directions."""
    candidates = []

    for bg in [0]:
        candidates.append(TransformationRule(
            f"gravity_down_bg={bg}",
            lambda g, _bg=bg: gravity_down(g, bg=_bg)
        ))

        # Gravity up
        def _gravity_up(g, _bg=bg):
            return mirror_vertical(gravity_down(mirror_vertical(g), bg=_bg))
        candidates.append(TransformationRule(f"gravity_up_bg={bg}", _gravity_up))

        # Gravity right
        def _gravity_right(g, _bg=bg):
            rotated = rotate90(g)
            dropped = gravity_down(rotated, bg=_bg)
            return rotate270(dropped)
        candidates.append(TransformationRule(f"gravity_right_bg={bg}", _gravity_right))

        # Gravity left
        def _gravity_left(g, _bg=bg):
            rotated = rotate270(g)
            dropped = gravity_down(rotated, bg=_bg)
            return rotate90(dropped)
        candidates.append(TransformationRule(f"gravity_left_bg={bg}", _gravity_left))

    return candidates


# =============================================================================
# Solver 10: Symmetry Completion
# =============================================================================

def solver_symmetry_completion(train_pairs):
    """Complete a partially-symmetric grid by enforcing symmetry."""
    candidates = []

    # Enforce horizontal symmetry (left→right)
    def _enforce_sym_h_lr(g):
        data = g.data.copy()
        h, w = data.shape
        for r in range(h):
            for c in range(w // 2):
                data[r, w - 1 - c] = data[r, c]
        return Grid(data)

    # Enforce horizontal symmetry (right→left)
    def _enforce_sym_h_rl(g):
        data = g.data.copy()
        h, w = data.shape
        for r in range(h):
            for c in range(w // 2):
                data[r, c] = data[r, w - 1 - c]
        return Grid(data)

    # Enforce vertical symmetry (top→bottom)
    def _enforce_sym_v_tb(g):
        data = g.data.copy()
        h, w = data.shape
        for r in range(h // 2):
            data[h - 1 - r, :] = data[r, :]
        return Grid(data)

    # Enforce vertical symmetry (bottom→top)
    def _enforce_sym_v_bt(g):
        data = g.data.copy()
        h, w = data.shape
        for r in range(h // 2):
            data[r, :] = data[h - 1 - r, :]
        return Grid(data)

    candidates.extend([
        TransformationRule("sym_h_left_to_right", _enforce_sym_h_lr),
        TransformationRule("sym_h_right_to_left", _enforce_sym_h_rl),
        TransformationRule("sym_v_top_to_bottom", _enforce_sym_v_tb),
        TransformationRule("sym_v_bottom_to_top", _enforce_sym_v_bt),
    ])

    return candidates


# =============================================================================
# Solver 11: Border / Frame Operations
# =============================================================================

def solver_border(train_pairs):
    """Detect border-related transformations."""
    candidates = []
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])

    # Add 1-cell border of a specific color
    for color in range(10):
        c = color
        candidates.append(TransformationRule(
            f"add_border_{c}",
            lambda g, _c=c: pad_grid(g, 1, 1, 1, 1, fill=_c)
        ))

    # Remove 1-cell border
    def _remove_border(g):
        if g.height < 3 or g.width < 3:
            return g
        return extract_subgrid(g, 1, 1, g.height - 2, g.width - 2)

    candidates.append(TransformationRule("remove_border", _remove_border))

    return candidates


# =============================================================================
# Solver 12: Input-Output Shape Analysis Based Rules
# =============================================================================

def solver_shape_based(train_pairs):
    """Generate rules based on input/output shape relationships."""
    candidates = []
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])
    ih, iw = inp.shape
    oh, ow = out.shape

    # Output is a single row → maybe max/min/mode per column
    if oh == 1 and ow == iw:
        # Mode per column
        def _mode_per_col(g):
            result = []
            for c in range(g.width):
                col = g.data[:, c]
                counts = collections.Counter(col.tolist())
                # Exclude background
                non_bg = {k: v for k, v in counts.items() if k != 0}
                if non_bg:
                    result.append(max(non_bg, key=non_bg.get))
                else:
                    result.append(counts.most_common(1)[0][0])
            return Grid(np.array([result]))

        candidates.append(TransformationRule("mode_per_column", _mode_per_col))

    # Output is a single column → mode per row
    if ow == 1 and oh == ih:
        def _mode_per_row(g):
            result = []
            for r in range(g.height):
                row = g.data[r, :]
                counts = collections.Counter(row.tolist())
                non_bg = {k: v for k, v in counts.items() if k != 0}
                if non_bg:
                    result.append([max(non_bg, key=non_bg.get)])
                else:
                    result.append([counts.most_common(1)[0][0]])
            return Grid(np.array(result))

        candidates.append(TransformationRule("mode_per_row", _mode_per_row))

    return candidates


# =============================================================================
# Solver 13: Marker/Seed Based Pattern (common ARC pattern)
# =============================================================================

def solver_marker_pattern(train_pairs):
    """
    Detect 'marker' pixels (rare non-background, non-border colors) and
    apply patterns relative to them.
    Common ARC pattern: a small marker indicates where to apply a transformation.
    """
    candidates = []
    pair = train_pairs[0]
    inp, out = Grid(pair['input']), Grid(pair['output'])

    if inp.shape != out.shape:
        return candidates

    bg = inp.background_color()
    in_colors = inp.color_counts()
    # Markers = rare colors (count ≤ 3)
    markers = [c for c, cnt in in_colors.items() if c != bg and cnt <= 3]

    for marker_color in markers:
        mc = marker_color
        diff = compute_diff(inp, out)
        if diff is None:
            continue

        # Check if marker positions are replaced with another color
        marker_positions = [(r, c) for r in range(inp.height) for c in range(inp.width) if inp.cell(r, c) == mc]
        if not marker_positions:
            continue

        # What color do markers become in output?
        out_colors_at_markers = set(out.cell(r, c) for r, c in marker_positions)
        if len(out_colors_at_markers) == 1:
            replacement = out_colors_at_markers.pop()
            if replacement != mc:
                repl = replacement
                candidates.append(TransformationRule(
                    f"replace_marker_{mc}→{repl}",
                    lambda g, _mc=mc, _repl=repl: color_swap(g, _mc, _repl)
                ))

    return candidates


# =============================================================================
# Solver 14: Subgrid Extraction by Divider Lines
# =============================================================================

def solver_subgrid_by_divider(train_pairs):
    """
    Detect if the grid has horizontal/vertical divider lines of a single color,
    and the output is one of the resulting subgrids.
    """
    candidates = []
    pair = train_pairs[0]
    inp = Grid(pair['input'])
    out = Grid(pair['output'])

    # Find horizontal divider rows (all same non-bg color)
    bg = inp.background_color()
    h_dividers = []
    for r in range(inp.height):
        row = inp.data[r, :]
        uniq = set(row.tolist())
        if len(uniq) == 1 and list(uniq)[0] != bg:
            h_dividers.append(r)

    # Find vertical divider columns
    v_dividers = []
    for c in range(inp.width):
        col = inp.data[:, c]
        uniq = set(col.tolist())
        if len(uniq) == 1 and list(uniq)[0] != bg:
            v_dividers.append(c)

    # Extract subgrids between dividers
    h_boundaries = [-1] + h_dividers + [inp.height]
    v_boundaries = [-1] + v_dividers + [inp.width]

    subgrid_index = 0
    for i in range(len(h_boundaries) - 1):
        for j in range(len(v_boundaries) - 1):
            r1 = h_boundaries[i] + 1
            r2 = h_boundaries[i + 1] - 1
            c1 = v_boundaries[j] + 1
            c2 = v_boundaries[j + 1] - 1
            if r1 <= r2 and c1 <= c2:
                _r1, _c1, _r2, _c2 = r1, c1, r2, c2
                idx = subgrid_index
                candidates.append(TransformationRule(
                    f"extract_subgrid_{idx}_({_r1},{_c1})-({_r2},{_c2})",
                    lambda g, a=_r1, b=_c1, c=_r2, d=_c2: extract_subgrid(g, a, b, c, d)
                ))
                subgrid_index += 1

    return candidates


# =============================================================================
# Solver 15: Overlay / Boolean Operations on Subgrids
# =============================================================================

def solver_overlay(train_pairs):
    """
    If grid has dividers, apply boolean-style ops (AND, OR, XOR) on subgrids.
    Common ARC pattern: two halves combined with a logical rule.
    """
    candidates = []
    pair = train_pairs[0]
    inp = Grid(pair['input'])
    out = Grid(pair['output'])
    bg = inp.background_color()

    # Try splitting grid in half horizontally
    if inp.height % 2 == 0:
        mid = inp.height // 2
        top = inp.data[:mid, :]
        bot = inp.data[mid:, :]
        if top.shape == bot.shape and (out.height, out.width) == top.shape:
            # OR: non-bg from either half
            def _or_h(g, _bg=bg):
                m = g.height // 2
                t, b = g.data[:m, :], g.data[m:, :]
                result = np.where(t != _bg, t, b)
                return Grid(result)

            # AND: non-bg in both halves
            def _and_h(g, _bg=bg):
                m = g.height // 2
                t, b = g.data[:m, :], g.data[m:, :]
                result = np.where((t != _bg) & (b != _bg), t, np.full_like(t, _bg))
                return Grid(result)

            # XOR: non-bg in exactly one half
            def _xor_h(g, _bg=bg):
                m = g.height // 2
                t, b = g.data[:m, :], g.data[m:, :]
                result = np.full_like(t, _bg)
                result[(t != _bg) & (b == _bg)] = t[(t != _bg) & (b == _bg)]
                result[(t == _bg) & (b != _bg)] = b[(t == _bg) & (b != _bg)]
                return Grid(result)

            candidates.extend([
                TransformationRule("halves_h_OR", _or_h),
                TransformationRule("halves_h_AND", _and_h),
                TransformationRule("halves_h_XOR", _xor_h),
            ])

    # Try splitting grid in half vertically
    if inp.width % 2 == 0:
        mid = inp.width // 2
        left = inp.data[:, :mid]
        right = inp.data[:, mid:]
        if left.shape == right.shape and (out.height, out.width) == left.shape:
            def _or_v(g, _bg=bg):
                m = g.width // 2
                l, r = g.data[:, :m], g.data[:, m:]
                return Grid(np.where(l != _bg, l, r))

            def _and_v(g, _bg=bg):
                m = g.width // 2
                l, r = g.data[:, :m], g.data[:, m:]
                return Grid(np.where((l != _bg) & (r != _bg), l, np.full_like(l, _bg)))

            def _xor_v(g, _bg=bg):
                m = g.width // 2
                l, r = g.data[:, :m], g.data[:, m:]
                result = np.full_like(l, _bg)
                result[(l != _bg) & (r == _bg)] = l[(l != _bg) & (r == _bg)]
                result[(l == _bg) & (r != _bg)] = r[(l == _bg) & (r != _bg)]
                return Grid(result)

            candidates.extend([
                TransformationRule("halves_v_OR", _or_v),
                TransformationRule("halves_v_AND", _and_v),
                TransformationRule("halves_v_XOR", _xor_v),
            ])

    return candidates


# =============================================================================
# Master Solver List
# =============================================================================

ALL_SOLVERS = [
    solver_identity,
    solver_geometric,
    solver_tiling,
    solver_scaling,
    solver_color_mapping,
    solver_crop,
    solver_object_extraction,
    solver_fill,
    solver_gravity,
    solver_symmetry_completion,
    solver_border,
    solver_shape_based,
    solver_marker_pattern,
    solver_subgrid_by_divider,
    solver_overlay,
]
