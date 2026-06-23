"""
ARC Prize 2026 — Core Solver Module
====================================
DSA-based heuristic solver using:
  - BFS/DFS connected components for object extraction
  - Node-based block analysis (graph of objects)
  - Mathematical symmetry / periodicity detection
  - Beam search over candidate transformation rules
"""

import numpy as np
import collections
import copy
from itertools import product


# =============================================================================
# Section 1 — Grid Representation
# =============================================================================

class Grid:
    """Immutable 2D grid of colors [0..9]."""

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data.astype(int)
        else:
            self.data = np.array(data, dtype=int)

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def shape(self):
        return (self.height, self.width)

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(self.data.tobytes())

    def __repr__(self):
        return f"Grid({self.height}x{self.width})"

    def to_list(self):
        return self.data.tolist()

    def copy(self):
        return Grid(self.data.copy())

    def unique_colors(self):
        return set(np.unique(self.data).tolist())

    def color_counts(self):
        colors, counts = np.unique(self.data, return_counts=True)
        return dict(zip(colors.tolist(), counts.tolist()))

    def background_color(self):
        """Most frequent color = background (heuristic)."""
        cc = self.color_counts()
        return max(cc, key=cc.get)

    def cell(self, r, c):
        return int(self.data[r, c])


# =============================================================================
# Section 2 — Object Extraction (BFS Connected Components)
# =============================================================================

class ArcObject:
    """Represents a connected component (block/node) in the grid."""

    def __init__(self, coords, grid):
        """
        coords: list of (row, col) tuples
        grid: the parent Grid
        """
        self.coords = coords
        self.color_map = {}  # (r, c) -> color
        for r, c in coords:
            self.color_map[(r, c)] = grid.cell(r, c)

        rows = [r for r, c in coords]
        cols = [c for r, c in coords]
        self.min_r, self.max_r = min(rows), max(rows)
        self.min_c, self.max_c = min(cols), max(cols)

    @property
    def bbox(self):
        return (self.min_r, self.min_c, self.max_r, self.max_c)

    @property
    def bbox_height(self):
        return self.max_r - self.min_r + 1

    @property
    def bbox_width(self):
        return self.max_c - self.min_c + 1

    @property
    def area(self):
        return len(self.coords)

    @property
    def colors(self):
        return set(self.color_map.values())

    @property
    def primary_color(self):
        """Most frequent color in this object."""
        cc = collections.Counter(self.color_map.values())
        return cc.most_common(1)[0][0]

    @property
    def centroid(self):
        rows = [r for r, c in self.coords]
        cols = [c for r, c in self.coords]
        return (sum(rows) / len(rows), sum(cols) / len(cols))

    def subgrid(self, bg=0):
        """Extract this object as a standalone grid with background fill."""
        h, w = self.bbox_height, self.bbox_width
        sub = np.full((h, w), bg, dtype=int)
        for (r, c), color in self.color_map.items():
            sub[r - self.min_r, c - self.min_c] = color
        return Grid(sub)

    def normalized_shape(self, bg=0):
        """Binary mask of the object shape (for shape comparison)."""
        h, w = self.bbox_height, self.bbox_width
        mask = np.zeros((h, w), dtype=int)
        for r, c in self.coords:
            mask[r - self.min_r, c - self.min_c] = 1
        return mask


def find_objects(grid, bg=0, connectivity=4):
    """
    BFS-based connected component extraction.
    Groups all non-background cells into objects.
    connectivity: 4 (cardinal) or 8 (cardinal + diagonal)
    Returns list of ArcObject.
    """
    h, w = grid.height, grid.width
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    if connectivity == 4:
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else:
        deltas = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if grid.cell(r, c) != bg and not visited[r, c]:
                # BFS flood
                queue = collections.deque([(r, c)])
                visited[r, c] = True
                component = []
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in deltas:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid.cell(nr, nc) != bg:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                objects.append(ArcObject(component, grid))

    return objects


def find_objects_by_color(grid, color):
    """Extract connected components of a single specific color."""
    h, w = grid.height, grid.width
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for r in range(h):
        for c in range(w):
            if grid.cell(r, c) == color and not visited[r, c]:
                queue = collections.deque([(r, c)])
                visited[r, c] = True
                component = []
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in deltas:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid.cell(nr, nc) == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                objects.append(ArcObject(component, grid))
    return objects


# =============================================================================
# Section 3 — Grid Transformations
# =============================================================================

def rotate90(grid):
    """Rotate grid 90° clockwise."""
    return Grid(np.rot90(grid.data, k=-1))

def rotate180(grid):
    return Grid(np.rot90(grid.data, k=2))

def rotate270(grid):
    return Grid(np.rot90(grid.data, k=1))

def mirror_horizontal(grid):
    """Flip left-right."""
    return Grid(np.fliplr(grid.data))

def mirror_vertical(grid):
    """Flip top-bottom."""
    return Grid(np.flipud(grid.data))

def transpose(grid):
    return Grid(grid.data.T)

def crop_to_content(grid, bg=0):
    """Crop to bounding box of all non-background cells."""
    rows = np.any(grid.data != bg, axis=1)
    cols = np.any(grid.data != bg, axis=0)
    if not np.any(rows) or not np.any(cols):
        return Grid(np.array([[bg]]))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return Grid(grid.data[rmin:rmax + 1, cmin:cmax + 1])

def scale_up(grid, factor):
    """Scale grid by integer factor (nearest-neighbor)."""
    return Grid(np.repeat(np.repeat(grid.data, factor, axis=0), factor, axis=1))

def tile_grid(grid, repeat_r, repeat_c):
    """Tile/repeat a grid pattern."""
    return Grid(np.tile(grid.data, (repeat_r, repeat_c)))

def color_swap(grid, old_color, new_color):
    """Replace all cells of old_color with new_color."""
    new_data = grid.data.copy()
    new_data[new_data == old_color] = new_color
    return Grid(new_data)

def color_map(grid, mapping):
    """Apply a color mapping dictionary {old: new}."""
    new_data = grid.data.copy()
    for old, new in mapping.items():
        new_data[grid.data == old] = new
    return Grid(new_data)

def flood_fill(grid, start_r, start_c, new_color):
    """BFS flood fill from a starting cell."""
    new_data = grid.data.copy()
    old_color = int(new_data[start_r, start_c])
    if old_color == new_color:
        return Grid(new_data)
    h, w = grid.height, grid.width
    queue = collections.deque([(start_r, start_c)])
    new_data[start_r, start_c] = new_color
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(new_data[nr, nc]) == old_color:
                new_data[nr, nc] = new_color
                queue.append((nr, nc))
    return Grid(new_data)

def overlay(base, overlay_grid, offset_r=0, offset_c=0, transparent=0):
    """Place overlay_grid on top of base at offset, treating transparent color as see-through."""
    result = base.data.copy()
    for r in range(overlay_grid.height):
        for c in range(overlay_grid.width):
            tr, tc = r + offset_r, c + offset_c
            if 0 <= tr < base.height and 0 <= tc < base.width:
                val = overlay_grid.cell(r, c)
                if val != transparent:
                    result[tr, tc] = val
    return Grid(result)

def gravity_down(grid, bg=0):
    """Drop all non-background cells downward (column-wise gravity)."""
    new_data = grid.data.copy()
    for c in range(grid.width):
        col = new_data[:, c]
        non_bg = col[col != bg]
        new_col = np.full(grid.height, bg, dtype=int)
        new_col[grid.height - len(non_bg):] = non_bg
        new_data[:, c] = new_col
    return Grid(new_data)

def extract_subgrid(grid, r1, c1, r2, c2):
    """Extract a rectangular subgrid [r1:r2+1, c1:c2+1]."""
    return Grid(grid.data[r1:r2 + 1, c1:c2 + 1])

def pad_grid(grid, top=0, bottom=0, left=0, right=0, fill=0):
    """Pad grid with a fill color."""
    return Grid(np.pad(grid.data, ((top, bottom), (left, right)),
                       mode='constant', constant_values=fill))


# =============================================================================
# Section 3b — Primitive Registry (for Genetic Algorithm)
# =============================================================================

def _fill_enclosed_default(grid, fill_color=1):
    """Fill all enclosed background regions with fill_color."""
    h, w = grid.height, grid.width
    bg = grid.background_color()
    result = grid.data.copy()

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

    for r in range(h):
        for c in range(w):
            if int(result[r, c]) == bg and not reachable[r, c]:
                result[r, c] = fill_color
    return Grid(result)


def _enforce_sym_h(grid):
    """Enforce horizontal symmetry (left→right)."""
    data = grid.data.copy()
    h, w = data.shape
    for r in range(h):
        for c in range(w // 2):
            data[r, w - 1 - c] = data[r, c]
    return Grid(data)


def _enforce_sym_v(grid):
    """Enforce vertical symmetry (top→bottom)."""
    data = grid.data.copy()
    h, w = data.shape
    for r in range(h // 2):
        data[h - 1 - r, :] = data[r, :]
    return Grid(data)


def _gravity_up(grid, bg=0):
    """Gravity upward."""
    return mirror_vertical(gravity_down(mirror_vertical(grid), bg=bg))


def _gravity_right(grid, bg=0):
    """Gravity rightward."""
    return rotate270(gravity_down(rotate90(grid), bg=bg))


def _gravity_left(grid, bg=0):
    """Gravity leftward."""
    return rotate90(gravity_down(rotate270(grid), bg=bg))


def _extract_largest_object(grid, bg=0):
    """Extract the largest connected component as a subgrid."""
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    largest = max(objs, key=lambda o: o.area)
    return largest.subgrid(bg=bg)


def _extract_smallest_object(grid, bg=0):
    """Extract the smallest connected component as a subgrid."""
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    smallest = min(objs, key=lambda o: o.area)
    return smallest.subgrid(bg=bg)


def _remove_border(grid):
    """Remove 1-cell border."""
    if grid.height < 3 or grid.width < 3:
        return grid
    return extract_subgrid(grid, 1, 1, grid.height - 2, grid.width - 2)


def _identity(grid):
    """Identity transform."""
    return grid


# =============================================================================
# Section 3c — Extended Primitives (Color, Boolean, Structural)
# =============================================================================

def _keep_color(grid, color):
    """Zero out all cells except those matching 'color'; keep background."""
    bg = grid.background_color()
    data = grid.data.copy()
    result = np.full_like(data, bg)
    result[data == color] = color
    return Grid(result)


def _fill_all_with_color(grid, color):
    """Replace every non-background cell with 'color'."""
    bg = grid.background_color()
    data = grid.data.copy()
    data[data != bg] = color
    return Grid(data)


def _remove_color(grid, color):
    """Replace every cell of 'color' with the background color."""
    bg = grid.background_color()
    data = grid.data.copy()
    data[data == color] = bg
    return Grid(data)


def _color_invert(grid):
    """Swap background color with the most frequent foreground color."""
    bg = grid.background_color()
    cc = grid.color_counts()
    non_bg = {k: v for k, v in cc.items() if k != bg}
    if not non_bg:
        return grid
    fg = max(non_bg, key=non_bg.get)
    data = grid.data.copy()
    result = data.copy()
    result[data == bg] = fg
    result[data == fg] = bg
    return Grid(result)


def _remap_by_frequency(grid):
    """Remap colors so most-frequent non-bg → 1, second → 2, etc."""
    bg = grid.background_color()
    cc = grid.color_counts()
    non_bg = {k: v for k, v in cc.items() if k != bg}
    if not non_bg:
        return grid
    sorted_colors = sorted(non_bg.keys(), key=lambda c: non_bg[c], reverse=True)
    mapping = {c: (i + 1) for i, c in enumerate(sorted_colors)}
    mapping[bg] = bg
    data = grid.data.copy()
    result = np.full_like(data, bg)
    for old_c, new_c in mapping.items():
        result[data == old_c] = new_c
    return Grid(result)


def _outline_objects(grid):
    """Keep only border cells of each object (hollow interiors)."""
    bg = grid.background_color()
    data = grid.data
    h, w = data.shape
    result = np.full_like(data, bg)
    for r in range(h):
        for c in range(w):
            if int(data[r, c]) != bg:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or int(data[nr, nc]) == bg:
                        result[r, c] = data[r, c]
                        break
    return Grid(result)


def _mirror_diag(grid):
    """Transpose (mirror along main diagonal) — only for square grids."""
    if grid.height == grid.width:
        return Grid(grid.data.T)
    return grid


def _mirror_anti_diag(grid):
    """Mirror along anti-diagonal — only for square grids."""
    if grid.height == grid.width:
        return Grid(np.rot90(np.fliplr(grid.data)))
    return grid


def _downscale_2x(grid):
    """Take every 2nd row and column (nearest-neighbor downscale by 2)."""
    return Grid(grid.data[::2, ::2])


def _downscale_3x(grid):
    """Take every 3rd row and column (nearest-neighbor downscale by 3)."""
    return Grid(grid.data[::3, ::3])


def _sort_rows_by_sum(grid):
    """Sort rows in ascending order of their color-value sum."""
    rows = list(grid.data)
    rows.sort(key=lambda r: int(np.sum(r)))
    return Grid(np.array(rows))


def _sort_cols_by_sum(grid):
    """Sort columns in ascending order of their color-value sum."""
    cols = list(grid.data.T)
    cols.sort(key=lambda c: int(np.sum(c)))
    return Grid(np.array(cols).T)


# Boolean halves — horizontal split
def _halves_h_or(grid):
    """Top half OR bottom half → non-bg from either half."""
    if grid.height % 2 != 0:
        return grid
    bg = grid.background_color()
    m = grid.height // 2
    t, b = grid.data[:m, :], grid.data[m:, :]
    result = np.where(t != bg, t, b)
    return Grid(result)


def _halves_h_and(grid):
    """Top half AND bottom half → non-bg only when both halves non-bg."""
    if grid.height % 2 != 0:
        return grid
    bg = grid.background_color()
    m = grid.height // 2
    t, b = grid.data[:m, :], grid.data[m:, :]
    result = np.where((t != bg) & (b != bg), t, np.full_like(t, bg))
    return Grid(result)


def _halves_h_xor(grid):
    """Top half XOR bottom half → non-bg in exactly one half."""
    if grid.height % 2 != 0:
        return grid
    bg = grid.background_color()
    m = grid.height // 2
    t, b = grid.data[:m, :], grid.data[m:, :]
    result = np.full_like(t, bg)
    result[(t != bg) & (b == bg)] = t[(t != bg) & (b == bg)]
    result[(t == bg) & (b != bg)] = b[(t == bg) & (b != bg)]
    return Grid(result)


# Boolean halves — vertical split
def _halves_v_or(grid):
    """Left half OR right half → non-bg from either half."""
    if grid.width % 2 != 0:
        return grid
    bg = grid.background_color()
    m = grid.width // 2
    l, r = grid.data[:, :m], grid.data[:, m:]
    return Grid(np.where(l != bg, l, r))


def _halves_v_and(grid):
    """Left half AND right half → non-bg only when both halves non-bg."""
    if grid.width % 2 != 0:
        return grid
    bg = grid.background_color()
    m = grid.width // 2
    l, r = grid.data[:, :m], grid.data[:, m:]
    return Grid(np.where((l != bg) & (r != bg), l, np.full_like(l, bg)))


def _halves_v_xor(grid):
    """Left half XOR right half → non-bg in exactly one half."""
    if grid.width % 2 != 0:
        return grid
    bg = grid.background_color()
    m = grid.width // 2
    l, r = grid.data[:, :m], grid.data[:, m:]
    result = np.full_like(l, bg)
    result[(l != bg) & (r == bg)] = l[(l != bg) & (r == bg)]
    result[(l == bg) & (r != bg)] = r[(l == bg) & (r != bg)]
    return Grid(result)


def _count_to_color(grid):
    """Return a 1×1 grid whose value is the number of distinct non-bg objects."""
    bg = grid.background_color()
    objs = find_objects(grid, bg=bg)
    count = min(len(objs), 9)  # Clamp to valid color range
    return Grid(np.array([[count]]))


def _unique_rows(grid):
    """Deduplicate rows, keeping first occurrence of each unique row."""
    seen = []
    result = []
    for row in grid.data:
        key = tuple(row.tolist())
        if key not in seen:
            seen.append(key)
            result.append(row)
    if not result:
        return grid
    return Grid(np.array(result))


def _repeat_border(grid):
    """Pad grid by repeating its border pixels once."""
    data = grid.data
    top = data[0:1, :]
    bot = data[-1:, :]
    mid = np.hstack([data[:, 0:1], data, data[:, -1:]])
    full = np.vstack([np.hstack([data[0:1, 0:1], top, data[0:1, -1:]]),
                      mid,
                      np.hstack([data[-1:, 0:1], bot, data[-1:, -1:]])])
    return Grid(full)


def _fill_bg_with_most_common_fg(grid):
    """Replace background with the single most-common foreground color."""
    bg = grid.background_color()
    cc = grid.color_counts()
    non_bg = {k: v for k, v in cc.items() if k != bg}
    if not non_bg:
        return grid
    fg = max(non_bg, key=non_bg.get)
    data = grid.data.copy()
    data[data == bg] = fg
    return Grid(data)



# The master registry: name → callable(Grid) → Grid
PRIMITIVE_REGISTRY = {
    # ── Identity ────────────────────────────────────────────────────
    "identity": _identity,

    # ── Geometric (rigid transforms) ────────────────────────────────
    "rotate90":            rotate90,
    "rotate180":           rotate180,
    "rotate270":           rotate270,
    "mirror_h":            mirror_horizontal,
    "mirror_v":            mirror_vertical,
    "transpose":           transpose,
    "mirror_diag":         _mirror_diag,
    "mirror_anti_diag":    _mirror_anti_diag,

    # ── Symmetry enforcement ────────────────────────────────────────
    "enforce_sym_h":       _enforce_sym_h,
    "enforce_sym_v":       _enforce_sym_v,

    # ── Crop / extract ──────────────────────────────────────────────
    "crop":                lambda g: crop_to_content(g, bg=0),
    "crop_bg1":            lambda g: crop_to_content(g, bg=1),
    "extract_largest":     _extract_largest_object,
    "extract_smallest":    _extract_smallest_object,
    "unique_rows":         _unique_rows,

    # ── Scaling ─────────────────────────────────────────────────────
    "scale_2x":            lambda g: scale_up(g, 2),
    "scale_3x":            lambda g: scale_up(g, 3),
    "downscale_2x":        _downscale_2x,
    "downscale_3x":        _downscale_3x,

    # ── Tiling ──────────────────────────────────────────────────────
    "tile_2x2":            lambda g: tile_grid(g, 2, 2),
    "tile_1x2":            lambda g: tile_grid(g, 1, 2),
    "tile_2x1":            lambda g: tile_grid(g, 2, 1),
    "tile_3x3":            lambda g: tile_grid(g, 3, 3),
    "tile_1x3":            lambda g: tile_grid(g, 1, 3),
    "tile_3x1":            lambda g: tile_grid(g, 3, 1),

    # ── Gravity ─────────────────────────────────────────────────────
    "gravity_down":        lambda g: gravity_down(g, bg=0),
    "gravity_up":          _gravity_up,
    "gravity_right":       _gravity_right,
    "gravity_left":        _gravity_left,

    # ── Fill operations ─────────────────────────────────────────────
    "fill_enclosed":       _fill_enclosed_default,
    "outline_objects":     _outline_objects,
    "fill_bg_with_fg":     _fill_bg_with_most_common_fg,

    # ── Border operations ───────────────────────────────────────────
    "remove_border":       _remove_border,
    "add_border_0":        lambda g: pad_grid(g, 1, 1, 1, 1, fill=0),
    "repeat_border":       _repeat_border,

    # ── Color analysis & remapping ──────────────────────────────────
    "color_invert":        _color_invert,
    "remap_by_frequency":  _remap_by_frequency,
    "count_to_color":      _count_to_color,

    # ── Sort ────────────────────────────────────────────────────────
    "sort_rows_by_sum":    _sort_rows_by_sum,
    "sort_cols_by_sum":    _sort_cols_by_sum,

    # ── Boolean half-grid operations ────────────────────────────────
    "halves_h_or":         _halves_h_or,
    "halves_h_and":        _halves_h_and,
    "halves_h_xor":        _halves_h_xor,
    "halves_v_or":         _halves_v_or,
    "halves_v_and":        _halves_v_and,
    "halves_v_xor":        _halves_v_xor,

    # ── Color-specific keep (zero-out everything except color N) ────
    **{f"keep_color_{c}": (lambda g, _c=c: _keep_color(g, _c)) for c in range(10)},

    # ── Color-specific fill (paint all fg cells with color N) ───────
    **{f"fill_color_{c}": (lambda g, _c=c: _fill_all_with_color(g, _c)) for c in range(1, 10)},

    # ── Color-specific remove (erase all cells of color N) ──────────
    **{f"remove_color_{c}": (lambda g, _c=c: _remove_color(g, _c)) for c in range(1, 10)},
}

# Ordered list of primitive names (the GA's alphabet)
PRIMITIVE_NAMES = list(PRIMITIVE_REGISTRY.keys())


def program_from_names(name_list):
    """
    Build a ComposedRule from a list of primitive names.
    Returns None if any name is invalid.
    """
    rules = []
    for name in name_list:
        fn = PRIMITIVE_REGISTRY.get(name)
        if fn is None:
            return None
        rules.append(TransformationRule(name, fn))
    if len(rules) == 0:
        return None
    if len(rules) == 1:
        return rules[0]
    return ComposedRule(rules)


# =============================================================================
# Section 4 — Node-Based Block Analysis
# =============================================================================

class BlockGraph:
    """
    Abstract graph where nodes = objects (connected components)
    and edges = spatial relationships.

    This enables higher-level reasoning about object arrangements.
    """

    def __init__(self, grid, bg=0):
        self.grid = grid
        self.bg = bg
        self.objects = find_objects(grid, bg=bg)
        self.edges = []
        self._build_edges()

    def _build_edges(self):
        """Compute spatial relationship edges between all object pairs."""
        for i, obj_a in enumerate(self.objects):
            for j, obj_b in enumerate(self.objects):
                if i >= j:
                    continue
                rel = self._relationship(obj_a, obj_b)
                if rel:
                    self.edges.append((i, j, rel))

    def _relationship(self, a, b):
        """Determine spatial relationship between two objects."""
        ca, cb = a.centroid, b.centroid
        rels = []
        # Vertical relationship
        if ca[0] < cb[0]:
            rels.append('above')
        elif ca[0] > cb[0]:
            rels.append('below')
        # Horizontal relationship
        if ca[1] < cb[1]:
            rels.append('left_of')
        elif ca[1] > cb[1]:
            rels.append('right_of')
        # Adjacency (touching bounding boxes)
        if self._bboxes_adjacent(a, b):
            rels.append('adjacent')
        # Containment
        if self._contains(a, b):
            rels.append('contains')
        elif self._contains(b, a):
            rels.append('contained_by')
        return rels if rels else None

    @staticmethod
    def _bboxes_adjacent(a, b):
        """Check if bounding boxes are touching (within 1 cell gap)."""
        return not (a.max_r + 1 < b.min_r or b.max_r + 1 < a.min_r or
                    a.max_c + 1 < b.min_c or b.max_c + 1 < a.min_c)

    @staticmethod
    def _contains(outer, inner):
        """Check if outer's bbox fully contains inner's bbox."""
        return (outer.min_r <= inner.min_r and outer.max_r >= inner.max_r and
                outer.min_c <= inner.min_c and outer.max_c >= inner.max_c)

    @property
    def num_objects(self):
        return len(self.objects)

    def objects_by_color(self, color):
        return [o for o in self.objects if o.primary_color == color]

    def objects_by_size(self, ascending=True):
        return sorted(self.objects, key=lambda o: o.area, reverse=not ascending)

    def largest_object(self):
        return max(self.objects, key=lambda o: o.area) if self.objects else None

    def smallest_object(self):
        return min(self.objects, key=lambda o: o.area) if self.objects else None


# =============================================================================
# Section 5 — Mathematical Analysis
# =============================================================================

def detect_symmetry(grid):
    """Detect grid symmetries: horizontal, vertical, 90° rotational."""
    symmetries = []
    if np.array_equal(grid.data, np.fliplr(grid.data)):
        symmetries.append('horizontal')
    if np.array_equal(grid.data, np.flipud(grid.data)):
        symmetries.append('vertical')
    if grid.height == grid.width:
        if np.array_equal(grid.data, np.rot90(grid.data, k=-1)):
            symmetries.append('rot90')
        if np.array_equal(grid.data, np.rot90(grid.data, k=2)):
            symmetries.append('rot180')
    return symmetries

def detect_scaling(input_grid, output_grid):
    """Detect if output is an integer scale of input."""
    ih, iw = input_grid.shape
    oh, ow = output_grid.shape
    if oh % ih == 0 and ow % iw == 0:
        sr, sc = oh // ih, ow // iw
        if sr == sc:
            scaled = scale_up(input_grid, sr)
            if scaled == output_grid:
                return sr
    return None

def detect_tiling(input_grid, output_grid):
    """Detect if output is a tiling of input."""
    ih, iw = input_grid.shape
    oh, ow = output_grid.shape
    if oh % ih == 0 and ow % iw == 0:
        tr, tc = oh // ih, ow // iw
        tiled = tile_grid(input_grid, tr, tc)
        if tiled == output_grid:
            return (tr, tc)
    return None

def detect_color_mapping(input_grid, output_grid):
    """Detect a consistent color mapping between input and output."""
    if input_grid.shape != output_grid.shape:
        return None
    mapping = {}
    for r in range(input_grid.height):
        for c in range(input_grid.width):
            ic = input_grid.cell(r, c)
            oc = output_grid.cell(r, c)
            if ic in mapping:
                if mapping[ic] != oc:
                    return None  # Inconsistent
            else:
                mapping[ic] = oc
    return mapping

def compute_diff(input_grid, output_grid):
    """Compute cell-level differences between two grids of same shape."""
    if input_grid.shape != output_grid.shape:
        return None
    diff = []
    for r in range(input_grid.height):
        for c in range(input_grid.width):
            if input_grid.cell(r, c) != output_grid.cell(r, c):
                diff.append((r, c, input_grid.cell(r, c), output_grid.cell(r, c)))
    return diff

def grid_similarity(a, b):
    """
    Compute similarity score between two grids.
    Returns 1.0 for exact match, 0.0 for completely different.
    """
    if a.shape != b.shape:
        # Penalize shape mismatch
        return 0.0
    total = a.height * a.width
    matching = np.sum(a.data == b.data)
    return matching / total


# =============================================================================
# Section 6 — Transformation Rule (Candidate for Beam Search)
# =============================================================================

class TransformationRule:
    """
    A candidate transformation rule.
    Wraps a function that maps Grid -> Grid plus metadata for beam search.
    """

    def __init__(self, name, transform_fn, score=0.0, params=None):
        self.name = name
        self.transform_fn = transform_fn
        self.score = score
        self.params = params or {}

    def apply(self, input_grid):
        """Apply this rule to produce an output grid."""
        try:
            result = self.transform_fn(input_grid)
            if isinstance(result, Grid):
                return result
            return Grid(result)
        except Exception:
            return None

    def __repr__(self):
        return f"Rule({self.name}, score={self.score:.3f})"


class ComposedRule(TransformationRule):
    """Pipeline of multiple rules applied sequentially."""

    def __init__(self, rules):
        self.rules = rules
        name = " → ".join(r.name for r in rules)
        super().__init__(name, self._apply_pipeline)

    def _apply_pipeline(self, input_grid):
        result = input_grid
        for rule in self.rules:
            result = rule.apply(result)
            if result is None:
                return None
        return result


# =============================================================================
# Section 7 — Beam Search Solver
# =============================================================================

def beam_search_solve(task, solvers, beam_width=10):
    """
    Beam search over candidate transformation rules.

    Args:
        task: dict with 'train' (list of {input, output}) and 'test' (list of {input})
        solvers: list of heuristic solver functions
        beam_width: number of candidates to keep at each step

    Returns:
        list of predictions, each a dict with 'attempt_1' and 'attempt_2' (as lists)
    """
    train_pairs = task['train']
    test_inputs = task['test']

    # Phase 1: Generate initial beam from all solvers
    beam = []
    for solver in solvers:
        try:
            candidates = solver(train_pairs)
            beam.extend(candidates)
        except Exception:
            continue

    if not beam:
        # Fallback: identity rule
        beam.append(TransformationRule("identity", lambda g: g))

    # Phase 2: Score every candidate against ALL training pairs
    for candidate in beam:
        total_score = 0.0
        for pair in train_pairs:
            inp = Grid(pair['input'])
            expected = Grid(pair['output'])
            predicted = candidate.apply(inp)
            if predicted is not None:
                total_score += grid_similarity(predicted, expected)
        candidate.score = total_score / len(train_pairs)

    # Sort and prune beam
    beam.sort(key=lambda c: c.score, reverse=True)
    beam = beam[:beam_width]

    # Phase 3: Try composing top candidates (depth-2 compositions)
    if beam[0].score < 1.0:
        composed = []
        top = beam[:min(5, len(beam))]
        for i, r1 in enumerate(top):
            for j, r2 in enumerate(top):
                if i != j:
                    comp = ComposedRule([r1, r2])
                    composed.append(comp)

        for candidate in composed:
            total_score = 0.0
            for pair in train_pairs:
                inp = Grid(pair['input'])
                expected = Grid(pair['output'])
                predicted = candidate.apply(inp)
                if predicted is not None:
                    total_score += grid_similarity(predicted, expected)
            candidate.score = total_score / len(train_pairs)

        beam.extend(composed)
        beam.sort(key=lambda c: c.score, reverse=True)
        beam = beam[:beam_width]

    # Phase 4: Apply top-2 rules to each test input
    predictions = []
    for test in test_inputs:
        inp = Grid(test['input'])

        # Attempt 1: best scoring rule
        attempt_1 = beam[0].apply(inp) if beam else None
        if attempt_1 is None:
            attempt_1 = inp  # Fallback: echo input

        # Attempt 2: second-best rule (if different, else same)
        attempt_2 = attempt_1
        if len(beam) > 1:
            a2 = beam[1].apply(inp)
            if a2 is not None:
                attempt_2 = a2

        predictions.append({
            'attempt_1': attempt_1.to_list(),
            'attempt_2': attempt_2.to_list()
        })

    return predictions, beam[0].score if beam else 0.0


def beam_search_solve_extended(task, solvers, beam_width=10):
    """
    Extended beam search that also returns the scored beam candidates.
    Used by the hybrid solver to seed the Genetic Algorithm.

    Returns:
        (predictions, best_score, beam) where beam is the list of
        scored TransformationRule/ComposedRule objects.
    """
    train_pairs = task['train']
    test_inputs = task['test']

    # Phase 1: Generate initial beam from all solvers
    beam = []
    for solver in solvers:
        try:
            candidates = solver(train_pairs)
            beam.extend(candidates)
        except Exception:
            continue

    if not beam:
        beam.append(TransformationRule("identity", lambda g: g))

    # Phase 2: Score every candidate against ALL training pairs
    for candidate in beam:
        total_score = 0.0
        for pair in train_pairs:
            inp = Grid(pair['input'])
            expected = Grid(pair['output'])
            predicted = candidate.apply(inp)
            if predicted is not None:
                total_score += grid_similarity(predicted, expected)
        candidate.score = total_score / len(train_pairs)

    beam.sort(key=lambda c: c.score, reverse=True)
    beam = beam[:beam_width]

    # Phase 3: Try composing top candidates (depth-2 compositions)
    if beam[0].score < 1.0:
        composed = []
        top = beam[:min(5, len(beam))]
        for i, r1 in enumerate(top):
            for j, r2 in enumerate(top):
                if i != j:
                    comp = ComposedRule([r1, r2])
                    composed.append(comp)

        for candidate in composed:
            total_score = 0.0
            for pair in train_pairs:
                inp = Grid(pair['input'])
                expected = Grid(pair['output'])
                predicted = candidate.apply(inp)
                if predicted is not None:
                    total_score += grid_similarity(predicted, expected)
            candidate.score = total_score / len(train_pairs)

        beam.extend(composed)
        beam.sort(key=lambda c: c.score, reverse=True)
        beam = beam[:beam_width]

    # Phase 4: Apply top-2 rules to each test input
    predictions = []
    for test in test_inputs:
        inp = Grid(test['input'])

        attempt_1 = beam[0].apply(inp) if beam else None
        if attempt_1 is None:
            attempt_1 = inp

        attempt_2 = attempt_1
        if len(beam) > 1:
            a2 = beam[1].apply(inp)
            if a2 is not None:
                attempt_2 = a2

        predictions.append({
            'attempt_1': attempt_1.to_list(),
            'attempt_2': attempt_2.to_list()
        })

    return predictions, beam[0].score if beam else 0.0, beam
