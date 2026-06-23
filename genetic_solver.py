"""
ARC Prize 2026 — Genetic Algorithm Solver (v2 — Island Model Redesign)
=======================================================================
Major improvements over v1:
  - Island model: 4 sub-populations (Explorer, Exploiter, ColorSpec, GeoSpec)
    with migration every N generations — prevents premature convergence
  - Schema-preserving crossover: hot genes (fitness contributors) are protected
  - Segment crossover: swaps gene blocks of the same semantic category
  - Diff-based task-aware seeding: analyzes input/output patterns to warm-start
  - Exact-match bonus in fitness (0.20 weight) — heavily rewards solving any pair
  - Local hill-climbing (Lamarckian step) on top-20% after crossover
  - Adaptive mutation rate: boosts when population stagnates
  - Program length expanded to 8 (from 5)
"""

import random
import numpy as np
import collections
from arc_solver import (
    Grid, TransformationRule, ComposedRule,
    PRIMITIVE_REGISTRY, PRIMITIVE_NAMES, program_from_names,
    grid_similarity, find_objects,
    mirror_horizontal, mirror_vertical,
    detect_color_mapping, detect_scaling, detect_tiling, detect_symmetry,
)


# =============================================================================
# Section 1 — Program Representation
# =============================================================================

class Program:
    """
    A candidate transformation program = ordered list of primitive names (genes).
    Tracks schema (which genes contributed positively to fitness).
    """

    def __init__(self, genes):
        self.genes = list(genes)
        self.fitness = 0.0
        self.exact_hits = 0          # Number of training pairs solved exactly
        self.hot_mask = None         # list[bool]: True = gene is a fitness contributor
        self._rule = None            # Lazily compiled rule

    @property
    def rule(self):
        if self._rule is None:
            self._rule = program_from_names(self.genes)
        return self._rule

    def invalidate_cache(self):
        self._rule = None
        self.hot_mask = None

    def apply(self, grid):
        r = self.rule
        if r is None:
            return None
        try:
            return r.apply(grid)
        except Exception:
            return None

    def copy(self):
        p = Program(list(self.genes))
        p.fitness = self.fitness
        p.exact_hits = self.exact_hits
        if self.hot_mask is not None:
            p.hot_mask = list(self.hot_mask)
        return p

    def __repr__(self):
        return f"Program({self.genes}, fit={self.fitness:.4f}, exact={self.exact_hits})"

    def __len__(self):
        return len(self.genes)

    def signature(self):
        return tuple(self.genes)


# =============================================================================
# Section 2 — Fitness Evaluation
# =============================================================================

def compute_program_fitness(program, train_pairs,
                             w_accuracy=0.55, w_shape=0.15,
                             w_exact=0.20, w_simplicity=0.10):
    """
    Multi-objective fitness with exact-match bonus.

    Components:
      - accuracy:    average cell-level similarity across all pairs (0..1)
      - shape_match: fraction of pairs where predicted shape == expected shape
      - exact_bonus: fraction of pairs solved perfectly (0 or 1 per pair)
      - simplicity:  1/(1+len) penalty for long programs

    The exact_bonus weight (0.20) strongly rewards programs that solve
    at least one training pair perfectly, steering away from mediocre
    partial-match local optima.
    """
    if not program.genes:
        return 0.0, 0

    total_accuracy = 0.0
    total_shape_match = 0.0
    exact_hits = 0
    n_pairs = len(train_pairs)

    for pair in train_pairs:
        inp = Grid(pair['input'])
        expected = Grid(pair['output'])

        try:
            predicted = program.apply(inp)
        except Exception:
            predicted = None

        if predicted is None:
            continue

        sim = grid_similarity(predicted, expected)
        total_accuracy += sim

        if predicted.shape == expected.shape:
            total_shape_match += 1.0

        if sim >= 1.0:
            exact_hits += 1

    accuracy = total_accuracy / n_pairs if n_pairs > 0 else 0.0
    shape_match = total_shape_match / n_pairs if n_pairs > 0 else 0.0
    exact_fraction = exact_hits / n_pairs if n_pairs > 0 else 0.0
    simplicity = 1.0 / (1.0 + len(program.genes))

    fitness = (w_accuracy * accuracy +
               w_shape * shape_match +
               w_exact * exact_fraction +
               w_simplicity * simplicity)

    return fitness, exact_hits


# =============================================================================
# Section 3 — Diff-Based Task-Aware Seeding
# =============================================================================

def analyze_task_and_seed(task, n_seeds=40):
    """
    Analyze input→output training pairs to generate targeted seed programs.

    Instead of starting the GA completely random, we inspect:
      - Shape ratios (→ scale, tile, halves, crop seeds)
      - Exact geometric matches (→ rotate/mirror seeds)
      - Color change patterns (→ color keep/fill/remap seeds)
      - Symmetry (→ enforce_sym seeds)
      - Object layout (→ gravity, extract seeds)

    Returns a list of Program objects tailored to this specific task.
    """
    train_pairs = task['train']
    seeds = []
    seen = set()

    def add_seed(gene_list):
        if not gene_list:
            return
        if not all(g in PRIMITIVE_REGISTRY for g in gene_list):
            return
        sig = tuple(gene_list)
        if sig not in seen:
            seen.add(sig)
            seeds.append(Program(gene_list))

    # Always add single-primitive seeds for the whole alphabet
    for name in PRIMITIVE_NAMES:
        add_seed([name])

    # Analyze each training pair
    for pair in train_pairs:
        inp = Grid(pair['input'])
        out = Grid(pair['output'])
        ih, iw = inp.shape
        oh, ow = out.shape

        # --- Same shape ---
        if oh == ih and ow == iw:
            add_seed(["enforce_sym_h"])
            add_seed(["enforce_sym_v"])
            add_seed(["fill_enclosed"])
            add_seed(["color_invert"])
            add_seed(["remap_by_frequency"])
            add_seed(["fill_bg_with_fg"])
            add_seed(["outline_objects"])

            # Color mapping seeds
            cmap = detect_color_mapping(inp, out)
            if cmap and any(k != v for k, v in cmap.items()):
                for c in range(10):
                    add_seed([f"keep_color_{c}"])
                    add_seed([f"fill_color_{c}"])
                    add_seed([f"remove_color_{c}"])

            # Symmetry-based seeds
            syms = detect_symmetry(inp)
            if 'horizontal' in syms:
                add_seed(["mirror_h"])
                add_seed(["mirror_h", "enforce_sym_v"])
            if 'vertical' in syms:
                add_seed(["mirror_v"])
                add_seed(["mirror_v", "enforce_sym_h"])

        # --- Output is smaller ---
        if oh < ih or ow < iw:
            add_seed(["crop"])
            add_seed(["crop_bg1"])
            add_seed(["extract_largest"])
            add_seed(["extract_smallest"])
            if ih > 0 and oh > 0 and ih % oh == 0:
                ratio = ih // oh
                if ratio == 2:
                    add_seed(["downscale_2x"])
                elif ratio == 3:
                    add_seed(["downscale_3x"])
            # Halves ops: output = top or bottom half
            if oh == ih // 2 and ow == iw and ih % 2 == 0:
                add_seed(["halves_h_or"])
                add_seed(["halves_h_and"])
                add_seed(["halves_h_xor"])
            if ow == iw // 2 and oh == ih and iw % 2 == 0:
                add_seed(["halves_v_or"])
                add_seed(["halves_v_and"])
                add_seed(["halves_v_xor"])

        # --- Output is larger ---
        if oh > ih or ow > iw:
            scale = detect_scaling(inp, out)
            if scale == 2:
                add_seed(["scale_2x"])
            elif scale == 3:
                add_seed(["scale_3x"])
            tiling = detect_tiling(inp, out)
            if tiling:
                tr, tc = tiling
                if tr == 2 and tc == 2:
                    add_seed(["tile_2x2"])
                elif tr == 1 and tc == 2:
                    add_seed(["tile_1x2"])
                elif tr == 2 and tc == 1:
                    add_seed(["tile_2x1"])
                elif tr == 3 and tc == 3:
                    add_seed(["tile_3x3"])
                elif tr == 1 and tc == 3:
                    add_seed(["tile_1x3"])
                elif tr == 3 and tc == 1:
                    add_seed(["tile_3x1"])

        # --- Exact geometric match detection ---
        for name, fn in [
            ("rotate90",    lambda g: Grid(np.rot90(g.data, k=-1))),
            ("rotate180",   lambda g: Grid(np.rot90(g.data, k=2))),
            ("rotate270",   lambda g: Grid(np.rot90(g.data, k=1))),
            ("mirror_h",    mirror_horizontal),
            ("mirror_v",    mirror_vertical),
        ]:
            try:
                if fn(inp) == out:
                    add_seed([name])
                    # Also try composing with color ops
                    add_seed([name, "fill_enclosed"])
                    add_seed([name, "remap_by_frequency"])
            except Exception:
                pass

        # --- Gravity analysis ---
        bg = inp.background_color()
        non_bg_rows = [r for r in range(inp.height)
                       if np.any(inp.data[r, :] != bg)]
        if non_bg_rows:
            if max(non_bg_rows) > inp.height * 0.6:
                add_seed(["gravity_down"])
            if min(non_bg_rows) < inp.height * 0.4:
                add_seed(["gravity_up"])

        # --- Two-step compositions from common patterns ---
        add_seed(["extract_largest", "rotate90"])
        add_seed(["extract_largest", "mirror_h"])
        add_seed(["crop", "scale_2x"])
        add_seed(["crop", "tile_2x2"])
        add_seed(["mirror_h", "mirror_v"])
        add_seed(["rotate90", "mirror_h"])
        add_seed(["fill_enclosed", "color_invert"])
        add_seed(["outline_objects", "fill_enclosed"])

    return seeds[:n_seeds]


# =============================================================================
# Section 4 — Genetic Operators
# =============================================================================

# Gene category mapping for segment crossover
_CATEGORY_MAP = {
    'rotate90': 'geo', 'rotate180': 'geo', 'rotate270': 'geo',
    'mirror_h': 'geo', 'mirror_v': 'geo', 'transpose': 'geo',
    'mirror_diag': 'geo', 'mirror_anti_diag': 'geo',
    'enforce_sym_h': 'sym', 'enforce_sym_v': 'sym',
    'gravity_down': 'grav', 'gravity_up': 'grav',
    'gravity_right': 'grav', 'gravity_left': 'grav',
    'fill_enclosed': 'fill', 'outline_objects': 'fill', 'fill_bg_with_fg': 'fill',
    'color_invert': 'color', 'remap_by_frequency': 'color', 'count_to_color': 'color',
    'crop': 'extract', 'crop_bg1': 'extract',
    'extract_largest': 'extract', 'extract_smallest': 'extract',
    'unique_rows': 'extract', 'sort_rows_by_sum': 'extract', 'sort_cols_by_sum': 'extract',
    'scale_2x': 'scale', 'scale_3x': 'scale',
    'downscale_2x': 'scale', 'downscale_3x': 'scale',
    'halves_h_or': 'bool', 'halves_h_and': 'bool', 'halves_h_xor': 'bool',
    'halves_v_or': 'bool', 'halves_v_and': 'bool', 'halves_v_xor': 'bool',
    'remove_border': 'border', 'add_border_0': 'border', 'repeat_border': 'border',
}
for _i in range(10):
    _CATEGORY_MAP[f'keep_color_{_i}'] = 'color'
    _CATEGORY_MAP[f'fill_color_{_i}'] = 'color'
    _CATEGORY_MAP[f'remove_color_{_i}'] = 'color'
for _i in range(1, 10):
    _CATEGORY_MAP[f'tile_{_i}x{_i}'] = 'tile'
    _CATEGORY_MAP[f'tile_1x{_i}'] = 'tile'
    _CATEGORY_MAP[f'tile_{_i}x1'] = 'tile'


def _get_category(gene):
    return _CATEGORY_MAP.get(gene, 'other')


def tournament_select(population, tournament_size=5):
    """Tournament selection: pick best from a random subset."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda p: p.fitness)


def schema_crossover(p1, p2, max_length=8):
    """
    Schema-preserving crossover.

    Hot genes (those that contributed positively to fitness) are protected
    from being overwritten. Cold genes are swapped between parents.

    Strategy:
      Child1 = p1 skeleton, cold positions replaced by p2's hot genes first
      Child2 = p2 skeleton, cold positions replaced by p1's hot genes first
    """
    g1 = list(p1.genes)
    g2 = list(p2.genes)
    hot1 = (p1.hot_mask if p1.hot_mask and len(p1.hot_mask) == len(g1)
            else [False] * len(g1))
    hot2 = (p2.hot_mask if p2.hot_mask and len(p2.hot_mask) == len(g2)
            else [False] * len(g2))

    cold_pos1 = [i for i, h in enumerate(hot1) if not h]
    hot_genes2 = [g for g, h in zip(g2, hot2) if h]
    cold_genes2 = [g for g, h in zip(g2, hot2) if not h]
    donor_to_1 = hot_genes2 + cold_genes2  # hot genes from p2 first

    child1_genes = list(g1)
    for idx, pos in enumerate(cold_pos1):
        if idx < len(donor_to_1):
            child1_genes[pos] = donor_to_1[idx]

    cold_pos2 = [i for i, h in enumerate(hot2) if not h]
    hot_genes1 = [g for g, h in zip(g1, hot1) if h]
    cold_genes1 = [g for g, h in zip(g1, hot1) if not h]
    donor_to_2 = hot_genes1 + cold_genes1

    child2_genes = list(g2)
    for idx, pos in enumerate(cold_pos2):
        if idx < len(donor_to_2):
            child2_genes[pos] = donor_to_2[idx]

    # Trim + safety
    child1_genes = child1_genes[:max_length] or [random.choice(PRIMITIVE_NAMES)]
    child2_genes = child2_genes[:max_length] or [random.choice(PRIMITIVE_NAMES)]

    return Program(child1_genes), Program(child2_genes)


def segment_crossover(p1, p2, max_length=8):
    """
    Segment crossover: swap blocks of same-category genes.

    Genes are grouped by semantic category (geometric, color, structural…).
    We swap a chosen category's genes between parents, keeping others intact.
    This preserves meaningful sub-sequences (e.g. a color pipeline).
    """
    g1, g2 = list(p1.genes), list(p2.genes)
    cats1 = [_get_category(g) for g in g1]
    cats2 = [_get_category(g) for g in g2]

    common_cats = set(cats1) & set(cats2)
    if not common_cats:
        return _single_point_crossover(p1, p2, max_length)

    chosen = random.choice(list(common_cats))

    seg1 = [g for g, c in zip(g1, cats1) if c == chosen]
    seg2 = [g for g, c in zip(g2, cats2) if c == chosen]
    rest1 = [g for g, c in zip(g1, cats1) if c != chosen]
    rest2 = [g for g, c in zip(g2, cats2) if c != chosen]

    c1 = (rest1 + seg2)[:max_length] or [random.choice(PRIMITIVE_NAMES)]
    c2 = (rest2 + seg1)[:max_length] or [random.choice(PRIMITIVE_NAMES)]
    return Program(c1), Program(c2)


def _single_point_crossover(p1, p2, max_length=8):
    """Standard single-point crossover (fallback when schema/segment fail)."""
    if not p1.genes or not p2.genes:
        return p1.copy(), p2.copy()
    cut1 = random.randint(1, len(p1.genes))
    cut2 = random.randint(1, len(p2.genes))
    g1 = (p1.genes[:cut1] + p2.genes[cut2:])[:max_length]
    g2 = (p2.genes[:cut2] + p1.genes[cut1:])[:max_length]
    return (Program(g1 or [random.choice(PRIMITIVE_NAMES)]),
            Program(g2 or [random.choice(PRIMITIVE_NAMES)]))


def mutate(program, max_length=8, mutation_strength=0.3):
    """
    Multi-operator mutation with hot-gene protection.

    Operators (chosen randomly):
      insert  — insert a random primitive at a random position
      delete  — remove a gene (preferring cold/non-hot positions)
      replace — swap one gene for another (preferring cold positions)
      reorder — swap two adjacent genes
      append  — add a gene at the end
    """
    program = program.copy()
    g = program.genes
    hot = (program.hot_mask if program.hot_mask and len(program.hot_mask) == len(g)
           else [False] * len(g))

    op = random.choice(['insert', 'delete', 'replace', 'reorder', 'append'])

    if op == 'insert' and len(g) < max_length:
        pos = random.randint(0, len(g))
        g.insert(pos, random.choice(PRIMITIVE_NAMES))

    elif op == 'delete' and len(g) > 1:
        cold_pos = [i for i, h in enumerate(hot) if not h]
        pos = random.choice(cold_pos) if cold_pos else random.randint(0, len(g) - 1)
        g.pop(pos)

    elif op == 'replace' and g:
        cold_pos = [i for i, h in enumerate(hot) if not h]
        if cold_pos and random.random() < 0.75:
            pos = random.choice(cold_pos)
        else:
            pos = random.randint(0, len(g) - 1)
        new_gene = random.choice(PRIMITIVE_NAMES)
        for _ in range(5):  # Try to pick a different gene
            if new_gene != g[pos]:
                break
            new_gene = random.choice(PRIMITIVE_NAMES)
        g[pos] = new_gene

    elif op == 'reorder' and len(g) > 1:
        pos = random.randint(0, len(g) - 2)
        g[pos], g[pos + 1] = g[pos + 1], g[pos]

    elif op == 'append' and len(g) < max_length:
        g.append(random.choice(PRIMITIVE_NAMES))

    program.invalidate_cache()
    return program


def local_hill_climb(program, train_pairs, max_steps=3, n_candidates=12):
    """
    Lamarckian local search: iteratively replace each gene with the best
    available alternative. Only applied to programs with fitness > 0.3.

    Parameters:
      max_steps:     maximum improvement rounds
      n_candidates:  number of replacement genes to try per position (random sample)
    """
    best = program.copy()
    best_fit, best_exact = compute_program_fitness(best, train_pairs)
    best.fitness = best_fit
    best.exact_hits = best_exact

    for _ in range(max_steps):
        improved = False
        positions = list(range(len(best.genes)))
        random.shuffle(positions)
        for i in positions:
            for new_gene in random.sample(PRIMITIVE_NAMES, min(n_candidates, len(PRIMITIVE_NAMES))):
                if new_gene == best.genes[i]:
                    continue
                candidate = best.copy()
                candidate.genes[i] = new_gene
                candidate.invalidate_cache()
                fit, exact = compute_program_fitness(candidate, train_pairs)
                if (fit > best_fit or
                        (abs(fit - best_fit) < 1e-6 and exact > best_exact)):
                    best = candidate
                    best.fitness = fit
                    best.exact_hits = exact
                    best_fit = fit
                    best_exact = exact
                    improved = True
                    break
        if not improved:
            break

    return best


def _update_hot_mask(program, train_pairs):
    """
    Approximate hot-mask by ablating each gene and measuring fitness drop.
    A gene is 'hot' if removing it reduces fitness by more than 0.02.
    This is run only on elite programs to keep cost low.
    """
    n = len(program.genes)
    if n <= 1:
        program.hot_mask = [True] * n
        return

    base_fit = program.fitness
    hot = []
    for i in range(n):
        reduced = program.genes[:i] + program.genes[i + 1:]
        if not reduced:
            hot.append(True)
            continue
        p_red = Program(reduced)
        fit, _ = compute_program_fitness(p_red, train_pairs)
        hot.append(base_fit - fit > 0.02)
    program.hot_mask = hot


# =============================================================================
# Section 5 — Island Model
# =============================================================================

class Island:
    """A sub-population with its own evolution hyperparameters."""

    def __init__(self, name, pop_size, crossover_rate, mutation_rate,
                 max_length, gene_filter=None, tournament_size=5):
        self.name = name
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.max_length = max_length
        self.gene_filter = gene_filter   # None = all genes allowed
        self.tournament_size = tournament_size
        self.population = []
        self.best_ever = None
        self.stagnation = 0

    def allowed_genes(self):
        if self.gene_filter:
            valid = [g for g in self.gene_filter if g in PRIMITIVE_REGISTRY]
            return valid if valid else PRIMITIVE_NAMES
        return PRIMITIVE_NAMES

    def random_program(self):
        pool = self.allowed_genes()
        length = random.randint(1, self.max_length)
        return Program([random.choice(pool) for _ in range(length)])


# =============================================================================
# Section 6 — GeneticSolver (Island Model)
# =============================================================================

class GeneticSolver:
    """
    Island-model Genetic Algorithm for ARC task solving.

    4 islands with different strategies:
      Explorer   — aggressive mutation, all genes, small tournament
      Exploiter  — conservative crossover, large tournament
      ColorSpec  — only color/fill primitives
      GeoSpec    — only geometric/structural primitives

    Migration every `migration_interval` generations allows good solutions
    to spread across islands without losing diversity.
    """

    # ── Gene pools per island ──────────────────────────────────────────────
    GEO_GENES = [
        "rotate90", "rotate180", "rotate270",
        "mirror_h", "mirror_v", "transpose",
        "mirror_diag", "mirror_anti_diag",
        "enforce_sym_h", "enforce_sym_v",
        "scale_2x", "scale_3x", "downscale_2x", "downscale_3x",
        "tile_2x2", "tile_1x2", "tile_2x1",
        "tile_3x3", "tile_1x3", "tile_3x1",
        "sort_rows_by_sum", "sort_cols_by_sum", "unique_rows",
        "gravity_down", "gravity_up", "gravity_right", "gravity_left",
        "halves_h_or", "halves_h_and", "halves_h_xor",
        "halves_v_or", "halves_v_and", "halves_v_xor",
        "crop", "crop_bg1", "extract_largest", "extract_smallest",
        "remove_border", "add_border_0", "repeat_border",
        "identity",
    ]

    COLOR_GENES = (
        ["color_invert", "remap_by_frequency", "fill_enclosed",
         "outline_objects", "fill_bg_with_fg", "count_to_color", "identity"]
        + [f"keep_color_{c}" for c in range(10)]
        + [f"fill_color_{c}" for c in range(1, 10)]
        + [f"remove_color_{c}" for c in range(1, 10)]
    )

    def __init__(self,
                 population_size=120,
                 max_generations=50,
                 crossover_rate=0.7,
                 mutation_rate=0.3,
                 elite_fraction=0.1,
                 max_program_length=8,
                 tournament_size=5,
                 migration_interval=5,
                 migration_size=3,
                 local_search_fraction=0.20,
                 verbose=False):

        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_fraction = elite_fraction
        self.max_program_length = max_program_length
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.local_search_fraction = local_search_fraction
        self.verbose = verbose

        island_pop = max(20, population_size // 4)

        self.islands = [
            # Island A: Explorer — high mutation, small tournament, all genes
            Island("Explorer",  island_pop,
                   crossover_rate=0.50, mutation_rate=0.60,
                   max_length=max_program_length, gene_filter=None,
                   tournament_size=3),

            # Island B: Exploiter — high crossover, large tournament, all genes
            Island("Exploiter", island_pop,
                   crossover_rate=0.90, mutation_rate=0.15,
                   max_length=max_program_length, gene_filter=None,
                   tournament_size=7),

            # Island C: ColorSpec — specialises in color transformations
            Island("ColorSpec", island_pop,
                   crossover_rate=0.70, mutation_rate=0.30,
                   max_length=max_program_length, gene_filter=self.COLOR_GENES,
                   tournament_size=5),

            # Island D: GeoSpec — specialises in geometric / structural transforms
            Island("GeoSpec",   island_pop,
                   crossover_rate=0.70, mutation_rate=0.30,
                   max_length=max_program_length, gene_filter=self.GEO_GENES,
                   tournament_size=5),
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self, task, seed_programs=None):
        """
        Evolve programs to solve an ARC task.

        Returns:
            Sorted list of Programs (best fitness first).
        """
        train_pairs = task['train']

        # ── Phase 1: Build initial seeds ──────────────────────────────
        diff_seeds = analyze_task_and_seed(task, n_seeds=50)
        all_seeds = list(diff_seeds)
        if seed_programs:
            all_seeds.extend(
                sp.copy() if isinstance(sp, Program) else Program(list(sp))
                for sp in seed_programs
            )

        # Evaluate seeds
        self._evaluate_list(all_seeds, train_pairs)
        all_seeds.sort(key=lambda p: p.fitness, reverse=True)

        # ── Phase 2: Initialise islands ───────────────────────────────
        for idx, island in enumerate(self.islands):
            island.population = []
            island_seeds = all_seeds[idx::len(self.islands)]
            for sp in island_seeds:
                if len(island.population) < island.pop_size // 3:
                    island.population.append(sp.copy())
            while len(island.population) < island.pop_size:
                island.population.append(island.random_program())
            self._evaluate_list(island.population, train_pairs)
            island.population.sort(key=lambda p: p.fitness, reverse=True)
            island.best_ever = island.population[0].copy() if island.population else None

        # ── Phase 3: Evolution ────────────────────────────────────────
        global_best = self._get_global_best()

        for gen in range(self.max_generations):
            for island in self.islands:
                self._evolve_island(island, train_pairs)
                top = island.population[0] if island.population else None
                if top and (island.best_ever is None or
                            top.fitness > island.best_ever.fitness):
                    island.best_ever = top.copy()
                    island.stagnation = 0
                else:
                    island.stagnation += 1

            new_best = self._get_global_best()
            if new_best and (global_best is None or
                             new_best.fitness > global_best.fitness):
                global_best = new_best.copy()

            if self.verbose:
                island_scores = [
                    f"{isl.name[:4]}:{isl.population[0].fitness:.3f}"
                    for isl in self.islands if isl.population
                ]
                gb_str = f"{global_best.fitness:.4f}[e={global_best.exact_hits}]" if global_best else "N/A"
                print(f"  Gen {gen:3d} | Best={gb_str} | {' '.join(island_scores)}")

            # Early stopping: perfect solution on all training pairs
            if (global_best and global_best.exact_hits == len(train_pairs)
                    and global_best.fitness >= 0.95):
                if self.verbose:
                    print(f"  ✓ Perfect solution found at generation {gen}!")
                break

            # Migration
            if (gen + 1) % self.migration_interval == 0:
                self._migrate(train_pairs)

        # ── Phase 4: Collect & return all unique candidates ───────────
        all_programs = []
        seen = set()
        for island in self.islands:
            for prog in island.population:
                sig = prog.signature()
                if sig not in seen:
                    seen.add(sig)
                    all_programs.append(prog)
            if island.best_ever:
                sig = island.best_ever.signature()
                if sig not in seen:
                    seen.add(sig)
                    all_programs.append(island.best_ever)
        if global_best:
            sig = global_best.signature()
            if sig not in seen:
                all_programs.append(global_best)

        self._evaluate_list(all_programs, train_pairs)
        all_programs.sort(key=lambda p: p.fitness, reverse=True)
        return all_programs

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_global_best(self):
        candidates = []
        for island in self.islands:
            if island.best_ever:
                candidates.append(island.best_ever)
            if island.population:
                candidates.append(island.population[0])
        return max(candidates, key=lambda p: p.fitness) if candidates else None

    def _evolve_island(self, island, train_pairs):
        """Run one generation of evolution on a single island."""
        pop = island.population
        if not pop:
            return

        elite_count = max(1, int(len(pop) * self.elite_fraction))
        pop.sort(key=lambda p: p.fitness, reverse=True)

        # Update hot masks for elite programs
        for prog in pop[:elite_count * 2]:
            if prog.hot_mask is None:
                _update_hot_mask(prog, train_pairs)

        # Adaptive mutation: boost when stagnating
        mut_rate = island.mutation_rate
        if island.stagnation >= 3:
            mut_rate = min(0.85, island.base_mutation_rate * (1.5 + island.stagnation * 0.1))

        # Elitism: carry forward unchanged
        next_gen = [p.copy() for p in pop[:elite_count]]

        # Offspring
        while len(next_gen) < island.pop_size:
            p1 = tournament_select(pop, island.tournament_size)
            p2 = tournament_select(pop, island.tournament_size)

            # Crossover
            if random.random() < island.crossover_rate:
                xover = random.random()
                if xover < 0.45:
                    c1, c2 = schema_crossover(p1, p2, island.max_length)
                elif xover < 0.80:
                    c1, c2 = segment_crossover(p1, p2, island.max_length)
                else:
                    c1, c2 = _single_point_crossover(p1, p2, island.max_length)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            if random.random() < mut_rate:
                c1 = mutate(c1, island.max_length, mut_rate)
            if random.random() < mut_rate:
                c2 = mutate(c2, island.max_length, mut_rate)

            next_gen.append(c1)
            if len(next_gen) < island.pop_size:
                next_gen.append(c2)

        # Evaluate offspring
        self._evaluate_list(next_gen, train_pairs)
        next_gen.sort(key=lambda p: p.fitness, reverse=True)

        # Local hill-climbing on top fraction
        n_climb = max(1, int(len(next_gen) * self.local_search_fraction))
        for i in range(n_climb):
            if next_gen[i].fitness > 0.25:
                next_gen[i] = local_hill_climb(next_gen[i], train_pairs,
                                               max_steps=2, n_candidates=10)

        island.population = next_gen

    def _migrate(self, train_pairs):
        """Ring migration: send top programs from each island to the next."""
        n = len(self.islands)
        emigrants = [
            [p.copy() for p in isl.population[:self.migration_size]]
            for isl in self.islands
        ]
        for i, island in enumerate(self.islands):
            # Receive from the previous island (ring topology)
            incoming = emigrants[(i - 1) % n]
            island.population.extend(incoming)
            island.population.sort(key=lambda p: p.fitness, reverse=True)
            island.population = island.population[:island.pop_size]
        if self.verbose:
            print(f"  [Migration] {self.migration_size} migrants exchanged across {n} islands")

    def _evaluate_list(self, population, train_pairs):
        """Evaluate a list of Programs with signature-level dedup cache."""
        cache = {}
        for program in population:
            sig = program.signature()
            if sig in cache:
                program.fitness, program.exact_hits = cache[sig]
            else:
                fit, exact = compute_program_fitness(program, train_pairs)
                program.fitness = fit
                program.exact_hits = exact
                cache[sig] = (fit, exact)


# =============================================================================
# Section 7 — Utility: Convert Beam Search Candidates → Seed Programs
# =============================================================================

def extract_seed_programs(beam_candidates):
    """
    Convert beam search TransformationRule / ComposedRule objects into
    Program objects that can seed the GA.

    Only keeps programs whose rule names exist in PRIMITIVE_REGISTRY,
    so the GA can mutate them meaningfully.
    """
    programs = []
    seen = set()

    for candidate in beam_candidates:
        genes = []
        if isinstance(candidate, ComposedRule):
            for r in candidate.rules:
                if r.name in PRIMITIVE_REGISTRY:
                    genes.append(r.name)
        elif isinstance(candidate, TransformationRule):
            if candidate.name in PRIMITIVE_REGISTRY:
                genes = [candidate.name]

        if genes:
            sig = tuple(genes)
            if sig not in seen:
                seen.add(sig)
                p = Program(genes)
                p.fitness = getattr(candidate, 'score', 0.0)
                programs.append(p)

    return programs
