"""
WSI Tile-Grid Gymnasium Environment for RL-based tumor detection.

The agent navigates a 2D grid of pre-computed tile embeddings extracted from
a Whole Slide Image (WSI).  Each tile is 224×224 px at 20× magnification.

Designed for use with Stable-Baselines3 >= 2.0 (gymnasium API).
"""

import gymnasium
from gymnasium import spaces
import numpy as np
import h5py
from scipy import ndimage


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STOP = 4  # only used in Multi-WSI / later phases


class WSIEnv(gymnasium.Env):
    """Tile-grid environment backed by an HDF5 tile database.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file produced by the preprocessing pipeline.
    max_steps : int
        Maximum number of steps per episode.
    embedding_suffix : str
        ``'_i'`` for ImageNet or ``'_s'`` for self-supervised embeddings.
    local_radius : int
        Half-size of the local visited map (default 5 → 11×11 = 121).
    enable_stop : bool
        Whether to include the STOP action (phase 4+).
    fixed_starts : list[tuple[int, int]] | None
        If provided, a list of (row, col) positions to cycle / sample from.
        Used in the fixed-start Single-WSI scenario.
    reward_cfg : dict | None
        Override default reward hyper-parameters.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        h5_path: str,
        max_steps: int = 2000,
        embedding_suffix: str = "_i",
        local_radius: int = 5,
        enable_stop: bool = False,
        fixed_starts: list | None = None,
        start_mode: str = "fixed",
        start_dist_range: tuple[int, int] | None = None,
        reward_cfg: dict | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.max_steps = max_steps
        self.embedding_suffix = embedding_suffix
        self.local_radius = local_radius
        self.enable_stop = enable_stop
        self.fixed_starts = fixed_starts
        self.start_mode = start_mode
        self.start_dist_range = start_dist_range
        self.render_mode = render_mode

        # -- reward defaults --------------------------------------------------
        default_reward = dict(
            step_penalty=-0.01,
            background_penalty=-0.5,
            revisit_penalty=-0.2,
            tumor_reward=50.0,
            timeout_penalty=-1.0,
            stop_correct=100.0,
            stop_wrong=-50.0,
        )
        self.reward_cfg = {**default_reward, **(reward_cfg or {})}

        # -- load HDF5 data ---------------------------------------------------
        self._load_h5(h5_path)

        # -- spaces ------------------------------------------------------------
        n_actions = 5 if self.enable_stop else 4
        self.action_space = spaces.Discrete(n_actions)

        local_dim = (2 * local_radius + 1) ** 2
        emb_dim = self.embed_20x.shape[1]  # 512
        obs_dim = emb_dim * 3 + 2 + local_dim + 1  # 1660 for emb_dim=512, r=5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # -- episode state (set in reset) --------------------------------------
        self.current_row = 0
        self.current_col = 0
        self.step_count = 0
        self.visited: set[tuple[int, int]] = set()
        self._start_idx = 0  # cycles through fixed_starts

        # -- pre-compute distance map for curriculum starts --------------------
        self._dist_to_tumor: np.ndarray | None = None
        self._start_pool_cache: dict[tuple[int, int], np.ndarray] = {}
        if self.start_mode in ("curriculum", "distance_band"):
            self._compute_distance_to_tumor()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_h5(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.coords = f["coords"][:]
            self.embed_20x = f[f"embeddings_20x{self.embedding_suffix}"][:]
            self.embed_10x = f[f"embeddings_10x{self.embedding_suffix}"][:]
            self.thumbnail_emb = f[f"thumbnail_embedding{self.embedding_suffix}"][:]
            self.tissue_mask = f["tissue_mask"][:]
            self.tumor_mask = f["tumor_mask"][:]
            self.slide_dims = f.attrs["slide_dimensions"]  # (W, H)

        # Build coord → index mappings
        xs_unique = np.unique(self.coords[:, 0])
        ys_unique = np.unique(self.coords[:, 1])
        self.n_cols = len(xs_unique)
        self.n_rows = len(ys_unique)
        self._x_to_col = {int(x): i for i, x in enumerate(xs_unique)}
        self._y_to_row = {int(y): i for i, y in enumerate(ys_unique)}
        self._col_vals = xs_unique  # for reverse mapping if needed
        self._row_vals = ys_unique

        # flat-index → (row, col) and vice versa
        self._flat_to_rc = np.empty((len(self.coords), 2), dtype=np.int32)
        self._rc_to_flat = np.full((self.n_rows, self.n_cols), -1, dtype=np.int32)
        for idx in range(len(self.coords)):
            c = self._x_to_col[int(self.coords[idx, 0])]
            r = self._y_to_row[int(self.coords[idx, 1])]
            self._flat_to_rc[idx] = (r, c)
            self._rc_to_flat[r, c] = idx

        # 2-D boolean grids
        self.tissue_grid = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        self.tumor_grid = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        for idx in range(len(self.coords)):
            r, c = self._flat_to_rc[idx]
            self.tissue_grid[r, c] = self.tissue_mask[idx]
            self.tumor_grid[r, c] = self.tumor_mask[idx]

        # Connected tumor regions (for analysis / starting-point selection)
        self.tumor_labeled, self.n_tumor_regions = ndimage.label(self.tumor_grid)

    # ------------------------------------------------------------------
    # Distance-to-tumor (grid BFS respecting tissue connectivity)
    # ------------------------------------------------------------------
    def _compute_distance_to_tumor(self):
        """BFS from all tumor tiles on the tissue grid → distance map."""
        from collections import deque

        dist = np.full((self.n_rows, self.n_cols), -1, dtype=np.int32)
        q: deque[tuple[int, int]] = deque()

        # Seed: all tumor tiles
        tumor_rc = np.argwhere(self.tumor_grid)
        for r, c in tumor_rc:
            dist[r, c] = 0
            q.append((r, c))

        # BFS on 4-connected tissue grid
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.n_rows
                    and 0 <= nc < self.n_cols
                    and self.tissue_grid[nr, nc]
                    and dist[nr, nc] == -1
                ):
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        self._dist_to_tumor = dist

    def get_start_pool(self, dist_min: int, dist_max: int) -> np.ndarray:
        """Return (N, 2) array of (row, col) tissue tiles within [dist_min, dist_max]."""
        key = (dist_min, dist_max)
        if key in self._start_pool_cache:
            return self._start_pool_cache[key]

        if self._dist_to_tumor is None:
            self._compute_distance_to_tumor()

        mask = (
            (self._dist_to_tumor >= dist_min)
            & (self._dist_to_tumor <= dist_max)
            & self.tissue_grid
            & ~self.tumor_grid
        )
        pool = np.argwhere(mask)
        self._start_pool_cache[key] = pool
        return pool

    # ------------------------------------------------------------------
    # Helper: get flat index from (row, col)
    # ------------------------------------------------------------------
    def _rc_to_idx(self, row: int, col: int) -> int:
        return int(self._rc_to_flat[row, col])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        idx = self._rc_to_idx(self.current_row, self.current_col)

        # Embeddings
        e20 = self.embed_20x[idx]              # (512,)
        e10 = self.embed_10x[idx]              # (512,)
        ethumb = self.thumbnail_emb             # (512,)

        # Normalised coordinate
        norm_coord = np.array([
            self.current_col / max(self.n_cols - 1, 1),
            self.current_row / max(self.n_rows - 1, 1),
        ], dtype=np.float32)

        # Local visited map
        local_map = self._get_local_visited_map()  # (121,)

        # Time budget
        time_budget = np.array(
            [self.step_count / self.max_steps], dtype=np.float32
        )

        obs = np.concatenate([e20, e10, ethumb, norm_coord, local_map, time_budget])
        return obs.astype(np.float32)

    def _get_local_visited_map(self) -> np.ndarray:
        r = self.local_radius
        size = 2 * r + 1
        local_map = np.zeros(size * size, dtype=np.float32)
        center_r, center_c = self.current_row, self.current_col

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = center_r + dr, center_c + dc
                li = (dr + r) * size + (dc + r)
                if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
                    local_map[li] = -1.0  # out of bounds
                elif not self.tissue_grid[nr, nc]:
                    local_map[li] = -1.0  # non-tissue / unreachable
                elif (nr, nc) in self.visited:
                    local_map[li] = 1.0   # visited
                # else: 0.0 — unvisited tissue
        return local_map

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.start_mode == "fixed" and self.fixed_starts is not None and len(self.fixed_starts) > 0:
            # Cycle through the provided list
            pos = self.fixed_starts[self._start_idx % len(self.fixed_starts)]
            self._start_idx += 1
            self.current_row, self.current_col = pos

        elif self.start_mode == "distance_band" and self.start_dist_range is not None:
            # Random tissue tile within the specified distance band
            d_min, d_max = self.start_dist_range
            pool = self.get_start_pool(d_min, d_max)
            if len(pool) == 0:
                # Fallback to any reachable tissue tile
                pool = np.argwhere(self.tissue_grid & ~self.tumor_grid)
            choice = self.np_random.integers(0, len(pool))
            self.current_row, self.current_col = pool[choice]

        elif self.start_mode == "random_tissue":
            # Any tissue tile (excluding tumor so agent must navigate)
            pool = np.argwhere(self.tissue_grid & ~self.tumor_grid)
            choice = self.np_random.integers(0, len(pool))
            self.current_row, self.current_col = pool[choice]

        else:
            # Default: random tissue tile
            tissue_indices = np.argwhere(self.tissue_grid)
            choice = self.np_random.integers(0, len(tissue_indices))
            self.current_row, self.current_col = tissue_indices[choice]

        self.step_count = 0
        self.visited = {(self.current_row, self.current_col)}

        obs = self._get_obs()
        info = self._make_info(reward=0.0, terminated=False, truncated=False)
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        reward = self.reward_cfg["step_penalty"]
        terminated = False
        truncated = False

        action = int(action)

        # ---- STOP action (only when enabled) ----
        if self.enable_stop and action == STOP:
            is_tumor = self.tumor_grid[self.current_row, self.current_col]
            reward = (
                self.reward_cfg["stop_correct"]
                if is_tumor
                else self.reward_cfg["stop_wrong"]
            )
            terminated = True
            obs = self._get_obs()
            info = self._make_info(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                success=bool(is_tumor),
            )
            return obs, reward, terminated, truncated, info

        # ---- Movement ----
        dr, dc = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}[action]
        new_r = self.current_row + dr
        new_c = self.current_col + dc

        # Boundary / tissue check
        moved = True
        if (
            0 <= new_r < self.n_rows
            and 0 <= new_c < self.n_cols
            and self.tissue_grid[new_r, new_c]
        ):
            self.current_row = new_r
            self.current_col = new_c
        else:
            # Invalid move: stay in place (wall bump)
            moved = False

        # ---- Reward components ----
        pos = (self.current_row, self.current_col)

        # Background penalty (non-tissue target — shouldn't happen with wall-bump, but guard)
        if not self.tissue_grid[self.current_row, self.current_col]:
            reward += self.reward_cfg["background_penalty"]

        # Revisit penalty
        if pos in self.visited and moved:
            reward += self.reward_cfg["revisit_penalty"]

        self.visited.add(pos)

        # ---- External termination: stepped on tumor ----
        if self.tumor_grid[self.current_row, self.current_col]:
            reward += self.reward_cfg["tumor_reward"]
            terminated = True

        # ---- Timeout ----
        if not terminated and self.step_count >= self.max_steps:
            reward += self.reward_cfg["timeout_penalty"]
            truncated = True

        obs = self._get_obs()
        info = self._make_info(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=terminated and self.tumor_grid[self.current_row, self.current_col],
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------
    def _make_info(self, reward, terminated, truncated, success=False):
        return {
            "row": self.current_row,
            "col": self.current_col,
            "step": self.step_count,
            "is_tumor": bool(
                self.tumor_grid[self.current_row, self.current_col]
            ),
            "is_tissue": bool(
                self.tissue_grid[self.current_row, self.current_col]
            ),
            "success": success,
            "n_visited": len(self.visited),
        }

    # ------------------------------------------------------------------
    # Rendering (optional)
    # ------------------------------------------------------------------
    def render(self):
        # Minimal: return a small grid image for logging
        return None


# ======================================================================
# Utility functions
# ======================================================================

def find_starts_near_tumor(env: WSIEnv, distance: int = 4, n_per_region: int = 1,
                           min_region_size: int = 20, seed: int = 42) -> list[tuple[int, int]]:
    """Find tissue tiles that are approximately *distance* steps from tumor.

    Only considers connected tumor regions with at least *min_region_size*
    tiles to avoid noisy single-tile annotations.

    Returns a list of (row, col) tuples.
    """
    rng = np.random.RandomState(seed)
    starts = []

    for region_id in range(1, env.n_tumor_regions + 1):
        region_mask = env.tumor_labeled == region_id
        if region_mask.sum() < min_region_size:
            continue

        # Dilate tumor region to find boundary band at given distance
        structure = ndimage.generate_binary_structure(2, 1)
        dilated = ndimage.binary_dilation(
            region_mask, structure=structure, iterations=distance
        )
        # Band = dilated minus closer dilations
        inner = ndimage.binary_dilation(
            region_mask, structure=structure, iterations=max(distance - 1, 0)
        )
        band = dilated & ~inner & env.tissue_grid & ~env.tumor_grid

        candidates = np.argwhere(band)
        if len(candidates) == 0:
            # Fallback: any tissue tile in the dilated ring
            band = dilated & env.tissue_grid & ~env.tumor_grid
            candidates = np.argwhere(band)
        if len(candidates) == 0:
            continue

        chosen = candidates[rng.choice(len(candidates), size=min(n_per_region, len(candidates)), replace=False)]
        for rc in chosen:
            starts.append((int(rc[0]), int(rc[1])))

    return starts
