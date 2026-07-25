"""Python port of the TU Delft WIRED-OR spatial spike sorting MATLAB reference.

Source: https://gitlab.tudelft.nl/... (Akhoundi et al. 2025 JSSC)
Ported components:
  - WiredOR compression (analog spatial readout simulation)
  - FindNeighborElec (hexagonal staggered 6-neighbor grid)
  - FeatureExtraction (spatial vector + amplitude coefficient)
  - SpikeDetection (VC-SPD pre-detection + NEO + central channel)
  - Classification (modified SOM with cluster migration)

All functions operate on 2D grid data shaped (n_rows, n_cols, n_samples).
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Wired-OR Compression
# ---------------------------------------------------------------------------

def wired_or(data_in: np.ndarray, wire_num: int) -> np.ndarray:
    """Simulate analog Wired-OR spatial compression.

    Groups rows into wires (every wire_num-th row shares a wire).
    For each time sample, a signal survives only if it is the sole
    active signal on its wire (collision-free).

    Args:
        data_in: (n_rows, n_cols, n_samples) integer recording data.
        wire_num: Number of wires (rows are grouped modulo wire_num).

    Returns:
        data_out: Same shape, with collided signals zeroed.
    """
    n_rows, n_cols, n_samples = data_in.shape
    data_out = np.zeros_like(data_in)

    for wire in range(wire_num):
        row_indices = np.arange(wire, n_rows, wire_num)
        group = data_in[row_indices, :, :]  # (n_group, n_cols, n_samples)

        for t in range(n_samples):
            frame = group[:, :, t]  # (n_group, n_cols)
            active = frame != 0

            # A signal is collision-free if it's the only active in its row OR column
            row_active_count = active.sum(axis=1)  # per group-row
            col_active_count = active.sum(axis=0)  # per col

            for gi, row_idx in enumerate(row_indices):
                for col in range(n_cols):
                    if not active[gi, col]:
                        continue
                    # collision-free: only one active in this group-row or this column
                    if row_active_count[gi] == 1 or col_active_count[col] == 1:
                        data_out[row_idx, col, t] = data_in[row_idx, col, t]

    return data_out


def wired_or_fast(data_in: np.ndarray, wire_num: int) -> np.ndarray:
    """Vectorized Wired-OR (faster for large arrays).

    Same semantics as wired_or() but uses vectorized operations per wire group.
    """
    n_rows, n_cols, n_samples = data_in.shape
    data_out = np.zeros_like(data_in)

    for wire in range(wire_num):
        row_indices = np.arange(wire, n_rows, wire_num)
        group = data_in[row_indices, :, :]  # (n_group, n_cols, n_samples)
        active = (group != 0).astype(np.int32)

        # row_active: (n_group, 1, n_samples), col_active: (1, n_cols, n_samples)
        row_active = active.sum(axis=1, keepdims=True)
        col_active = active.sum(axis=0, keepdims=True)

        # collision-free mask: active AND (sole in row OR sole in col)
        collision_free = active * ((row_active == 1) | (col_active == 1))

        data_out[row_indices, :, :] = group * collision_free

    return data_out


# ---------------------------------------------------------------------------
# Hexagonal Neighbor Lookup (staggered grid)
# ---------------------------------------------------------------------------

# Neighbor positions in um for 36um pitch, 18um stagger
_NEIGHBOR_DX = np.array([-60, 60, -30, 30, -30, 30], dtype=np.float64)
_NEIGHBOR_DY = np.array([0, 0, 60, 60, -60, -60], dtype=np.float64)
_NEIGHBOR_ANGLES = np.arctan2(_NEIGHBOR_DY, _NEIGHBOR_DX)
_NEIGHBOR_COS = np.cos(_NEIGHBOR_ANGLES)
_NEIGHBOR_SIN = np.sin(_NEIGHBOR_ANGLES)


def find_neighbor_elec(main_row: int, main_col: int, n_rows: int, n_cols: int):
    """Find 6 hexagonal neighbors for a staggered grid electrode.

    The grid uses row-staggering: odd rows are offset by half a column.
    Neighbor positions (in um): (-60,0), (60,0), (-30,60), (30,60), (-30,-60), (30,-60)

    Args:
        main_row: 0-based row index of the main electrode.
        main_col: 0-based column index.
        n_rows: Total number of rows.
        n_cols: Total number of columns.

    Returns:
        neighbor_rows: (6,) array of neighbor row indices.
        neighbor_cols: (6,) array of neighbor column indices.
        cos_vals: (6,) cosine of direction to each neighbor.
        sin_vals: (6,) sine of direction to each neighbor.
        valid: (6,) boolean mask — False if neighbor is out of bounds.
    """
    neighbor_rows = np.zeros(6, dtype=np.int32)
    neighbor_cols = np.zeros(6, dtype=np.int32)

    # Left and right (same row)
    neighbor_rows[0] = main_row
    neighbor_cols[0] = main_col - 1
    neighbor_rows[1] = main_row
    neighbor_cols[1] = main_col + 1

    # Diagonal neighbors depend on row parity (0-based: even row = MATLAB odd row)
    if main_row % 2 == 0:
        # MATLAB mod(row,2)==1 (1-based odd = 0-based even)
        neighbor_rows[2] = main_row + 1
        neighbor_cols[2] = main_col - 1
        neighbor_rows[3] = main_row + 1
        neighbor_cols[3] = main_col
        neighbor_rows[4] = main_row - 1
        neighbor_cols[4] = main_col - 1
        neighbor_rows[5] = main_row - 1
        neighbor_cols[5] = main_col
    else:
        neighbor_rows[2] = main_row + 1
        neighbor_cols[2] = main_col
        neighbor_rows[3] = main_row + 1
        neighbor_cols[3] = main_col + 1
        neighbor_rows[4] = main_row - 1
        neighbor_cols[4] = main_col
        neighbor_rows[5] = main_row - 1
        neighbor_cols[5] = main_col + 1

    # Check bounds
    valid = (
        (neighbor_rows >= 0) & (neighbor_rows < n_rows) &
        (neighbor_cols >= 0) & (neighbor_cols < n_cols)
    )

    cos_vals = _NEIGHBOR_COS.copy()
    sin_vals = _NEIGHBOR_SIN.copy()

    # Invalid neighbors: zero out direction, point to self
    cos_vals[~valid] = 0.0
    sin_vals[~valid] = 0.0
    neighbor_rows[~valid] = main_row
    neighbor_cols[~valid] = main_col

    return neighbor_rows, neighbor_cols, cos_vals, sin_vals, valid


def build_neighbor_table(n_rows: int, n_cols: int):
    """Pre-compute neighbor lookup for all electrodes.

    Returns:
        rows_table: (n_rows, n_cols, 6) neighbor row indices.
        cols_table: (n_rows, n_cols, 6) neighbor col indices.
        cos_table: (n_rows, n_cols, 6) direction cosines.
        sin_table: (n_rows, n_cols, 6) direction sines.
        valid_table: (n_rows, n_cols, 6) validity mask.
    """
    rows_table = np.zeros((n_rows, n_cols, 6), dtype=np.int32)
    cols_table = np.zeros((n_rows, n_cols, 6), dtype=np.int32)
    cos_table = np.zeros((n_rows, n_cols, 6), dtype=np.float64)
    sin_table = np.zeros((n_rows, n_cols, 6), dtype=np.float64)
    valid_table = np.zeros((n_rows, n_cols, 6), dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            nr, nc, cv, sv, v = find_neighbor_elec(r, c, n_rows, n_cols)
            rows_table[r, c] = nr
            cols_table[r, c] = nc
            cos_table[r, c] = cv
            sin_table[r, c] = sv
            valid_table[r, c] = v

    return rows_table, cols_table, cos_table, sin_table, valid_table


# ---------------------------------------------------------------------------
# Feature Extraction (spatial vector + amplitude coefficient)
# ---------------------------------------------------------------------------

def extract_features_reference(
    wired_or_data: np.ndarray,
    detected_spikes: np.ndarray,
    n_rows: int,
    n_cols: int,
    sin_cos_width: int = 4,
    divider_width: int = 12,
    spatial_resolution: int = 5,
):
    """Extract spatial features using the reference MATLAB algorithm.

    For each detected spike, computes (deltaX, deltaY) displacement features
    using the hexagonal neighbor amplitudes and direction vectors.

    Args:
        wired_or_data: (n_rows, n_cols, n_samples) Wired-OR compressed data.
        detected_spikes: (n_rows, n_cols, n_samples) boolean spike detection map.
        n_rows: Grid rows.
        n_cols: Grid columns.
        sin_cos_width: Quantization bits for cos/sin (default 4).
        divider_width: Fixed-point division precision (default 12).
        spatial_resolution: Output resolution bits (default 5, range [-31,31]).

    Returns:
        features_x: list of arrays, per (row,col) deltaX values.
        features_y: list of arrays, per (row,col) deltaY values.
        features_time: list of arrays, per (row,col) spike sample indices.
    """
    n_samples = wired_or_data.shape[2]
    clamp_max = 2 ** spatial_resolution - 1  # 31

    # Quantized cos/sin
    cos_q = np.round(_NEIGHBOR_COS * (2 ** sin_cos_width)).astype(np.int32)
    sin_q = np.round(_NEIGHBOR_SIN * (2 ** sin_cos_width)).astype(np.int32)

    features_x = [[None] * n_cols for _ in range(n_rows)]
    features_y = [[None] * n_cols for _ in range(n_rows)]
    features_time = [[None] * n_cols for _ in range(n_rows)]

    for row in range(n_rows):
        for col in range(n_cols):
            spike_times = np.where(detected_spikes[row, col, :])[0]
            if len(spike_times) == 0:
                features_x[row][col] = np.array([], dtype=np.int32)
                features_y[row][col] = np.array([], dtype=np.int32)
                features_time[row][col] = np.array([], dtype=np.int64)
                continue

            # Filter: need at least 1 sample before and 2 after
            valid_mask = (spike_times >= 1) & (spike_times <= n_samples - 3)
            spike_times = spike_times[valid_mask]
            if len(spike_times) == 0:
                features_x[row][col] = np.array([], dtype=np.int32)
                features_y[row][col] = np.array([], dtype=np.int32)
                features_time[row][col] = np.array([], dtype=np.int64)
                continue

            # Main channel: min over 4 samples [t-1, t, t+1, t+2]
            main_window = np.stack([
                wired_or_data[row, col, spike_times - 1],
                wired_or_data[row, col, spike_times],
                wired_or_data[row, col, spike_times + 1],
                wired_or_data[row, col, spike_times + 2],
            ], axis=0)  # (4, n_spikes)
            main_amp = np.abs(main_window.min(axis=0))  # (n_spikes,)

            # Neighbor amplitudes
            nr, nc, cv, sv, valid = find_neighbor_elec(row, col, n_rows, n_cols)
            n_neighbors = 6

            neighbor_amps = np.zeros((n_neighbors, len(spike_times)))
            for ni in range(n_neighbors):
                if not valid[ni]:
                    continue
                nb_window = np.stack([
                    wired_or_data[nr[ni], nc[ni], spike_times - 1],
                    wired_or_data[nr[ni], nc[ni], spike_times],
                    wired_or_data[nr[ni], nc[ni], spike_times + 1],
                    wired_or_data[nr[ni], nc[ni], spike_times + 2],
                ], axis=0)
                neighbor_amps[ni] = np.abs(nb_window.min(axis=0))

            # Max neighbor amplitude
            max_neighbor_amp = neighbor_amps.max(axis=0)  # (n_spikes,)

            # Spatial vector: sum of (cos_q + j*sin_q) * neighbor_amp
            # Using quantized cos/sin
            spatial_re = np.zeros(len(spike_times), dtype=np.float64)
            spatial_im = np.zeros(len(spike_times), dtype=np.float64)
            for ni in range(n_neighbors):
                spatial_re += cos_q[ni] * neighbor_amps[ni]
                spatial_im += sin_q[ni] * neighbor_amps[ni]
            # Divide by 2^sin_cos_width to dequantize
            spatial_re /= (2 ** sin_cos_width)
            spatial_im /= (2 ** sin_cos_width)

            # |spatial_vector| approximation: max(|re|,|im|) + min(|re|,|im|)/2
            abs_re = np.abs(spatial_re)
            abs_im = np.abs(spatial_im)
            spatial_abs = np.maximum(abs_re, abs_im) + np.minimum(abs_re, abs_im) / 2.0

            # Amplitude coefficient
            # AmpCoef = max_neighbor_amp * 2^divider_width / (main_amp * spatial_abs)
            with np.errstate(divide='ignore', invalid='ignore'):
                amp_coef = np.where(
                    (main_amp > 0) & (spatial_abs > 0),
                    max_neighbor_amp * (2 ** divider_width) / (main_amp * spatial_abs),
                    0.0
                )
            amp_coef[max_neighbor_amp == 0] = 0.0

            # Final features: delta = spatial_component * AmpCoef / 2^(divider_width - spatial_resolution)
            scale = 2 ** (divider_width - spatial_resolution)
            delta_x = np.round(spatial_re * amp_coef / scale).astype(np.int32)
            delta_y = np.round(spatial_im * amp_coef / scale).astype(np.int32)

            # Remove spikes where main_amp == 0
            keep = main_amp > 0
            delta_x = delta_x[keep]
            delta_y = delta_y[keep]
            spike_times_kept = spike_times[keep]

            # Clamp to [-31, 31]
            delta_x = np.clip(delta_x, -clamp_max, clamp_max)
            delta_y = np.clip(delta_y, -clamp_max, clamp_max)

            features_x[row][col] = delta_x
            features_y[row][col] = delta_y
            features_time[row][col] = spike_times_kept

    return features_x, features_y, features_time


# ---------------------------------------------------------------------------
# Spike Detection (VC-SPD + NEO + central channel)
# ---------------------------------------------------------------------------

def detect_spikes_reference(
    wired_or_data: np.ndarray,
    n_rows: int,
    n_cols: int,
    fs: int = 20000,
    non_collision_threshold: int = 7,
    pre_detection_threshold: int = 3,
    amp_threshold: float = 0.0,
    neo_threshold: float = 40.0,
    central_channel_check: bool = True,
    duration_samples: int = None,
):
    """Detect spikes using the reference VC-SPD + NEO algorithm.

    Operates on Wired-OR compressed data. Detection pipeline:
    1. VC-SPD: count consecutive non-zero samples per electrode
    2. Pre-detection: count >= threshold AND signal != 0
    3. Local minimum finder (negative peak)
    4. Amplitude threshold
    5. NEO threshold
    6. Central channel check (main channel must be largest)

    Args:
        wired_or_data: (n_rows, n_cols, n_samples) Wired-OR data.
        n_rows, n_cols: Grid dimensions.
        fs: Sampling frequency (for batch sizing).
        non_collision_threshold: Max counter value for VC-SPD.
        pre_detection_threshold: Consecutive non-zero count to trigger pre-detection.
        amp_threshold: Minimum amplitude (absolute value).
        neo_threshold: NEO energy threshold.
        central_channel_check: Whether to verify central channel dominance.
        duration_samples: Process only this many samples (None = all).

    Returns:
        detected: (n_rows, n_cols, n_samples) boolean detection map.
    """
    n_samples = wired_or_data.shape[2]
    if duration_samples is not None:
        n_samples = min(n_samples, duration_samples)

    detected = np.zeros((n_rows, n_cols, n_samples), dtype=bool)
    batch_extend = 10

    # Process in batches of 1 second
    batch_size = fs
    n_batches = int(np.ceil(n_samples / batch_size))

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_samples)
        actual_len = end - start

        # We need batch_extend samples before + actual_len + 2 after
        total_needed = batch_extend + actual_len + 2
        data_batch = np.zeros((n_rows, n_cols, total_needed), dtype=wired_or_data.dtype)

        # Fill from available data
        src_start = max(0, start - batch_extend)
        src_end = min(wired_or_data.shape[2], end + 2)
        dst_offset = src_start - (start - batch_extend)
        src_len = src_end - src_start
        data_batch[:, :, dst_offset:dst_offset + src_len] = wired_or_data[:, :, src_start:src_end]

        # VC-SPD: count consecutive non-zero samples
        non_collision_counter = np.zeros((n_rows, n_cols), dtype=np.int32)
        pre_detected_batch = np.zeros((n_rows, n_cols, actual_len), dtype=bool)

        for t in range(actual_len):
            frame = data_batch[:, :, t + batch_extend]
            nonzero = frame != 0

            # Increment counter where signal is non-zero and counter < threshold
            inc_mask = (non_collision_counter < non_collision_threshold) & nonzero
            non_collision_counter[inc_mask] += 1

            # Decrement counter where signal is zero and counter > 0
            dec_mask = (non_collision_counter > 0) & ~nonzero
            non_collision_counter[dec_mask] -= 1

            # Pre-detection
            pre_detected_batch[:, :, t] = (
                (non_collision_counter >= pre_detection_threshold) & nonzero
            )

        # Local minimum finder (negative peak detection)
        # Need t-2 relative to batch, which is at index batch_extend + t - 2 in data_batch
        if actual_len < 4:
            continue

        for t in range(actual_len):
            idx = batch_extend + t
            if idx < 2 or idx >= data_batch.shape[2] - 1:
                continue
            center = data_batch[:, :, idx - 2]
            # Local min: center < left, center <= right, center <= next
            local_min = (
                (center < data_batch[:, :, idx - 3]) &
                (center <= data_batch[:, :, idx - 1]) &
                (center <= data_batch[:, :, idx])
            )
            pre_detected_batch[:, :, t] &= local_min

        # Amplitude threshold
        for t in range(actual_len):
            idx = batch_extend + t
            if idx < 2:
                continue
            amp = np.abs(data_batch[:, :, idx - 2].astype(np.float64))
            pre_detected_batch[:, :, t] &= (amp > amp_threshold)

        # NEO threshold: |x[n]^2 - x[n-1]*x[n+1]| > threshold
        for t in range(actual_len):
            idx = batch_extend + t
            if idx < 3 or idx >= data_batch.shape[2] - 1:
                pre_detected_batch[:, :, t] = False
                continue
            x_n = data_batch[:, :, idx - 2].astype(np.float64)
            x_n1 = data_batch[:, :, idx - 3].astype(np.float64)
            x_n_1 = data_batch[:, :, idx - 1].astype(np.float64)
            neo_energy = np.abs(x_n * x_n - x_n1 * x_n_1)
            pre_detected_batch[:, :, t] &= (neo_energy > neo_threshold)

        detected[:, :, start:end] = pre_detected_batch

    # Central channel check
    if central_channel_check:
        _central_channel_filter(wired_or_data, detected, n_rows, n_cols, n_samples)

    return detected


def _central_channel_filter(
    wired_or_data: np.ndarray,
    detected: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_samples: int,
):
    """Remove detections where a neighbor has larger amplitude than main channel."""
    for row in range(n_rows):
        for col in range(n_cols):
            spike_times = np.where(detected[row, col, :])[0]
            if len(spike_times) == 0:
                continue

            nr, nc, _, _, valid = find_neighbor_elec(row, col, n_rows, n_cols)

            for t in spike_times:
                if t < 3 or t >= n_samples:
                    detected[row, col, t] = False
                    continue

                # Main channel min over [t-3, t-2, t-1, t]
                main_min = np.abs(wired_or_data[row, col, t - 3:t + 1].min())

                # Neighbor min over same window
                neighbor_max = 0.0
                for ni in range(6):
                    if not valid[ni]:
                        continue
                    nb_min = np.abs(wired_or_data[nr[ni], nc[ni], t - 3:t + 1].min())
                    neighbor_max = max(neighbor_max, nb_min)

                if neighbor_max > main_min:
                    detected[row, col, t] = False


# ---------------------------------------------------------------------------
# Classification (Modified SOM with cluster migration)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceSOM:
    """Modified SOM classifier faithful to the MATLAB reference.

    Grid is 2x oversampled: (2*n_rows) x (2*n_cols).
    Position encoding uses 5-bit sub-cell resolution (32 levels per cell).
    """
    n_rows: int
    n_cols: int
    remove_scale: int = 8  # beta: pruning aggressiveness
    sub_cell_bits: int = 5  # 2^5 = 32 sub-cell positions

    # Internal state
    grid_rows: int = field(init=False)
    grid_cols: int = field(init=False)
    delta_x_mem: np.ndarray = field(init=False, repr=False)
    delta_y_mem: np.ndarray = field(init=False, repr=False)
    valid_mem: np.ndarray = field(init=False, repr=False)
    counter_mem: np.ndarray = field(init=False, repr=False)
    clusters: np.ndarray = field(init=False, repr=False)
    converged: bool = field(init=False, default=False)

    def __post_init__(self):
        self.grid_rows = self.n_rows * 2
        self.grid_cols = self.n_cols * 2
        self.reset()

    def reset(self):
        """Reset all grid state (called on first batch)."""
        self.delta_x_mem = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        self.delta_y_mem = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        self.valid_mem = np.ones((self.grid_rows, self.grid_cols), dtype=np.int32)
        self.counter_mem = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        self.clusters = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        self.converged = False

    def classify_batch(
        self,
        delta_x: np.ndarray,
        delta_y: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Process one batch of spikes through the SOM.

        Args:
            delta_x: (N,) deltaX features (integer, [-31, 31]).
            delta_y: (N,) deltaY features.
            rows: (N,) electrode row indices (0-based).
            cols: (N,) electrode column indices (0-based).
            training: If True, apply pruning after batch.

        Returns:
            cluster_index: (N,) assigned cluster labels (-1 = unassigned).
        """
        n_spikes = len(delta_x)
        cluster_index = np.full(n_spikes, -1, dtype=np.int32)
        sub = 2 ** self.sub_cell_bits  # 32

        valid_num_before = int(self.valid_mem.sum())

        if training:
            self.counter_mem[:, :] = 0

        # Compute absolute positions
        # locationX = (2*col + mod(row,2)) * 32 + deltaX
        # locationY = (2*row) * 32 + deltaY
        loc_x = (2 * cols + (rows % 2)) * sub + delta_x.astype(np.float64)
        loc_y = (2 * rows) * sub + delta_y.astype(np.float64)

        # Initial grid position for BMU search
        row_start = 2 * rows - 2
        col_start = 2 * cols + (rows % 2) - 2

        for i in range(n_spikes):
            found = False
            min_dist = np.inf
            best_r, best_c = -1, -1
            best_loc_x, best_loc_y = 0.0, 0.0

            # Search 5x5 neighborhood
            for dr in range(5):
                for dc in range(5):
                    r_addr = row_start[i] + dr
                    c_addr = col_start[i] + dc

                    if r_addr < 0 or r_addr >= self.grid_rows:
                        continue
                    if c_addr < 0 or c_addr >= self.grid_cols:
                        continue
                    if self.valid_mem[r_addr, c_addr] == 0:
                        continue

                    # Read stored position
                    read_x = c_addr * sub + self.delta_x_mem[r_addr, c_addr]
                    read_y = r_addr * sub + self.delta_y_mem[r_addr, c_addr]

                    dist = np.sqrt((loc_x[i] - read_x) ** 2 + (loc_y[i] - read_y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_r, best_c = r_addr, c_addr
                        best_loc_x, best_loc_y = read_x, read_y
                        found = True

            if not found:
                continue

            cluster_index[i] = self.clusters[best_r, best_c]

            # BMU update: new = (15*old + new) / 16  (alpha = 16)
            new_loc_x = np.round((best_loc_x * 15 + loc_x[i]) / 16)
            new_loc_y = np.round((best_loc_y * 15 + loc_y[i]) / 16)

            # Determine new grid cell
            new_row = int(np.round(new_loc_y / sub))
            new_col = int(np.round(new_loc_x / sub))

            # Clamp to grid bounds
            new_row = max(0, min(new_row, self.grid_rows - 1))
            new_col = max(0, min(new_col, self.grid_cols - 1))

            new_delta_x = new_loc_x - new_col * sub
            new_delta_y = new_loc_y - new_row * sub

            if best_r == new_row and best_c == new_col:
                # Stay in same cell: update centroid
                self.delta_x_mem[best_r, best_c] = new_delta_x
                self.delta_y_mem[best_r, best_c] = new_delta_y
                self.counter_mem[best_r, best_c] += 1
            else:
                # Migration: kill old cell, activate new cell
                self.valid_mem[best_r, best_c] = 0
                self.delta_x_mem[best_r, best_c] = 0
                self.delta_y_mem[best_r, best_c] = 0

                self.valid_mem[new_row, new_col] = 1
                self.clusters[new_row, new_col] = self.clusters[best_r, best_c]
                self.clusters[best_r, best_c] = 0

                self.delta_x_mem[new_row, new_col] = new_delta_x
                self.delta_y_mem[new_row, new_col] = new_delta_y

                self.counter_mem[new_row, new_col] += self.counter_mem[best_r, best_c] + 1
                self.counter_mem[best_r, best_c] = 0

        # Pruning
        if training:
            valid_num = int(self.valid_mem.sum())
            if valid_num > 0:
                counter_avg = int(np.ceil(n_spikes / valid_num / self.remove_scale))
                self.valid_mem[self.counter_mem < counter_avg] = 0

            valid_num_after = int(self.valid_mem.sum())
            if valid_num_after == valid_num_before:
                self.converged = True
                self._post_convergence_merge()

        return cluster_index

    def _post_convergence_merge(self):
        """2x2 neighborhood max-count merge after convergence."""
        for r in range(self.grid_rows - 1):
            for c in range(self.grid_cols - 1):
                block = self.counter_mem[r:r + 2, c:c + 2]
                max_count = int(np.floor(block.max() / self.remove_scale))
                mask = block > max_count
                self.valid_mem[r:r + 2, c:c + 2] *= mask.astype(np.int32)
                # Assign cluster IDs
                self.clusters[r, c] = r * 2 * self.grid_cols + c

        self.clusters *= self.valid_mem

    def train(self, features_x, features_y, features_time, n_rows, n_cols,
              fs: int = 20000, batch_duration_s: float = 6.0):
        """Run the full training loop until convergence.

        Args:
            features_x: (n_rows, n_cols) list of arrays — deltaX per electrode.
            features_y: (n_rows, n_cols) list of arrays — deltaY per electrode.
            features_time: (n_rows, n_cols) list of arrays — spike times per electrode.
            n_rows, n_cols: Grid dimensions.
            fs: Sampling frequency.
            batch_duration_s: Batch size in seconds.

        Returns:
            valid_num_history: List of active cluster counts per batch.
        """
        # Flatten and sort by time
        all_dx, all_dy, all_rows, all_cols, all_times = [], [], [], [], []
        for r in range(n_rows):
            for c in range(n_cols):
                fx = features_x[r][c]
                fy = features_y[r][c]
                ft = features_time[r][c]
                if len(ft) > 0:
                    all_dx.append(fx)
                    all_dy.append(fy)
                    all_rows.append(np.full(len(ft), r, dtype=np.int32))
                    all_cols.append(np.full(len(ft), c, dtype=np.int32))
                    all_times.append(ft)

        if not all_times:
            return []

        all_dx = np.concatenate(all_dx).astype(np.float64)
        all_dy = np.concatenate(all_dy).astype(np.float64)
        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_times = np.concatenate(all_times)

        # Sort by time
        sort_idx = np.argsort(all_times)
        all_dx = all_dx[sort_idx]
        all_dy = all_dy[sort_idx]
        all_rows = all_rows[sort_idx]
        all_cols = all_cols[sort_idx]
        all_times = all_times[sort_idx]

        batch_size = int(batch_duration_s * fs)
        max_time = int(all_times[-1])
        sample_num = (max_time // fs) * fs

        self.reset()
        valid_num_history = [int(self.valid_mem.sum())]
        reset_done = False

        while not self.converged:
            batch_start = int((len(valid_num_history) - 1) % max(1, sample_num // batch_size)) * batch_size
            batch_end = batch_start + batch_size

            mask = (all_times >= batch_start) & (all_times < batch_end)
            if mask.sum() == 0:
                # Wrap around
                if len(valid_num_history) > sample_num // batch_size + 2:
                    break
                continue

            self.classify_batch(
                all_dx[mask], all_dy[mask],
                all_rows[mask], all_cols[mask],
                training=True,
            )
            valid_num_history.append(int(self.valid_mem.sum()))

            if len(valid_num_history) > 100:
                break

        return valid_num_history

    def assign_all(self, features_x, features_y, features_time, n_rows, n_cols):
        """Assign cluster labels to all spikes (inference mode).

        Returns:
            cluster_indices: (N,) cluster label per spike.
            times: (N,) spike time per assignment.
            rows: (N,) electrode row.
            cols: (N,) electrode col.
        """
        all_dx, all_dy, all_rows, all_cols, all_times = [], [], [], [], []
        for r in range(n_rows):
            for c in range(n_cols):
                fx = features_x[r][c]
                fy = features_y[r][c]
                ft = features_time[r][c]
                if len(ft) > 0:
                    all_dx.append(fx)
                    all_dy.append(fy)
                    all_rows.append(np.full(len(ft), r, dtype=np.int32))
                    all_cols.append(np.full(len(ft), c, dtype=np.int32))
                    all_times.append(ft)

        if not all_times:
            return np.array([]), np.array([]), np.array([]), np.array([])

        all_dx = np.concatenate(all_dx).astype(np.float64)
        all_dy = np.concatenate(all_dy).astype(np.float64)
        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_times = np.concatenate(all_times)

        sort_idx = np.argsort(all_times)
        all_dx = all_dx[sort_idx]
        all_dy = all_dy[sort_idx]
        all_rows = all_rows[sort_idx]
        all_cols = all_cols[sort_idx]
        all_times = all_times[sort_idx]

        cluster_indices = self.classify_batch(
            all_dx, all_dy, all_rows, all_cols, training=False
        )

        return cluster_indices, all_times, all_rows, all_cols


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_reference_pipeline(
    raw_data: np.ndarray,
    n_rows: int,
    n_cols: int,
    fs: int = 20000,
    wire_num: int = 4,
    lsb_step: int = 4,
    remove_scale: int = 8,
    amp_threshold: float = 0.0,
    neo_threshold: float = 40.0,
    batch_duration_s: float = 6.0,
    verbose: bool = True,
):
    """Run the complete reference pipeline: WiredOR -> Detect -> Features -> Classify.

    Args:
        raw_data: (n_rows * n_cols, n_samples) or (n_rows, n_cols, n_samples) raw recording.
        n_rows, n_cols: Grid dimensions.
        fs: Sampling frequency.
        wire_num: Number of wires for Wired-OR.
        lsb_step: LSB quantization step.
        remove_scale: SOM pruning parameter (beta).
        amp_threshold: Detection amplitude threshold.
        neo_threshold: Detection NEO threshold.
        batch_duration_s: SOM training batch duration.
        verbose: Print progress.

    Returns:
        dict with keys: cluster_indices, spike_times, spike_rows, spike_cols,
                        detected_map, wired_or_data, valid_num_history.
    """
    # Reshape if needed
    if raw_data.ndim == 2:
        n_samples = raw_data.shape[1]
        raw_3d = raw_data.reshape(n_rows, n_cols, n_samples)
    else:
        raw_3d = raw_data
        n_samples = raw_3d.shape[2]

    # Quantize
    if verbose:
        print(f"[1/4] Quantizing (lsb_step={lsb_step})...")
    data_q = np.round(raw_3d / lsb_step).astype(np.int16)

    # Wired-OR
    if verbose:
        print(f"[2/4] Wired-OR compression (wire_num={wire_num})...")
    wor_data = wired_or_fast(data_q.astype(np.float64), wire_num)

    # Detection
    if verbose:
        print(f"[3/4] Spike detection (amp_thr={amp_threshold}, neo_thr={neo_threshold})...")
    detected = detect_spikes_reference(
        wor_data, n_rows, n_cols, fs=fs,
        amp_threshold=amp_threshold,
        neo_threshold=neo_threshold,
    )
    n_detected = int(detected.sum())
    if verbose:
        print(f"       Detected {n_detected} spikes")

    # Feature extraction
    if verbose:
        print(f"[4/4] Feature extraction + SOM classification (beta={remove_scale})...")
    feat_x, feat_y, feat_t = extract_features_reference(
        wor_data, detected, n_rows, n_cols
    )

    # SOM training
    som = ReferenceSOM(n_rows, n_cols, remove_scale=remove_scale)
    history = som.train(feat_x, feat_y, feat_t, n_rows, n_cols,
                        fs=fs, batch_duration_s=batch_duration_s)
    if verbose:
        print(f"       SOM converged: {som.converged}, "
              f"active clusters: {int(som.valid_mem.sum())}, "
              f"batches: {len(history)}")

    # Assignment
    cluster_idx, times, rows, cols = som.assign_all(
        feat_x, feat_y, feat_t, n_rows, n_cols
    )

    return {
        "cluster_indices": cluster_idx,
        "spike_times": times,
        "spike_rows": rows,
        "spike_cols": cols,
        "detected_map": detected,
        "wired_or_data": wor_data,
        "valid_num_history": history,
        "som": som,
    }
