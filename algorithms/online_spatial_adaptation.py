"""Integer shift-based online adaptation of spatial WTA centroids and theta.

Streaming updates use a trunc-toward-zero shift-add EMA with a power-of-two
learning rate:

    c_u <- c_u + trunc((q - c_u) / 2^{lr_shift})
    theta_u <- theta_u + trunc((d_u - theta_u) / 2^{lr_shift})

No divider hardware is required: magnitude is right-shifted and the sign is
restored, which is unbiased between positive and negative errors (unlike a
signed arithmetic ``>>``, which ratchets centroids downward for small residuals).

No float, no test labels.  Predictions drive the update, and each event is
predicted against the *current* state before that event's write is applied, so
later events see earlier adaptations.  Every centroid / theta write is counted
for the hardware update-traffic ledger.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .spatial_wta import SpatialWTAClassifier


class OnlineSpatialAdapter:
    """Wrap a fitted ``SpatialWTAClassifier`` with streaming integer EMA updates.

    Parameters
    ----------
    classifier:
        Already fitted (and optionally calibrated) WTA classifier.  Integer
        ``n_bits`` is required; float classifiers are rejected.
    lr_shift:
        Learning-rate as a right-shift amount.  ``lr_shift=4`` means EMA
        coefficient ``1/16``.  Must be in ``[1, 8]``.
    update_theta:
        When true, also adapt ``theta_u`` from the predicted unit's distance.
    clip_codes:
        Keep centroids inside the quantizer code range after every update.
    """

    def __init__(
        self,
        classifier: SpatialWTAClassifier,
        *,
        lr_shift: int = 4,
        update_theta: bool = True,
        clip_codes: bool = True,
    ) -> None:
        if not getattr(classifier, "is_fitted_", False):
            raise RuntimeError("classifier must be fitted before online adaptation")
        if classifier.n_bits is None:
            raise ValueError("OnlineSpatialAdapter requires an integer quantizer")
        if isinstance(lr_shift, bool) or int(lr_shift) != lr_shift or not 1 <= int(lr_shift) <= 8:
            raise ValueError("lr_shift must be an integer in [1, 8]")
        self.classifier = classifier
        self.lr_shift = int(lr_shift)
        self.update_theta = bool(update_theta)
        self.clip_codes = bool(clip_codes)
        self.centroid_writes_ = 0
        self.theta_writes_ = 0
        self.n_updates_ = 0
        self.n_predictions_ = 0
        self.code_max = classifier.quantizer.code_max  # type: ignore[union-attr]

    @staticmethod
    def trunc_shift(error: np.ndarray | int, lr_shift: int) -> np.ndarray:
        """Signed trunc-toward-zero right shift: ``sign(e) * (|e| >> lr_shift)``.

        Equivalent to ``trunc(e / 2**lr_shift)`` without floating point.  Unlike
        arithmetic ``e >> lr_shift``, small positive and negative residuals both
        map to zero, so the update is not a one-way ratchet.
        """
        e = np.asarray(error, dtype=np.int64)
        mag = np.abs(e) >> int(lr_shift)
        return np.where(e >= 0, mag, -mag).astype(np.int64, copy=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict without updating state."""
        pred = self.classifier.predict(features)
        self.n_predictions_ += int(np.asarray(pred).size)
        return pred

    def predict_and_update(self, features: np.ndarray) -> np.ndarray:
        """Predict each row against current state, then adapt the winner.

        Events are processed strictly in order so that event ``i+1`` observes
        any centroid / theta writes from event ``i``.
        """
        values = np.asarray(features)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        n = int(values.shape[0])
        centroids = self.classifier.centroids_
        theta = self.classifier.theta_
        units = self.classifier.units_
        predictions = np.empty(n, dtype=units.dtype)

        for i in range(n):
            code = self.classifier.transform(values[i : i + 1])[0]
            distances = self.classifier._row_distances(code[np.newaxis, :])[0]
            scores = distances - theta
            row = int(np.argmin(scores))
            predictions[i] = units[row]
            self.n_predictions_ += 1

            # Integer EMA with trunc-toward-zero: c <- c + trunc((q-c)/2^s)
            delta = self.trunc_shift(
                code.astype(np.int64) - centroids[row].astype(np.int64),
                self.lr_shift,
            )
            if np.any(delta != 0):
                centroids[row] = centroids[row] + delta
                if self.clip_codes:
                    np.clip(centroids[row], 0, self.code_max, out=centroids[row])
                self.centroid_writes_ += 1
            if self.update_theta:
                d = int(distances[row])
                t_delta = int(self.trunc_shift(d - int(theta[row]), self.lr_shift))
                if t_delta != 0:
                    theta[row] = int(theta[row]) + t_delta
                    self.theta_writes_ += 1
            self.n_updates_ += 1
        return predictions

    def update_accounting(self) -> dict[str, int | float]:
        """Emit update-write traffic for the hardware ledger."""
        n_dims = int(self.classifier.n_features_in_)
        n_bits = int(self.classifier.n_bits)  # type: ignore[arg-type]
        centroid_bits_per_write = n_dims * n_bits
        theta_bits = int(
            self.classifier.memory_accounting()["theta_bits_per_row"]
        )
        return {
            "n_predictions": int(self.n_predictions_),
            "n_updates": int(self.n_updates_),
            "centroid_writes": int(self.centroid_writes_),
            "theta_writes": int(self.theta_writes_),
            "centroid_bits_written": int(self.centroid_writes_ * centroid_bits_per_write),
            "theta_bits_written": int(self.theta_writes_ * theta_bits),
            "total_update_bits_written": int(
                self.centroid_writes_ * centroid_bits_per_write
                + self.theta_writes_ * theta_bits
            ),
            "lr_shift": int(self.lr_shift),
            "ema_coefficient_reciprocal": int(1 << self.lr_shift),
            "shift_mode": "trunc_toward_zero",
        }

    def snapshot_state(self) -> dict[str, Any]:
        """Copy current centroid / theta for audit."""
        return {
            "centroids": self.classifier.centroids_.copy(),
            "theta": self.classifier.theta_.copy(),
            "units": self.classifier.units_.copy(),
        }
