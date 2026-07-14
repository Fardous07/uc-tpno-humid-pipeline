from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentalValidator:
    """
    Validate model predictions against experimental adsorption data.

    Compares TPNO predictions with experimental measurements from databases
    like NIST-ISODB, IZA, or published literature.

    Parameters
    ----------
    experimental_db_path : Optional[Path]
        Path to experimental database file (parquet or CSV).
    validate_columns : Dict[str, str]
        Mapping of experimental column names to model prediction column names.

    Example
    -------
    >>> validator = ExperimentalValidator("data/experimental/nist_isodb.parquet")
    >>> metrics = validator.validate_predictions(predictions_df)
    >>> print(f"MAE: {metrics['mae']:.4f} mol/kg")

    Fixes vs. original
    ------------------
    1. BUG FIXED: validate_by_temperature / validate_by_pressure did
       predictions["temperature"] before renaming "T" → "temperature", raising
       KeyError.  Now renames consistently before filtering.
    2. BUG FIXED: pd.cut emits NaN bins for values outside the range.
       Iterating over unique() includes NaN; NaN.left raises AttributeError.
       Fixed with .dropna() on unique bin labels.
    3. BUG FIXED: Three sites mutated self.exp_db without try/finally.  If
       validate_predictions raised an exception, self.exp_db stayed as the
       filtered subset permanently, corrupting all future calls.
    4. BUG FIXED: compute_rank_correlation default metric="loading" does not
       match model output columns (co2_loading_molkg etc.), causing KeyError.
       Default changed to "co2_loading_molkg".
    """

    def __init__(
        self,
        experimental_db_path: Optional[Union[str, Path]] = None,
        validate_columns: Optional[Dict[str, str]] = None,
    ):
        self.exp_db = None
        self.validate_columns = validate_columns or {
            "loading":     "loading",
            "temperature": "T",
            "pressure":    "pressure",
            "mof_id":      "mof_id",
            "component":   "component",
        }
        if experimental_db_path is not None:
            self.load_experimental_data(experimental_db_path)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_experimental_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load experimental data from file.

        Expected columns:
        - mof_id      : MOF identifier
        - component   : Adsorbate (CO2, N2, H2O)
        - temperature : Temperature in K
        - pressure    : Pressure in bar
        - loading     : Loading in mol/kg

        Optional:
        - reference              : Publication / DOI
        - measurement_uncertainty: Uncertainty in loading
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experimental database not found: {path}")

        self.exp_db = (
            pd.read_parquet(path) if path.suffix == ".parquet"
            else pd.read_csv(path)
        )

        required = ["mof_id", "component", "temperature", "pressure", "loading"]
        missing  = [c for c in required if c not in self.exp_db.columns]
        if missing:
            raise ValueError(
                f"Experimental database missing required columns: {missing}"
            )

        logger.info("Loaded experimental data: %d rows", len(self.exp_db))
        return self.exp_db

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_temperature(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has a 'temperature' column.
        Model outputs use 'T'; this renames it if needed.
        Returns a copy so the caller's DataFrame is not mutated.
        """
        if "temperature" not in df.columns and "T" in df.columns:
            return df.rename(columns={"T": "temperature"})
        return df.copy()

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------

    def validate_predictions(
        self,
        predictions: pd.DataFrame,
        merge_on: Optional[List[str]] = None,
        component_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare model predictions against experimental data.

        Parameters
        ----------
        predictions : pd.DataFrame
            Must contain mof_id, T (or temperature), pressure, and component
            loading columns.
        merge_on : List[str], optional
            Columns to merge on.  Default: ['mof_id', 'temperature', 'pressure']
        component_mapping : Dict[str, str], optional
            Map component name → prediction column.
            Default: {'CO2': 'co2_loading_molkg', ...}

        Returns
        -------
        Dict with mae, rmse, r2, mape, n_points, per_component,
        error_statistics, outlier_fraction, (outliers).
        """
        if self.exp_db is None:
            raise RuntimeError(
                "No experimental data loaded. Call load_experimental_data() first."
            )

        component_mapping = component_mapping or {
            "CO2": "co2_loading_molkg",
            "N2":  "n2_loading_molkg",
            "H2O": "h2o_loading_molkg",
        }

        if predictions is None or len(predictions) == 0:
            return {"error": "No predictions provided"}

        # FIX 1: rename "T" → "temperature" before any merge/filter
        pred_df = self._normalise_temperature(predictions)

        merge_on = merge_on or ["mof_id", "temperature", "pressure"]

        # Melt wide-format predictions to long format
        if all(col in pred_df.columns for col in component_mapping.values()):
            melted_parts: List[pd.DataFrame] = []
            for comp, col_name in component_mapping.items():
                if col_name in pred_df.columns:
                    subset = pred_df[merge_on + [col_name]].copy()
                    subset["component"] = comp
                    subset["loading"]   = subset.pop(col_name)
                    melted_parts.append(subset)

            if not melted_parts:
                return {"error": "No prediction columns found"}
            pred_melted = pd.concat(melted_parts, ignore_index=True)
        else:
            pred_melted = pred_df

        # Merge
        merged = pd.merge(
            self.exp_db,
            pred_melted,
            on=merge_on + ["component"],
            suffixes=("_exp", "_pred"),
            how="inner",
        )

        if len(merged) == 0:
            return {
                "error": "No overlapping points found. Check MOF IDs and conditions.",
                "exp_mofs":  self.exp_db["mof_id"].unique().tolist()[:10],
                "pred_mofs": pred_melted["mof_id"].unique().tolist()[:10],
            }

        y_true = merged["loading_exp"].values
        y_pred = merged["loading_pred"].values

        metrics = self._compute_metrics(y_true, y_pred)
        metrics["n_points"] = int(len(merged))

        # Per-component metrics
        per_comp: Dict[str, Any] = {}
        for comp in merged["component"].unique():
            mask = merged["component"] == comp
            per_comp[comp] = self._compute_metrics(
                merged.loc[mask, "loading_exp"].values,
                merged.loc[mask, "loading_pred"].values,
            )
        metrics["per_component"] = per_comp

        # Error distribution
        errors     = y_true - y_pred
        abs_errors = np.abs(errors)
        metrics["error_statistics"] = {
            "mean_error":       float(np.mean(errors)),
            "std_error":        float(np.std(errors)),
            "skewness":         float(stats.skew(errors)),
            "kurtosis":         float(stats.kurtosis(errors)),
            "mean_abs_error":   float(np.mean(abs_errors)),
            "median_abs_error": float(np.median(abs_errors)),
            "max_abs_error":    float(np.max(abs_errors)),
            "q1_abs_error":     float(np.percentile(abs_errors, 25)),
            "q3_abs_error":     float(np.percentile(abs_errors, 75)),
        }

        # Outliers (> 2σ)
        error_std    = float(np.std(errors))
        outlier_mask = np.abs(errors) > 2.0 * error_std
        metrics["outlier_fraction"] = float(np.mean(outlier_mask))
        if outlier_mask.any():
            outliers = merged[outlier_mask].copy()
            outliers["abs_error"] = np.abs(
                outliers["loading_exp"] - outliers["loading_pred"]
            )
            metrics["outliers"] = (
                outliers[
                    ["mof_id", "component", "temperature", "pressure",
                     "loading_exp", "loading_pred", "abs_error"]
                ]
                .sort_values("abs_error", ascending=False)
                .head(10)
                .to_dict("records")
            )

        logger.info(
            "Experimental validation: MAE=%.4f, RMSE=%.4f, R²=%.4f, N=%d",
            metrics["mae"], metrics["rmse"], metrics["r2"], metrics["n_points"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Metrics helper
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if len(y_true) == 0:
            return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mape": np.nan}

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        mape = float(
            100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))
        )
        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

    # ------------------------------------------------------------------
    # Per-MOF validation
    # ------------------------------------------------------------------

    def validate_mof(
        self,
        mof_id: str,
        predictions: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Validate predictions for a single MOF."""
        pred_subset = predictions[predictions["mof_id"] == mof_id]
        if len(pred_subset) == 0:
            return {"error": f"No predictions found for MOF: {mof_id}"}
        return self.validate_predictions(pred_subset)

    # ------------------------------------------------------------------
    # Temperature-binned validation
    # ------------------------------------------------------------------

    def validate_by_temperature(
        self,
        predictions: pd.DataFrame,
        temperature_bins: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate predictions binned by temperature.

        Parameters
        ----------
        predictions : pd.DataFrame
            Model predictions (may have 'T' or 'temperature' column).
        temperature_bins : List[float], optional
            Bin edges in K.  Default: [273.15, 298.15, 313.15, 333.15, 373.15]
        """
        if self.exp_db is None:
            raise RuntimeError("No experimental data loaded.")

        if temperature_bins is None:
            temperature_bins = [273.15, 298.15, 313.15, 333.15, 373.15]

        # FIX 1: normalise column name before filtering
        pred_norm = self._normalise_temperature(predictions)

        exp_copy = self.exp_db.copy()
        exp_copy["temp_bin"] = pd.cut(
            exp_copy["temperature"],
            bins=temperature_bins,
            include_lowest=True,
        )

        results: Dict[str, Any] = {}

        # FIX 2: dropna() skips NaN bins produced by pd.cut for out-of-range values
        for bin_label in exp_copy["temp_bin"].dropna().unique():
            bin_exp = exp_copy[exp_copy["temp_bin"] == bin_label].drop(
                columns=["temp_bin"]
            )

            # FIX 1 (continued): filter on renamed column
            if "temperature" not in pred_norm.columns:
                continue
            pred_filtered = pred_norm[
                pred_norm["temperature"].between(bin_label.left, bin_label.right)
            ]

            if len(pred_filtered) == 0 or len(bin_exp) == 0:
                continue

            # FIX 3: try/finally so self.exp_db is always restored
            original_db = self.exp_db
            try:
                self.exp_db = bin_exp
                metrics = self.validate_predictions(pred_filtered)
            finally:
                self.exp_db = original_db

            results[str(bin_label)] = metrics

        return results

    # ------------------------------------------------------------------
    # Pressure-binned validation
    # ------------------------------------------------------------------

    def validate_by_pressure(
        self,
        predictions: pd.DataFrame,
        pressure_bins: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate predictions binned by pressure.

        Parameters
        ----------
        predictions : pd.DataFrame
            Model predictions.
        pressure_bins : List[float], optional
            Bin edges in bar.  Default: [0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        """
        if self.exp_db is None:
            raise RuntimeError("No experimental data loaded.")

        if pressure_bins is None:
            pressure_bins = [0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

        # FIX 1: normalise temperature column (predictions may use "T")
        pred_norm = self._normalise_temperature(predictions)

        exp_copy = self.exp_db.copy()
        exp_copy["press_bin"] = pd.cut(
            exp_copy["pressure"],
            bins=pressure_bins,
            include_lowest=True,
        )

        results: Dict[str, Any] = {}

        # FIX 2: dropna() avoids AttributeError on NaN bin labels
        for bin_label in exp_copy["press_bin"].dropna().unique():
            bin_exp = exp_copy[exp_copy["press_bin"] == bin_label].drop(
                columns=["press_bin"]
            )

            if "pressure" not in pred_norm.columns:
                continue
            pred_filtered = pred_norm[
                pred_norm["pressure"].between(bin_label.left, bin_label.right)
            ]

            if len(pred_filtered) == 0 or len(bin_exp) == 0:
                continue

            # FIX 3: try/finally
            original_db = self.exp_db
            try:
                self.exp_db = bin_exp
                metrics = self.validate_predictions(pred_filtered)
            finally:
                self.exp_db = original_db

            results[str(bin_label)] = metrics

        return results

    # ------------------------------------------------------------------
    # Rank correlation
    # ------------------------------------------------------------------

    def compute_rank_correlation(
        self,
        predictions: pd.DataFrame,
        # FIX 4: default changed from "loading" (doesn't exist in model output)
        # to "co2_loading_molkg" which matches ThermodynamicPotentialNO output.
        metric: str = "co2_loading_molkg",
    ) -> Dict[str, Any]:
        """
        Compute rank correlation between model predictions and experimental data.

        Useful for assessing whether the model correctly identifies
        high-performing MOFs.

        Parameters
        ----------
        predictions : pd.DataFrame
            Model predictions.
        metric : str
            Column to rank by.  Default: 'co2_loading_molkg'.

        Returns
        -------
        Dict with spearman_r, kendall_tau, spearman_p_value, n_mofs.
        """
        if self.exp_db is None:
            raise RuntimeError("No experimental data loaded.")

        if metric not in predictions.columns:
            raise KeyError(
                f"Column '{metric}' not found in predictions. "
                f"Available: {list(predictions.columns)}"
            )

        exp_agg  = self.exp_db.groupby("mof_id")["loading"].mean().reset_index()
        pred_agg = predictions.groupby("mof_id")[metric].mean().reset_index()

        merged = pd.merge(
            exp_agg, pred_agg,
            on="mof_id",
            suffixes=("_exp", "_pred"),
        )

        if len(merged) < 2:
            return {"error": "Not enough MOFs for rank correlation (need ≥ 2)"}

        spearman_r, spearman_p = stats.spearmanr(
            merged["loading_exp"],
            merged[f"{metric}_pred"],
        )
        kendall_tau, _ = stats.kendalltau(
            merged["loading_exp"],
            merged[f"{metric}_pred"],
        )

        return {
            "spearman_r":       float(spearman_r),
            "spearman_p_value": float(spearman_p),
            "kendall_tau":      float(kendall_tau),
            "n_mofs":           int(len(merged)),
        }

    # ------------------------------------------------------------------
    # Comprehensive report
    # ------------------------------------------------------------------

    def create_validation_report(
        self,
        predictions: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Create a comprehensive validation report.

        Parameters
        ----------
        predictions : pd.DataFrame
            Model predictions.
        output_path : Optional[Path]
            Save the report as JSON at this path.

        Returns
        -------
        Dict with overall, by_temperature, by_pressure, rank_correlation,
        and per_mof metrics.
        """
        report: Dict[str, Any] = {
            "overall":          self.validate_predictions(predictions),
            "by_temperature":   self.validate_by_temperature(predictions),
            "by_pressure":      self.validate_by_pressure(predictions),
            "rank_correlation": self.compute_rank_correlation(predictions),
        }

        # Per-MOF metrics for MOFs with ≥ 3 experimental points
        mof_metrics: Dict[str, Any] = {}
        for mof_id in self.exp_db["mof_id"].unique():
            mof_exp = self.exp_db[self.exp_db["mof_id"] == mof_id]
            if len(mof_exp) < 3:
                continue
            pred_mof = predictions[predictions["mof_id"] == mof_id]
            if len(pred_mof) < 3:
                continue

            # FIX 3: try/finally so self.exp_db is always restored
            original_db = self.exp_db
            try:
                self.exp_db = mof_exp
                mof_metrics[mof_id] = self.validate_predictions(pred_mof)
            finally:
                self.exp_db = original_db

        report["per_mof"] = mof_metrics

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            def _convert(obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                raise TypeError(f"Cannot serialize {type(obj)}")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=_convert)
            logger.info("Validation report saved → %s", output_path)

        return report

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def add_experimental_data(
        self,
        data: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """Add or replace experimental data."""
        if self.exp_db is None or overwrite:
            self.exp_db = data.copy()
        else:
            self.exp_db = pd.concat([self.exp_db, data], ignore_index=True)
        logger.info("Updated experimental database: %d rows", len(self.exp_db))

    def save_experimental_data(self, path: Union[str, Path]) -> None:
        """Save the experimental database to file."""
        if self.exp_db is None:
            raise RuntimeError("No experimental data to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            self.exp_db.to_parquet(path, index=False)
        else:
            self.exp_db.to_csv(path, index=False)
        logger.info("Experimental data saved → %s", path)


__all__ = ["ExperimentalValidator"]