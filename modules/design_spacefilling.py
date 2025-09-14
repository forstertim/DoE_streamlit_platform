"""
Backend module providing the designer class for space-filling design.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from pyDOE3 import lhs

from modules.design_base import DesignerBaseClass


# ###########################
# Space-filling designer class
# ###########################
class SpaceFillingDesigner(DesignerBaseClass):
    """
    Generate a space-filling candidate set for training models (e.g., GP)
    using maximin LHS for continuous variables and respecting mixture and categorical constraints.
    """

    def __init__(
        self,
        cont_names: List[str],
        cont_bounds: List[Tuple[float, float]],
        cat_defs: List[Tuple[str, List[str]]],
        mix_names: List[str],
        mixture_bounds: Optional[List[Tuple[float, float]]] = None,
        n_samples: int = 10,
        random_state: Optional[int] = None,
    ):
        """Initialize a space-filling design generator.

        Parameters
        ----------
        cont_names : list of str
            Names of continuous variables.
        cont_bounds : list of (float, float)
            Bounds for each continuous variable.
        cat_defs : list of (str, list of str)
            Definitions of categorical variables as (name, levels).
        mix_names : list of str
            Names of mixture components.
        mixture_bounds : list of (float, float), optional
            Bounds for each mixture component. Defaults to (0, 1) for all.
        n_samples : int, default=10
            Number of samples (rows) in the design.
        random_state : int, optional
            Seed for reproducibility.
        """

        self.cont_names = cont_names
        self.cont_bounds = cont_bounds
        self.cat_defs = cat_defs
        self.cat_levels = [levels for _, levels in cat_defs]
        self.mix_names = mix_names
        self.n_samples = n_samples
        self.random_state = random_state
        self.mixture_bounds = (
            mixture_bounds if mixture_bounds else [(0, 1)] * len(mix_names)
        )
        if random_state is not None:
            np.random.seed(random_state)

        self.design_df: Optional[pd.DataFrame] = None

    def generate(self, iters: int = 1000) -> pd.DataFrame:
        """Generate a space-filling design using LHS (continuous), Dirichlet sampling (mixtures),
        and random assignment (categoricals).

        Parameters
        ----------
        iters : int, default=1000
            Maximum attempts per sample to generate valid mixture points within bounds.

        Returns
        -------
        pandas.DataFrame
            Generated design DataFrame containing continuous, mixture, and categorical variables.
        """

        # Continuous LHS
        n_cont = len(self.cont_names)
        if n_cont > 0:
            # Generate Latin Hypercube samples in [0, 1]
            lhs_cont = lhs(n_cont, samples=self.n_samples, criterion="maximin")
            # Rescale each column to its specified bounds
            for i, (lo, hi) in enumerate(self.cont_bounds):
                lhs_cont[:, i] = lo + lhs_cont[:, i] * (hi - lo)
        else:
            # If no continuous variables, just create empty array with correct row count
            lhs_cont = np.zeros((self.n_samples, 0))

        # Mixture samples (Dirichlet sampling with bounds)
        n_mix = len(self.mix_names)
        mix_samples = np.zeros((self.n_samples, n_mix))
        for s in range(self.n_samples):
            for _ in range(iters):
                # Sample mixture proportions from a Dirichlet distribution
                mix = np.random.dirichlet(np.ones(n_mix))
                # Check whether each component respects its bounds
                ok = all(
                    self.mixture_bounds[m][0] <= mix[m] <= self.mixture_bounds[m][1]
                    for m in range(n_mix)
                )
                if ok:
                    mix_samples[s, :] = mix
                    break

            # Fallback: if no valid mixture found after all attempts
            # (take midpoints of bounds and normalize to sum to 1)
            else:
                mix_samples[s, :] = np.array(
                    [(lo + hi) / 2 for lo, hi in self.mixture_bounds]
                )
                mix_samples[s, :] /= mix_samples[s, :].sum()  # normalize

        # Construct dataframe
        df = pd.DataFrame(lhs_cont, columns=self.cont_names)
        df_mixes = pd.DataFrame(mix_samples, columns=self.mix_names)
        df = pd.concat([df, df_mixes], axis=1)

        # Add categorical variables randomly
        for name, levels in self.cat_defs:
            df[name] = np.random.choice(levels, size=self.n_samples)

        # Make sure design is not too large
        df, clip_warning = self._clip_numeric(df)

        # Store design and return it
        self.design_df = df
        return df, clip_warning

    # =====================
    # Utilities
    # =====================
    def _get_design_dataframe(self) -> pd.DataFrame:
        """Return the generated design DataFrame.

        Returns
        -------
        pandas.DataFrame
            Design DataFrame containing all generated points.

        Raises
        ------
        RuntimeError
            If design has not been generated yet.
        """

        if self.design_df is None:
            raise RuntimeError("Design not generated yet")
        return self.design_df

    def _get_metadata(self) -> dict:
        """Return metadata about the current design configuration.

        Returns
        -------
        dict
            Dictionary containing factor names, bounds, and sample size.
        """

        return {
            "cont_names": self.cont_names,
            "cont_bounds": self.cont_bounds,
            "cat_defs": self.cat_defs,
            "mix_names": self.mix_names,
            "n_samples": self.n_samples,
        }

    def _clip_numeric(self, df, threshold=1e6):
        """
        Clip overly large/small numeric values to avoid instability.
        Warn the user when clipping occurs.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        clip_warning = None
        for col in numeric_cols:
            max_val = df[col].abs().max()
            if max_val > threshold:
                clip_warning = "Detected very large value, which might lead to numerical \
                    instabilities! Bring numerical values to reasonable scales! \
                    Best is to leave them in a dimensionless space, and after downloading, \
                    convert them to your desired scale!"
                # df[col] = df[col].clip(lower=-threshold, upper=threshold) <-- TODO: could clip value here, but leave it out for now and just display message
        return df, clip_warning
