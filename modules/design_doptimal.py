"""
Backend module providing the D-optimal designer class.

This implements candidate generation, model-matrix construction,
and two Fedorov-exchange/Genetic algorithm selection methods:
 - naive (recompute det(X'X) for each swap)
 - SMW-accelerated (Sherman-Morrison updates for fast rank-1 updates)
 - Genetic algorithm for candidate selection via evolution.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import random

from modules.design_base import DesignerBaseClass

# ###########################
# D-optimal designer class
# ###########################
class DOOptimalDesigner(DesignerBaseClass):
    def __init__(
        self,
        cont_names: List[str],
        cont_bounds: List[Tuple[float, float]],
        cat_defs: List[Tuple[str, List[str]]],
        mix_names: List[str],
        i_cont,
        j_cat,
        k_mix,
        N,
        cont_grid_points: int = 5,
        mixture_resolution: int = 6,
        n_candidates=1000,
        mixture_bounds=None,
        random_state: Optional[int] = None,
    ):
        """Initialize the D-optimal design generator.

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
        i_cont : int
            Number of continuous variables.
        j_cat : int
            Number of categorical variables.
        k_mix : int
            Number of mixture components.
        N : int
            Desired number of runs in the final design.
        cont_grid_points : int, default=5
            Number of grid points per continuous variable for factorial design.
        mixture_resolution : int, default=6
            Resolution for mixture lattice generation.
        n_candidates : int, default=1000
            Number of random candidates to generate.
        mixture_bounds : list of (float, float), optional
            Bounds for each mixture component. Defaults to (0, 1) for all.
        random_state : int, optional
            Seed for reproducibility.
        """

        self.cont_names = cont_names
        self.cont_bounds = cont_bounds
        self.cat_defs = cat_defs
        self.cat_levels = [levels for _, levels in cat_defs]
        self.mix_names = mix_names
        self.cont_grid_points = int(cont_grid_points)
        self.mixture_resolution = int(mixture_resolution)
        self.random_state = random_state
        self.i_cont = i_cont
        self.j_cat = j_cat
        self.k_mix = k_mix
        self.N = N
        self.n_candidates = n_candidates
        self.mixture_bounds = mixture_bounds if mixture_bounds else [(0, 1)] * k_mix
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # placeholders filled after candidate-generation / design-matrix build
        self.cand_df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None  # candidate design matrix (m x p)
        self.X_df: Optional[pd.DataFrame] = None
        self.p: Optional[int] = None

    def _compositions(self, s: int, k: int):
        """Generate all nonnegative integer k-tuples summing to s.

        Parameters
        ----------
        s : int
            Target sum.
        k : int
            Number of parts in the composition.

        Yields
        ------
        tuple of int
            One composition of s into k parts.
        """

        # generate nonnegative integer k-tuples summing to s
        if k == 1:
            yield (s,)
            return
        for i in range(s + 1):
            for tail in self._compositions(s - i, k - 1):
                yield (i,) + tail

    def generate_mixture_lattice(self):
        """Generate a mixture lattice design based on resolution and bounds.

        Returns
        -------
        list of numpy.ndarray
            List of feasible mixture compositions satisfying bounds.
        """

        k = len(self.mix_names)
        s = int(self.mixture_resolution)
        if k == 0:
            return [()]
        if k == 1:
            val = 1.0
            lo, hi = self.mixture_bounds[0]
            return [np.array([val])] if lo <= val <= hi else []

        comps = []
        for parts in self._compositions(s, k):
            arr = np.array(parts, dtype=float) / float(s)
            # keep only if all mixture bounds are satisfied
            if all(
                self.mixture_bounds[m][0] <= arr[m] <= self.mixture_bounds[m][1]
                for m in range(k)
            ):
                comps.append(arr)

        if len(comps) == 0:
            # fallback: midpoint of bounds, normalized
            arr = np.array([(lo + hi) / 2 for lo, hi in self.mixture_bounds])
            arr = arr / arr.sum()
            comps = [arr]

        uniq = np.unique(np.array(comps), axis=0)
        return [u for u in uniq]

    def generate_candidates(self, include_full_factorial: bool = False):
        """Generate a mixture lattice design based on resolution and bounds.

        Returns
        -------
        list of numpy.ndarray
            List of feasible mixture compositions satisfying bounds.
        """

        candidates = []
        cont_bounds = self.cont_bounds
        cat_levels = self.cat_levels

        # ---------------------------
        # Random candidate generation
        # ---------------------------
        for _ in range(self.n_candidates):
            row = {}

            # Continuous factors
            for c, (low, high) in enumerate(cont_bounds):
                name = self.cont_names[c] if c < len(self.cont_names) else f"cont{c+1}"
                row[name] = np.random.uniform(low, high)

            # Categorical factors
            for c, levels in enumerate(cat_levels):
                name = self.cat_defs[c][0] if c < len(self.cat_defs) else f"cat{c+1}"
                row[name] = np.random.choice(levels)

            # Mixture factors
            if self.k_mix > 0:
                for _ in range(2000):
                    mix = np.random.dirichlet(np.ones(self.k_mix))
                    if all(
                        self.mixture_bounds[m][0] <= mix[m] <= self.mixture_bounds[m][1]
                        for m in range(self.k_mix)
                    ):
                        for m in range(self.k_mix):
                            name = (
                                self.mix_names[m]
                                if m < len(self.mix_names)
                                else f"mix{m+1}"
                            )
                            row[name] = mix[m]
                        break
                else:
                    # fallback: midpoint of bounds, normalized
                    mix = np.array([(lo + hi) / 2 for lo, hi in self.mixture_bounds])
                    mix = mix / mix.sum()
                    for m in range(self.k_mix):
                        name = (
                            self.mix_names[m]
                            if m < len(self.mix_names)
                            else f"mix{m+1}"
                        )
                        row[name] = mix[m]

            candidates.append(row)

        df_random = pd.DataFrame(candidates)

        # ---------------------------
        # Full factorial candidates
        # ---------------------------
        if include_full_factorial:
            factorial_candidates = []
            import itertools

            # Continuous factors on a grid
            if len(cont_bounds) > 0:
                grids = [
                    np.linspace(lo, hi, self.cont_grid_points) for lo, hi in cont_bounds
                ]
            else:
                grids = [[None]]  # dummy iterable

            # Categorical factors
            if self.cat_defs:
                cat_grids = [levels for _, levels in self.cat_defs]
            else:
                cat_grids = [[None]]

            # Mixture lattice
            if self.k_mix > 0:
                mix_grids = self.generate_mixture_lattice()
            else:
                mix_grids = [[None]]

            for cont_vals in itertools.product(*grids):
                for cat_vals in itertools.product(*cat_grids):
                    for mix_vals in mix_grids:
                        row = {}

                        # continuous
                        if cont_vals[0] is not None:
                            for c, val in enumerate(cont_vals):
                                name = (
                                    self.cont_names[c]
                                    if c < len(self.cont_names)
                                    else f"cont{c+1}"
                                )
                                row[name] = val

                        # categorical
                        if cat_vals[0] is not None:
                            for c, val in enumerate(cat_vals):
                                name = (
                                    self.cat_defs[c][0]
                                    if c < len(self.cat_defs)
                                    else f"cat{c+1}"
                                )
                                row[name] = val

                        # mixture
                        if mix_vals[0] is not None:
                            for m, val in enumerate(mix_vals):
                                name = (
                                    self.mix_names[m]
                                    if m < len(self.mix_names)
                                    else f"mix{m+1}"
                                )
                                row[name] = val

                        factorial_candidates.append(row)

            df_factorial = pd.DataFrame(factorial_candidates)

            # Combine both sets
            df = pd.concat([df_random, df_factorial], ignore_index=True)
        else:
            df = df_random

        self.cand_df = df
        return df

    def build_design_matrix(
        self, include_intercept: bool = True, model_type: str = "linear"
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Construct the design matrix (X) for regression modeling.

        Parameters
        ----------
        include_intercept : bool, default=True
            Whether to include an intercept column in the design matrix.
        model_type : {"linear", "quadratic"}, default="linear"
            Type of model to build. Quadratic includes squares and interaction terms.

        Returns
        -------
        X : numpy.ndarray
            Numerical design matrix.
        X_df : pandas.DataFrame
            DataFrame version of the design matrix with column names.
        """

        if self.cand_df is None:
            raise RuntimeError(
                "Candidates not generated yet. Call generate_candidates() first."
            )
        df = self.cand_df.copy()
        parts = []

        # Intercept
        if include_intercept:
            parts.append(pd.Series(1.0, index=df.index, name="Intercept"))

        # Continuous and mixture variables
        num_vars = []
        if len(self.cont_names) > 0:
            for c in self.cont_names:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            parts.append(df[self.cont_names].astype(float))
            num_vars += self.cont_names

        if len(self.mix_names) > 0:
            k = len(self.mix_names)
            mix_cols = self.mix_names[:-1] if k > 1 else self.mix_names
            for m in mix_cols:
                df[m] = pd.to_numeric(df[m], errors="coerce")
            parts.append(df[mix_cols].astype(float))
            num_vars += mix_cols

        # Quadratic and interaction terms
        if model_type.lower() == "quadratic" and len(num_vars) > 0:
            # squares
            for var in num_vars:
                parts.append((df[var] ** 2).rename(f"{var}^2"))
            # interactions
            for i in range(len(num_vars)):
                for j in range(i + 1, len(num_vars)):
                    vi, vj = num_vars[i], num_vars[j]
                    parts.append((df[vi] * df[vj]).rename(f"{vi}*{vj}"))

        # Categorical -> dummies
        if len(self.cat_defs) > 0:
            cat_cols = [name for name, _ in self.cat_defs]
            dummies = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
            if dummies.shape[1] > 0:
                parts.append(dummies.astype(float))

        if len(parts) == 0:
            raise RuntimeError(
                "No model terms were constructed. Check factor definitions."
            )

        X_df = pd.concat(parts, axis=1)
        
        # Make sure X_df is compliant (types, and size)
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        X_df, clip_warning = self._clip_numeric(X_df)
        
        # Store
        X = X_df.values.astype(float)

        self.X = X
        self.X_df = X_df
        self.p = X.shape[1]
        return X, X_df, clip_warning

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

    # =====================
    # Selection algorithms
    # =====================
    def _slogdet(self, M: np.ndarray, regularization: float =1e-12) -> Tuple[float, float]:
        """
        Compute the sign and log-determinant of a matrix M.
        Falls back to pseudo-inverse with regularization if M is singular.

        Parameters
        ----------
        M : numpy.ndarray
            Square matrix.

        Returns
        -------
        sign : float
            Sign of the determinant.
        logdet : float
            Natural logarithm of the absolute determinant.
        """

        try:
            sign, ld = np.linalg.slogdet(M)
            if sign > 0:
                return sign, ld
            else:
                raise np.linalg.LinAlgError("Matrix not positive definite.")
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse with tiny ridge regularization
            M_reg = M + np.eye(M.shape[0]) * regularization
            try:
                sign, ld = np.linalg.slogdet(M_reg)
                return sign, ld
            except Exception:
                # Last resort: use determinant of pseudo-inverse
                M_pinv = np.linalg.pinv(M)
                sign, ld = np.linalg.slogdet(M_pinv + np.eye(M.shape[0]) * regularization)
                return sign, ld

    def select_design(
        self,
        N: int,
        method: str = "auto",
        max_iter: int = 5000,
        tol: float = 1e-12,
        smw_threshold: int = 2000,
        random_state: Optional[int] = None,
        allow_regularization: bool = True,
    ) -> Tuple[List[int], float, List[float]]:
        """Select N candidate runs using Fedorov exchange.

        Parameters
        ----------
        N : int
            Number of runs to select.
        method : {"auto", "naive", "smw"}, default="auto"
            Exchange algorithm to use.
        max_iter : int, default=5000
            Maximum number of exchange iterations.
        tol : float, default=1e-12
            Convergence tolerance for improvement in logdet.
        smw_threshold : int, default=2000
            Candidate size threshold for switching to SMW updates.
        random_state : int, optional
            Seed for reproducibility.
        allow_regularization : bool, default=True
            Whether to allow ridge regularization if the information matrix is singular.

        Returns
        -------
        indices : list of int
            Indices of selected candidates.
        final_logdet : float
            Log-determinant of the information matrix for the final design.
        history : list of float
            Convergence history of log-determinant values.
        """

        # Store thresholding value for choice of algorithm
        self.smw_threshold = int(smw_threshold)

        if self.X is None or self.cand_df is None:
            raise RuntimeError(
                "Must generate candidates and build design matrix before selection."
            )
        X = self.X
        m, p = X.shape
        N = int(N)
        if N > m:
            raise ValueError(f"Requested N={N} larger than number of candidates m={m}")
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        if method == "auto":
            method = "smw" if m > self.smw_threshold else "naive"

        if method == "naive":
            return self._fedorov_naive(
                X, N, max_iter=max_iter, tol=tol, random_state=random_state
            )
        elif method == "smw":
            return self._fedorov_smw(
                X,
                N,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                allow_regularization=allow_regularization,
            )
        else:
            raise ValueError("Unknown method; choose 'auto', 'naive', or 'smw'.")

    def select_design_ga(
        self,
        N: int,
        n_generations: int = 200,
        pop_size: int = 50,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        """Genetic Algorithm (GA) selection for D-optimal design.

        Binary chromosome: 1 = include candidate, 0 = exclude.
        Must have exactly N ones.

        Parameters
        ----------
        N : int
            Number of runs to select.
        n_generations : int, default=200
            Number of GA generations.
        pop_size : int, default=50
            Population size.
        crossover_rate : float, default=0.7
            Probability of performing crossover.
        mutation_rate : float, default=0.1
            Probability of performing mutation.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        indices : list of int
            Selected candidate indices.
        best_logdet : float
            Log-determinant of the best design found.
        best_fitness_history : list of float
            Best log-determinant values across generations.
        """

        if self.X is None or self.cand_df is None:
            raise RuntimeError(
                "Must generate candidates and build design matrix before selection."
            )

        m, p = self.X.shape
        if N < p:
            raise ValueError(
                f"N={N} < p={p}. The information matrix will be singular. Increase N or reduce model complexity."
            )

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        # --- helper functions ---
        def init_chromosome():
            chrom = np.zeros(m, dtype=int)
            idx = np.random.choice(m, N, replace=False)
            chrom[idx] = 1
            return chrom

        def fitness(chrom):
            idx = np.where(chrom == 1)[0]
            Xsel = self.X[idx, :]
            M = Xsel.T @ Xsel
            sign, ld = self._slogdet(M)
            return ld if sign > 0 else -np.inf

        def crossover(parent1, parent2):
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, m - 1)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
                # repair children
                child1 = repair(child1)
                child2 = repair(child2)
                return child1, child2
            else:
                return parent1.copy(), parent2.copy()

        def mutate(chrom):
            if np.random.rand() < mutation_rate:
                ones = np.where(chrom == 1)[0]
                zeros = np.where(chrom == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    i = np.random.choice(ones)
                    j = np.random.choice(zeros)
                    chrom[i], chrom[j] = 0, 1
            return chrom

        def repair(chrom):
            """Ensure exactly N ones."""
            ones = np.sum(chrom)
            if ones > N:
                idx = np.where(chrom == 1)[0]
                chrom[np.random.choice(idx, ones - N, replace=False)] = 0
            elif ones < N:
                idx = np.where(chrom == 0)[0]
                chrom[np.random.choice(idx, N - ones, replace=False)] = 1
            return chrom

        # --- initialize population ---
        population = [init_chromosome() for _ in range(pop_size)]
        fitnesses = [fitness(ch) for ch in population]

        best_fitness_history = []
        best_chrom = population[np.argmax(fitnesses)].copy()
        best_fit = max(fitnesses)

        # --- evolution loop ---
        for gen in range(n_generations):
            new_pop = []
            while len(new_pop) < pop_size:
                # tournament selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                parent1 = (
                    population[idx1]
                    if fitnesses[idx1] > fitnesses[idx2]
                    else population[idx2]
                )

                idx3, idx4 = np.random.choice(pop_size, 2, replace=False)
                parent2 = (
                    population[idx3]
                    if fitnesses[idx3] > fitnesses[idx4]
                    else population[idx4]
                )

                # crossover + mutation
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)

                new_pop.append(child1)
                if len(new_pop) < pop_size:
                    new_pop.append(child2)

            population = new_pop
            fitnesses = [fitness(ch) for ch in population]

            gen_best_fit = max(fitnesses)
            if gen_best_fit > best_fit:
                best_fit = gen_best_fit
                best_chrom = population[np.argmax(fitnesses)].copy()

            best_fitness_history.append(best_fit)

        indices = np.where(best_chrom == 1)[0].tolist()
        return indices, float(best_fit), best_fitness_history

    def _fedorov_naive(
        self,
        X: np.ndarray,
        N: int,
        max_iter: int = 2000,
        tol: float = 1e-12,
        random_state: Optional[int] = None,
    ):
        """Naive implementation of the Fedorov exchange algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            Candidate design matrix (m × p).
        N : int
            Number of runs to select.
        max_iter : int, default=2000
            Maximum number of iterations.
        tol : float, default=1e-12
            Convergence tolerance.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        indices : list of int
            Indices of selected candidates.
        final_logdet : float
            Log-determinant of the selected design's information matrix.
        history : list of float
            Log-determinant values across iterations.
        """

        m, p = X.shape
        if N < p:
            raise ValueError(
                f"N={N} < p={p}. The information matrix will be singular. Increase N or reduce model complexity."
            )
        # initialize
        indices = set(random.sample(range(m), N))
        Xsel = X[list(indices), :]
        M = Xsel.T @ Xsel
        sign, current_ld = self._slogdet(M)
        if sign <= 0:
            current_ld = -np.inf

        # Record convergence
        best_history = [current_ld]

        # Perform the Federov exchange algorithm
        # After initializing, it will iteratively exchange some candidates from the subset
        # in order to improve the log-determinant of the resulting candidate matrix X
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for sel in list(indices):
                if improved:
                    break
                for out in range(m):
                    if out in indices:
                        continue
                    new_indices = set(indices)
                    new_indices.remove(sel)
                    new_indices.add(out)
                    Xnew = X[list(new_indices), :]
                    Mnew = Xnew.T @ Xnew
                    sign, ld = self._slogdet(Mnew)
                    if sign <= 0:
                        ld = -np.inf
                    if ld > current_ld + 1e-12:
                        indices = new_indices
                        current_ld = ld
                        improved = True
                        break
            best_history.append(current_ld)
        return sorted(indices), float(current_ld), best_history

    def _fedorov_smw(
        self,
        X: np.ndarray,
        N: int,
        max_iter: int = 2000,
        tol: float = 1e-12,
        random_state: Optional[int] = None,
        allow_regularization: bool = True,
    ):
        """Fedorov exchange algorithm using Sherman–Morrison–Woodbury (SMW) updates.

        Parameters
        ----------
        X : numpy.ndarray
            Candidate design matrix (m × p).
        N : int
            Number of runs to select.
        max_iter : int, default=2000
            Maximum number of iterations.
        tol : float, default=1e-12
            Convergence tolerance.
        random_state : int, optional
            Seed for reproducibility.
        allow_regularization : bool, default=True
            Whether to add ridge regularization if the information matrix is singular.

        Returns
        -------
        indices : list of int
            Indices of selected candidates.
        final_logdet : float
            Log-determinant of the selected design's information matrix.
        history : list of float
            Log-determinant values across iterations.
        """

        m, p = X.shape
        if N < p:
            raise ValueError(
                f"N={N} < p={p}. The information matrix will be singular. Increase N or reduce model complexity."
            )

        # initialize
        indices = set(random.sample(range(m), N))
        Xsel = X[list(indices), :]
        M = Xsel.T @ Xsel

        # Define inversion calculation function with pseudo-inversion fallback
        def _calc_inversion(M_):
            try:
                Minv_ = np.linalg.inv(M_)
            except np.linalg.LinAlgError:
                Minv_ = np.linalg.pinv(M_)
            return Minv_

        # try to invert M; if not invertible, add tiny ridge for numerical stability
        try:
            sign, current_ld = self._slogdet(M)
            if sign <= 0:
                raise np.linalg.LinAlgError(
                    "Initial information matrix not positive definite"
                )
            M_inv = _calc_inversion(M)
        except Exception:
            if not allow_regularization:
                raise
            ridge = 1e-8 * np.trace(M) * np.eye(p) # Apparently, using the trace makes it numerically more stable
            M_reg = M + ridge
            sign, current_ld = self._slogdet(M_reg)
            if sign <= 0:
                # fallback: set very small logdet and continue but treat as bad
                current_ld = -np.inf
            M_inv = _calc_inversion(M_reg)

        # Record convergence
        best_history = [current_ld]

        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # iterate possible swaps
            for sel in list(indices):
                if improved:
                    break
                u = X[sel, :].reshape(p, 1)
                for out in range(m):
                    if out in indices:
                        continue
                    v = X[out, :].reshape(p, 1)
                    # Step 1: remove u u^T -> compute scalar s = u^T M_inv u
                    s = float((u.T @ M_inv @ u).squeeze())
                    if 1.0 - s <= tol:
                        # would make matrix singular or numerically unstable
                        continue
                    # M1_inv = M_inv + (M_inv u u^T M_inv)/(1 - s)
                    M1_inv = M_inv + (M_inv @ u @ u.T @ M_inv) / (1.0 - s)
                    # Step 2: add v v^T -> compute t = v^T M1_inv v
                    t = float((v.T @ M1_inv @ v).squeeze())
                    if 1.0 + t <= tol:
                        continue
                    # determinant update: det(M') = det(M) * (1 - s) * (1 + t)
                    new_ld = current_ld + np.log(1.0 - s) + np.log(1.0 + t)
                    if new_ld > current_ld + 1e-12:
                        # accept swap and update inverse (first M1_inv then rank-1 update)
                        M_inv = M1_inv - (M1_inv @ v @ v.T @ M1_inv) / (1.0 + t)
                        current_ld = new_ld
                        indices.remove(sel)
                        indices.add(out)
                        improved = True
                        break
            best_history.append(current_ld)
        return sorted(indices), float(current_ld), best_history

    # =====================
    # Utilities
    # =====================
    def get_selected_dataframe(self, indices: List[int]) -> pd.DataFrame:
        """Return a DataFrame of candidates corresponding to the selected indices.

        Parameters
        ----------
        indices : list of int
            Indices of selected candidates.

        Returns
        -------
        pandas.DataFrame
            Subset of candidate DataFrame containing only selected rows.
        """

        if self.cand_df is None:
            raise RuntimeError("Candidates not generated yet")
        return self.cand_df.iloc[sorted(indices)].reset_index(drop=True)

    def _get_design_dataframe(self) -> pd.DataFrame:
        """Return the full candidate DataFrame of all generated candidates.

        Returns
        -------
        pandas.DataFrame
            DataFrame of all candidates.
        """

        if self.cand_df is None:
            raise RuntimeError("Candidates not generated yet")
        return self.cand_df

    def _get_metadata(self) -> dict:
        """Return metadata about the current designer configuration.

        Returns
        -------
        dict
            Dictionary containing factor definitions, grid resolution,
            mixture resolution, and candidate count.
        """

        return {
            "cont_names": self.cont_names,
            "cont_bounds": self.cont_bounds,
            "cat_defs": self.cat_defs,
            "mix_names": self.mix_names,
            "cont_grid_points": self.cont_grid_points,
            "mixture_resolution": self.mixture_resolution,
            "candidate_count": len(self.cand_df) if self.cand_df is not None else None,
        }