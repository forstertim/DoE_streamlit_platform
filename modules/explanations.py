"""
This file is a part of the DoE Generator in Streamlit.
It contains the explanations that are shown to the user, just to keep this long text separate from
the actual app, for better readability.
"""


def design_explanation():
    return r"""
        ## D-optimal design

        ### Federov

        D-optimal designs aim to **maximize the information content** for fitting a linear (or quadratic) model.  
        The key idea is to maximize the determinant of the information matrix:

        $$
        \mathbf{X}^\top \mathbf{X},
        $$

        where $\mathbf{X}$ is the design matrix containing the coded continuous, mixture, and categorical variables (with dummy coding).

        - For a linear model: main effects only.
        - For a quadratic model: squares ($x_i^2$) and two-way interactions ($x_i x_j$) are included.

        **Fedorov exchange algorithm** iteratively improves a candidate set:

        1. Start with an initial set of $N$ candidate points.
        2. Swap a selected point $x_s$ with a candidate $x_c$ if it increases $\det(\mathbf{X}^\top \mathbf{X})$.
        3. Repeat until convergence or maximum iterations.

        With Sherman-Morrison (SMW) updates, we can efficiently update the inverse $(\mathbf{X}^\top \mathbf{X})^{-1}$ after a single swap:

        $$
        (\mathbf{M} + \mathbf{u}\mathbf{u}^\top)^{-1} 
        = \mathbf{M}^{-1} - \frac{\mathbf{M}^{-1}\mathbf{u}\mathbf{u}^\top \mathbf{M}^{-1}}{1 + \mathbf{u}^\top \mathbf{M}^{-1} \mathbf{u}}
        $$

        ### Genetic Algorithm (GA)

        The **Genetic Algorithm (GA)** is a stochastic optimization method inspired by natural selection.  
        Instead of local swaps like Fedorov, GA **evolves a population** of candidate designs over generations.

        - Each design is represented as a **binary chromosome**, where a `1` means the candidate point is included and `0` means excluded.  
        - The **fitness** of a chromosome is the log-determinant of the information matrix:

        $$
        f(D) = \log \det(\mathbf{X}_D^\top \mathbf{X}_D),
        $$

        where $\mathbf{X}_D$ is the design matrix for the selected subset $D$.

        **GA procedure**:

        1. Initialize a population of random designs (subsets of size $N$).  
        2. Evaluate fitness $f(D)$ for each design.  
        3. Select parents using tournament or roulette selection.  
        4. Apply **crossover** (swap subsets of points between parents).  
        5. Apply **mutation** (randomly swap in/out some points).  
        6. Repair chromosomes so that exactly $N$ points are selected.  
        7. Repeat for several generations, keeping the best designs.

        This allows **global exploration** of the candidate space and avoids local optima.

        ---

        ## Space-filling design (maximin LHS)

        For models like **Gaussian Processes**, we often want a design that **covers the input space evenly**.  

        - **Latin Hypercube Sampling (LHS)**: Each variable is divided into $N$ equally probable intervals; one sample is taken per interval.
        - **Maximin criterion**: The minimum distance between any two points is maximized to ensure even coverage.

        Mathematically, let $D = \{x_1, ..., x_N\}$ be the design points and $d_{ij} = \|x_i - x_j\|$ the Euclidean distance. The maximin objective is:

        $$
        \max_D \min_{i \neq j} d_{ij}.
        $$

        Categorical variables are sampled randomly, while mixture components are sampled respecting their constraints (e.g., sum to 1).

        ---

        ## Summary

        - **D-optimal design** → Efficient for estimating parameters of linear/quadratic models. Useful for factor importances and "classical DoE tasks". 
        - **Space-filling LHS** → Maximizes coverage, well-suited for nonlinear and advanced models (e.g., Gaussian Processes).  
    """
