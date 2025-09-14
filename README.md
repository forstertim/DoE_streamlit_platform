# Design of Experiments platform 🚀

Use the following link to access the Design of experiments (DoE) platform on Streamlit.

[...]

## Implementation 💻
The platform incorporates two options to create either a space-filling design (using a Latin Hypercube Sampling - LHS) approach, or a D-optimal design. 

The design variables might be continouos or categorical, for which both sound appropriate bounds can be indicated.
Furthermore, the user can define a mixture by choosing mixture variables. There, the assumption is that *one* mixture is present in the system, which contains `k` components, where the values of those `k` components sum to one (for all the implementations below, the sampling is done from a Dirichlet distribution).

### Space-filling design 🪐
For the space-filling design, a maximin-LHS approach is implemented using `pyDOE3`. 

### D-optimal design 🌄
To find the D-optimal design, an optimization routine tries to maximize the determinant (or log-determinant). More information are visible at the bottom of the Streamlit app. There are three possible optimization approaches implemented: The federov exchange algorithm (naive or via Sherman-Morrison-Woodbury updates), or a genetic algorithm (GA), which is coded in very simple ways using numpy. 

## Usage 🎺
Open the design platform via the link provided above.

### Sidebar settings 🛠️
In the sidebar you can configure the design generation:

- **Design type (choose based on your use case)**:
    - ***D-optimal***: A "classical" DoE approach, used to estimate model parameters (linear or quadratic).
    - ***Space-filling (maximin LHS)***: Better suited for training nonlinear models (e.g., Gaussian processes). Ensures uniform coverage of the design space.
- **Main settings**:
    - ***Number of continuous factors*** (variables with numeric ranges, e.g., temperature, pressure).
    - ***Number of categorical factors*** (discrete options, e.g., vessel type: glass, steel, plastic).
    - ***Number of mixture components*** (assumes exactly one mixture; variables constrained to sum = 1).
    - ***Number of experiments*** (your experimental budget, i.e., how many runs to select).
- **Model settings (D-optimal only)**:
    - ***Model type***: linear or quadratic (quadratic adds interactions and squared terms).
- **Optimizer settings (D-optimal only)**:
    - ***Algorithm***: Fedorov/SMW (deterministic exchange algorithm) or Genetic Algorithm (GA) (stochastic global search).
- **Advanced candidate generation** (inside expander):
    - ***Random seed*** (set to 0 for random each run).
    - ***Continuous grid resolution***: number of discrete points per continuous factor. Example: with bounds [0, 10] and resolution = 3 → points at 0, 5, 10. Higher values give more candidate points and better coverage, but increase computation.
    - ***Mixture lattice resolution***: resolution for enumerating mixture candidates. Example: with 3 mixture components and resolution = 6, all integer triplets (n1, n2, n3) with n1+n2+n3=6 are enumerated → normalized to fractions that sum to 1. Larger values give finer grids inside the simplex, but may generate thousands of candidates.
    - ***Extra random candidates***: number of additional randomly sampled candidate points (beyond the systematic grid/lattice). This increases the search space diversity and may improve designs, especially with nonlinear interactions.
    - ***For Fedorov/SMW: maximum iterations***, and threshold for switching to Sherman–Morrison–Woodbury (SMW) updates.
    - ***For GA: number of generations***, population size, crossover and mutation rates.
    - ***For LHS: number of random search iterations***.

### Main body settings 📋
Here you define the variables:
- ***Continuous variables***: enter names and bounds. (If the upper bound ≤ lower bound, the app shows an error.)
- ***Categorical variables***: enter names and levels (comma-separated).
- ***Mixture components***: enter names and bounds. (Bounds are clipped to [0, 1]. Upper must be greater than lower.)

> 💡 Note: Streamlit updates the labels dynamically when variable names are changed. Pressing Tab to move through fields sometimes needs repeating due to UI refresh.

### Running the design ▶️
Click `Generate design`. The system will:

- Validate inputs (warns you if bounds are invalid or `N` < `number of model parameters`).
- Warn you to check extremely large values to prevent numerical issues.
- Run the chosen optimizer (D-optimal or LHS).
- Provide convergence history (D-optimal only), correlation plots, and mixture ternary diagrams (if `k`=3).

### Export 📂
After generating a design:

- Preview the design (first 10 rows).
- Download it as Excel (.xlsx).
- Or download as CSV via the Streamlit built-in option in the data table menu.


## Disclaimer 🚧
- The algorithms are certainly not optimally created, and there would be tools avaialble that are more sophisticated. These codes were also created to understand the algorithms, so also bugs might still be in there. 
- Some parts were created using LLM support (also used for this README file). 😊



