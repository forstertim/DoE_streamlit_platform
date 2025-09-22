"""
Streamlit app for a "Design of Experiments" method.
The app combines two main approaches to create experiments:
    - D-optimal design (for generating samples that maximize the parameter estimation accuracy of linear models)
    - Latin Hypercube Sampling via a maximin approach (Space-filling design, better suited for nonlinear model training)
The user interface should be kept on a level so people without deep knowledge of DoE can use it.

This file only includes the front-end, where the actual calculations are done by a separate class.
"""

import streamlit as st
import io

from modules.design_doptimal import DOOptimalDesigner
from modules.design_spacefilling import SpaceFillingDesigner
from modules.plotting import Plotting
from modules.explanations import design_explanation


# ###########################
# General Streamlit settings
# ###########################
st.set_page_config(page_title="Design generator", layout="wide")
st.title("Design generator")


# ###########################
# SIDEBAR: Design type and main settings
# ###########################
st.sidebar.header("Design type")
design_type = st.sidebar.selectbox(
    "Design type",
    options=["D-optimal", "Space-filling (maximin LHS)"],
    index=0,
    help="""
        A D-optimal design creates experiments so that, statistically, the \
        parameters of a model (here a linear one) can be estimated as accurately as possible. \
        A space-filling design (like the latin hypercube sampling (LHS) approach), is suited \
        if the user wants to collect data to train a nonlinear model like a Gaussian Process. \
        This design makes sure that the design space is evenly covered.
    """,
)

# Number of continous factors
st.sidebar.header("Main settings")
i = st.sidebar.number_input(
    "Number of continuous factors (i)",
    min_value=0,
    max_value=10,
    value=1,
    step=1,
    help="""
        A variable which can be set continously (i.e., a Temperature or Pressure).
    """,
)
# Number of categorical factors
j = st.sidebar.number_input(
    "Number of categorical (or discrete numeric) factors (j)",
    min_value=0,
    max_value=10,
    value=1,
    step=1,
    help="""
        A categorical factor is a variable from a finite set, where only one element of this set \
        can be executed for one experiment (i.e., You have 3 vessel types, steel, plastic, and glass, and \
        per experiment you need to choose one).
    """,
)
# Number of mixture components (assume only one mixture))
k = st.sidebar.number_input(
    "Number of mixture components (k)",
    min_value=0,
    max_value=6,
    value=3,
    step=1,
    help="""
        It is assumed that there is one mixture involved in the process, \
        which contains $k$ components.
    """,
)
# Number of total final experiments in the design
N = st.sidebar.number_input(
    "Number of experiments to select (N)",
    min_value=1,
    max_value=2000,
    value=10,
    step=1,
    help="""
        This defines how many experiments should be created in the final design, \
        and therefore represents the experimental budget.
    """,
)


# ###########################
# SIDEBAR: Model / algorithm settings
# ###########################
if design_type == "D-optimal":

    # Model settings
    st.sidebar.markdown("---")
    st.sidebar.header("Model settings")
    model_type = st.sidebar.selectbox(
        "Model type",
        options=["linear", "quadratic"],
        index=0,
        help="""
            Linear: main effects only. Quadratic: adds squares and \
            two-way interactions for continuous and mixture variables.
        """,
    )

    # Optimizer settings
    st.sidebar.markdown("---")
    st.sidebar.header("Optimzizer settings")
    algo_type = st.sidebar.selectbox(
        "Selection algorithm",
        options=["Fedorov/SMW", "Genetic Algorithm (GA)"],
        index=1,
        help="""
            Choose how to optimize the design. The GA is the default algorithm, \
            where it may be slower with large numbers of candidates, but explores globally.
        """,
    )


# ###########################
# SIDEBAR: Additional advanced settings
# ###########################
st.sidebar.markdown("---")
st.sidebar.header("Advanced candidate generation settings")
with st.sidebar.expander("Change defaults", expanded=False):

    # General advanced settings
    random_seed = st.number_input(
        "Random seed (0 for random)", min_value=0, max_value=99999, value=0
    )

    # Advanced settings for D-optimal design
    # ======================================
    if design_type == "D-optimal":

        # General advanced settings
        # ----------------------------------
        cont_grid_points = st.number_input(
            "Grid points per continuous factor",
            min_value=2,
            max_value=20,
            value=3,
            step=1,
            help="""
                For generating the full factorial design, the 'levels' of the variables are required. \
                The two bounds (lower and upper) are taken into consideration by default together with \
                a center point (hence, value=3). \
                If additional grid points should be considered, increase the value. \
                To estimate curvature, at least one middle point is needed.
            """,
        )
        mixture_resolution = st.number_input(
            "Mixture lattice resolution (integer)",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            help="""
                The lattice resolution defines the enumeration points on a simplex (i.e., \
                a triangle for a 3-component mixture, which is represented by the 3-component \
                diagram). The algorithm enumerates points of the form [$n_1/s, n_2/s, ..., n_k/s$], with integer $n_i>=0$\
                and $sum(n_k)=s$. The higher this integer, the greater the 'resolution', but the more possible \
                enumeration candidates are generated.
            """,
        )
        n_candidates = st.number_input(
            "Additional random samples",
            min_value=0,
            max_value=50000,
            value=2000,
            step=1000,
            help="""
                A full-factorial design will be generated, where additional random \
                samples might be added to increase the potential of getting a better design.
            """,
        )

        # If Federov/SMV algorithm is chosen
        # ----------------------------------
        if algo_type == "Fedorov/SMW":

            max_federov_iterations = st.number_input(
                "Max Federov iterations",
                min_value=1000,
                max_value=20000,
                value=5000,
                step=500,
                help="""
                    During each iteration one candidate of the subset is exchanged by another one. \
                    If the new subset's determinant is larger than before, the new sample is accepted. \
                    Check at the bottom of the main page for more details.
                """,
            )
            smw_threshold = st.number_input(
                "Switch to SMW if candidate number exceeds",
                min_value=100,
                max_value=20000,
                value=5000,
                step=100,
                help="""
                    The default algorithm to create the design is the Federov exchange algorithm \
                    (check at the bottom of the page for more information). If the amount of candidates \
                    is too large, the algorithm becomes less efficient. Then, the Sherman-Morrison-Woodbury (SMW) \
                    approach becomes more efficient and robust. The threshold when to switch to the SMW approach \
                    can be changed here. For more informatoin about SMW, check at the main page's bottom. 
                """,
            )

        # If GA is chosen, show GA advanced settings
        # -----------------------------------
        elif algo_type == "Genetic Algorithm (GA)":
            ga_generations = st.number_input(
                "GA generations", min_value=50, max_value=2000, value=500, step=50
            )
            ga_pop_size = st.number_input(
                "GA population size", min_value=10, max_value=500, value=100, step=10
            )
            ga_crossover_rate = st.number_input(
                "GA crossover rate",
                min_value=0.01,
                max_value=1.00,
                value=0.7,
                step=0.01,
            )
            ga_mutation_rate = st.number_input(
                "GA mutation rate", min_value=0.01, max_value=1.00, value=0.1, step=0.01
            )

    # Advanced settings for LHS
    # ======================================
    elif design_type == "Space-filling (maximin LHS)":

        random_search_iterations = st.number_input(
            "Search iterations", min_value=1000, max_value=20000, value=5000, step=500
        )


# ###########################
# Factor definitions
# ###########################
st.header("Define continuous factors (if any)")
cont_names, cont_bounds = [], []
if i > 0:
    cols = st.columns(min(i, 4))
    for idx in range(i):
        with cols[idx % len(cols)]:
            name = st.text_input(
                f"Continuous name #{idx+1}",
                value=f"Cont{idx+1}",
                key=f"cont_name_{idx}",
            )
            lo = st.number_input(f"{name} lower bound", value=0.0, key=f"cont_lo_{idx}")
            hi = st.number_input(f"{name} upper bound", value=1.0, key=f"cont_hi_{idx}")
            if hi <= lo:
                st.error(f"For {name}, upper bound must be > lower bound.")
            cont_names.append(name)
            cont_bounds.append((lo, hi))
else:
    st.info("No continuous factors.")

st.header("Define categorical factors (if any)")
cat_defs = []
if j > 0:
    cols = st.columns(min(j, 4))
    for idx in range(j):
        with cols[idx % len(cols)]:
            name = st.text_input(
                f"Categorical name #{idx+1}",
                value=f"Cat{idx+1}",
                key=f"cat_name_{idx}",
            )
            levels_s = st.text_input(
                f"Levels for {name} (comma-separated without spacing)",
                value="A,B",
                key=f"cat_levels_{idx}",
                help="If a variable includes a space or special character, use an underscore!",
            )
            levels = [l.strip() for l in levels_s.split(",") if l.strip()]
            cat_defs.append((name, levels))
else:
    st.info("No categorical factors.")

st.header("Define mixture components (if any)")
mix_names, mixture_bounds = [], []
if k > 0:
    cols = st.columns(min(k, 6))
    for idx in range(k):
        with cols[idx % len(cols)]:
            name = st.text_input(
                f"Mixture component #{idx+1} name",
                value=f"x{idx+1}",
                key=f"mix_name_{idx}",
            )
            lo = st.number_input(
                f"{name} lower bound",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                key=f"mix_lo_{idx}",
            )
            hi = st.number_input(
                f"{name} upper bound",
                value=1.0,
                min_value=0.0,
                max_value=1.0,
                key=f"mix_hi_{idx}",
            )
            if hi <= lo:
                st.error(f"For {name}, upper bound must be > lower bound.")
            mix_names.append(name)
            mixture_bounds.append((lo, hi))
else:
    st.info("No mixture components.")


st.markdown("---")


# ###########################
# Run button
# ###########################
if st.button("Generate design"):

    # Store design type in session (in case things are changed in the side bar, need to update those)
    st.session_state['design_type'] = design_type

    # Define seed
    seed = None if random_seed == 0 else int(random_seed)

    # Initialize helper variables
    st.session_state['large_enough_N'] = True

    # D-optimal design approach
    # ======================================
    if st.session_state['design_type'] == "D-optimal":

        # Again store D-optimal design specific things to session state
        st.session_state['model_type'] = model_type

        # Instantiate designer
        designer = DOOptimalDesigner(
            cont_names=cont_names,
            cont_bounds=cont_bounds,
            cat_defs=cat_defs,
            mix_names=mix_names,
            i_cont=i,
            j_cat=j,
            k_mix=k,
            N=int(N),
            cont_grid_points=cont_grid_points,
            mixture_resolution=mixture_resolution,
            n_candidates=int(n_candidates),
            mixture_bounds=mixture_bounds,
            random_state=seed,
        )

        # Generate all the candidates (full-matrix)
        with st.spinner("Generating candidates..."):
            cand_df = designer.generate_candidates(include_full_factorial=True)
        st.write(f"Candidate points generated: {len(cand_df)}")

        # Check that we have more candidates than requested experiments.
        if len(cand_df) < N:
            st.error(
                f"Number of candidates ({len(cand_df)}) < requested N ({N}). Increase candidate resolution or reduce N."
            )

        # If check is ok, start to build the final design matrix X
        else:
            with st.spinner("Building design matrix..."):
                X, X_df, clip_warning = designer.build_design_matrix(model_type=model_type)
                st.write(f"Model matrix has p = {X.shape[1]} parameters")
                if clip_warning is not None:
                    st.warning(clip_warning)

            # If the user requests too many experiments, show a warning
            if N < X.shape[1]:
                st.error(
                    f"You selected N={N} but model has p={X.shape[1]} parameters. "
                    f"D-optimal designs typically require N ≥ p."
                )
                st.session_state['large_enough_N'] = False

            # Start design selection
            # -------------------------------------
            if st.session_state['large_enough_N']:
                with st.spinner("Running selection..."):

                    # Federov approach
                    if algo_type == "Fedorov/SMW":
                        indices, final_ld, best_history = designer.select_design(
                            N=int(N),
                            method="auto",
                            max_iter=max_federov_iterations,
                            smw_threshold=smw_threshold,
                            random_state=seed,
                        )

                    # Genetic algorithm approach
                    else:
                        indices, final_ld, best_history = designer.select_design_ga(
                            N=int(N),
                            n_generations=ga_generations,
                            pop_size=ga_pop_size,
                            crossover_rate=ga_crossover_rate,
                            mutation_rate=ga_mutation_rate,
                            random_state=seed,
                        )

                    # Update the user about success
                    st.success(f"Selection finished. logdet(X'X) = {final_ld:.4f}")

                    # Get final design
                    out_df = designer.get_selected_dataframe(indices)

                    # Define output filename
                    st.session_state["export_design_filename"] = "D_optimal_design.xlsx"
        
                    # Store congergence information
                    st.session_state["best_history"] = best_history

    # Space-filling design approach
    # ======================================
    elif st.session_state['design_type'] == "Space-filling (maximin LHS)":

        # Start design selection
        # -------------------------------------
        with st.spinner("Running selection..."):
            designer = SpaceFillingDesigner(
                cont_names=cont_names,
                cont_bounds=cont_bounds,
                cat_defs=cat_defs,
                mix_names=mix_names,
                mixture_bounds=mixture_bounds,
                n_samples=int(N),
                random_state=seed,
            )

            # Get final design
            out_df, clip_warning = designer.generate(iters=random_search_iterations)                
            if clip_warning is not None:
                    st.warning(clip_warning)

        # Define LHS-specific things for session state
        st.session_state["export_design_filename"] = "spacefilling_design.xlsx"
        st.session_state['model_type'] = None

        # No convergence available for this approach
        st.session_state['best_history'] = None

    # Store required variables to session state
    if st.session_state['large_enough_N']:
        st.session_state["out_df"] = out_df
        st.session_state["designer"] = designer



# ###########################
# Display design and make download available
# ###########################
# The if-statement prevents a direct reloading of the app that when we
# change some buttons in the plots below
if "out_df" in st.session_state and st.session_state['large_enough_N']:

    st.markdown("---")

    out_df = st.session_state["out_df"]
    designer = st.session_state["designer"]

    # Display dataframe
    preview_n_rows = 10 if out_df.shape[0] > 10 else out_df.shape[0]
    st.subheader(f"Selected design (first {preview_n_rows} rows)")
    st.dataframe(out_df.head(preview_n_rows))

    # Excel download (now unified via DesignExporter)
    towrite = io.BytesIO()
    designer.export_to_excel(towrite)
    towrite.seek(0)
    st.download_button(
        label="Download design as Excel (.xlsx)",
        data=towrite,
        file_name=st.session_state["export_design_filename"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    st.header("Design summary")

    # ===========================
    # Design summary
    # ===========================
    st.subheader("General summary")
    st.caption("""
        The design above and the figures below refer to the following information! \
        When changing any settings, the design and the figures are not automatically updated! \
        Use the button "Generate design" to then update the design upon changing the settings!
        """
    )
    design_summary_cols = st.columns([1,1,1,3], gap='small')
    with design_summary_cols[0]:
        st.write(f'- Number of continuous features: {len(st.session_state['designer'].cont_names)}')
        st.write(f'- Number of categorical features: {len(st.session_state['designer'].cat_levels)}')
        st.write(f'- Number of mixture features: {len(st.session_state['designer'].mix_names)}')
    with design_summary_cols[1]:
        st.write(f'- Number experiments: {st.session_state['out_df'].shape[0]}')
        st.write(f'- Design type: {st.session_state['design_type']}')
    with design_summary_cols[2]:
        if st.session_state['design_type'] == 'D-optimal':
            st.write(f'- Model type: {st.session_state['model_type']}')
            st.write(f'- Logdet={st.session_state["best_history"][-1]:.1f}')


    # ===========================
    # Plotting
    # ===========================
    # Instantiate plotter
    # --------------------------------------
    plotter = Plotting()
    plot_cols = st.columns([1,1], gap='small')

    # Plotting for D-optimal designs
    # --------------------------------------
    if st.session_state['design_type'] == "D-optimal":
    
        # Plot history if available
        if st.session_state["best_history"] is not None:
            with plot_cols[0]:
                st.subheader("Convergence evolution of iterative design search")
                fig = plotter.get_best_history_plot(history=st.session_state["best_history"])
                st.plotly_chart(fig, use_container_width=True)

        # Correlation plot of design matrix
        with plot_cols[1]:
            st.subheader("Correlation plot of model terms")
            fig = plotter.get_model_term_correlation_plot(
                st.session_state["designer"].X_df.drop("Intercept", axis=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Plot ternary diagram for mixture distribution
    # --------------------------------------
    if k == 3:
        st.subheader("Ternary diagram of the components in the mixture")
        ternary_plot_cols = st.columns([1, 1], gap="small")
        with ternary_plot_cols[0]:
            _color_name = st.selectbox(
                "Color name",
                [designer.cat_defs[i_][0] for i_ in range(len(designer.cat_defs))],
            )
        with ternary_plot_cols[1]:
            _size_name = st.selectbox("Size name", designer.cont_names)
        fig = plotter.get_ternary_diagram(
            df=st.session_state["out_df"],
            designer=st.session_state["designer"],
            color_name=_color_name,
            size_name=_size_name,
        )
        st.plotly_chart(fig, use_container_width=True)


# ###########################
# Some additional information for the user
# ###########################
st.markdown("---")
st.caption("To reset everything, just reload this page!")
st.caption(
    """
        Notes: If the Federov approach is used for the D-optimal designs, the algorithm \
        switches from a naive update rule to Sherman-Morrison-Woodbury (SMW) updates \
        automatically when candidate count exceeds the given threshold. \
        Space-filling designs use maximin LHS for continuous variables, where the values \
        for the mixture variables are sampled from a Dirichlet distribution.
    """
)

# Show explanation of methods
with st.expander("Explanation of design methods", expanded=False):
    st.markdown(design_explanation(), unsafe_allow_html=False)
