import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


class Plotting:
    """
    A utility class for generating various plots using Plotly.
    Includes methods for convergence history plots, ternary diagrams,
    and correlation heatmaps of model terms.
    """

    def __init__(self):
        """Initialize the Plotting class."""
        pass

    def get_best_history_plot(self, history):
        """
        Generate a line plot of the best log-determinant history
        across generations/iterations.

        Parameters
        ----------
        history : list or array-like
            Sequence of best logdet values from iterative design search.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A Plotly figure showing the convergence evolution.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=history,
                mode="lines+markers",  # show both line and data points
                name="Best logdet",
            )
        )
        fig.update_layout(
            xaxis_title="Generation/Iteration",
            yaxis_title="Best logdet(X'X)",
        )
        return fig

    def get_ternary_diagram(
        self, df: pd.DataFrame, designer, color_name: str, size_name: str
    ):
        """
        Generate a ternary scatter plot of mixture designs.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing mixture proportions and attributes.
        designer : object
            An object that has `mix_names` attribute (list of three mixture component names).
        color_name : str
            Column name in `df` used to color points.
        size_name : str
            Column name in `df` used to scale marker sizes.

        Returns
        -------
        fig : plotly.express.Figure
            A Plotly ternary scatter plot.
        """
        # Normalize size column to range [5, 15] for consistent marker sizing
        size_col = df[size_name].to_numpy(dtype=float)
        if np.ptp(size_col) > 0:  # avoid division by zero if values are constant
            scaled_size = 5 + (size_col - size_col.min()) * (15 - 5) / (
                size_col.max() - size_col.min()
            )
        else:
            scaled_size = np.full_like(
                size_col, 10.0
            )  # use uniform size if values are constant

        # Use a qualitative color map with highly distinguishable colors
        qualitative_palette = (
            px.colors.qualitative.Set1
        )  # alternatives: Set2, Dark2, Bold

        # Plot ternary diagram with scaled marker sizes and custom color palette
        fig = px.scatter_ternary(
            df,
            a=designer.mix_names[0],
            b=designer.mix_names[1],
            c=designer.mix_names[2],
            color=color_name,
            size=scaled_size,  # pass normalized size values
            size_max=15,
            color_discrete_sequence=qualitative_palette,  # enforce consistent colors
        )
        return fig

    def get_model_term_correlation_plot(self, X_df: pd.DataFrame):
        """
        Generate a heatmap showing the correlation between model terms.

        Parameters
        ----------
        X_df : pandas.DataFrame
            DataFrame containing model terms (columns of the design matrix).

        Returns
        -------
        fig : plotly.express.Figure
            A Plotly heatmap of the correlation matrix.
        """
        # Compute correlation matrix of model terms
        corr = X_df.corr()

        # Create a heatmap with correlation values displayed
        fig = px.imshow(
            corr,
            text_auto=".2f",  # show values with 2 decimal places
            color_continuous_scale="RdBu_r",  # red-blue diverging scale
            zmin=-1,
            zmax=1,  # correlation range
            aspect="auto",
        )
        fig.update_layout(
            xaxis_title="Model terms",
            yaxis_title="Model terms",
        )
        return fig
