#! /usr/bin/env python3
"""
Visualizations Module
Handles creating charts and graphs for portfolio analysis.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.patheffects as path_effects


class Visualizer:
    """Manages creation of portfolio visualizations with unified styling."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)
        self.config = config.get("visualization", {})
        self.export_path = (
            Path(config.get("exports", {}).get("path", "data/exports/")) / "charts"
        )
        self.export_path.mkdir(exist_ok=True, parents=True)
        self.chart_style = self.config.get("chart_style", "seaborn-v0_8")
        self.color_palette = self.config.get("color_palette", "husl")
        self.figure_size = self.config.get("figure_size", [15, 12])
        self.dpi = self.config.get("dpi", 300)
        self.formats = self.config.get("formats", ["png"])

        # Define consistent color palette for both Plotly and Matplotlib
        self.colors = {
            "primary": "#1f77b4",  # Blue
            "secondary": "#ff7f0e",  # Orange
            "success": "#2ca02c",  # Green
            "danger": "#d62728",  # Red
            "purple": "#9467bd",  # Purple
            "brown": "#8c564b",  # Brown
            "pink": "#e377c2",  # Pink
            "gray": "#7f7f7f",  # Gray
            "olive": "#bcbd22",  # Olive
            "cyan": "#17becf",  # Cyan
        }

        plt.style.use(self.chart_style)
        sns.set_palette(self.color_palette)
        self.logger.debug("Visualizer initialized.")

    def _save_chart(self, fig, filename_prefix: str):
        """Saves the figure in configured formats to disk."""
        # Add timestamp to filename to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.export_path / f"{filename_prefix}_{timestamp}"
        for fmt in self.formats:
            filepath = base_path.with_suffix(f".{fmt}")
            try:
                fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
                self.logger.debug(f"Chart saved to: {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save chart {filepath}: {e}")
        plt.close(fig)

    def _get_chart_bytes(self, fig, format: str = "png") -> bytes:
        """Returns chart as bytes for download."""
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format=format, dpi=self.dpi, bbox_inches="tight")
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            self.logger.error(f"Failed to convert chart to bytes: {e}")
            return b""
        finally:
            plt.close(fig)

    def create_portfolio_allocation_pie(
        self,
        holdings_df: pd.DataFrame,
        metrics: Dict[str, Any],
        save_to_disk: bool = True,
        return_bytes: bool = False,
    ) -> Optional[bytes]:
        """Creates a pie chart showing portfolio allocation by value."""
        if holdings_df.empty or "value_usd" not in holdings_df.columns:
            self.logger.warning("Cannot create allocation pie: No holdings data.")
            return None

        # Filter out zero or very small values that would show as 0.0%
        min_threshold = holdings_df["value_usd"].sum() * 0.005  # 0.5% minimum
        significant_holdings = holdings_df[
            holdings_df["value_usd"] >= min_threshold
        ].copy()

        # Group small holdings into "Others" category if any exist
        small_holdings = holdings_df[holdings_df["value_usd"] < min_threshold]
        if not small_holdings.empty:
            others_value = small_holdings["value_usd"].sum()
            others_row = pd.DataFrame(
                {"symbol": ["Others"], "value_usd": [others_value]}
            )
            significant_holdings = pd.concat(
                [significant_holdings, others_row], ignore_index=True
            )

        # Make it larger for better readability
        fig, ax = plt.subplots(figsize=(16, 14))  # Even larger size

        data = significant_holdings.set_index("symbol")["value_usd"]
        total_value = data.sum()

        # Calculate percentages manually for better control
        percentages = (data / total_value * 100).round(1)

        # Use consistent colors
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["success"],
            self.colors["danger"],
            self.colors["purple"],
            self.colors["brown"],
            self.colors["pink"],
            self.colors["gray"],
            self.colors["olive"],
            self.colors["cyan"],
        ]

        # Custom autopct function to ensure proper percentage display
        def make_autopct(values):
            def my_autopct(pct):
                if pct < 1.0:  # For very small slices, don't show percentage
                    return ""
                return f"{pct:.1f}%"

            return my_autopct

        wedges, texts, autotexts = ax.pie(
            data,
            autopct=make_autopct(data),
            startangle=90,
            pctdistance=0.75,  # Move percentages closer to edge for better readability
            colors=colors[: len(data)],
            textprops={
                "fontsize": 16,
                "weight": "bold",
                "color": "white",
            },  # Larger, bold white text
            labeldistance=1.1,  # Move labels further from pie
        )

        # Set title with larger font
        ax.set_title(
            "Portfolio Allocation (by Value)", fontsize=24, pad=30, weight="bold"
        )

        # Create custom legend with values and percentages
        legend_labels = []
        for symbol, value, pct in zip(data.index, data.values, percentages):
            legend_labels.append(f"{symbol}: ${value:,.0f} ({pct:.1f}%)")

        # Position legend to the right side instead of bottom
        legend = ax.legend(
            wedges,
            legend_labels,
            title="Assets (Value & %)",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),  # Position to the right
            fontsize=14,
            title_fontsize=16,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Make percentage text more readable with better contrast
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(16)
            autotext.set_weight("bold")
            # Add subtle outline for better readability
            autotext.set_path_effects(
                [plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")]
            )

        # Make slice labels more readable
        for text in texts:
            text.set_fontsize(14)
            text.set_weight("bold")
            text.set_color("black")

        ax.axis("equal")

        # Adjust layout to accommodate the right-side legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.65)  # Make room for legend on right

        if save_to_disk:
            self._save_chart(fig, "portfolio_allocation_pie")

        if return_bytes:
            return self._get_chart_bytes(fig, "png")

        return None

        def create_interactive_allocation_pie(
            self, holdings_df: pd.DataFrame
        ) -> go.Figure:
            """Creates an interactive pie chart for web UI."""
            if holdings_df.empty or "value_usd" not in holdings_df.columns:
                return go.Figure()

            # Apply same filtering logic as static version
            min_threshold = holdings_df["value_usd"].sum() * 0.005  # 0.5% minimum
            significant_holdings = holdings_df[
                holdings_df["value_usd"] >= min_threshold
            ].copy()

            # Group small holdings into "Others" category if any exist
            small_holdings = holdings_df[holdings_df["value_usd"] < min_threshold]
            if not small_holdings.empty:
                others_value = small_holdings["value_usd"].sum()
                others_row = pd.DataFrame(
                    {"symbol": ["Others"], "value_usd": [others_value]}
                )
                significant_holdings = pd.concat(
                    [significant_holdings, others_row], ignore_index=True
                )

            data = significant_holdings.set_index("symbol")["value_usd"]

            # Use the same colors as static version
            colors = [
                self.colors["primary"],
                self.colors["secondary"],
                self.colors["success"],
                self.colors["danger"],
                self.colors["purple"],
                self.colors["brown"],
                self.colors["pink"],
                self.colors["gray"],
                self.colors["olive"],
                self.colors["cyan"],
            ]

            # Create custom hover text with both values and percentages
            total_value = data.sum()
            hover_text = []
            for symbol, value in zip(data.index, data.values):
                pct = value / total_value * 100
                hover_text.append(
                    f"{symbol}<br>Value: ${value:,.0f}<br>Percentage: {pct:.1f}%"
                )

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=data.index,
                        values=data.values,
                        hole=0,
                        textinfo="label+percent",
                        textposition="auto",  # Let Plotly choose best position
                        textfont=dict(size=14, color="white"),  # Larger, white text
                        marker=dict(
                            colors=colors[: len(data)],
                            line=dict(
                                color="white", width=2
                            ),  # Add white borders between slices
                        ),
                        hovertext=hover_text,
                        hoverinfo="text",  # Use custom hover text
                    )
                ]
            )

            fig.update_layout(
                title=dict(
                    text="Portfolio Allocation (by Value)",
                    font=dict(size=20, family="Arial Black"),
                ),
                showlegend=True,
                legend=dict(
                    orientation="v",  # Vertical legend
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,  # Position to the right
                    font=dict(size=12),
                ),
                height=600,
                width=900,  # Make wider to accommodate legend
                margin=dict(r=200),  # Add right margin for legend
            )

            return fig

    def create_allocation_comparison_bar(
        self,
        holdings_df: pd.DataFrame,
        target_allocation: Dict[str, float],
        save_to_disk: bool = True,
        return_bytes: bool = False,
    ) -> Optional[bytes]:
        """Creates a bar chart comparing current vs. target allocation."""
        if holdings_df.empty or not target_allocation:
            self.logger.warning(
                "Cannot create allocation comparison: No holdings or target."
            )
            return None

        current_alloc = holdings_df.set_index("symbol")["allocation"] * 100
        target_alloc = pd.Series(target_allocation) * 100
        comparison_df = pd.DataFrame(
            {"Current (%)": current_alloc, "Target (%)": target_alloc}
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1] / 2))

        # Use consistent colors
        comparison_df.plot(
            kind="bar",
            ax=ax,
            width=0.8,
            color=[self.colors["primary"], self.colors["secondary"]],
        )

        ax.set_title("Current vs. Target Portfolio Allocation", fontsize=16, pad=20)
        ax.set_ylabel("Allocation (%)")
        ax.set_xlabel("Assets")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Allocation Type")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_to_disk:
            self._save_chart(fig, "allocation_comparison_bar")

        if return_bytes:
            return self._get_chart_bytes(fig, "png")

        return None

    def create_pl_by_asset_bar(
        self,
        holdings_df: pd.DataFrame,
        save_to_disk: bool = True,
        return_bytes: bool = False,
    ) -> Optional[bytes]:
        """Creates a bar chart showing P/L per asset."""
        if holdings_df.empty or "unrealized_pl_usd" not in holdings_df.columns:
            self.logger.warning("Cannot create P/L bar: No P/L data.")
            return None

        data = holdings_df.set_index("symbol")["unrealized_pl_usd"].sort_values()
        colors = [
            self.colors["danger"] if x < 0 else self.colors["success"] for x in data
        ]

        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1] / 2))
        data.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Unrealized Profit/Loss (P/L) by Asset", fontsize=16, pad=20)
        ax.set_ylabel("P/L (USD)")
        ax.set_xlabel("Assets")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()

        if save_to_disk:
            self._save_chart(fig, "pl_by_asset_bar")

        if return_bytes:
            return self._get_chart_bytes(fig, "png")

        return None

    def create_portfolio_value_history(
        self,
        snapshots_df: pd.DataFrame,
        save_to_disk: bool = True,
        return_bytes: bool = False,
    ) -> Optional[bytes]:
        """Creates a line chart showing portfolio value over time."""
        if snapshots_df.empty:
            self.logger.warning("Cannot create value history: No snapshot data.")
            return None

        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1] / 2))

        # Use consistent blue color
        snapshots_df.plot(
            kind="line",
            y="total_value_usd",
            ax=ax,
            marker="o",
            color=self.colors["primary"],
            linewidth=2,
            markersize=6,
        )

        ax.set_title("Portfolio Value Over Time", fontsize=16, pad=20)
        ax.set_ylabel("Total Value (USD)")
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend().set_visible(False)

        # Ensure y-axis starts from 0
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save_to_disk:
            self._save_chart(fig, "portfolio_value_history")

        if return_bytes:
            return self._get_chart_bytes(fig, "png")

        return None

    # New methods for interactive charts (Plotly)
    def create_interactive_allocation_pie(self, holdings_df: pd.DataFrame) -> go.Figure:
        """Creates an interactive pie chart for web UI."""
        if holdings_df.empty or "value_usd" not in holdings_df.columns:
            return go.Figure()

        data = holdings_df.set_index("symbol")["value_usd"]

        # Use the same colors as static version
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["success"],
            self.colors["danger"],
            self.colors["purple"],
            self.colors["brown"],
            self.colors["pink"],
            self.colors["gray"],
            self.colors["olive"],
            self.colors["cyan"],
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=data.index,
                    values=data.values,
                    hole=0,  # No hole
                    textinfo="label+percent",
                    textposition="inside",
                    marker=dict(colors=colors[: len(data)]),
                )
            ]
        )
        fig.update_layout(
            title="Portfolio Allocation (by Value)",
            showlegend=True,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=-0.2,  # Move legend below the chart
                xanchor="center",
                x=0.5,  # Center the legend
            ),
            height=600,  # Make the chart taller
        )
        return fig

    def create_allocation_comparison_bar(
        self,
        holdings_df: pd.DataFrame,
        target_allocation: Dict[str, float],
        save_to_disk: bool = True,
        return_bytes: bool = False,
    ) -> Optional[bytes]:
        """Creates a bar chart comparing current vs. target allocation."""
        if holdings_df.empty or not target_allocation:
            self.logger.warning(
                "Cannot create allocation comparison: No holdings or target."
            )
            return None

        current_alloc = holdings_df.set_index("symbol")["allocation"] * 100
        target_alloc = pd.Series(target_allocation) * 100
        comparison_df = pd.DataFrame(
            {"Current (%)": current_alloc, "Target (%)": target_alloc}
        ).fillna(0)

        # Make chart larger for better readability
        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1] * 0.8))

        # Use consistent colors with better styling
        bars = comparison_df.plot(
            kind="bar",
            ax=ax,
            width=0.7,
            color=[self.colors["primary"], self.colors["secondary"]],
            edgecolor="white",
            linewidth=1.5,
        )

        # Enhanced title and labels
        ax.set_title(
            "Current vs. Target Portfolio Allocation",
            fontsize=18,
            pad=25,
            weight="bold",
        )
        ax.set_ylabel("Allocation (%)", fontsize=14, weight="bold")
        ax.set_xlabel("Assets", fontsize=14, weight="bold")

        # Improve tick formatting
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        # Enhanced legend
        legend = ax.legend(
            title="Allocation Type",
            fontsize=12,
            title_fontsize=13,
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Better grid styling
        ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
        ax.set_axisbelow(True)  # Put grid behind bars

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f%%",
                fontsize=10,
                weight="bold",
                padding=3,
                color="black",
            )

        # Set y-axis to start from 0 with proper range
        y_max = max(comparison_df.max()) * 1.1
        ax.set_ylim(0, y_max)

        plt.tight_layout()

        if save_to_disk:
            self._save_chart(fig, "allocation_comparison_bar")

        if return_bytes:
            return self._get_chart_bytes(fig, "png")

        return None

    def create_interactive_allocation_comparison(
        self, holdings_df: pd.DataFrame, target_allocation: Dict[str, float]
    ) -> go.Figure:
        """Creates an interactive bar chart for web UI."""
        if holdings_df.empty or not target_allocation:
            return go.Figure()

        current_alloc = holdings_df.set_index("symbol")["allocation"] * 100
        target_alloc = pd.Series(target_allocation) * 100
        comparison_df = pd.DataFrame(
            {"Current (%)": current_alloc, "Target (%)": target_alloc}
        ).fillna(0)

        fig = go.Figure()

        # Add Current allocation bars
        fig.add_trace(
            go.Bar(
                x=comparison_df.index,
                y=comparison_df["Current (%)"],
                name="Current (%)",
                marker_color=self.colors["primary"],
                marker_line=dict(color="white", width=1.5),
                text=[f"{val:.1f}%" for val in comparison_df["Current (%)"]],
                textposition="outside",
                textfont=dict(size=12, color="white"),
            )
        )

        # Add Target allocation bars
        fig.add_trace(
            go.Bar(
                x=comparison_df.index,
                y=comparison_df["Target (%)"],
                name="Target (%)",
                marker_color=self.colors["secondary"],
                marker_line=dict(color="white", width=1.5),
                text=[f"{val:.1f}%" for val in comparison_df["Target (%)"]],
                textposition="outside",
                textfont=dict(size=12, color="white"),
            )
        )

        fig.update_layout(
            title=dict(
                text="Current vs. Target Portfolio Allocation",
                font=dict(size=18, family="Arial Black"),
            ),
            barmode="group",
            xaxis_title="Assets",
            yaxis_title="Allocation (%)",
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
            height=500,
            showlegend=True,
        )

        return fig

    def create_interactive_pl_by_asset(self, holdings_df: pd.DataFrame) -> go.Figure:
        """Creates an interactive P/L bar chart for web UI."""
        if holdings_df.empty or "unrealized_pl_usd" not in holdings_df.columns:
            return go.Figure()

        data = holdings_df.set_index("symbol")["unrealized_pl_usd"].sort_values()
        colors = [
            self.colors["danger"] if x < 0 else self.colors["success"] for x in data
        ]

        # Create custom hover text
        hover_text = [
            f"{symbol}<br>P/L: ${value:,.0f}"
            for symbol, value in zip(data.index, data.values)
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=data.index,
                    y=data.values,
                    marker_color=colors,
                    marker_line=dict(color="white", width=1.5),
                    text=[f"${val:,.0f}" for val in data.values],
                    textposition="outside",
                    textfont=dict(size=12, color="white"),
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text="Unrealized Profit/Loss (P/L) by Asset",
                font=dict(size=18, family="Arial Black"),
            ),
            xaxis_title="Assets",
            yaxis_title="P/L (USD)",
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(
                tickfont=dict(size=12),
                zeroline=True,
                zerolinecolor="black",
                zerolinewidth=2,
                tickformat="$,.0f",
            ),
            height=500,
            showlegend=False,
        )

        return fig

    def create_interactive_value_history(self, snapshots_df: pd.DataFrame) -> go.Figure:
        """Creates an interactive line chart for web UI."""
        if snapshots_df.empty:
            return go.Figure()

        # Check if we have a timestamp index or column
        if isinstance(snapshots_df.index, pd.DatetimeIndex):
            dates = snapshots_df.index
        elif "timestamp" in snapshots_df.columns:
            dates = pd.to_datetime(snapshots_df["timestamp"])
        else:
            dates = list(range(len(snapshots_df)))

        # Filter out any NaN values
        valid_mask = ~pd.isna(snapshots_df["total_value_usd"])
        if valid_mask.sum() == 0:
            return go.Figure()

        if isinstance(dates, pd.Series) or isinstance(dates, pd.DatetimeIndex):
            valid_dates = dates[valid_mask]
        else:
            valid_dates = [dates[i] for i in range(len(dates)) if valid_mask.iloc[i]]

        valid_values = snapshots_df["total_value_usd"][valid_mask]

        # Format dates for display if they are datetime
        if isinstance(valid_dates, pd.DatetimeIndex):
            x_values = valid_dates.strftime("%Y-%m-%d")
            xaxis_title = "Date"
        else:
            x_values = valid_dates
            xaxis_title = "Snapshot Number"

        # Calculate y-axis range
        y_min = 0
        y_max = valid_values.max()
        y_max_rounded = ((y_max // 20) + 1) * 20

        # Create custom hover text
        hover_text = [
            f"Date: {date}<br>Value: ${value:,.0f}"
            for date, value in zip(x_values, valid_values)
        ]

        fig = go.Figure()

        # Add area fill
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=valid_values,
                mode="none",
                fill="tozeroy",
                fillcolor=f"rgba({int(self.colors['primary'][1:3], 16)}, {int(self.colors['primary'][3:5], 16)}, {int(self.colors['primary'][5:7], 16)}, 0.2)",
                name="Portfolio Value Area",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add line with markers
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=valid_values,
                mode="lines+markers",
                name="Portfolio Value",
                line=dict(color=self.colors["primary"], width=3),
                marker=dict(
                    color="white",
                    size=8,
                    line=dict(color=self.colors["primary"], width=2),
                ),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

        fig.update_layout(
            title=dict(
                text="Portfolio Value Over Time",
                font=dict(size=18, family="Arial Black"),
            ),
            xaxis_title=xaxis_title,
            yaxis_title="Total Value (USD)",
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=12),
                tickmode="auto",
                nticks=min(10, len(x_values)),
            ),
            yaxis=dict(
                range=[y_min, y_max_rounded],
                zeroline=True,
                zerolinecolor="lightgray",
                tickmode="linear",
                dtick=20,
                tickfont=dict(size=12),
                tickformat="$,.0f",
            ),
            hovermode="x unified",
            height=500,
            showlegend=False,
        )

        return fig

    def generate_all_charts(
        self,
        holdings_df: pd.DataFrame,
        metrics: Dict[str, Any],
        target_allocation: Dict[str, float],
        snapshots_df: pd.DataFrame,
        save_to_disk: bool = True,
    ):
        """Generates all configured charts."""
        self.logger.info("Generating all charts...")
        self.create_portfolio_allocation_pie(holdings_df, metrics, save_to_disk)
        self.create_allocation_comparison_bar(
            holdings_df, target_allocation, save_to_disk
        )
        self.create_pl_by_asset_bar(holdings_df, save_to_disk)
        self.create_portfolio_value_history(snapshots_df, save_to_disk)
        self.logger.info("Chart generation complete.")

    def get_chart_bytes(
        self, chart_type: str, data: Dict[str, Any], format: str = "png"
    ) -> Optional[bytes]:
        """Get chart as bytes for download."""
        holdings_df = data.get("holdings_df")
        snapshots_df = data.get("snapshots_df")
        target_allocation = data.get("target_allocation", {})

        if chart_type == "allocation_pie":
            return self.create_portfolio_allocation_pie(
                holdings_df, data, save_to_disk=False, return_bytes=True
            )
        elif chart_type == "allocation_comparison":
            return self.create_allocation_comparison_bar(
                holdings_df, target_allocation, save_to_disk=False, return_bytes=True
            )
        elif chart_type == "pl_by_asset":
            return self.create_pl_by_asset_bar(
                holdings_df, save_to_disk=False, return_bytes=True
            )
        elif chart_type == "value_history":
            return self.create_portfolio_value_history(
                snapshots_df, save_to_disk=False, return_bytes=True
            )
        else:
            self.logger.error(f"Unknown chart type: {chart_type}")
            return None
