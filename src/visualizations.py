"""
Visualizations Module
Handles creating charts and graphs for portfolio analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Visualizer:
    """Manages creation of portfolio visualizations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualizer."""
        self.config = config.get("visualization", {})
        self.export_path = Path(config.get("exports", {}).get("path", "data/exports/"))
        self.chart_style = self.config.get("chart_style", "seaborn-v0_8")
        self.color_palette = self.config.get("color_palette", "husl")
        self.figure_size = self.config.get("figure_size", [15, 12])
        self.dpi = self.config.get("dpi", 300)
        self.formats = self.config.get("formats", ["png"])
        plt.style.use(self.chart_style)
        sns.set_palette(self.color_palette)
        logger.info("Visualizer initialized.")

    def _save_chart(self, fig, filename_prefix: str):
        """Saves the figure in configured formats."""
        base_path = self.export_path / filename_prefix
        for fmt in self.formats:
            filepath = base_path.with_suffix(f".{fmt}")
            try: fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight'); logger.info(f"Chart saved to: {filepath}")
            except Exception as e: logger.error(f"Failed to save chart {filepath}: {e}")
        plt.close(fig)

    def create_portfolio_allocation_pie(self, holdings_df: pd.DataFrame, metrics: Dict[str, Any]):
        """Creates a pie chart showing portfolio allocation by value."""
        if holdings_df.empty or 'value_usd' not in holdings_df.columns: logger.warning("Cannot create allocation pie: No holdings data."); return
        fig, ax = plt.subplots(figsize=(self.figure_size[0]/2, self.figure_size[1]/2)); data = holdings_df.set_index('symbol')['value_usd']
        wedges, texts, autotexts = ax.pie(data, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        ax.set_title('Portfolio Allocation (by Value)', fontsize=16, pad=20); fig.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
        ax.legend(wedges, data.index, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)); plt.setp(autotexts, size=8, weight="bold", color="white"); ax.axis('equal')
        self._save_chart(fig, "portfolio_allocation_pie")

    def create_allocation_comparison_bar(self, holdings_df: pd.DataFrame, target_allocation: Dict[str, float]):
        """Creates a bar chart comparing current vs. target allocation."""
        if holdings_df.empty or not target_allocation: logger.warning("Cannot create allocation comparison: No holdings or target."); return
        current_alloc = holdings_df.set_index('symbol')['allocation'] * 100; target_alloc = pd.Series(target_allocation) * 100
        comparison_df = pd.DataFrame({'Current (%)': current_alloc, 'Target (%)': target_alloc}).fillna(0)
        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1]/2)); comparison_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Current vs. Target Portfolio Allocation', fontsize=16, pad=20); ax.set_ylabel('Allocation (%)'); ax.set_xlabel('Assets')
        ax.tick_params(axis='x', rotation=45); ax.legend(title='Allocation Type'); ax.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        self._save_chart(fig, "allocation_comparison_bar")

    def create_pl_by_asset_bar(self, holdings_df: pd.DataFrame):
        """Creates a bar chart showing P/L per asset."""
        if holdings_df.empty or 'unrealized_pl_usd' not in holdings_df.columns: logger.warning("Cannot create P/L bar: No P/L data."); return
        data = holdings_df.set_index('symbol')['unrealized_pl_usd'].sort_values(); colors = ['red' if x < 0 else 'green' for x in data]
        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1]/2)); data.plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Unrealized Profit/Loss (P/L) by Asset', fontsize=16, pad=20); ax.set_ylabel('P/L (USD)'); ax.set_xlabel('Assets')
        ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', linestyle='--', alpha=0.7); ax.axhline(0, color='black', linewidth=0.8); plt.tight_layout()
        self._save_chart(fig, "pl_by_asset_bar")

    def create_portfolio_value_history(self, snapshots_df: pd.DataFrame):
        """Creates a line chart showing portfolio value over time."""
        if snapshots_df.empty: logger.warning("Cannot create value history: No snapshot data."); return
        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1]/2)); snapshots_df.plot(kind='line', y='total_value_usd', ax=ax, marker='o')
        ax.set_title('Portfolio Value Over Time', fontsize=16, pad=20); ax.set_ylabel('Total Value (USD)'); ax.set_xlabel('Date')
        ax.tick_params(axis='x', rotation=45); ax.grid(True, linestyle='--', alpha=0.7); ax.legend().set_visible(False); plt.tight_layout()
        self._save_chart(fig, "portfolio_value_history")

    def generate_all_charts(self, holdings_df: pd.DataFrame, metrics: Dict[str, Any], target_allocation: Dict[str, float], snapshots_df: pd.DataFrame):
        """Generates all configured charts."""
        logger.info("Generating all charts...")
        self.create_portfolio_allocation_pie(holdings_df, metrics); self.create_allocation_comparison_bar(holdings_df, target_allocation)
        self.create_pl_by_asset_bar(holdings_df); self.create_portfolio_value_history(snapshots_df)
        logger.info("Chart generation complete.")
