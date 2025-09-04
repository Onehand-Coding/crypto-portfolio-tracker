"""
Report Generator - Handles all reporting and visualization operations
Moved from CryptoPortfolioTracker to separate concerns.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from .visualizations import Visualizer
from .exporters import ExcelExporter, HtmlExporter, CsvExporter


class ReportGenerator:
    """
    Handles all reporting and visualization operations including:
    - Portfolio chart creation
    - Data export in multiple formats
    - Trend report generation
    - Backup data export
    """

    def __init__(self, config: Dict[str, Any], db_manager=None,
                 visualizer: Optional[Visualizer] = None,
                 excel_exporter: Optional[ExcelExporter] = None,
                 html_exporter: Optional[HtmlExporter] = None,
                 csv_exporter: Optional[CsvExporter] = None):
        """
        Initialize ReportGenerator with necessary dependencies.

        Args:
            config: Configuration dictionary
            db_manager: Database manager instance
            visualizer: Visualizer instance for chart generation
            excel_exporter: ExcelExporter instance
            html_exporter: HtmlExporter instance
            csv_exporter: CsvExporter instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.visualizer = visualizer
        self.excel_exporter = excel_exporter
        self.html_exporter = html_exporter
        self.csv_exporter = csv_exporter

    def create_portfolio_charts(self, chart_type: str, metrics: Dict[str, Any]) -> bool:
        """
        Create a specific chart based on the chart type using unified visualizer.

        Args:
            chart_type: Type of chart to create ('allocation_pie', 'allocation_comparison', 'pl_by_asset', 'value_history')
            metrics: Portfolio metrics dictionary

        Returns:
            bool: True if chart was created successfully, False otherwise
        """
        if not self.visualizer:
            self.logger.error("Visualizer not available for chart creation")
            return False

        holdings_df = metrics.get("holdings_df")
        if holdings_df is None:
            self.logger.warning("No holdings data for chart generation.")
            return False

        try:
            snapshots_df = self.db_manager.get_all_snapshots() if self.db_manager else pd.DataFrame()
            target_alloc = self.config.get("target_allocation", {})

            if chart_type == "allocation_pie":
                self.visualizer.create_portfolio_allocation_pie(
                    holdings_df, metrics, save_to_disk=True
                )
                return True
            elif chart_type == "allocation_comparison":
                self.visualizer.create_allocation_comparison_bar(
                    holdings_df, target_alloc, save_to_disk=True
                )
                return True
            elif chart_type == "pl_by_asset":
                self.visualizer.create_pl_by_asset_bar(holdings_df, save_to_disk=True)
                return True
            elif chart_type == "value_history":
                self.visualizer.create_portfolio_value_history(
                    snapshots_df, save_to_disk=True
                )
                return True
            else:
                self.logger.error(f"Unknown chart type: {chart_type}")
                return False
        except Exception as e:
            self.logger.error(f"Error creating chart {chart_type}: {e}")
            return False

    def create_portfolio_charts_all(self, metrics: Dict[str, Any]):
        """Generate all portfolio charts."""
        if not self.visualizer:
            self.logger.error("Visualizer not available for chart creation")
            return

        holdings_df = metrics.get("holdings_df")
        if holdings_df is not None:
            snapshots_df = self.db_manager.get_all_snapshots() if self.db_manager else pd.DataFrame()
            target_alloc = self.config.get("target_allocation", {})

            self.visualizer.generate_all_charts(
                holdings_df, metrics, target_alloc, snapshots_df
            )
        else:
            self.logger.warning("No holdings data for chart generation.")

    def export_portfolio_summary(self, metrics: Dict[str, Any], format: str):
        """Export portfolio summary to specified format."""
        format_lower = format.lower()

        if format_lower == "html" and self.html_exporter:
            self.html_exporter.export(
                metrics=metrics, holdings_df=metrics.get("holdings_df")
            )
        elif format_lower == "excel" and self.excel_exporter:
            self.excel_exporter.export(
                metrics=metrics, holdings_df=metrics.get("holdings_df")
            )
        elif format_lower == "csv" and self.csv_exporter:
            self.csv_exporter.export(
                metrics=metrics, holdings_df=metrics.get("holdings_df")
            )
        else:
            if format_lower == "html" and not self.html_exporter:
                raise ValueError("HTML exporter not available")
            elif format_lower == "excel" and not self.excel_exporter:
                raise ValueError("Excel exporter not available")
            elif format_lower == "csv" and not self.csv_exporter:
                raise ValueError("CSV exporter not available")
            else:
                raise ValueError(f"Unsupported export format: {format}")

    def export_portfolio_summary_all_formats(self, metrics: Dict[str, Any]):
        """Export portfolio summary to all available formats."""
        available_formats = []
        if self.html_exporter:
            available_formats.append("html")
        if self.excel_exporter:
            available_formats.append("excel")
        if self.csv_exporter:
            available_formats.append("csv")

        for format_type in available_formats:
            try:
                self.export_portfolio_summary(metrics, format_type)
            except Exception as e:
                self.logger.error(f"Failed to export {format_type}: {e}")

    def export_data_backup(self, data_type: str, format: str):
        """Export transactions/holdings to backup file with robust timestamp handling."""
        if not self.db_manager:
            raise ValueError("Database manager not available for data backup")

        format_lower = format.lower()

        if data_type == "transactions" and format_lower == "csv" and self.csv_exporter:
            return self.csv_exporter.export(
                transactions_df=self.db_manager.get_all_transactions(),
                data_type="transactions",
            )
        elif data_type == "holdings" and format_lower == "csv" and self.csv_exporter:
            return self.csv_exporter.export(
                holdings_df=self.db_manager.get_holdings(),
                data_type="holdings"
            )
        elif data_type == "transactions" and format_lower == "excel" and self.excel_exporter:
            return self.excel_exporter.export(
                transactions_df=self.db_manager.get_all_transactions(),
                data_type="transactions",
            )
        elif data_type == "holdings" and format_lower == "excel" and self.excel_exporter:
            return self.excel_exporter.export(
                holdings_df=self.db_manager.get_holdings(),
                data_type="holdings"
            )
        else:
            if data_type not in ["transactions", "holdings"]:
                raise ValueError(f"Unsupported data type: {data_type}")
            elif format_lower == "csv" and not self.csv_exporter:
                raise ValueError("CSV exporter not available")
            elif format_lower == "excel" and not self.excel_exporter:
                raise ValueError("Excel exporter not available")
            else:
                raise ValueError(f"Unsupported format: {format}")

    def export_all_data_backups(self):
        """Export all data backups to all available formats with robust timestamp handling."""
        if not self.db_manager:
            self.logger.error("Database manager not available for data backup")
            return {}

        results = {}
        data_types = ["transactions", "holdings"]
        formats = []

        if self.csv_exporter:
            formats.append("csv")
        if self.excel_exporter:
            formats.append("excel")

        for data_type in data_types:
            for format_type in formats:
                try:
                    result = self.export_data_backup(data_type, format_type)
                    results[f"{data_type}_{format_type}"] = result
                except Exception as e:
                    self.logger.error(f"Failed to export {data_type} as {format_type}: {e}")
                    results[f"{data_type}_{format_type}"] = None

        return results

    def export_trend_report(
        self, report: Dict[str, Any], timeframe: str, export_format: str = "HTML"
    ) -> Optional[Path]:
        """
        Exports a trend analysis report to various formats.

        Args:
            report: The trend analysis report dictionary from CryptoTrendAnalyzer
            timeframe: The timeframe of the analysis (e.g., 'long_term', 'swing', 'day')
            export_format: The export format ('CSV', 'JSON', 'HTML')

        Returns:
            Path to the exported file, or None if export failed
        """
        try:
            # Get export directory from config
            export_dir = Path(
                self.config.get("exports", {}).get("path", "data/exports/")
            )
            export_dir.mkdir(parents=True, exist_ok=True)

            # Create DataFrame from coin analyses for export
            coin_analyses = report.get("coin_analyses", {})
            df_export = pd.DataFrame(
                [
                    {
                        "Symbol": symbol,
                        "Price": analysis.get("current_price", 0),
                        "Change (%)": analysis.get("price_change_pct", 0),
                        "RSI": analysis.get("rsi", 0),
                        "Support": analysis.get("support_level", 0),
                        "Resistance": analysis.get("resistance_level", 0),
                        "Active Conditions": ", ".join(
                            analysis.get("active_conditions", [])
                        ),
                    }
                    for symbol, analysis in coin_analyses.items()
                ]
            )

            # Generate timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_report_{timeframe}_{timestamp}.{export_format.lower()}"
            exported_file = export_dir / filename

            if export_format.upper() == "CSV":
                df_export.to_csv(exported_file, index=False)
                self.logger.info(f"Trend report exported to CSV: {exported_file}")

            elif export_format.upper() == "JSON":
                with open(exported_file, "w") as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Trend report exported to JSON: {exported_file}")

            elif export_format.upper() == "HTML":
                if not self.html_exporter:
                    self.logger.error("HTML exporter not available for trend report")
                    return None

                exported_file = self.html_exporter.export_trend_report(
                    report, df_export
                )
                if exported_file:
                    self.logger.info(f"Trend report exported to HTML: {exported_file}")
                else:
                    self.logger.error("Failed to export trend report to HTML")
                    return None
            else:
                self.logger.error(f"Unsupported export format: {export_format}")
                return None

            return exported_file

        except Exception as e:
            self.logger.error(f"Error exporting trend report: {e}", exc_info=True)
            return None

    def export_trend_report_all_formats(
        self, report: Dict[str, Any], timeframe: str
    ) -> Dict[str, Optional[Path]]:
        """
        Exports a trend analysis report to all available formats.

        Args:
            report: The trend analysis report dictionary
            timeframe: The timeframe of the analysis

        Returns:
            Dictionary mapping format to exported file path
        """
        results = {}
        available_formats = ["CSV", "JSON"]

        if self.html_exporter:
            available_formats.append("HTML")

        for format_type in available_formats:
            try:
                file_path = self.export_trend_report(report, timeframe, format_type)
                results[format_type] = file_path
            except Exception as e:
                self.logger.error(f"Failed to export {format_type}: {e}")
                results[format_type] = None

        return results

    def cleanup_old_data(self):
        """Clean up old data using the database manager."""
        if self.db_manager:
            self.db_manager.cleanup_old_data()
        else:
            self.logger.warning("Database manager not available for cleanup")
