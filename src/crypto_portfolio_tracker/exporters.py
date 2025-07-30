"""
Exporters Module
Handles exporting portfolio data to various formats like Excel, HTML, and CSV.
"""

import shutil
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from crypto_portfolio_tracker.utils import clean_export_df


class Exporter:
    """Base class for data exporters."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        export_config = self.config.get("exports", {})
        self.export_dir = Path(export_config.get("path", "data/exports/"))
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, name_prefix: str, extension: str) -> Path:
        """Generate a timestamped filepath for exports."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_{timestamp}.{extension}"
        return self.export_dir / filename

    def _prepare_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for export with proper timestamp handling.
        This incorporates the Web UI's robust timestamp handling logic.
        """
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        export_df = df.copy()
        
        # Handle timezone-aware datetime columns for Excel/CSV compatibility
        for col in export_df.select_dtypes(["datetimetz"]).columns:
            export_df[col] = export_df[col].dt.tz_localize(None)
        
        return export_df

    def export(self, data: Any, **kwargs):
        """Base export method to be overridden by subclasses."""
        raise NotImplementedError


class ExcelExporter(Exporter):
    """Exports data to Excel files with robust timestamp handling."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.excel_config = self.config.get("formats", {}).get("excel", {})

    def export(self, **kwargs):
        """Exports data to Excel with proper timestamp handling."""
        if not self.excel_config.get("enabled", True):
            self.logger.info("Excel export is disabled.")
            return None
        
        transactions_df = kwargs.get("transactions_df")
        holdings_df = kwargs.get("holdings_df")
        data_type = kwargs.get("data_type", "data")  # "transactions", "holdings", or "data"
        
        try:
            if data_type == "transactions" and transactions_df is not None:
                filepath = self._get_filepath("transactions_backup", "xlsx")
                export_df = self._prepare_dataframe_for_export(transactions_df)
                export_df.to_excel(filepath, index=False, engine="openpyxl")
                self.logger.info(f"Transactions Excel exported to: {filepath}")
                return filepath
                
            elif data_type == "holdings" and holdings_df is not None:
                filepath = self._get_filepath("holdings_backup", "xlsx")
                # Apply holdings-specific cleaning
                holdings_df = clean_export_df(holdings_df)
                export_df = self._prepare_dataframe_for_export(holdings_df)
                export_df.to_excel(filepath, index=False, engine="openpyxl")
                self.logger.info(f"Holdings Excel exported to: {filepath}")
                return filepath
                
            else:
                # Original portfolio report logic
                filepath = self._get_filepath("portfolio_report", "xlsx")
                metrics = kwargs.get("metrics", {})
                summary_df = kwargs.get("summary_df")
                
                # Clean DataFrames before export
                if holdings_df is not None:
                    holdings_df = clean_export_df(holdings_df)
                if summary_df is not None:
                    summary_df = clean_export_df(summary_df)
                
                with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
                    if summary_df is not None:
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    if holdings_df is not None:
                        holdings_df.to_excel(writer, sheet_name="Holdings", index=False)
                    pd.DataFrame(
                        {
                            "Metric": list(metrics.keys()),
                            "Value": [
                                str(v)
                                if not isinstance(v, (pd.DataFrame, pd.Series))
                                else "See Sheet"
                                for k, v in metrics.items()
                            ],
                        }
                    ).to_excel(writer, sheet_name="Metrics", index=False)
                self.logger.info(f"Excel report exported successfully to: {filepath}")
                return filepath
                
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            return None


class CsvExporter(Exporter):
    """Exports data to CSV files with robust timestamp handling."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.csv_config = self.config.get("formats", {}).get("csv", {})

    def export(self, **kwargs):
        """Exports data to CSV with proper timestamp handling."""
        if not self.csv_config.get("enabled", True):
            self.logger.info("CSV export is disabled.")
            return None
        
        transactions_df = kwargs.get("transactions_df")
        holdings_df = kwargs.get("holdings_df")
        data_type = kwargs.get("data_type", "data")  # "transactions", "holdings", or "data"
        
        try:
            if data_type == "transactions" and transactions_df is not None:
                filepath = self._get_filepath("transactions_backup", "csv")
                export_df = self._prepare_dataframe_for_export(transactions_df)
                export_df.to_csv(filepath, index=False)
                self.logger.info(f"Transactions CSV exported to: {filepath}")
                return filepath
                
            elif data_type == "holdings" and holdings_df is not None:
                filepath = self._get_filepath("holdings_backup", "csv")
                # Apply holdings-specific cleaning
                holdings_df = clean_export_df(holdings_df)
                export_df = self._prepare_dataframe_for_export(holdings_df)
                export_df.to_csv(filepath, index=False)
                self.logger.info(f"Holdings CSV exported to: {filepath}")
                return filepath
                
            else:
                # Original portfolio report logic
                filepath = self._get_filepath("portfolio_report", "csv")
                metrics = kwargs.get("metrics", {})
                summary_df = kwargs.get("summary_df")
                
                # Clean DataFrames before export
                if holdings_df is not None:
                    holdings_df = clean_export_df(holdings_df)
                if summary_df is not None:
                    summary_df = clean_export_df(summary_df)
                
                # For CSV, we'll create a simple summary
                summary_data = []
                for key, value in metrics.items():
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        summary_data.append([key, "See separate file"])
                    else:
                        summary_data.append([key, str(value)])
                
                summary_df_export = pd.DataFrame(summary_data, columns=["Metric", "Value"])
                summary_df_export.to_csv(filepath, index=False)
                self.logger.info(f"CSV report exported successfully to: {filepath}")
                return filepath
                
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return None


class HtmlExporter(Exporter):
    """Exports data to HTML files."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.html_config = self.config.get("formats", {}).get("html", {})
        templates_path = Path(__file__).parent / "templates"
        templates_path.mkdir(exist_ok=True, parents=True)
        self.jinja_env = Environment(loader=FileSystemLoader(templates_path))

    def export(self, metrics: Dict[str, Any], **kwargs):
        """Exports portfolio metrics and holdings to an HTML file."""
        if not self.html_config.get("enabled", True):
            self.logger.info("HTML export is disabled.")
            return None
        filepath = self._get_filepath("portfolio_report", "html")
        holdings_df = kwargs.get("holdings_df")
        summary_df = kwargs.get("summary_df")
        # --- Clean DataFrames before export ---
        if holdings_df is not None:
            holdings_df = clean_export_df(holdings_df)
        if summary_df is not None:
            summary_df = clean_export_df(summary_df)
        try:
            template = self.jinja_env.get_template("report_template.html")
            html_content = template.render(
                metrics=metrics,
                holdings_table=holdings_df.to_html(
                    index=False, classes="table table-striped"
                )
                if holdings_df is not None
                else "",
                summary_table=summary_df.to_html(
                    index=False, classes="table table-striped"
                )
                if summary_df is not None
                else "",
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            with open(filepath, "w") as f:
                f.write(html_content)
            self.logger.info(f"HTML report exported successfully to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error exporting to HTML: {e}")
            self.logger.warning("Make sure 'templates/report_template.html' exists.")
            return None

    def export_trend_report(self, report: dict, df_export: pd.DataFrame):
        """
        Exports a market trend report to an HTML file using the trend_report_template.html.
        """
        filepath = self._get_filepath("trend_report", "html")
        try:
            template = self.jinja_env.get_template("trend_report_template.html")
            html_content = template.render(
                report=report,
                coin_table=df_export.to_html(index=False, classes="table table-striped")
                if df_export is not None
                else "",
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            with open(filepath, "w") as f:
                f.write(html_content)
            self.logger.info(f"Trend HTML report exported successfully to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error exporting trend report to HTML: {e}")
            return None
