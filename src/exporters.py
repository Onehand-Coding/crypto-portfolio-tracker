"""
Exporters Module
Handles exporting portfolio data to various formats like Excel, HTML, and CSV.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import datetime
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class Exporter:
    """Base class for exporters."""
    def __init__(self, config: Dict[str, Any]):
        self.export_path = Path(config.get("exports", {}).get("path", "data/exports/"))
        self.config = config.get("exports", {})
        self.export_path.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_filepath(self, name_prefix: str, extension: str) -> Path:
        """Generate a timestamped filepath."""
        return self.export_path / f"{name_prefix}_{self.timestamp}.{extension}"

    def export(self, data: Any, **kwargs):
        """Main export method to be implemented by subclasses."""
        raise NotImplementedError

class ExcelExporter(Exporter):
    """Exports data to an Excel file."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config); self.excel_config = self.config.get("formats", {}).get("excel", {})

    def export(self, metrics: Dict[str, Any], **kwargs):
        """Exports portfolio metrics and holdings to an Excel file."""
        if not self.excel_config.get("enabled", True): logger.info("Excel export is disabled."); return
        filepath = self._get_filepath("portfolio_report", "xlsx")
        holdings_df = kwargs.get("holdings_df"); summary_df = kwargs.get("summary_df")
        try:
            with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                if summary_df is not None: summary_df.to_excel(writer, sheet_name='Summary', index=False)
                if holdings_df is not None: holdings_df.to_excel(writer, sheet_name='Holdings', index=False)
                pd.DataFrame({"Metric": list(metrics.keys()), "Value": [str(v) if not isinstance(v, (pd.DataFrame, pd.Series)) else "See Sheet" for k, v in metrics.items()]}).to_excel(writer, sheet_name='Metrics', index=False)
            logger.info(f"Excel report exported successfully to: {filepath}")
        except Exception as e: logger.error(f"Error exporting to Excel: {e}")

class HtmlExporter(Exporter):
    """Exports data to an HTML file."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config); self.html_config = self.config.get("formats", {}).get("html", {})
        templates_path = Path(__file__).parent / "templates"
        templates_path.mkdir(exist_ok=True, parents=True)
        self.jinja_env = Environment(loader=FileSystemLoader(templates_path))

    def export(self, metrics: Dict[str, Any], **kwargs):
        """Exports portfolio metrics and holdings to an HTML file."""
        if not self.html_config.get("enabled", True): logger.info("HTML export is disabled."); return
        filepath = self._get_filepath("portfolio_report", "html")
        holdings_df = kwargs.get("holdings_df"); summary_df = kwargs.get("summary_df")
        try:
            template = self.jinja_env.get_template("report_template.html")
            html_content = template.render(metrics=metrics, holdings_table=holdings_df.to_html(index=False, classes='table table-striped') if holdings_df is not None else "", summary_table=summary_df.to_html(index=False, classes='table table-striped') if summary_df is not None else "", timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            with open(filepath, 'w') as f: f.write(html_content)
            logger.info(f"HTML report exported successfully to: {filepath}")
        except Exception as e: logger.error(f"Error exporting to HTML: {e}"); logger.warning("Make sure 'templates/report_template.html' exists.")

class CsvExporter(Exporter):
    """Exports data to CSV files."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config); self.csv_config = self.config.get("formats", {}).get("csv", {})

    def export(self, **kwargs):
        """Exports holdings, transactions, etc., to CSV files."""
        if not self.csv_config.get("enabled", True): logger.info("CSV export is disabled."); return
        transactions_df = kwargs.get("transactions_df"); holdings_df = kwargs.get("holdings_df")
        try:
            if transactions_df is not None: filepath = self._get_filepath("transactions_backup", "csv"); transactions_df.to_csv(filepath, index=False); logger.info(f"Transactions CSV exported to: {filepath}")
            if holdings_df is not None: filepath = self._get_filepath("holdings_backup", "csv"); holdings_df.to_csv(filepath, index=False); logger.info(f"Holdings CSV exported to: {filepath}")
        except Exception as e: logger.error(f"Error exporting to CSV: {e}")
