"""
Configuration Management Module
Handles loading configuration from environment variables, files, and defaults.
"""

import os
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import colorlog

class ConfigManager:
    """Manages application configuration from multiple sources"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "config" / "default_config.json"
        self.env_path = self.project_root / ".env"

        # Load environment variables
        load_dotenv(self.env_path)

        # Load configuration
        self.config = self._load_config()

        # Create necessary directories
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables"""
        # Load base configuration from JSON file
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {self.config_path}. Loading defaults.")
            config = self._get_default_config()
            self._save_default_config(config, self.project_root / "config" / "default_config.json")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}. Loading defaults.")
            config = self._get_default_config()

        # Override with environment variables
        config = self._apply_env_overrides(config)

        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        default_config_path = self.project_root / "config" / "default_config.json"
        try:
            with open(default_config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
             logging.warning("Default config file not found or invalid. Using hardcoded defaults.")
             return {
                "version": "2.0.0",
                "target_allocation": {"BTC": 0.40, "ETH": 0.25, "SOL": 0.15, "RNDR": 0.10, "TAO": 0.10},
                "apis": {
                    "coingecko": {"base_url": "https://api.coingecko.com/api/v3", "rate_limit": 100, "timeout": 30},
                    "binance": {"rate_limit": 1200, "timeout": 30, "testnet": False}
                },
                "database": {"path": "data/portfolio.db", "backup_enabled": True, "backup_interval_hours": 24, "cleanup_days": 90},
                "logging": {
                    "level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file_config": {"enabled": True, "path": "logs/portfolio_tracker.log", "max_size_mb": 10, "backup_count": 5},
                    "console_config": {"enabled": True, "colored": True}
                },
                "exports": {"path": "data/exports/", "cleanup": {"enabled": True, "keep_days": 30}},
                "portfolio": {"minimum_value_usd": 1.0, "stablecoin_symbols": ["USDT", "USDC", "BUSD", "DAI"]},
            }

    def _save_default_config(self, config: Dict[str, Any], path: Path):
        """Saves the default config if one doesn't exist."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Saved default config to {path}")
        except Exception as e:
            logging.error(f"Could not save default config to {path}: {e}")

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables"""
        if os.getenv("BINANCE_API_KEY"):
            config.setdefault("api_keys", {})["binance_key"] = os.getenv("BINANCE_API_KEY")
        if os.getenv("BINANCE_API_SECRET"):
            config.setdefault("api_keys", {})["binance_secret"] = os.getenv("BINANCE_API_SECRET")
        if os.getenv("COINGECKO_API_KEY"):
             config.setdefault("api_keys", {})["coingecko_key"] = os.getenv("COINGECKO_API_KEY")

        if os.getenv("LOG_LEVEL"):
            config["logging"]["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_TO_CONSOLE"):
            config["logging"]["console_config"]["enabled"] = os.getenv("LOG_TO_CONSOLE").lower() == "true"
        if os.getenv("LOG_TO_FILE"):
            config["logging"]["file_config"]["enabled"] = os.getenv("LOG_TO_FILE").lower() == "true"

        if os.getenv("DATABASE_PATH"):
            config["database"]["path"] = os.getenv("DATABASE_PATH")

        if os.getenv("EXPORT_PATH"):
            config["exports"]["path"] = os.getenv("EXPORT_PATH")

        boolean_env_vars = {
            "BINANCE_TESTNET": ["apis", "binance", "testnet"],
            "DB_BACKUP_ENABLED": ["database", "backup_enabled"],
            "LOG_COLORED": ["logging", "console_config", "colored"],
            "EXPORT_CLEANUP_ENABLED": ["exports", "cleanup", "enabled"]
        }

        for env_var, keys in boolean_env_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                nested = config
                for key in keys[:-1]:
                    nested = nested.setdefault(key, {})
                nested[keys[-1]] = value.lower() == "true"

        return config

    def _create_directories(self):
        """Create necessary directories based on config"""
        paths_to_create = [
            Path(self.config["database"]["path"]).parent,
            Path(self.config["logging"]["file_config"]["path"]).parent,
            Path(self.config["exports"]["path"])
        ]
        for path in paths_to_create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create directory {path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

def setup_logging(level: str = "INFO", config: Optional[Dict[str, Any]] = None):
    """Setup application logging"""
    if config is None:
        temp_config_manager = ConfigManager()
        config = temp_config_manager.config.get("logging", {})

    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if config.get("console_config", {}).get("enabled", True):
        console_handler: logging.Handler
        if config.get("console_config", {}).get("colored", True):
            handler_format = colorlog.ColoredFormatter(
                f"%(log_color)s{log_format}",
                log_colors={
                    'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                    'ERROR': 'red', 'CRITICAL': 'red,bg_white',
                }
            )
            console_handler = colorlog.StreamHandler()
            console_handler.setFormatter(handler_format)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

    file_config = config.get("file_config", {})
    if file_config.get("enabled", True):
        log_path = Path(file_config.get("path", "logs/portfolio_tracker.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        max_size = file_config.get("max_size_mb", 10) * 1024 * 1024
        backup_count = file_config.get("backup_count", 5)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_size, backupCount=backup_count
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    logging.info("Logging setup complete.")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration for the application"""
    manager = ConfigManager(config_path)
    return manager.config
