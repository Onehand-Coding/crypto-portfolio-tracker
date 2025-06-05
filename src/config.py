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
        self.config_file_path = config_path or self.project_root / "config" / "config.json"
        self.env_path = self.project_root / ".env"
        load_dotenv(self.env_path)
        self.config = self._load_config()
        self._create_directories()

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Converts all relative paths in the config to absolute paths from the project root."""
        paths_to_resolve = {
            ("database", "path"),
            ("logging", "file_config", "path"),
            ("exports", "path"),
            ("portfolio", "binance_csv_path"),
            ("portfolio", "copy_trading_csv_path")
        }
        for path_keys in paths_to_resolve:
            temp_config = config
            key_exists = True
            for key in path_keys[:-1]:
                if key in temp_config:
                    temp_config = temp_config[key]
                else:
                    key_exists = False
                    break
            if key_exists and path_keys[-1] in temp_config:
                final_key = path_keys[-1]
                original_path = Path(temp_config[final_key])
                if not original_path.is_absolute():
                    temp_config[final_key] = str(self.project_root / original_path)
        return config

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, apply environment overrides, and resolve paths."""
        try:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            default_config_path = self.project_root / "config" / "default_config.json"
            try:
                with open(default_config_path, 'r') as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                 # Cannot use logger here as it's not configured yet.
                 # This will print to stderr and exit.
                 raise SystemExit(f"Fatal: Default configuration is missing or corrupt: {e}") from e

        config = self._apply_env_overrides(config)
        config = self._resolve_paths(config)
        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables in a safe way."""
        # API Keys
        if os.getenv("BINANCE_API_KEY"):
            config.setdefault("api_keys", {})["binance_key"] = os.getenv("BINANCE_API_KEY")
        if os.getenv("BINANCE_API_SECRET"):
            config.setdefault("api_keys", {})["binance_secret"] = os.getenv("BINANCE_API_SECRET")

        # Logging Settings
        logging_config = config.setdefault("logging", {})

        if os.getenv("LOG_LEVEL"):
            logging_config["level"] = os.getenv("LOG_LEVEL")

        console_config = logging_config.setdefault("console_config", {})
        if os.getenv("LOG_TO_CONSOLE"):
            console_config["enabled"] = os.getenv("LOG_TO_CONSOLE").lower() == "true"

        file_config = logging_config.setdefault("file_config", {})
        if os.getenv("LOG_TO_FILE"):
            file_config["enabled"] = os.getenv("LOG_TO_FILE").lower() == "true"

        return config

    def _create_directories(self):
        """Create necessary directories based on config using absolute paths."""
        paths_to_create = [
            Path(self.config["database"]["path"]).parent,
            Path(self.config["logging"]["file_config"]["path"]).parent,
            Path(self.config["exports"]["path"])
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

def setup_logging(level: str = "INFO", config: Optional[Dict[str, Any]] = None):
    """Setup application logging"""
    if config is None:
        # This will work now because load_config no longer tries to log prematurely
        temp_config_manager = ConfigManager()
        config = temp_config_manager.config.get("logging", {})

    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(log_level)

    if config.get("console_config", {}).get("enabled", True):
        handler_format = colorlog.ColoredFormatter(
            f"%(log_color)s{log_format}",
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                'ERROR': 'red', 'CRITICAL': 'red,bg_white',
            }
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(handler_format)
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
        root_logger.addHandler(file_handler)

    # This will now be the first message logged by the configured logger
    logging.info("Logging setup complete.")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration for the application"""
    manager = ConfigManager(config_path)
    return manager.config
