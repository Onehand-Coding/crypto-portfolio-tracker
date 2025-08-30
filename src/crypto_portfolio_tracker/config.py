"""
Configuration Management Module
Handles loading configuration from environment variables, files, and defaults.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

from .symbol_mapper import SymbolMapper


class ConfigManager:
    """
    Manages application configuration by loading non-sensitive settings
    from a JSON file and sensitive secrets from environment variables.
    
    This class handles the loading, parsing, and management of application configuration.
    It separates sensitive data (API keys, secrets) which are loaded from environment
    variables, from non-sensitive configuration which is loaded from a JSON file.
    
    The config manager also handles path resolution, directory creation, and provides
    convenient accessors for common configuration values like database paths and API keys.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize and load all configurations.
        
        Args:
            config_path: Optional path to the configuration file. If not provided,
                        defaults to config/default_config.json in the project root.
                        
        Raises:
            FileNotFoundError: If the project root cannot be found or config file is missing
            SystemExit: If the configuration file is corrupt or missing
        """
        current_dir = Path(__file__).parent
        while not (current_dir / "pyproject.toml").exists():
            if current_dir == current_dir.parent:
                raise FileNotFoundError(
                    "Could not find project root containing 'pyproject.toml'."
                )
            current_dir = current_dir.parent
        self.project_root = current_dir
        self.config_file_path = (
            config_path or self.project_root / "config" / "default_config.json"
        )
        self.env_path = self.project_root / ".env"

        load_dotenv(self.env_path)

        self.config: Dict[str, Any] = self._load_json_config()
        
        # Validate the configuration
        self.config = self._validate_config(self.config)

        # --- Load the CoinGecko API key from .env ---
        if "apis" in self.config and "coingecko" in self.config["apis"]:
            self.config["apis"]["coingecko"]["api_key"] = os.getenv("COINGECKO_API_KEY")

        self.config = self._resolve_paths(self.config)

        self.main_api_keys: Dict[str, Optional[str]] = {
            "api_key": os.getenv("MAIN_API_KEY"),
            "api_secret": os.getenv("MAIN_API_SECRET"),
        }
        self.testnet_api_keys: Dict[str, Optional[str]] = {
            "api_key": os.getenv("TESTNET_API_KEY"),
            "api_secret": os.getenv("TESTNET_API_SECRET"),
        }
        self.sub_accounts: List[Dict[str, Any]] = self._load_sub_accounts()

        # --- Add API keys and sub accounts to config ---
        self.config["main_api_keys"] = self.main_api_keys
        self.config["sub_accounts"] = self.sub_accounts

        self.symbol_mapper = SymbolMapper(self.config)

        self._create_directories()

    @property
    def is_live(self):
        """Check if the application is in live mode"""
        return self.config.get("portfolio", {}).get("live_trading_enabled", False)

    @property
    def is_testnet_mode(self):
        """Check if the application is in testnet mode"""
        return self.config.get("portfolio", {}).get("testnet_mode", False)

    def get_database_path(self) -> Path:
        """Get the appropriate database path based on testnet mode"""
        if self.is_testnet_mode:
            db_path = self.config.get("database", {}).get(
                "testnet_path", "data/testnet_portfolio.db"
            )
            logging.debug("TESTNET mode active. Using testnet database.")
        else:
            db_path = self.config.get("database", {}).get("path", "data/portfolio.db")
            logging.debug("MAINNET mode active. Using production database.")

        return Path(db_path)

    def get_backup_dir(self) -> Path:
        """Get the backup directory path"""
        db_path = self.get_database_path()
        return db_path.parent / "db_backups"

    def _load_json_config(self) -> Dict[str, Any]:
        """Loads the base configuration from the JSON file."""
        try:
            with open(self.config_file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            default_config_path = self.project_root / "config" / "default_config.json"
            try:
                with open(default_config_path, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise SystemExit(
                    f"FATAL: Default configuration is missing or corrupt: {e}"
                )

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the configuration and applies defaults where necessary.
        
        Args:
            config: The configuration dictionary to validate
            
        Returns:
            Dict[str, Any]: The validated configuration dictionary
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Validate required top-level keys
        required_keys = ["target_allocation", "portfolio"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: '{key}'")
        
        # Validate target allocation sums to approximately 100%
        target_allocation = config.get("target_allocation", {})
        if target_allocation:
            total_allocation = sum(target_allocation.values())
            if not 0.95 <= total_allocation <= 1.05:
                logging.warning(
                    f"Target allocation sums to {total_allocation:.2%}, which is outside "
                    f"the typical range of 95-105%. This may cause unexpected behavior."
                )
        
        # Validate portfolio settings
        portfolio_config = config.get("portfolio", {})
        if "testnet_mode" not in portfolio_config:
            config["portfolio"]["testnet_mode"] = True  # Default to testnet for safety
            logging.info("Setting default testnet_mode=true for safety")
            
        if "live_trading_enabled" not in portfolio_config:
            config["portfolio"]["live_trading_enabled"] = False  # Default to disabled
            logging.info("Setting default live_trading_enabled=false for safety")
        
        # Validate numeric values
        numeric_validations = [
            ("portfolio.minimum_trade_usd", 0.0),
            ("database.connection_timeout", 1),
            ("database.cleanup_days", 0),
        ]
        
        for path, min_value in numeric_validations:
            keys = path.split(".")
            value = config
            try:
                for key in keys:
                    value = value[key]
                if not isinstance(value, (int, float)) or value < min_value:
                    logging.warning(
                        f"Configuration value '{path}' should be a number >= {min_value}. "
                        f"Found: {value}. This may cause unexpected behavior."
                    )
            except (KeyError, TypeError):
                # Key doesn't exist, which is fine as we'll use defaults
                pass
        
        # Validate file paths
        path_validations = [
            "database.path",
            "database.testnet_path",
            "logging.file_config.path",
            "exports.path",
            "cache.path",
        ]
        
        for path in path_validations:
            keys = path.split(".")
            value = config
            try:
                for key in keys:
                    value = value[key]
                if value:  # Only validate if a value is set
                    # We'll resolve paths later, so just check if it's a string
                    if not isinstance(value, str):
                        logging.warning(
                            f"Configuration path '{path}' should be a string. "
                            f"Found: {value}. This may cause unexpected behavior."
                        )
            except (KeyError, TypeError):
                # Key doesn't exist, which is fine as we'll use defaults
                pass
        
        logging.info("Configuration validation completed successfully")
        return config

    def _load_sub_accounts(self) -> List[Dict[str, Any]]:
        """Loads sub-account keys from environment variables."""
        accounts = []
        if os.getenv("SWING_API_KEY") and os.getenv("SWING_API_SECRET"):
            accounts.append(
                {
                    "name": "Swing Trading Account",
                    "type": "swing",
                    "binance_key": os.getenv("SWING_API_KEY"),
                    "binance_secret": os.getenv("SWING_API_SECRET"),
                }
            )
        if os.getenv("DAY_API_KEY") and os.getenv("DAY_API_SECRET"):
            accounts.append(
                {
                    "name": "Day Trading Account",
                    "type": "day",
                    "binance_key": os.getenv("DAY_API_KEY"),
                    "binance_secret": os.getenv("DAY_API_SECRET"),
                }
            )
        return accounts

    def get_binance_keys(
        self, account_name: Optional[str] = "Main Account"
    ) -> Dict[str, Optional[str]]:
        """
        Returns the appropriate API keys based on testnet mode and selected account.
        """
        if self.is_testnet_mode:
            logging.debug(
                "TESTNET mode is active. Using testnet keys for all operations."
            )
            return self.testnet_api_keys

        # Live mode logic
        if account_name == "Main Account":
            return self.main_api_keys

        for acc in self.sub_accounts:
            if acc["name"] == account_name:
                return {
                    "api_key": acc["binance_key"],
                    "api_secret": acc["binance_secret"],
                }

        logging.warning(
            f"No keys found for account '{account_name}'. Defaulting to main keys."
        )
        return self.main_api_keys

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Converts all relative paths in the config to absolute paths."""
        paths_to_resolve = {
            ("database", "path"),
            ("database", "testnet_path"),
            ("logging", "file_config", "path"),
            ("exports", "path"),
            ("cache", "path"),
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

    def _create_directories(self):
        """Create necessary directories based on config."""
        paths_to_create = [
            Path(self.config["database"]["path"]).parent,
            Path(self.config["logging"]["file_config"]["path"]).parent,
            Path(self.config["exports"]["path"]),
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self):
        """Persist the current config dictionary to the config file."""
        try:
            with open(self.config_file_path, "w") as f:
                # Remove sensitive data from config.
                config = self.config
                del config["main_api_keys"]
                del config["sub_accounts"]
                del config["apis"]["coingecko"]["api_key"]

                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")
