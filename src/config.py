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

from . symbol_mapper import SymbolMapper


class ConfigManager:
    """
    Manages application configuration by loading non-sensitive settings
    from a JSON file and sensitive secrets from environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize and load all configurations."""
        self.project_root = Path(__file__).parent.parent
        self.config_file_path = config_path or self.project_root / "config" / "default_config.json"
        self.env_path = self.project_root / ".env"

        load_dotenv(self.env_path)

        self.config: Dict[str, Any] = self._load_json_config()

        # --- Load the CoinGecko API key from .env ---
        if "apis" in self.config and "coingecko" in self.config["apis"]:
            self.config["apis"]["coingecko"]["api_key"] = os.getenv("COINGECKO_API_KEY")

        # Determine testnet status from environment *before* resolving paths
        self.is_testnet_mode: bool = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'

        # If in testnet mode, swap the database path with the testnet path
        if self.is_testnet_mode:
            logging.warning("TESTNET mode active. Switching to testnet database.")
            db_config = self.config.get("database", {})
            testnet_db_path = db_config.get("testnet_path")
            if testnet_db_path:
                db_config["path"] = testnet_db_path
            else:
                logging.warning("Testnet mode is active, but no 'testnet_path' found in database config.")

        self.config = self._resolve_paths(self.config)

        self.main_api_keys: Dict[str, Optional[str]] = {
            "api_key": os.getenv("MAIN_API_KEY"),
            "api_secret": os.getenv("MAIN_API_SECRET")
        }
        self.testnet_api_keys: Dict[str, Optional[str]] = {
            "api_key": os.getenv("TESTNET_API_KEY"),
            "api_secret": os.getenv("TESTNET_API_SECRET")
        }
        self.sub_accounts: List[Dict[str, Any]] = self._load_sub_accounts()

        self.symbol_mapper = SymbolMapper(self.config)

        self._create_directories()

    def _load_json_config(self) -> Dict[str, Any]:
        """Loads the base configuration from the JSON file."""
        try:
            with open(self.config_file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            default_config_path = self.project_root / "config" / "default_config.json"
            try:
                with open(default_config_path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise SystemExit(f"FATAL: Default configuration is missing or corrupt: {e}")

    def _load_sub_accounts(self) -> List[Dict[str, Any]]:
        """Loads sub-account keys from environment variables."""
        accounts = []
        if os.getenv("SWING_API_KEY") and os.getenv("SWING_API_SECRET"):
            accounts.append({
                "name": "Swing Trading Account",
                "type": "swing",
                "binance_key": os.getenv("SWING_API_KEY"),
                "binance_secret": os.getenv("SWING_API_SECRET")
            })
        if os.getenv("DAY_API_KEY") and os.getenv("DAY_API_SECRET"):
            accounts.append({
                "name": "Day Trading Account",
                "type": "day",
                "binance_key": os.getenv("DAY_API_KEY"),
                "binance_secret": os.getenv("DAY_API_SECRET")
            })
        return accounts

    def get_binance_keys(self, account_name: Optional[str] = "Main Account") -> Dict[str, Optional[str]]:
        """
        Returns the appropriate API keys based on testnet mode and selected account.
        """
        if self.is_testnet_mode:
            logging.info("TESTNET mode is active. Using testnet keys for all operations.")
            return self.testnet_api_keys

        # Live mode logic
        if account_name == "Main Account":
            return self.main_api_keys

        for acc in self.sub_accounts:
            if acc['name'] == account_name:
                return {"api_key": acc['binance_key'], "api_secret": acc['binance_secret']}

        logging.warning(f"No keys found for account '{account_name}'. Defaulting to main keys.")
        return self.main_api_keys

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Converts all relative paths in the config to absolute paths."""
        paths_to_resolve = {
            ("database", "path"),
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
            Path(self.config["exports"]["path"])
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
