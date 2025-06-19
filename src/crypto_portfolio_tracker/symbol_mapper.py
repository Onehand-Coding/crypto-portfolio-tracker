# src/symbol_mapper.py

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class SymbolMapper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mappings_file_path = Path(config.get("database", {}).get("path", "data/portfolio.db")).parent / "coingecko_mappings.json"
        self.coingecko_api_config = config.get("apis", {}).get("coingecko", {})
        self._mappings = self._load_mappings()
        self._norm_map = {k.upper(): v.upper() for k, v in self.config.get("symbol_normalization_map", {}).items()}
        self._coins_list_cache: Optional[List[Dict[str, str]]] = None
        self._coins_list_last_updated: float = 0
        self.derivative_keywords = ['bridged', 'wrapped', 'peg', 'wormhole']

    def __getitem__(self, symbol: str) -> str:
        normalized_symbol = self.normalize_symbol(symbol)
        if normalized_symbol in self._mappings:
            return self._mappings[normalized_symbol]

        self.logger.warning(f"Symbol '{symbol}' not found. Triggering discovery.")
        self.discover_mappings([normalized_symbol])

        if normalized_symbol in self._mappings:
            return self._mappings[normalized_symbol]

        raise KeyError(f"Discovery failed for symbol '{symbol}' (normalized: '{normalized_symbol}')")

    def get(self, symbol: str, default: Any = None) -> Optional[str]:
        """
        Gets a CoinGecko ID for a symbol. If not found in the local cache,
        it triggers the discovery process to find it from the API.
        """
        normalized_symbol = self.normalize_symbol(symbol)
        # First, check the existing mappings
        if normalized_symbol in self._mappings:
            return self._mappings[normalized_symbol]

        # If not found, trigger the discovery process for just this symbol
        self.logger.warning(f"Symbol '{symbol}' not found via .get(). Triggering API discovery.")
        self.discover_mappings([normalized_symbol])

        # Check again after discovery has run
        if normalized_symbol in self._mappings:
            return self._mappings[normalized_symbol]

        # If it's still not found after discovery, return the default value
        self.logger.error(f"Discovery failed for '{symbol}'. Returning default.")
        return default

    @retry(
        retry=retry_if_exception_type(HTTPError),
        stop=stop_after_attempt(5),  # Try up to 5 times
        wait=wait_exponential(multiplier=1, min=4, max=60),  # Wait 4s, then 8s, 16s, etc.
        reraise=True  # If all retries fail, raise the last exception
    )
    def _fetch_market_data_chunk_with_retry(self, chunk: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches market data for a chunk of coin IDs with robust retry logic
        to handle API rate limiting (429 errors).
        """
        url = f"{self.coingecko_api_config.get('base_url')}/coins/markets"
        params = {'vs_currency': 'usd', 'ids': ",".join(chunk)}

        api_key = self.coingecko_api_config.get("api_key")
        if api_key:
            params['x_cg_demo_api_key'] = api_key

        response = requests.get(url, params=params, timeout=30)

        # Specifically check for 429 and log it before tenacity handles the retry
        if response.status_code == 429:
            self.logger.warning(f"Rate limited (429) fetching market data. Tenacity will retry...")

        # This will raise an HTTPError for any 4xx or 5xx status code,
        # which will be caught by the @retry decorator.
        response.raise_for_status()
        return response.json()

    def get_all_mappings(self) -> Dict[str, str]:
        """Return a copy of all current mappings."""
        return self._mappings.copy()

    def _load_mappings(self) -> Dict[str, str]:
        base_mappings = self.config.get("symbol_mappings", {}).get("coingecko_ids", {})
        if self.mappings_file_path.exists():
            try:
                with open(self.mappings_file_path, 'r') as f:
                    discovered_mappings = json.load(f)
                base_mappings.update(discovered_mappings)
            except (json.JSONDecodeError, IOError):
                pass # Ignore errors, start fresh
        return {k.upper(): v for k, v in base_mappings.items()}

    def _save_mappings(self):
        try:
            with open(self.mappings_file_path, 'w') as f:
                json.dump(self._mappings, f, indent=4, sort_keys=True)
            self.logger.info(f"Saved updated mappings to {self.mappings_file_path}")
        except IOError as e:
            self.logger.error(f"Could not save mappings file: {e}")

    def _fetch_master_coins_list(self) -> List[Dict[str, str]]:
        if self._coins_list_cache and (time.time() - self._coins_list_last_updated < 86400):
            return self._coins_list_cache
        url = f"{self.coingecko_api_config.get('base_url')}/coins/list"
        self.logger.info("Fetching master coin list from CoinGecko...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self._coins_list_cache = response.json()
            self._coins_list_last_updated = time.time()
            return self._coins_list_cache
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch master coin list from CoinGecko: {e}")
            return []

    def _score_candidate(self, candidate: Dict[str, Any]) -> tuple:
        is_derivative = any(keyword in candidate['id'] for keyword in self.derivative_keywords)
        market_cap = candidate.get('market_cap') or 0
        return (is_derivative, -market_cap)

    def discover_mappings(self, symbols_to_find: List[str]):
        """
        Efficiently discovers mappings for a list of unknown symbols,
        handling large numbers of requests by chunking them.
        """
        self.logger.info(f"Attempting to discover mappings for {len(symbols_to_find)} symbols.")
        coins_list = self._fetch_master_coins_list()
        if not coins_list: return

        ambiguous_symbols = {}
        all_candidate_ids = set()

        # Find all unique symbols that need discovery
        unique_symbols = set(s for s in symbols_to_find if s not in self._mappings)

        for symbol in unique_symbols:
            candidates = [c for c in coins_list if c.get('symbol', '').upper() == symbol]
            if not candidates:
                self.logger.error(f"AUTOMAP: No candidates found for '{symbol}'.")
                continue

            if len(candidates) == 1:
                self._mappings[symbol] = candidates[0]['id']
                self.logger.info(f"AUTOMAP: Found unique match for '{symbol}': {candidates[0]['id']}")
            else:
                ambiguous_symbols[symbol] = candidates
                all_candidate_ids.update([c['id'] for c in candidates])

        # If we found new unique matches, save them before proceeding
        self._save_mappings()

        if ambiguous_symbols:
            self.logger.warning(f"Found {len(ambiguous_symbols)} ambiguous symbols. Resolving in chunks...")
            market_data_map = {}
            all_ids_list = list(all_candidate_ids)
            chunk_size = 200  # Safe chunk size for CoinGecko API

            for i in range(0, len(all_ids_list), chunk_size):
                chunk = all_ids_list[i:i + chunk_size]
                self.logger.info(f"Fetching market data for chunk {i//chunk_size + 1}...")
                try:
                    market_data_list = self._fetch_market_data_chunk_with_retry(chunk)
                    for item in market_data_list:
                        market_data_map[item['id']] = item
                except Exception as e:
                    self.logger.error(f"AUTOMAP: API error on chunk {i//chunk_size + 1} after all retries: {e}.")

            for symbol, candidates in ambiguous_symbols.items():
                scored = [
                    (self._score_candidate(market_data_map.get(c['id'])), c['id'])
                    for c in candidates if market_data_map.get(c['id'])
                ]
                if scored:
                    scored.sort(key=lambda x: x[0])
                    best_id = scored[0][1]
                    self._mappings[symbol] = best_id
                    self.logger.info(f"AUTOMAP: Resolved '{symbol}' to '{best_id}'")

            self._save_mappings()

    def normalize_symbol(self, symbol: str) -> str:
        s_upper = symbol.upper()
        s_upper = self._norm_map.get(s_upper, s_upper)
        if s_upper.startswith("LD") and len(s_upper) > 2:
            s_upper = s_upper[2:]
        return s_upper
