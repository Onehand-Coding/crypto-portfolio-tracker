import time
import json
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import talib
from threading import Thread
import smtplib
from email.mime.text import MimeText

class LiveTradingBot:
    def __init__(self, config_file='trading_config.json'):
        """
        Live trading bot that implements the swing trading strategy
        """
        self.config = self.load_config(config_file)
        self.client = Client(
            self.config['api_key'], 
            self.config['api_secret'],
            testnet=self.config.get('testnet', True)
        )
        
        # Initialize strategy
        from swing_trading_strategy import SwingTradingStrategy  # Import our strategy class
        self.strategy = SwingTradingStrategy(
            self.config['api_key'], 
            self.config['api_secret'], 
            testnet=self.config.get('testnet', True)
        )
        
        # Trading parameters
        self.symbols = self.config.get('symbols', ['BTCUSDT'])
        self.timeframe = self.config.get('timeframe', '4h')
        self.check_interval = self.config.get('check_interval', 300)  # 5 minutes
        
        # Risk management
        self.max_positions = self.config.get('max_positions', 3)
        self.account_risk = self.config.get('account_risk', 0.02)  # 2% per trade
        
        # Active positions tracking
        self.active_positions = {}
        self.trade_log = []
        
        # Setup logging
        self.setup_logging
