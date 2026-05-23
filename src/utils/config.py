"""
Configuration helpers and .env loader
"""

from dotenv import load_dotenv
import os

load_dotenv()

def get_env(key: str, default=None):
    return os.getenv(key, default)
