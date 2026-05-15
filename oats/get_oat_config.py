"""Lazy singleton loader for the OAT (Open Agent Tools) configuration."""
import os
import ujson as json
from typing import Tuple
from oats.models import OatConfig

OAT_CONFIG = None 

def get_oat_config() -> OatConfig:
    """Return the process-wide OatConfig singleton, creating it on first call."""
    global OAT_CONFIG
    if OAT_CONFIG is None:
        OAT_CONFIG = OatConfig()
    return OAT_CONFIG
