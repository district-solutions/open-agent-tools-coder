import os
import ujson as json
from typing import Tuple
from oats.models import OatConfig

OAT_CONFIG = None 

def get_oat_config() -> OatConfig:
    global OAT_CONFIG
    if OAT_CONFIG is None:
        OAT_CONFIG = OatConfig()
    return OAT_CONFIG
