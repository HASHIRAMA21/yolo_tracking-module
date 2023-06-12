import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
TRACKERS = ROOT / 'trackers'
DETECTION = ROOT / 'detection'
WEIGHTS = ROOT / 'detection' / 'weights'
REQUIREMENTS = ROOT / 'requirements.txt'

# global logger
from loguru import logger
logger.remove()
logger.add(sys.stderr, colorize=True)