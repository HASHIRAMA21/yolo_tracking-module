__version__ = '10.0.14'

from pathlib import Path
from trackers.strongsort.strong_sort import StrongSORT
from trackers.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from trackers.ocsort.ocsort import OCSort as OCSORT
from trackers.botsort.bot_sort import BoTSORT
from trackers.bytetrack.byte_tracker import BYTETracker
from trackers.deepocsort.ocsort import OCSort as DeepOCSORT
from multi_tracker import get_tracker_config, create_tracker
from trackers.deep.reid_multibackend import ReIDDetectMultiBackend


__all__ = '__version__', \
          'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT', 'DeepOCSORT', 'DeepSort'
