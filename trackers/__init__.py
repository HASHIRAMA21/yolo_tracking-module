__version__ = '10.0.14'

from pathlib import Path
from yolo_tracking.trackers.strongsort.strong_sort import StrongSORT
from yolo_tracking.trackers.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from yolo_tracking.trackers.ocsort.ocsort import OCSort as OCSORT
from yolo_tracking.trackers.botsort.bot_sort import BoTSORT
from yolo_tracking.trackers.bytetrack.byte_tracker import BYTETracker
from yolo_tracking.trackers.deepocsort.ocsort import OCSort as DeepOCSORT
from multi_tracker import get_tracker_config, create_tracker
from yolo_tracking.trackers.deep.reid_multibackend import ReIDDetectMultiBackend


__all__ = '__version__', \
          'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT', 'DeepOCSORT', 'DeepSort'
