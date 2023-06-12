from pathlib import Path
import yaml
from types import SimpleNamespace
from utils.__init__ import TRACKERS


def get_tracker_config(tracker_type):
    tracking_config = \
        TRACKERS / \
        tracker_type / \
        'configs' / \
        (tracker_type + '.yaml')
    return tracking_config


def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']

    if tracker_type == 'strongsort':
        from yolo_tracking.trackers.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.max_dist,
            max_iou_dist=cfg.max_iou_dist,
            max_age=cfg.max_age,
            max_unmatched_preds=cfg.max_unmatched_preds,
            n_init=cfg.n_init,
            nn_budget=cfg.nn_budget,
            mc_lambda=cfg.mc_lambda,
            ema_alpha=cfg.ema_alpha,

        )
        return strongsort

    elif tracker_type == 'ocsort':
        from yolo_tracking.trackers.ocsort.ocsort import OCSort
        ocsort = OCSort(
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
            use_byte=cfg.use_byte,
        )
        return ocsort

    elif tracker_type == 'bytetrack':
        from yolo_tracking.trackers.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            frame_rate=cfg.frame_rate
        )
        return bytetracker

    elif tracker_type == 'botsort':
        from yolo_tracking.trackers.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.track_high_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method=cfg.cmc_method,
            frame_rate=cfg.frame_rate,
            lambda_=cfg.lambda_
        )
        return botsort
    elif tracker_type == 'deepocsort':
        from yolo_tracking.trackers.deepocsort.ocsort import OCSort
        deepocsort = OCSort(
            reid_weights,
            device,
            half,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
        )
        return deepocsort
    elif tracker_type == 'deepsort':
        from yolo_tracking.trackers.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
        from deep_sort_pytorch.utils.parser import get_configs
        cfg_deep = get_configs()
        cfg_deep.merge_from_file("trackers/deep_sort_pytorch/config/deep_sort.yaml")
        deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        return deepsort
    else:
        print('No such tracker')
        exit()
