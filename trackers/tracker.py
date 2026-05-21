from ultralytics import YOLO
import supervision as sv
import pickle
import os

from .drawing import DrawingMixin
from .identity import IdentityMixin
from .invalid_roles import InvalidRoleMixin
from .track_appearance import TrackAppearanceMixin
from .tracking_geometry import TrackingGeometryMixin


class Tracker(
    TrackingGeometryMixin,
    TrackAppearanceMixin,
    IdentityMixin,
    InvalidRoleMixin,
    DrawingMixin
):
    def __init__(self, model_path, min_track_consecutive_frames=5):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.15,
            lost_track_buffer=90,
            minimum_matching_threshold=0.65,
            frame_rate=24,
            minimum_consecutive_frames=3
        )
        self.player_class_id = self._get_class_id("player")
        self.goalkeeper_class_id = self._get_class_id("goalkeeper")
        self.referee_class_id = self._get_class_id("referee")
        self.tracked_class_ids = [
            self.player_class_id,
            self.goalkeeper_class_id,
            self.referee_class_id
        ]
        self.invalid_player_class_ids = {
            self.goalkeeper_class_id,
            self.referee_class_id
        }
        self.min_track_consecutive_frames = min_track_consecutive_frames
        self.max_reid_gap = 45
        self.max_reid_distance = 120
        self.max_size_change = 0.45
        self.max_switch_gap = 30
        self.max_switch_distance = 90
        self.interruption_gap_threshold = 2
        self.interruption_appearance_threshold = 0.48
        self.close_recovery_distance = 55
        self.nearest_recovery_weight = 90
        self.max_cross_team_appearance_distance = 0.58
        self.suspicious_appearance_distance = 0.38
        self.cross_team_appearance_penalty = 520
        self.appearance_update_max_distance = 0.50
        self.max_interrupted_appearance_distance = 0.54
        self.mature_identity_frames = 8
        self.mature_identity_appearance_distance = 0.44
        self.mature_interrupted_appearance_distance = 0.38
        self.mature_identity_penalty = 700
        self.fragment_merge_appearance_threshold = 0.36
        self.fragment_merge_appearance_penalty = 520
        self.fragment_merge_appearance_samples = 5
        self.duplicate_iou_threshold = 0.82
        self.duplicate_containment_threshold = 0.92
        self.continuity_iou_threshold = 0.55
        self.invalid_class_iou_threshold = 0.35
        self.invalid_class_containment_threshold = 0.6
        self.invalid_track_memory = 30
        self.invalid_track_distance = 35
        self.invalid_track_size_change = 0.6
        self.invalid_projected_iou_threshold = 0.12
        self.invalid_projected_containment_threshold = 0.32
        self.invalid_state_match_distance = 90
        self.invalid_state_match_iou = 0.12
        self.invalid_track_hit_threshold = 6
        self.invalid_track_overlap_ratio = 0.35

    def _get_class_id(self, class_name):
        for class_id, model_class_name in self.model.names.items():
            if model_class_name == class_name:
                return class_id

        raise ValueError(f"Class '{class_name}' was not found in the YOLO model.")

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size],
                conf=0.1,
                classes=self.tracked_class_ids
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            tracks = self.ensure_track_appearances(frames, tracks)
            tracks = self.stabilize_player_ids(tracks)
            tracks = self.merge_fragmented_player_tracks(tracks)
            return self.filter_short_player_tracks(tracks)

        detections = self.detect_frames(frames)

        tracks={
            "players":[]
        }
        invalid_track_states = {}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            invalid_detections = []
            for frame_detection in detection_with_tracks:
                cls_id = frame_detection[3]
                if cls_id not in self.invalid_player_class_ids:
                    continue

                invalid_detections.append({
                    "bbox": frame_detection[0].tolist(),
                    "track_id": frame_detection[4],
                    "class_id": cls_id
                })

            self._update_invalid_track_states(
                invalid_track_states,
                invalid_detections,
                frame_num
            )

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                invalid_overlap = self._conflicts_with_invalid_track(
                    bbox,
                    track_id,
                    invalid_track_states,
                    frame_num
                )

                if (
                    cls_id == cls_names_inv['player'] and
                    not invalid_overlap
                ):
                    tracks["players"][frame_num][track_id] = {
                        "bbox":bbox,
                        "appearance": self._extract_appearance(frames[frame_num], bbox),
                        "invalid_overlap": invalid_overlap
                    }
                elif cls_id == cls_names_inv['player'] and invalid_overlap:
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "appearance": self._extract_appearance(frames[frame_num], bbox),
                        "invalid_overlap": invalid_overlap
                    }

        tracks = self.stabilize_player_ids(tracks)
        tracks = self.merge_fragmented_player_tracks(tracks)
        tracks = self._remove_invalid_player_tracks(tracks)
        tracks = self.filter_short_player_tracks(tracks)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
