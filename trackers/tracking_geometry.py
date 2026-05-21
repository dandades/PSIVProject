import numpy as np

from utils import get_foot_position


class TrackingGeometryMixin:
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def _bbox_size(self, bbox):
        return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

    def _bbox_area(self, bbox):
        width, height = self._bbox_size(bbox)
        return width * height

    def _bbox_intersection_area(self, bbox_a, bbox_b):
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _bbox_iou(self, bbox_a, bbox_b):
        if bbox_a is None or bbox_b is None:
            return 0

        intersection = self._bbox_intersection_area(bbox_a, bbox_b)
        union = self._bbox_area(bbox_a) + self._bbox_area(bbox_b) - intersection
        if union <= 0:
            return 0

        return intersection / union

    def _bbox_containment(self, bbox_a, bbox_b):
        intersection = self._bbox_intersection_area(bbox_a, bbox_b)
        smaller_area = min(self._bbox_area(bbox_a), self._bbox_area(bbox_b))
        if smaller_area <= 0:
            return 0

        return intersection / smaller_area

    def _track_summary(self, object_tracks):
        summary = {}

        for frame_num, frame_tracks in enumerate(object_tracks):
            for track_id, track_info in frame_tracks.items():
                bbox = track_info["bbox"]
                summary.setdefault(track_id, []).append({
                    "frame_num": frame_num,
                    "position": get_foot_position(bbox),
                    "size": self._bbox_size(bbox),
                    "appearance": track_info.get("appearance")
                })

        return summary

    def _segment_appearance(self, observations, from_end=False):
        if from_end:
            selected_observations = observations[-self.fragment_merge_appearance_samples:]
        else:
            selected_observations = observations[:self.fragment_merge_appearance_samples]

        appearances = [
            observation["appearance"]
            for observation in selected_observations
            if observation.get("appearance") is not None
        ]
        if not appearances:
            return None

        return np.median(np.array(appearances, dtype=np.float32), axis=0).tolist()

    def _estimate_position(self, observations, target_frame_num):
        last_observation = observations[-1]
        if len(observations) < 2:
            return last_observation["position"]

        previous_observation = observations[-2]
        frame_delta = max(
            1,
            last_observation["frame_num"] - previous_observation["frame_num"]
        )
        vx = (
            last_observation["position"][0] - previous_observation["position"][0]
        ) / frame_delta
        vy = (
            last_observation["position"][1] - previous_observation["position"][1]
        ) / frame_delta
        target_delta = target_frame_num - last_observation["frame_num"]

        return (
            last_observation["position"][0] + vx * target_delta,
            last_observation["position"][1] + vy * target_delta
        )

    def _distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _appearance_distance(self, old_appearance, new_appearance):
        if old_appearance is None or new_appearance is None:
            return 0.8

        old_appearance = np.array(old_appearance, dtype=np.float32)
        new_appearance = np.array(new_appearance, dtype=np.float32)
        return float(np.linalg.norm(old_appearance - new_appearance))

    def _size_change(self, old_size, new_size):
        width_change = abs(old_size[0] - new_size[0]) / max(old_size[0], new_size[0])
        height_change = abs(old_size[1] - new_size[1]) / max(old_size[1], new_size[1])
        return max(width_change, height_change)

    def _find_root_id(self, id_mapping, track_id):
        while id_mapping.get(track_id, track_id) != track_id:
            track_id = id_mapping[track_id]
        return track_id

    def _max_consecutive_observations(self, observations):
        if not observations:
            return 0

        max_streak = 1
        current_streak = 1
        previous_frame_num = observations[0]["frame_num"]

        for observation in observations[1:]:
            frame_num = observation["frame_num"]
            if frame_num == previous_frame_num + 1:
                current_streak += 1
            else:
                current_streak = 1

            max_streak = max(max_streak, current_streak)
            previous_frame_num = frame_num

        return max_streak
