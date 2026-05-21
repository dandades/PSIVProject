from utils import get_foot_position


class InvalidRoleMixin:
    def _overlaps_invalid_class(self, bbox, invalid_bboxes):
        for invalid_bbox in invalid_bboxes:
            if (
                self._bbox_iou(bbox, invalid_bbox) >= self.invalid_class_iou_threshold or
                self._bbox_containment(bbox, invalid_bbox) >= self.invalid_class_containment_threshold
            ):
                return True

        return False

    def _get_predicted_bbox(self, state, frame_num):
        gap = max(0, frame_num - state["last_frame"])
        dx = state["velocity"][0] * gap
        dy = state["velocity"][1] * gap
        return (
            state["bbox"][0] + dx,
            state["bbox"][1] + dy,
            state["bbox"][2] + dx,
            state["bbox"][3] + dy
        )

    def _invalid_state_match_cost(self, state, detection, frame_num):
        predicted_bbox = self._get_predicted_bbox(state, frame_num)
        predicted_position = get_foot_position(predicted_bbox)
        detection_position = get_foot_position(detection["bbox"])
        distance = self._distance(predicted_position, detection_position)
        iou = self._bbox_iou(predicted_bbox, detection["bbox"])
        containment = self._bbox_containment(predicted_bbox, detection["bbox"])
        size_change = self._size_change(state["size"], self._bbox_size(detection["bbox"]))
        same_tracker_id = state.get("source_track_id") == detection["track_id"]

        if same_tracker_id:
            return distance - 80

        if size_change > 0.9:
            return None

        if (
            distance > self.invalid_state_match_distance and
            iou < self.invalid_state_match_iou and
            containment < self.invalid_projected_containment_threshold
        ):
            return None

        return distance + size_change * 45 - iou * 90 - containment * 45

    def _update_invalid_track_states(self, invalid_track_states, invalid_detections, frame_num):
        for detection in invalid_detections:
            best_state_id = None
            best_cost = None

            for state_id, state in invalid_track_states.items():
                gap = frame_num - state["last_frame"]
                if gap < 0 or gap > self.invalid_track_memory:
                    continue

                cost = self._invalid_state_match_cost(state, detection, frame_num)
                if cost is None:
                    continue

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_state_id = state_id

            bbox = detection["bbox"]
            position = get_foot_position(bbox)
            previous_state = (
                invalid_track_states.get(best_state_id)
                if best_state_id is not None
                else None
            )

            if previous_state is None:
                velocity = (0, 0)
                state_id = max(invalid_track_states.keys(), default=0) + 1
                hits = 1
                class_counts = {}
            else:
                state_id = best_state_id
                frame_gap = max(1, frame_num - previous_state["last_frame"])
                velocity = (
                    (position[0] - previous_state["position"][0]) / frame_gap,
                    (position[1] - previous_state["position"][1]) / frame_gap
                )
                hits = previous_state.get("hits", 0) + 1
                class_counts = dict(previous_state.get("class_counts", {}))

            class_counts[detection["class_id"]] = class_counts.get(detection["class_id"], 0) + 1

            invalid_track_states[state_id] = {
                "bbox": bbox,
                "position": position,
                "velocity": velocity,
                "size": self._bbox_size(bbox),
                "last_frame": frame_num,
                "class_id": detection["class_id"],
                "source_track_id": detection["track_id"],
                "hits": hits,
                "class_counts": class_counts
            }

        for track_id in list(invalid_track_states.keys()):
            if frame_num - invalid_track_states[track_id]["last_frame"] > self.invalid_track_memory:
                del invalid_track_states[track_id]

    def _conflicts_with_invalid_track(self, bbox, track_id, invalid_track_states, frame_num):
        size = self._bbox_size(bbox)

        for invalid_track_id, state in invalid_track_states.items():
            gap = frame_num - state["last_frame"]
            if gap < 0 or gap > self.invalid_track_memory:
                continue

            if track_id == state.get("source_track_id"):
                return True

            if self._overlaps_invalid_class(bbox, [state["bbox"]]):
                return True

            if gap == 0:
                continue

            predicted_bbox = self._get_predicted_bbox(state, frame_num)
            size_change = self._size_change(state["size"], size)
            if size_change > self.invalid_track_size_change:
                continue

            projected_iou = self._bbox_iou(bbox, predicted_bbox)
            projected_containment = self._bbox_containment(bbox, predicted_bbox)
            if (
                projected_iou >= self.invalid_projected_iou_threshold or
                projected_containment >= self.invalid_projected_containment_threshold
            ):
                return True

        return False

    def _remove_invalid_player_tracks(self, tracks):
        player_tracks = tracks["players"]
        summary = {}

        for frame_tracks in player_tracks:
            for track_id, track_info in frame_tracks.items():
                summary.setdefault(track_id, {"frames": 0, "invalid_hits": 0})
                summary[track_id]["frames"] += 1
                if track_info.get("invalid_overlap"):
                    summary[track_id]["invalid_hits"] += 1

        invalid_track_ids = {
            track_id
            for track_id, stats in summary.items()
            if (
                stats["invalid_hits"] >= self.invalid_track_hit_threshold and
                stats["invalid_hits"] / max(1, stats["frames"]) >= self.invalid_track_overlap_ratio
            )
        }

        if not invalid_track_ids:
            return tracks

        for frame_tracks in player_tracks:
            for track_id in list(frame_tracks.keys()):
                if track_id in invalid_track_ids:
                    del frame_tracks[track_id]

        return tracks
