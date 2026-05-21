import numpy as np

from utils import get_bbox_width, get_foot_position


class IdentityMixin:
    def _duplicate_detection_score(self, detection, stable_states, frame_num):
        bbox = detection["track_info"]["bbox"]
        position = detection["position"]
        best_score = None

        for state in stable_states.values():
            gap = frame_num - state["last_frame"]
            if gap <= 0 or gap > self.max_switch_gap:
                continue

            predicted_position = (
                state["position"][0] + state["velocity"][0] * gap,
                state["position"][1] + state["velocity"][1] * gap
            )
            score = self._distance(predicted_position, position)
            score -= self._bbox_iou(state.get("bbox"), bbox) * 100

            if state.get("last_original_track_id") == detection["original_track_id"]:
                score -= 45

            if best_score is None or score < best_score:
                best_score = score

        if best_score is None:
            best_score = -self._bbox_area(bbox) * 0.005

        return best_score

    def _remove_duplicate_detections(self, detections, stable_states, frame_num):
        if len(detections) < 2:
            return detections

        keep = [True] * len(detections)

        for first_index in range(len(detections)):
            if not keep[first_index]:
                continue

            for second_index in range(first_index + 1, len(detections)):
                if not keep[second_index]:
                    continue

                first_bbox = detections[first_index]["track_info"]["bbox"]
                second_bbox = detections[second_index]["track_info"]["bbox"]
                iou = self._bbox_iou(first_bbox, second_bbox)
                containment = self._bbox_containment(first_bbox, second_bbox)

                if (
                    iou < self.duplicate_iou_threshold and
                    containment < self.duplicate_containment_threshold
                ):
                    continue

                first_score = self._duplicate_detection_score(
                    detections[first_index],
                    stable_states,
                    frame_num
                )
                second_score = self._duplicate_detection_score(
                    detections[second_index],
                    stable_states,
                    frame_num
                )

                if second_score < first_score:
                    keep[first_index] = False
                    break

                keep[second_index] = False

        return [
            detection
            for detection_index, detection in enumerate(detections)
            if keep[detection_index]
        ]

    def stabilize_player_ids(self, tracks):
        stable_states = {}
        next_stable_id = 1
        stabilized_tracks = {"players": []}

        for frame_num, frame_tracks in enumerate(tracks["players"]):
            detections = []
            for original_track_id, track_info in frame_tracks.items():
                bbox = track_info["bbox"]
                track_info["original_track_id"] = original_track_id
                detections.append({
                    "original_track_id": original_track_id,
                    "track_info": track_info,
                    "position": get_foot_position(bbox),
                    "size": self._bbox_size(bbox),
                    "appearance": track_info.get("appearance")
                })

            detections = self._remove_duplicate_detections(
                detections,
                stable_states,
                frame_num
            )

            candidate_matches = []
            interrupted_distance_ranks = {}
            for detection_index, detection in enumerate(detections):
                for stable_id, state in stable_states.items():
                    gap = frame_num - state["last_frame"]
                    if gap <= 0 or gap > self.max_switch_gap:
                        continue

                    interrupted = gap >= self.interruption_gap_threshold
                    predicted_position = (
                        state["position"][0] + state["velocity"][0] * gap,
                        state["position"][1] + state["velocity"][1] * gap
                    )
                    distance = self._distance(predicted_position, detection["position"])
                    appearance_distance = self._appearance_distance(
                        state.get("appearance"),
                        detection.get("appearance")
                    )
                    bbox_iou = self._bbox_iou(
                        state.get("bbox"),
                        detection["track_info"]["bbox"]
                    )
                    same_original_id = (
                        state.get("last_original_track_id") ==
                        detection["original_track_id"]
                    )
                    mature_identity = (
                        state.get("observation_count", 1) >=
                        self.mature_identity_frames
                    )

                    allowed_distance = self.max_switch_distance + gap * 8
                    if gap == 1 and bbox_iou >= self.continuity_iou_threshold:
                        allowed_distance = max(allowed_distance, self.max_switch_distance * 1.8)

                    if interrupted and appearance_distance <= self.interruption_appearance_threshold:
                        allowed_distance = max(allowed_distance, 160 + gap * 10)

                    if distance > allowed_distance:
                        continue

                    if interrupted and appearance_distance > self.max_interrupted_appearance_distance:
                        continue

                    if (
                        mature_identity and
                        interrupted and
                        appearance_distance > self.mature_interrupted_appearance_distance
                    ):
                        continue

                    if (
                        mature_identity and
                        not same_original_id and
                        appearance_distance > self.mature_identity_appearance_distance
                    ):
                        continue

                    if (
                        appearance_distance > self.max_cross_team_appearance_distance and
                        not (gap == 1 and bbox_iou >= self.continuity_iou_threshold)
                    ):
                        continue

                    size_change = self._size_change(state["size"], detection["size"])
                    if size_change > 0.7:
                        continue

                    if (
                        interrupted and
                        appearance_distance > self.interruption_appearance_threshold
                    ):
                        continue

                    if interrupted:
                        interrupted_distance_ranks.setdefault(stable_id, []).append(
                            (distance, detection_index)
                        )

                    new_velocity = (
                        (detection["position"][0] - state["position"][0]) / gap,
                        (detection["position"][1] - state["position"][1]) / gap
                    )
                    velocity_change = self._distance(new_velocity, state["velocity"])
                    original_id_bonus = -25 if same_original_id else 0
                    overlap_bonus = 0
                    if gap == 1 and bbox_iou >= self.continuity_iou_threshold:
                        overlap_bonus = -95
                    elif gap <= 3 and bbox_iou >= 0.4:
                        overlap_bonus = -45
                    appearance_penalty = 0
                    if appearance_distance > self.suspicious_appearance_distance:
                        appearance_penalty = (
                            appearance_distance - self.suspicious_appearance_distance
                        ) * self.cross_team_appearance_penalty
                    if mature_identity and not same_original_id:
                        appearance_penalty += (
                            appearance_distance *
                            self.mature_identity_penalty
                        )

                    if interrupted:
                        predicted_state = stable_states[stable_id]
                        predicted_position = (
                            predicted_state["position"][0] + predicted_state["velocity"][0] * gap,
                            predicted_state["position"][1] + predicted_state["velocity"][1] * gap
                        )
                        previous_position = predicted_state["position"]
                        distance_from_last_seen = self._distance(
                            previous_position,
                            detection["position"]
                        )
                        cost = (
                            distance * 2.2 +
                            distance_from_last_seen * 0.35 +
                            size_change * 45 +
                            appearance_distance * 70 +
                            velocity_change * 5 +
                            gap * 2 +
                            original_id_bonus +
                            overlap_bonus +
                            appearance_penalty
                        )
                    else:
                        cost = (
                            distance +
                            size_change * 80 +
                            appearance_distance * 160 +
                            velocity_change * 10 +
                            gap * 4 +
                            original_id_bonus +
                            overlap_bonus +
                            appearance_penalty
                        )
                    candidate_matches.append((cost, stable_id, detection_index))

            nearest_detection_by_stable_id = {}
            for stable_id, distances in interrupted_distance_ranks.items():
                distances.sort(key=lambda item: item[0])
                nearest_detection_by_stable_id[stable_id] = distances[0][1]

            reranked_matches = []
            for cost, stable_id, detection_index in candidate_matches:
                nearest_detection = nearest_detection_by_stable_id.get(stable_id)
                if nearest_detection is not None and detection_index != nearest_detection:
                    cost += self.nearest_recovery_weight
                reranked_matches.append((cost, stable_id, detection_index))

            candidate_matches = sorted(reranked_matches, key=lambda item: item[0])
            assigned_stable_ids = set()
            assigned_detection_indices = set()
            frame_output = {}

            for _, stable_id, detection_index in candidate_matches:
                if stable_id in assigned_stable_ids or detection_index in assigned_detection_indices:
                    continue

                detection = detections[detection_index]
                frame_output[stable_id] = detection["track_info"]
                assigned_stable_ids.add(stable_id)
                assigned_detection_indices.add(detection_index)

            for detection_index, detection in enumerate(detections):
                if detection_index in assigned_detection_indices:
                    continue

                stable_id = next_stable_id
                next_stable_id += 1
                frame_output[stable_id] = detection["track_info"]
                assigned_stable_ids.add(stable_id)

            for stable_id, track_info in frame_output.items():
                position = get_foot_position(track_info["bbox"])
                previous_state = stable_states.get(stable_id)
                if previous_state is None:
                    velocity = (0, 0)
                else:
                    frame_gap = max(1, frame_num - previous_state["last_frame"])
                    velocity = (
                        (position[0] - previous_state["position"][0]) / frame_gap,
                        (position[1] - previous_state["position"][1]) / frame_gap
                    )

                old_appearance = previous_state.get("appearance") if previous_state else None
                new_appearance = track_info.get("appearance")
                interrupted = (
                    previous_state is not None and
                    frame_num - previous_state["last_frame"] >= self.interruption_gap_threshold
                )

                appearance_distance = self._appearance_distance(old_appearance, new_appearance)
                if old_appearance is not None and new_appearance is not None:
                    if appearance_distance > self.appearance_update_max_distance:
                        appearance = old_appearance
                    elif interrupted:
                        appearance = (
                            np.array(old_appearance) * 0.65 +
                            np.array(new_appearance) * 0.35
                        ).tolist()
                    else:
                        appearance = (
                            np.array(old_appearance) * 0.85 +
                            np.array(new_appearance) * 0.15
                        ).tolist()
                elif interrupted and new_appearance is not None:
                    appearance = new_appearance
                else:
                    appearance = new_appearance or old_appearance

                stable_states[stable_id] = {
                    "last_frame": frame_num,
                    "position": position,
                    "velocity": velocity,
                    "size": self._bbox_size(track_info["bbox"]),
                    "bbox": track_info["bbox"],
                    "appearance": appearance,
                    "last_original_track_id": track_info.get("original_track_id"),
                    "observation_count": (
                        previous_state.get("observation_count", 0) + 1
                        if previous_state
                        else 1
                    )
                }

            stabilized_tracks["players"].append(frame_output)

        return stabilized_tracks

    def filter_short_player_tracks(self, tracks):
        object_tracks = tracks["players"]
        summary = self._track_summary(object_tracks)
        valid_track_ids = {
            track_id
            for track_id, observations in summary.items()
            if self._max_consecutive_observations(observations)
            >= self.min_track_consecutive_frames
        }

        for frame_tracks in object_tracks:
            for track_id in list(frame_tracks.keys()):
                if track_id not in valid_track_ids:
                    del frame_tracks[track_id]

        return tracks

    def merge_fragmented_player_tracks(self, tracks):
        object_tracks = tracks["players"]
        summary = self._track_summary(object_tracks)
        id_mapping = {}

        ordered_track_ids = sorted(
            summary.keys(),
            key=lambda track_id: summary[track_id][0]["frame_num"]
        )

        for new_track_id in ordered_track_ids:
            new_observations = summary[new_track_id]
            new_start = new_observations[0]
            best_old_id = None
            best_score = None

            for old_track_id in ordered_track_ids:
                if old_track_id == new_track_id:
                    continue

                old_observations = summary[old_track_id]
                old_end = old_observations[-1]
                gap = new_start["frame_num"] - old_end["frame_num"]

                if gap <= 0 or gap > self.max_reid_gap:
                    continue

                predicted_position = self._estimate_position(
                    old_observations,
                    new_start["frame_num"]
                )
                distance = self._distance(predicted_position, new_start["position"])
                allowed_distance = self.max_reid_distance + gap * 4
                size_change = self._size_change(old_end["size"], new_start["size"])
                old_appearance = self._segment_appearance(
                    old_observations,
                    from_end=True
                )
                new_appearance = self._segment_appearance(new_observations)
                appearance_distance = self._appearance_distance(
                    old_appearance,
                    new_appearance
                )

                if (
                    distance > allowed_distance or
                    size_change > self.max_size_change or
                    appearance_distance > self.fragment_merge_appearance_threshold
                ):
                    continue

                score = (
                    distance +
                    gap * 2 +
                    size_change * 100 +
                    appearance_distance * self.fragment_merge_appearance_penalty
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_old_id = old_track_id

            if best_old_id is not None:
                id_mapping[new_track_id] = self._find_root_id(id_mapping, best_old_id)

        if not id_mapping:
            return tracks

        for frame_tracks in object_tracks:
            remapped_tracks = {}
            for track_id, track_info in frame_tracks.items():
                root_id = self._find_root_id(id_mapping, track_id)

                if root_id not in remapped_tracks:
                    remapped_tracks[root_id] = track_info
                    continue

                current_width = get_bbox_width(track_info["bbox"])
                existing_width = get_bbox_width(remapped_tracks[root_id]["bbox"])
                if current_width > existing_width:
                    remapped_tracks[root_id] = track_info

            frame_tracks.clear()
            frame_tracks.update(remapped_tracks)

        return tracks
