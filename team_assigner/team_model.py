import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TeamModelMixin:
    def _sample_frame_nums(self, player_tracks, sample_every):
        sampled_frame_nums = list(range(0, len(player_tracks), sample_every))
        if len(sampled_frame_nums) < len(player_tracks):
            sampled_frame_nums.extend(
                frame_num
                for frame_num in range(len(player_tracks))
                if frame_num not in sampled_frame_nums
            )

        return sampled_frame_nums

    def _build_track_level_samples(self, frames, player_tracks, sample_every):
        features_by_track = {}
        colors_by_track = {}

        for frame_num in self._sample_frame_nums(player_tracks, sample_every):
            if frame_num >= len(frames):
                break

            frame = frames[frame_num]
            for player_id, player_detection in player_tracks[frame_num].items():
                feature_vector, representative_color = self.extract_player_features(
                    frame,
                    player_detection["bbox"]
                )

                if feature_vector is None:
                    continue

                features_by_track.setdefault(player_id, []).append(feature_vector)
                colors_by_track.setdefault(player_id, []).append(representative_color)

        return features_by_track, colors_by_track

    def _select_clean_samples(self, feature_samples, color_samples):
        features = np.array(feature_samples, dtype=np.float32)
        colors = np.array(color_samples, dtype=np.float32)

        if len(features) <= 3:
            return features, colors

        feature_center = np.median(features, axis=0)
        color_center = np.median(colors, axis=0)

        feature_mad = np.median(np.abs(features - feature_center), axis=0) + 1e-6
        color_distances = np.linalg.norm(colors - color_center, axis=1)
        feature_distances = np.median(np.abs(features - feature_center) / feature_mad, axis=1)

        color_threshold = np.percentile(color_distances, 70)
        feature_threshold = np.percentile(feature_distances, 70)
        clean_mask = (
            (color_distances <= color_threshold) &
            (feature_distances <= feature_threshold)
        )

        if clean_mask.sum() < max(2, len(features) // 3):
            keep_count = max(2, len(features) // 2)
            combined_score = (
                color_distances / (np.median(color_distances) + 1e-6) +
                feature_distances / (np.median(feature_distances) + 1e-6)
            )
            keep_indices = np.argsort(combined_score)[:keep_count]
            clean_mask = np.zeros(len(features), dtype=bool)
            clean_mask[keep_indices] = True

        return features[clean_mask], colors[clean_mask]

    def _build_clean_sample_set(self, features_by_track, colors_by_track):
        sample_features = []
        sample_colors = []
        sample_track_ids = []
        clean_features_by_track = {}
        clean_colors_by_track = {}

        for player_id, feature_samples in features_by_track.items():
            if len(feature_samples) < 2:
                continue

            clean_features, clean_colors = self._select_clean_samples(
                feature_samples,
                colors_by_track[player_id]
            )

            if len(clean_features) < 2:
                continue

            clean_features_by_track[player_id] = clean_features
            clean_colors_by_track[player_id] = clean_colors

            for feature_vector, representative_color in zip(clean_features, clean_colors):
                sample_features.append(feature_vector)
                sample_colors.append(representative_color)
                sample_track_ids.append(player_id)

        return (
            sample_track_ids,
            sample_features,
            sample_colors,
            clean_features_by_track,
            clean_colors_by_track
        )

    def _assign_tracks_by_votes(self, sample_track_ids, labels, distances):
        votes_by_track = {}

        for player_id, label, sample_distances in zip(sample_track_ids, labels, distances):
            label = int(label)
            margin = float(np.partition(sample_distances, 1)[1] - sample_distances[label])
            weight = max(0.1, margin)
            votes_by_track.setdefault(player_id, {0: 0.0, 1: 0.0})
            votes_by_track[player_id][label] += weight

        player_team_dict = {}
        for player_id, votes in votes_by_track.items():
            player_team_dict[player_id] = 0 if votes[0] >= votes[1] else 1

        return player_team_dict, votes_by_track

    def _robust_upper_threshold(self, values, percentile=85, mad_scale=3.0):
        values = np.array(values, dtype=np.float32)
        if len(values) == 0:
            return float("inf")

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) + 1e-6
        robust_threshold = median + mad_scale * 1.4826 * mad
        percentile_threshold = float(np.percentile(values, percentile))

        return max(robust_threshold, percentile_threshold)

    def _mark_invalid_player_tracks(
        self,
        sample_track_ids,
        labels,
        distances,
        clean_colors_by_track
    ):
        distance_by_track = {}
        for player_id, label, sample_distances in zip(sample_track_ids, labels, distances):
            assigned_distance = float(sample_distances[int(label)])
            distance_by_track.setdefault(player_id, []).append(assigned_distance)

        all_assigned_distances = [
            distance
            for track_distances in distance_by_track.values()
            for distance in track_distances
        ]
        feature_threshold = self._robust_upper_threshold(
            all_assigned_distances,
            percentile=86,
            mad_scale=3.2
        )

        color_distances = []
        color_distance_by_track = {}
        for player_id, clean_colors in clean_colors_by_track.items():
            if player_id not in self.player_team_dict:
                continue

            team_id = self.player_team_dict[player_id]
            player_color = np.median(clean_colors, axis=0)
            team_color = np.array(self.team_colors[team_id], dtype=np.float32)
            color_distance = float(np.linalg.norm(player_color - team_color))
            color_distance_by_track[player_id] = color_distance
            color_distances.append(color_distance)

        color_threshold = max(
            45.0,
            self._robust_upper_threshold(
                color_distances,
                percentile=84,
                mad_scale=2.8
            )
        )

        self.invalid_player_ids = set()
        self.valid_player_ids = set(self.player_team_dict.keys())
        self.track_outlier_scores = {}

        for player_id, track_distances in distance_by_track.items():
            team_id = self.player_team_dict.get(player_id)
            if team_id is None:
                continue

            feature_distance = float(np.median(track_distances))
            color_distance = color_distance_by_track.get(player_id, 0.0)
            votes = self.player_team_votes.get(player_id, {0: 0.0, 1: 0.0})
            total_votes = votes.get(0, 0.0) + votes.get(1, 0.0) + 1e-6
            vote_margin = abs(votes.get(0, 0.0) - votes.get(1, 0.0)) / total_votes

            feature_ratio = feature_distance / max(feature_threshold, 1e-6)
            color_ratio = color_distance / max(color_threshold, 1e-6)
            self.track_outlier_scores[player_id] = {
                "team": team_id,
                "feature_distance": feature_distance,
                "feature_threshold": feature_threshold,
                "color_distance": color_distance,
                "color_threshold": color_threshold,
                "vote_margin": vote_margin
            }

            is_feature_outlier = feature_distance > feature_threshold * 1.25
            is_color_outlier = color_distance > color_threshold * 1.15
            is_extreme_feature_outlier = feature_distance > feature_threshold * 1.9
            is_extreme_color_outlier = color_distance > color_threshold * 1.9

            if (
                (is_feature_outlier and is_color_outlier and vote_margin < 0.45) or
                (is_extreme_feature_outlier and is_color_outlier) or
                (is_extreme_color_outlier and is_feature_outlier)
            ):
                self.invalid_player_ids.add(player_id)

        self.valid_player_ids -= self.invalid_player_ids

    def assign_team_color_from_tracks(self, frames, player_tracks, sample_every=10):
        features_by_track, colors_by_track = self._build_track_level_samples(
            frames,
            player_tracks,
            sample_every
        )
        (
            sample_track_ids,
            features,
            colors,
            clean_features_by_track,
            clean_colors_by_track
        ) = self._build_clean_sample_set(features_by_track, colors_by_track)

        if len(features) < 2:
            raise ValueError(
                "Not enough clean player samples to classify teams. "
                "Try a video segment with at least two visible players."
            )

        features = np.array(features, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=50, random_state=0)
        labels = self.kmeans.fit_predict(scaled_features)
        distances = self.kmeans.transform(scaled_features)
        self.team_centers = self.kmeans.cluster_centers_

        self.track_features = {
            player_id: np.median(clean_features, axis=0)
            for player_id, clean_features in clean_features_by_track.items()
        }
        self.player_team_dict, self.player_team_votes = self._assign_tracks_by_votes(
            sample_track_ids,
            labels,
            distances
        )

        for team_id in (0, 1):
            team_colors = colors[labels == team_id]
            if len(team_colors) == 0:
                self.team_colors[team_id] = (0, 0, 255)
                continue

            color = np.median(team_colors, axis=0)
            self.team_colors[team_id] = tuple(int(channel) for channel in color)

        for player_id, clean_colors in clean_colors_by_track.items():
            if player_id not in self.player_team_dict:
                continue

            team_id = self.player_team_dict[player_id]
            votes = self.player_team_votes.get(player_id, {})
            if votes.get(team_id, 0) < votes.get(1 - team_id, 0) * 1.2:
                continue

            player_color = np.median(clean_colors, axis=0)
            team_color = np.array(self.team_colors[team_id])
            self.team_colors[team_id] = tuple(
                int(channel)
                for channel in ((team_color * 0.8) + (player_color * 0.2))
            )

        self._mark_invalid_player_tracks(
            sample_track_ids,
            labels,
            distances,
            clean_colors_by_track
        )
