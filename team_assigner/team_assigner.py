from .player_features import PlayerFeatureExtractor
from .team_model import TeamModelMixin


class TeamAssigner(PlayerFeatureExtractor, TeamModelMixin):
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.scaler = None
        self.track_features = {}
        self.team_centers = None
        self.player_team_votes = {}
        self.invalid_player_ids = set()
        self.valid_player_ids = set()
        self.track_outlier_scores = {}

    def filter_invalid_players(self, player_tracks):
        if not self.invalid_player_ids:
            return player_tracks

        for frame_tracks in player_tracks:
            for player_id in list(frame_tracks.keys()):
                if player_id in self.invalid_player_ids:
                    del frame_tracks[player_id]

        return player_tracks

    def assign_team_color(self, frame, player_detections):
        self.assign_team_color_from_tracks([frame], [player_detections], sample_every=1)

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.invalid_player_ids:
            return None

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None or self.scaler is None:
            return 0

        feature_vector, _ = self.extract_player_features(frame, player_bbox)
        if feature_vector is None:
            return 0

        scaled_feature = self.scaler.transform(feature_vector.reshape(1, -1))
        team_id = int(self.kmeans.predict(scaled_feature)[0])

        self.player_team_dict[player_id] = team_id

        return team_id
