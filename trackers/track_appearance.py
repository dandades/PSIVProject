import cv2
import numpy as np


class TrackAppearanceMixin:
    def _clip_bbox(self, frame, bbox):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _extract_appearance(self, frame, bbox):
        clipped_bbox = self._clip_bbox(frame, bbox)
        if clipped_bbox is None:
            return None

        x1, y1, x2, y2 = clipped_bbox
        player_image = frame[y1:y2, x1:x2]
        if player_image.size == 0:
            return None

        height, width = player_image.shape[:2]
        torso_y1 = int(height * 0.15)
        torso_y2 = max(torso_y1 + 1, int(height * 0.65))
        torso_x1 = int(width * 0.12)
        torso_x2 = max(torso_x1 + 1, int(width * 0.88))
        torso = player_image[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        b, g, r = cv2.split(torso)

        grass_mask = (
            (hsv[:, :, 0] >= 35) &
            (hsv[:, :, 0] <= 90) &
            (hsv[:, :, 1] >= 35) &
            (hsv[:, :, 2] >= 35)
        )
        excess_green = (
            (g.astype(np.int16) - r.astype(np.int16) > 20) &
            (g.astype(np.int16) - b.astype(np.int16) > 20)
        )
        valid_mask = ~(grass_mask | excess_green)

        if np.count_nonzero(valid_mask) < torso.shape[0] * torso.shape[1] * 0.1:
            valid_mask = np.ones(torso.shape[:2], dtype=bool)

        selected_hsv = hsv[valid_mask]
        selected_lab = lab[valid_mask]
        if len(selected_hsv) == 0:
            return None

        hue = selected_hsv[:, 0]
        saturation = selected_hsv[:, 1]
        value = selected_hsv[:, 2]
        median_hsv = np.median(selected_hsv, axis=0)
        median_lab = np.median(selected_lab, axis=0)

        appearance = np.array([
            median_hsv[0] / 180.0,
            median_hsv[1] / 255.0,
            median_hsv[2] / 255.0,
            median_lab[0] / 255.0,
            median_lab[1] / 255.0,
            median_lab[2] / 255.0,
            np.mean((saturation < 55) & (value > 135)),
            np.mean(((hue < 12) | (hue > 165)) & (saturation > 70) & (value > 55)),
            np.mean((hue >= 18) & (hue <= 42) & (saturation > 60) & (value > 70)),
            np.mean((hue >= 90) & (hue <= 135) & (saturation > 45) & (value > 45)),
        ], dtype=np.float32)

        return appearance.tolist()

    def ensure_track_appearances(self, frames, tracks):
        for frame_num, frame_tracks in enumerate(tracks["players"]):
            if frame_num >= len(frames):
                break

            frame = frames[frame_num]
            for track_info in frame_tracks.values():
                if track_info.get("appearance") is None:
                    track_info["appearance"] = self._extract_appearance(
                        frame,
                        track_info["bbox"]
                    )

        return tracks
