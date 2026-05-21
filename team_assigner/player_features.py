import cv2
import numpy as np
from sklearn.cluster import KMeans


class PlayerFeatureExtractor:
    def _clip_bbox(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, min(int(x1), frame_w - 1))
        x2 = max(0, min(int(x2), frame_w))
        y1 = max(0, min(int(y1), frame_h - 1))
        y2 = max(0, min(int(y2), frame_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def get_player_top_image(self, frame, bbox):
        clipped_bbox = self._clip_bbox(frame, bbox)
        if clipped_bbox is None:
            return None

        x1, y1, x2, y2 = clipped_bbox
        player_image = frame[y1:y2, x1:x2]

        if player_image.size == 0:
            return None

        height, width = player_image.shape[:2]
        torso_y1 = int(height * 0.15)
        torso_y2 = max(torso_y1 + 1, int(height * 0.62))
        torso_x1 = int(width * 0.12)
        torso_x2 = max(torso_x1 + 1, int(width * 0.88))
        top_image = player_image[torso_y1:torso_y2, torso_x1:torso_x2]

        if top_image.size == 0:
            return None

        return top_image

    def remove_grass(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(image)

        lower_green = np.array([35, 35, 35])
        upper_green = np.array([90, 255, 255])
        hsv_grass = cv2.inRange(hsv, lower_green, upper_green)

        excess_green = (
            (g.astype(np.int16) - r.astype(np.int16) > 20) &
            (g.astype(np.int16) - b.astype(np.int16) > 20)
        ).astype(np.uint8) * 255

        grass_mask = cv2.bitwise_or(hsv_grass, excess_green)
        kernel = np.ones((3, 3), np.uint8)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)

        player_mask = cv2.bitwise_not(grass_mask)
        valid_pixels = cv2.countNonZero(player_mask)

        if valid_pixels < image.shape[0] * image.shape[1] * 0.1:
            player_mask = np.full(image.shape[:2], 255, dtype=np.uint8)

        return player_mask

    def _sample_pixels(self, pixels, max_pixels=2000):
        if len(pixels) <= max_pixels:
            return pixels

        sample_indices = np.linspace(0, len(pixels) - 1, max_pixels).astype(int)
        return pixels[sample_indices]

    def _get_shirt_pixels(self, image, player_mask):
        pixels = image[player_mask > 0]
        if len(pixels) == 0:
            return None

        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        high_saturation = (hsv_pixels[:, 1] > 35) & (hsv_pixels[:, 2] > 35)
        white_like = (hsv_pixels[:, 1] < 55) & (hsv_pixels[:, 2] > 135)
        useful_pixels = pixels[high_saturation | white_like]

        if np.mean(white_like) > 0.35:
            useful_pixels = pixels[white_like | high_saturation]

        if len(useful_pixels) < max(20, len(pixels) * 0.2):
            useful_pixels = pixels

        sampled_pixels = self._sample_pixels(useful_pixels)
        n_clusters = min(3, len(sampled_pixels))
        if n_clusters < 2:
            return useful_pixels

        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=5, random_state=0)
        labels = kmeans.fit_predict(sampled_pixels.astype(np.float32))

        cluster_scores = []
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_pixels = sampled_pixels[labels == cluster_id]
            if len(cluster_pixels) == 0:
                cluster_scores.append(-1)
                cluster_stats.append((0, 0, 0, np.zeros(3)))
                continue

            cluster_hsv = cv2.cvtColor(
                cluster_pixels.reshape(-1, 1, 3),
                cv2.COLOR_BGR2HSV
            ).reshape(-1, 3)
            area_ratio = len(cluster_pixels) / len(sampled_pixels)
            saturation = np.median(cluster_hsv[:, 1]) / 255.0
            value = np.median(cluster_hsv[:, 2]) / 255.0
            center = np.median(cluster_pixels, axis=0)

            white_bonus = 0.35 if saturation < 0.22 and value > 0.62 else 0.0
            dark_penalty = 0.2 if value < 0.12 else 0.0
            cluster_scores.append(area_ratio + 0.35 * saturation + white_bonus - dark_penalty)
            cluster_stats.append((area_ratio, saturation, value, center))

        saturated_clusters = [
            stat
            for stat in cluster_stats
            if stat[0] >= 0.15 and stat[1] >= 0.25 and stat[2] >= 0.2
        ]
        if len(saturated_clusters) >= 2:
            centers = np.array([stat[3] for stat in saturated_clusters])
            max_center_distance = 0
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    max_center_distance = max(
                        max_center_distance,
                        np.linalg.norm(centers[i] - centers[j])
                    )

            if max_center_distance > 55:
                return useful_pixels

        shirt_cluster = int(np.argmax(cluster_scores))
        shirt_center = kmeans.cluster_centers_[shirt_cluster]
        distances = np.linalg.norm(useful_pixels.astype(np.float32) - shirt_center, axis=1)
        distance_threshold = np.percentile(distances, 85)
        shirt_pixels = useful_pixels[distances <= distance_threshold]

        if len(shirt_pixels) < 20:
            return useful_pixels

        return shirt_pixels

    def extract_player_features(self, frame, bbox):
        top_image = self.get_player_top_image(frame, bbox)
        if top_image is None:
            return None, None

        player_mask = self.remove_grass(top_image)
        shirt_pixels = self._get_shirt_pixels(top_image, player_mask)
        if shirt_pixels is None or len(shirt_pixels) == 0:
            return None, None

        hsv_pixels = cv2.cvtColor(
            shirt_pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)
        lab_pixels = cv2.cvtColor(
            shirt_pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2LAB
        ).reshape(-1, 3)

        bgr_median = np.median(shirt_pixels, axis=0)
        hsv_median = np.median(hsv_pixels, axis=0)
        hsv_std = np.std(hsv_pixels, axis=0)
        lab_median = np.median(lab_pixels, axis=0)
        lab_std = np.std(lab_pixels, axis=0)

        gray = cv2.cvtColor(top_image, cv2.COLOR_BGR2GRAY)
        gabor_kernel = cv2.getGaborKernel(
            (9, 9),
            4.0,
            np.pi / 4,
            10.0,
            0.5,
            0,
            ktype=cv2.CV_32F
        )
        gabor_response = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
        gabor_value = np.mean(np.abs(gabor_response[player_mask > 0])) / 255.0

        contours, _ = cv2.findContours(
            player_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contour_area = sum(cv2.contourArea(contour) for contour in contours)
        contour_value = contour_area / max(1, cv2.countNonZero(player_mask))

        hue = hsv_pixels[:, 0]
        saturation = hsv_pixels[:, 1]
        value = hsv_pixels[:, 2]
        lightness_p25 = np.percentile(lab_pixels[:, 0], 25)
        lightness_p75 = np.percentile(lab_pixels[:, 0], 75)

        white_ratio = np.mean((saturation < 45) & (value > 145))
        dark_ratio = np.mean(value < 70)
        red_ratio = np.mean(((hue < 12) | (hue > 165)) & (saturation > 70) & (value > 55))
        yellow_ratio = np.mean((hue >= 18) & (hue <= 42) & (saturation > 60) & (value > 70))
        blue_ratio = np.mean((hue >= 90) & (hue <= 135) & (saturation > 45) & (value > 45))

        feature_vector = np.array([
            hsv_median[0],
            hsv_median[1],
            hsv_median[2],
            hsv_std[0],
            hsv_std[1],
            hsv_std[2],
            lab_median[0],
            lab_median[1],
            lab_median[2],
            lab_std[0],
            lab_std[1],
            lab_std[2],
            lightness_p25,
            lightness_p75,
            white_ratio * 255,
            dark_ratio * 255,
            red_ratio * 255,
            yellow_ratio * 255,
            blue_ratio * 255,
            gabor_value,
            contour_value
        ], dtype=np.float32)

        representative_color = bgr_median

        return feature_vector, representative_color
