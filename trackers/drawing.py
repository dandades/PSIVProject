import cv2


TEAM_DRAW_COLORS = {
    0: (255, 0, 0),
    1: (0, 0, 255),
}


class DrawingMixin:
    def draw_player_box(self,frame,bbox,color,track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if track_id is None:
            return frame

        label = f"{track_id}"
        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )
        label_width = text_size[0] + 12
        label_height = text_size[1] + 10
        label_y1 = max(0, y1 - label_height)
        label_y2 = label_y1 + label_height
        label_x2 = min(frame.shape[1] - 1, x1 + label_width)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        return frame

    def draw_annotations(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = TEAM_DRAW_COLORS.get(player.get("team"), (255, 255, 255))
                frame = self.draw_player_box(frame, player["bbox"],color, track_id)

            output_video_frames.append(frame)

        return output_video_frames
