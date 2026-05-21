import argparse

from extract_players import extract_player_tracks
from labeling import label_player_teams
from utils import read_video, save_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run football video analysis on a selected input video."
    )
    parser.add_argument(
        "--input",
        default="input_videos/110.mp4",
        help="Path to the input video to analyze."
    )
    parser.add_argument(
        "--output",
        default="output_videos/output_video.avi",
        help="Path where the annotated output video will be saved."
    )
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to the YOLO model."
    )
    parser.add_argument(
        "--no-stubs",
        action="store_true",
        help="Recalculate detections instead of loading/saving cached track stubs."
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Minimum consecutive frames required to keep a player track."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_frames = read_video(args.input)

    tracker, tracks, _ = extract_player_tracks(
        video_frames,
        input_video_path=args.input,
        model_path=args.model,
        use_stubs=not args.no_stubs,
        min_track_frames=args.min_track_frames
    )

    label_player_teams(video_frames, tracks)

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, args.output)

if __name__ == '__main__':
    main()
