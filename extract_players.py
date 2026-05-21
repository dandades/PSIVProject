import os

from trackers import Tracker


def get_track_stub_path(input_video_path, min_track_frames):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    return os.path.join(
        "stubs",
        f"track_stubs_fragment_appearance_min{min_track_frames}_{video_name}.pkl"
    )


def extract_player_tracks(
    frames,
    input_video_path,
    model_path,
    use_stubs=True,
    min_track_frames=5,
    stub_path=None
):
    tracker = Tracker(
        model_path,
        min_track_consecutive_frames=min_track_frames
    )
    resolved_stub_path = (
        None
        if not use_stubs
        else stub_path or get_track_stub_path(input_video_path, min_track_frames)
    )

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=use_stubs,
        stub_path=resolved_stub_path
    )
    tracker.add_position_to_tracks(tracks)

    return tracker, tracks, resolved_stub_path
