# Football Player Detection, Tracking & Team Classification in Video

This project analyzes football videos to detect valid players, keep a stable
identifier for each player, and classify each player as Team A or Team B based
on shirt colors.

## Pipeline

1. `main.py` reads the video and coordinates the full execution.
2. `extract_players.py` detects players with YOLO, applies ByteTrack, and uses `stubs/` as a track cache.
3. `trackers/` stabilizes IDs, merges fragmented tracks, and filters goalkeepers/referees.
4. `labeling.py` assigns a stable team to each `track_id`.
5. `team_assigner/` extracts visual shirt features and applies K-Means to separate Team A and Team B.
6. `trackers/drawing.py` draws the final inferences on the output video: boxes, IDs, and assigned team.
7. `manual_track_accuracy.py` generates evidence to manually evaluate tracks and teams.

## Libraries

- `OpenCV`: reading, writing, and manipulating video frames.
- `Ultralytics YOLO`: detecting players, goalkeepers, and referees.
- `Supervision`: ByteTrack integration for object tracking.
- `NumPy`: numerical calculations for boxes, positions, and visual features.
- `Scikit-learn`: K-Means and normalization for team classification.
- `PyTorch`: backend used by the YOLO model.

## Execution

Process the default video:

```bash
python main.py
```

Set input and output paths:

```bash
python main.py --input input_videos/110.mp4 --output output_videos/110_result.avi
```

Recalculate tracks without using cache:

```bash
python main.py --input input_videos/110.mp4 --output output_videos/110_result.avi --no-stubs
```

## Manual Evaluation

Generate one evidence sample for each `track_id` and save results as CSV/JSON:

```bash
python manual_track_accuracy.py --video input_videos/110.mp4
```

Options during the review:

| Key | Meaning |
| --- | --- |
| `s` | Valid player and correct team. |
| `n` | Valid player, incorrect team. |
| `t` | Not a valid player. |
| `k` | Skip ID. |
| `q` | Quit and save. |
