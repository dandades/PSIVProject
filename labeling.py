from team_assigner import TeamAssigner


def label_player_teams(frames, tracks, sample_every=10):
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color_from_tracks(
        frames,
        tracks["players"],
        sample_every=sample_every
    )

    for frame_num, player_track in enumerate(tracks["players"]):
        if frame_num >= len(frames):
            break

        for player_id, track in list(player_track.items()):
            team = team_assigner.get_player_team(
                frames[frame_num],
                track["bbox"],
                player_id
            )
            if team is None:
                continue

            track["team"] = team
            track["team_color"] = team_assigner.team_colors.get(team)

    team_assigner.filter_invalid_players(tracks["players"])
    return team_assigner
