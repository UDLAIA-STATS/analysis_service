import numpy as np
from analisis.entities.tracks.track_detail import TrackDetailBase, TrackPlayerDetail
from analisis.tasks.analysis.analysis_components import AnalysisComponents
from typing import List
from cv2.typing import MatLike

def assign_processing(components: AnalysisComponents, video_frames: List[MatLike]):
    # Assign players team
    """
    Assign team and ball acquisition to players in a video.

    Parameters
    ----------
    components : AnalysisComponents
        Object containing the tracks collection and team assigner.
    video_frames : List[MatLike]
        List of frames from the video.

    Returns
    -------
    team_ball_control : np.ndarray
        Array of length equal to the number of frames in the video, where each element is the team id of the player who has the ball in that frame.
    """
    for frame_num, track_content in components.tracks_collection.tracks["players"].items():
        print("Assigning player teams...", components.tracks_collection.tracks["players"][frame_num].values())
        components.team_assigner.assign_team_color(
            video_frames[0],
            components.tracks_collection.tracks["players"][frame_num])
        continue

    for frame_num, player_track in components.tracks_collection.tracks["players"].items():
        for player_id, track in player_track.items():
            team = components.team_assigner.get_player_team(
                video_frames[frame_num],
                track.bbox,
                player_id
            )
            player_tracker = TrackPlayerDetail(**track.model_dump())
            player_tracker.update(team=team, team_color=components.team_assigner.team_colors[team])
            # player_tracker.team = team
            # player_tracker.team_color = team_assigner.team_colors[team]
            components.tracks_collection.update_track(
                entity_type="players",
                frame_num=frame_num,
                track_id=player_id,
                track_detail=player_tracker
            )

    # Assign Ball Acquisition
    team_ball_control = []
    print("Assigning ball to players...")
    print("Total frames to assign ball: ", len(components.tracks_collection.tracks["players"]))
    for frame_num, player_track in components.tracks_collection.tracks["players"].items():
        print("Processing frame number: ", frame_num)
        print("Ball tracks length: ", components.tracks_collection.tracks['ball'].values())
        ball_frame_tracks = components.tracks_collection.tracks.get('ball', {}).get(int(frame_num))
        
        if not ball_frame_tracks or 1 not in ball_frame_tracks:
            print("No ball track for this frame, appending -1")
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
            continue
        
        ball_detail = next(iter(ball_frame_tracks.values()))
        ball_bbox = ball_detail.bbox
        assigned_player = components.player_assigner.assign_ball_to_player(
            player_track, ball_bbox)
        print("Assigned player: ", assigned_player)

        if assigned_player != -1:
            player_base: TrackDetailBase = player_track[assigned_player]
            print("Actual player track: ", player_base)
            dict_player = player_base.model_dump()
            print("Updated player team: ", dict_player['team'])
            print("Updated player team color: ", dict_player['team_color'])
            print("Dict player: ", TrackPlayerDetail.model_validate(dict_player))
            player: TrackPlayerDetail = TrackPlayerDetail(**dict_player)
            player.update(has_ball=True)
            # player.has_ball = True
            team_ball_control.append(player.team)
            components.tracks_collection.update_track(
                entity_type="players",
                frame_num=frame_num,
                track_id=assigned_player,
                track_detail=player
            )
        else:
            # Handle first frame case
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else -1)

    team_ball_control = np.array(team_ball_control)
    
    return team_ball_control