from typing import Dict
from analisis.entities.tracks.track_detail import TrackDetailBase

def player_tracks_to_json(player_tracks: Dict[int, TrackDetailBase]) -> Dict[int, dict]:
    return {track_id: track.to_json() for track_id, track in player_tracks.items()}

def player_frames_to_json(player_frames: Dict[int, Dict[int, TrackDetailBase]]) -> Dict[int, Dict[int, dict]]:
    return {frame_num: player_tracks_to_json(tracks) for frame_num, tracks in player_frames.items()}