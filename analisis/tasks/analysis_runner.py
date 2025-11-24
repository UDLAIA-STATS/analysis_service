from pathlib import Path
import numpy as np
from typing import Dict, Mapping
from analisis.entities.trackers.ball_tracker import BallTracker
from analisis.entities.tracks.track_detail import TrackDetailBase, TrackPlayerDetail
from analisis.infraestructure.services.video_processing_service import extract_player_images, read_video
from analisis.services.r2_downloader import R2Downloader
from analisis.tasks.analysis.analysis_components import AnalysisComponents
from analisis.tasks.analysis import ( preprocessing, post_processing, assign_processing )
from analisis.entities.utils.json_transform import player_frames_to_json, player_tracks_to_json
from analisis.tasks.analysis.verify_model import prepare_model
from celery import shared_task
from decouple import config

@shared_task(bind=True)
def run_analysis(video_path: str):
    """
    Analiza un video y extrae información de los jugadores y del balón.
    
    Parameters:
    video_path (str): Ruta del archivo de video a analizar.
    
    Returns:
    None
    """

    prepare_model(
        model_path=AnalysisComponents.get_model_path(),
        source_path=AnalysisComponents.get_model_path().parent)


    downloader = R2Downloader({
        "BUCKET": config("R2_BUCKET"),
        "ACCESS_KEY_ID": config("R2_ACCESS_KEY_ID"),
        "SECRET_ACCESS_KEY": config("R2_SECRET_ACCESS_KEY"),
        "ENDPOINT": config("S3_CLIENT_ACCOUNT_ENDPOINT"),
    })

    downloader.stream_download(key=video_path, destination_path="../res/r2")
    download_path = downloader.build_destination_path(key=video_path)

    
    video_frames = read_video(download_path.as_posix())

    if not video_frames:
        print("No frames found in the video.")
        raise ValueError("No se pudo analizar el video, no se obtuvieron frames. Verifique el archivo de video e intente nuevamente.")

    components = AnalysisComponents(video_frames[0])

    preprocessing(components, video_frames)

    # Trackers post-processing
    post_processing(components, video_frames)

    team_ball_control = assign_processing(components, video_frames)
    player_tracks_json = player_frames_to_json(components.tracks_collection.tracks["players"])
    extract_player_images(video_frames, components.tracks_collection, '../res/output')

    return {
        "player_tracks": player_tracks_json,
        "team_ball_control": team_ball_control
    }
    
    