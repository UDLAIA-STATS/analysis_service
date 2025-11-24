from analisis.entities.trackers.ball_tracker import BallTracker
from analisis.tasks.analysis.analysis_components import AnalysisComponents
from typing import List
from cv2.typing import MatLike

def post_processing(components: AnalysisComponents, video_frames: List[MatLike]):
    """
    Process the output of the analysis.

    Prints the ball detection rate and interpolates the ball positions between frames.

    :param components: AnalysisComponents instance
    :param video_frames: List of frames of the video
    """
    
    ball_tracker = components.tracker.get_tracker("ball")

    if not isinstance(ball_tracker, BallTracker):
        raise TypeError("El tracker de balón no es una instancia de BallTracker.")

    detected_frames = sum(
        1
        for frame_tracks in components.tracks_collection.tracks["ball"].values()
        if 1 in frame_tracks and getattr(frame_tracks[1], "bbox", None) is not None
    )

    total_frames = len(video_frames)
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
    print(f"Ball detection rate: {detection_rate:.2%} ({detected_frames}/{total_frames} frames)")

    ball_tracker.interpolate_ball_positions(
        components.tracks_collection.tracks["ball"],
    )

    # Speed and distance estimation
    components.speed_and_distance_estimator.add_speed_and_distance_to_tracks(
        components.tracks_collection
    )