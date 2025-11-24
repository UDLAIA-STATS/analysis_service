from typing import List
from analisis.tasks.analysis.analysis_components import AnalysisComponents
from cv2.typing import MatLike


def preprocessing(components: AnalysisComponents, video_frames: List[MatLike]):
    """
    Preprocess video frames to get tracks and estimate camera movement.

    Steps:
    1. Get object tracks from video frames.
    2. Add position to tracks.
    3. Estimate camera movement from video frames.
    4. Add adjusted positions to tracks.
    5. Add transformed position to tracks.
    """
    components.tracker.get_object_tracks(
        frames=video_frames,
        read_from_stub=False,
        tracks_collection=components.tracks_collection
    )
    components.tracker.add_position_to_tracks(components.tracks_collection)


    #Estimate camera movement
    camera_movement_per_frame = components.camera_movement_estimator.get_camera_movement(
        frames=video_frames,
        read_from_stub=False
        )
    components.camera_movement_estimator.add_adjust_positions_to_tracks(
        camera_movement_per_frame,
        components.tracks_collection
    )
    components.view_transformer.add_transformed_position_to_tracks(
        components.tracks_collection
    )