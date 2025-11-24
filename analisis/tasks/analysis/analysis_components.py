from pathlib import Path

from analisis.entities.collection.track_collection import TrackCollection
from analisis.entities.trackers.ball_tracker import BallTracker
from analisis.entities.trackers.player_tracker import PlayerTracker
from analisis.entities.utils.singleton import Singleton
from analisis.infraestructure.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from analisis.infraestructure.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from analisis.infraestructure.speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
from analisis.infraestructure.team_assigner.team_assigner import TeamAssigner
from analisis.infraestructure.trackers.services.tracker_service import TrackerService
from analisis.infraestructure.view_transformer.view_transformer import ViewTransformer
from cv2.typing import MatLike


model_path = Path("../res/models/football_model.torchscript")

class AnalysisComponents(metaclass=Singleton):
    def __init__(self, video_frame: MatLike) -> None:
        self.tracker: TrackerService = TrackerService(model_path.as_posix())
        self.tracker.create_tracker("player", PlayerTracker)
        self.tracker.create_tracker("ball", BallTracker)

        self.tracks_collection: TrackCollection = TrackCollection()
        self.view_transformer: ViewTransformer = ViewTransformer()
        self.speed_and_distance_estimator: SpeedAndDistanceEstimator = SpeedAndDistanceEstimator()
        self.team_assigner: TeamAssigner = TeamAssigner()
        self.player_assigner: PlayerBallAssigner = PlayerBallAssigner()
        self.camera_movement_estimator: CameraMovementEstimator = CameraMovementEstimator(video_frame)


    @staticmethod
    def get_model_path() -> Path:
        return model_path