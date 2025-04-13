from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class KalmanBallTracker:
    def __init__(self, init_bbox):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01
        cx, cy = get_center_of_bbox(init_bbox)
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)

    def update(self, observed_bbox, confidence=1.0):
        self.set_covariances(confidence)
        cx, cy = get_center_of_bbox(observed_bbox)
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measurement)

    def predict(self):
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])

    def set_covariances(self, confidence):
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * (0.05 + (1 - confidence) * 0.1)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * (0.1 + (1 - confidence) * 0.2)

    def get_predicted_bbox(self, width=30):
        pred_x, pred_y = self.predict()
        return [pred_x - width // 2, pred_y - width // 2, pred_x + width // 2, pred_y + width // 2]


class Tracker:
    def __init__(self, model_path, ball_conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.ball_conf_threshold = ball_conf_threshold
        self.kalman_ball_tracker = None

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['goalkeeper']:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                conf = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball'] and conf >= self.ball_conf_threshold:
                    tracks["ball"][frame_num][1] = {"bbox": bbox, "conf": conf}
                    if self.kalman_ball_tracker is None:
                        self.kalman_ball_tracker = KalmanBallTracker(bbox)
                    else:
                        self.kalman_ball_tracker.update(bbox, confidence=conf)

            if 1 not in tracks["ball"][frame_num]:
                byte_track_ball = None
                for track in detection_with_tracks:
                    bbox = track[0].tolist()
                    cls_id = track[3]
                    track_id = track[4]
                    if cls_id == cls_names_inv['ball'] and track_id == 1:
                        byte_track_ball = bbox
                        break

                if byte_track_ball:
                    if self.kalman_ball_tracker is None:
                        self.kalman_ball_tracker = KalmanBallTracker(byte_track_ball)
                    else:
                        self.kalman_ball_tracker.update(byte_track_ball, confidence=0.5)
                    w = get_bbox_width(byte_track_ball)
                    pred_bbox = self.kalman_ball_tracker.get_predicted_bbox(width=w)
                    tracks["ball"][frame_num][1] = {"bbox": pred_bbox, "conf": -1, "predicted": True}
                elif self.kalman_ball_tracker:
                    pred_bbox = self.kalman_ball_tracker.get_predicted_bbox(width=30)
                    tracks["ball"][frame_num][1] = {"bbox": pred_bbox, "conf": -1, "predicted": True}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 20:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_traingle(self, frame, bbox, color, predicted=False):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        if predicted:
            color = (128, 128, 128)

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, player_dict):
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_frames = (team_ball_control_till_frame == 1).sum()
        team_2_frames = (team_ball_control_till_frame == 2).sum()
        total = team_1_frames + team_2_frames + 1e-6

        if not hasattr(self, 'team1_color'):
            self.team1_color = None
            self.team2_color = None
            for player in player_dict.values():
                color = player.get("team_color", (0, 0, 255))
                if self.team1_color is None and np.array_equal(color, (0, 0, 255)):
                    self.team1_color = color
                elif self.team2_color is None and not np.array_equal(color, (0, 0, 255)):
                    self.team2_color = color
                if self.team1_color and self.team2_color:
                    break

        team1_color = self.team1_color if hasattr(self, "team1_color") else (255, 0, 0)
        team2_color = self.team2_color if hasattr(self, 'team2_color') else (0, 255, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Team 1
        cv2.rectangle(frame, (1370, 890), (1390, 910), team1_color, -1)
        cv2.putText(frame, f"Team 1 Ball Control: {team_1_frames / total * 100:.2f}%",
                    (1410, 905), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Team 2
        cv2.rectangle(frame, (1370, 940), (1390, 960), team2_color, -1)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_frames / total * 100:.2f}%",
                    (1410, 955), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for track_id, gk in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, gk["bbox"], (0, 255, 0), track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                predicted = ball.get("predicted", False)
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0), predicted=predicted)

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, player_dict)
            output_video_frames.append(frame)

        return output_video_frames