# FootballSystem

### utils
Define methods to measure distances and read/save vedios.

### TeamAssigner 
Use the K-Means clustering algorithm to extract color information from images and perform clustering, thereby determining which team each player belongs to.

### player_ball_assigner
Assign the ball to specific player who is closed to it for further calculating of ball control rate.

### camera_movement_estimator
Estimate the movement of cameras between continuous frames by LK optical flow.

### view_transformer
Transfrom the pixel coordinate to meter coordinate with Perspective Transform, but it needs to revise based on concrete size of football field

### SpeedAndDistance_Estimator
Calculate speed and distance of players between each 5 frames

### trackers
This code forms the core of the detection and tracking functionality. It first uses a finetuned YOLOv8x model to perform object detection on each frame, then identify players, goalkeepers, referees, and the football on the field. After detection, the system applies the ByteTrack multi-object tracking algorithm to associate the same targets across consecutive frames, thereby assigning consistent Track IDs to each object for continuous tracking.

For the football, which is prone to occlusion or missed detections during the game, the system integrates an additional Kalman filter. If the ball is successfully detected in the current frame, the Kalman filter is updated using the detection results. If the ball is not detected, the Kalman filter predicts its position, and the predicted bounding box is added to the tracking results for the current frame.
