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
Transfrom the 2D coordinate to 3D coordinate with Perspective Transform, but it needs to revise based on concrete size of football field
