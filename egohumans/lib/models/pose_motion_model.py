import numpy as np
from pykalman import KalmanFilter

class PoseMotionModel:
    def __init__(self, num_keypoints=17):
        self.num_keypoints = num_keypoints
        self.kf = KalmanFilter(transition_matrices=np.eye(self.num_keypoints*3), observation_matrices=np.eye(self.num_keypoints*3))
        self.state_covariance = None  # Add this line to initialize the state covariance
        return

    def train(self, pose3d_sequence):
        measurements = np.reshape(pose3d_sequence, (pose3d_sequence.shape[0], -1))
        (smoothed_state_means, smoothed_state_covariances) = self.kf.smooth(measurements)
        self.kf = KalmanFilter(
            transition_matrices=np.eye(self.num_keypoints*3), 
            observation_matrices=np.eye(self.num_keypoints*3), 
            initial_state_mean=smoothed_state_means[0], 
            initial_state_covariance=smoothed_state_covariances[0]
        )
        self.state_covariance = smoothed_state_covariances[0]  # Update the state covariance after training
        return

    def predict_next_pose(self, current_pose):
        current_pose = np.reshape(current_pose, -1)
        next_state_mean, self.state_covariance = self.kf.filter_update(current_pose, self.state_covariance)  # Update the state covariance
        return np.reshape(next_state_mean, (self.num_keypoints, 3))
