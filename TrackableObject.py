# trackable_object.py
# December 2020
# Storing the center coordinates locations and other info for tracked objects.

from typing import Tuple


class TrackableObject:
    def __init__(
            self,
            objectID,
            centroid,
            radius,
            timestamp,
            label='',
            attribute_smoothing=1,  # seconds
            out_of_frame_persistence=5):
        self.objectID = objectID
        self.radius = radius
        self.centroids = {}  # Maybe change to Ordered Dict
        self.velocity = {}  # Maybe change to Ordered Dict
        self.acceleration = {}  # Maybe change to Ordered Dict
        self.centroids[timestamp] = centroid
        self.label = label
        self.attribute_smoothing = attribute_smoothing
        self.oof_persistence = out_of_frame_persistence
        self.oof_count = 0
        self.predicted: Tuple = None

    ### NEED TO IMPLEMENT ###
    # Predicts the objects position at timestamp based on current attributes
    def predict_position(self, timestamp):
        self.predicted = None

    ### NEED TO IMPLEMENT - error margins are in meters ###
    # Determines if the predicted position is close enough to the passed in
    # centroid/radius to be considered the same object.
    def is_collided(self,
                    other_centroid,
                    other_radius,
                    other_label='',
                    error_margin_centroid=1,
                    error_margin_radius=.5) -> bool:
        return False

    # Updates object with new information
    def update_object(self, new_centroid, new_radius, timestamp):
        self.centroids[timestamp] = new_centroid
        self.radius = new_radius
        self.oof_count = 0

    ### NEED TO IMPLEMENT - 'attribute_smoothing' should be considered when
    # calculating data (e,g,. velocity is change in speed from one smoothing
    # time to the next, not one frame to the next) ###
    # Calculates object velocity/acceleration for current timestamp
    # Predicts next position
    def calculate_attributes(self, timestamp):

        # do stuff #

        self.predict_position(timestamp)
        self.oof_count += 1
        pass

    def is_out_of_frame(self) -> bool:
        return self.oof_count > self.oof_persistence


##
