import logging
from collections import deque
from typing import List

import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment


from .unit_object import UnitObject
from .base_tracker import BaseTracker
from .kalman_tracker import KalmanTracker

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARN)


def calculate_iou(box1, box2):
    """
    Calculate intersection over union
    :param box1: a[0], a[1], a[2], a[3] <-> left, top, right, bottom
    :param box2: b[0], b[1], b[2], b[3] <-> left, top, right, bottom
    """

    w_intsec = np.maximum(0, (np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])))
    h_intsec = np.maximum(0, (np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1])))
    s_intsec = w_intsec * h_intsec

    s_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_b = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)

class Tracking:
    """
    Class that connects detection and tracking
    """

    def __init__(self, minHits = 3, maxAge = 5):
        self.max_age = maxAge
        self.min_hits = minHits
        self.tracker_list: List[BaseTracker] = []
        self.track_id_list = deque(list(map(str, range(25))))
        self.tracker = KalmanTracker()

    def update(self, unit_detections):


        unit_trackers = []

        for trk in self.tracker_list:
            unit_trackers.append(trk.unit_object)

        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(unit_trackers, unit_detections,
                                                                                     iou_thrd=0.3)

        print('Detection: ' + str(unit_detections))
        print('x_box: ' + str(unit_trackers))
        print('matched:' + str(matched))
        print('unmatched_det:' + str(unmatched_dets))
        print('unmatched_trks:' + str(unmatched_trks))

        # Matched Detections
        for trk_idx, det_idx in matched:
            z = unit_detections[det_idx].box
            z = np.expand_dims(z, axis=0).T
            tmp_trk = self.tracker_list[trk_idx]
            tmp_trk.predict_and_update(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            unit_trackers[trk_idx].box = xx
            unit_trackers[trk_idx].class_id = unit_detections[det_idx].class_id
            tmp_trk.unit_object = unit_trackers[trk_idx]
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

        # Unmatched Detections
        for idx in unmatched_dets:
            z = unit_detections[idx].box
            z = np.expand_dims(z, axis=0).T
            tmp_trk = KalmanTracker() #self.tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.unit_object.box = xx
            tmp_trk.unit_object.class_id = unit_detections[idx].class_id
            tmp_trk.tracking_id = self.track_id_list.popleft()  # assign an ID for the tracker
            self.tracker_list.append(tmp_trk)
            unit_trackers.append(tmp_trk.unit_object)

        # Unmatched trackers
        for trk_idx in unmatched_trks:
            tmp_trk = self.tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.unit_object.box = xx
            unit_trackers[trk_idx] = tmp_trk.unit_object

        # The list of tracks to be annotated
        good_tracker_list = []
        for trk in self.tracker_list:
            if (trk.hits >= self.min_hits) and (trk.no_losses <= self.max_age):
                good_tracker_list.append(trk)
                # img = utils.drawing.draw_box_label(img, trk, self.detector.class_names)

        # Manage Tracks to be deleted
        deleted_tracks = filter(lambda x: x.no_losses > self.max_age, self.tracker_list)

        for trk in deleted_tracks:
            self.track_id_list.append(trk.tracking_id)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses <= self.max_age]


    @staticmethod
    def assign_detections_to_trackers(unit_trackers: List[UnitObject], unit_detections: List[UnitObject], iou_thrd=0.3):
        """
        Matches Trackers and Detections
        :param unit_trackers: trackers
        :param unit_detections: detections
        :param iou_thrd: threshold to qualify as a match
        :return: matches, unmatched_detections, unmatched_trackers
        """
        IOU_mat = np.zeros((len(unit_trackers), len(unit_detections)), dtype=np.float32)
        for t, trk in enumerate(unit_trackers):
            for d, det in enumerate(unit_detections):
                if trk.class_id == det.class_id:
                    IOU_mat[t, d] = calculate_iou(trk.box, det.box)

        # Finding Matches using Hungarian Algorithm
        row_ind, col_ind = linear_assignment(-IOU_mat)
        # matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t, trk in enumerate(unit_trackers):
            if t not in row_ind: # matched_idx[:, 0]:
                unmatched_trackers.append(t)

        for d, det in enumerate(unit_detections):
            if d not in col_ind: #matched_idx[:, 1]:
                unmatched_detections.append(d)

        matches = []

        # Checking quality of matched by comparing with threshold
        for i in range(len(row_ind)):
            if IOU_mat[row_ind[i], col_ind[i]] < iou_thrd:
                unmatched_trackers.append(row_ind[i])
                unmatched_detections.append(col_ind[i])
            else:
                m = np.array([row_ind[i], col_ind[i]])
                matches.append([m.reshape(1, 2)])

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)


        matches = matches.reshape(len(matches), 2)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
