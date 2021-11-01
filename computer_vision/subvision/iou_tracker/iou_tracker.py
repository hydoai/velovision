import numpy as np
from lapsolver import solve_dense
from time import time

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def associate(tracks, detections, sigma_iou):
    """ perform association between tracks and detections in a frame.
    Args:
        tracks (list): input tracks
        detections (list): input detections
        sigma_iou (float): minimum intersection-over-union of a valid association

    Returns:
        (tuple): tuple containing:

        track_ids (numpy.array): 1D array with indexes of the tracks
        det_ids (numpy.array): 1D array of the associated indexes of the detections
    """
    costs = np.empty(shape=(len(tracks), len(detections)), dtype=np.float32)
    for row, track in enumerate(tracks):
        for col, detection in enumerate(detections):
            costs[row, col] = 1 - iou(track['bboxes'], detection['bbox'])

    np.nan_to_num(costs)
    costs[costs > 1 - sigma_iou] = np.nan
    track_ids, det_ids = solve_dense(costs)
    return track_ids, det_ids

class IOUTracker:
    def __init__(self, sigma_l=0.5, sigma_h=1.0, sigma_iou=0.3, t_min=7, ttl=1):
        """
        Args:
            sigma_l (float) : low detection threshold
            sigma_h (float) : high detection threshold
            sigma_iou (float) : IOU threshold
            t_min (float) : minimum track length in frames
            ttl (float) : maximum number of frames to perform visual tracking. This can fill 'gaps' of up to 2*ttl frames (ttl times forward and backward).

        Usage:

        tracker = IOUTracker()

        for frame in video:
            detections = object_detector()
            tracker.update(detections)
            ...(do something with tracked detections)
        """
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.ttl = ttl

        self.tracking_id = 0

        self.tracks_active = []
        self.tracks_extendable = []
        self.tracks_finished = []
        self.frame_num = 0

    def update(self, detections):
        '''
        Args:
            detections: a list of dictionaries, for example:
                [{'bbox': (520.0, 208.0, 645.0, 266.0),
                  'score': 0.96,
                  'class': 'pedestrian'},
                 {'bbox': (783.0, 162.0, 807.0, 209.0),
                  'score': 0.88,
                  'class': 'pedestrian'}]
        '''

        self.frame_num += 1

        # apply low threshold to detections
        dets = [det for det in detections if det['score'] >= self.sigma_l]

        track_ids, det_ids = associate(self.tracks_active, dets, self.sigma_iou)

        updated_tracks = []

        for track_id, det_id in zip(track_ids, det_ids):
            # This upstream code keeps all the past information about bounding box locations. This is not needed for my purpuosese.
            #self.tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
            #self.tracks_active[track_id]['max_score'] = max(self.tracks_active[track_id]['max_score'], dets[det_id]['score'])
            #self.tracks_active[track_id]['classes'].append(dets[det_id]['class'])
            #self.tracks_active[track_id]['det_counter'] += 1
            self.tracks_active[track_id]['bboxes'] = dets[det_id]['bbox']
            self.tracks_active[track_id]['score'] = dets[det_id]['score']
            self.tracks_active[track_id]['classes'] = dets[det_id]['class']
            self.tracks_active[track_id]['det_counter'] += 1

            if self.tracks_active[track_id]['ttl'] != self.ttl:
                # reset visual tracker if active
                self.tracks_active[track_id]['ttl'] = self.ttl
                self.tracks_active[track_id]['visual_tracker'] = None

            updated_tracks.append(self.tracks_active[track_id])

        tracks_not_updated = [self.tracks_active[idx] for idx in set(range(len(self.tracks_active))).difference(set(track_ids))]

        for track in tracks_not_updated:
            if track['ttl'] > 0:
                self.tracks_extendable.append(track)

        # update the list of extenable tracks.
        # tracks that are too old are deleted
        # this should not be necessary but may improve the performance for large numbers of tracks
        tracks_extendable_updated = []
        for track in self.tracks_extendable:
            if track['start_frame'] + len(track['bboxes']) + self.ttl - track['ttl'] >= self.frame_num:
                # still hope for revival?
                tracks_extendable_updated.append(track)
            elif track['score'] >= self.sigma_h and track['det_counter'] >= self.t_min:
                # too old!
                del track
                #self.tracks_finished.append(track)

        self.tracks_extendable = tracks_extendable_updated

        new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
        dets_for_new = new_dets

        # create new tracks
        new_tracks = []
        for det in new_dets:
            self.tracking_id += 1
            new_tracks.append({
                'bboxes' : det['bbox'],
                'score' : det['score'],
                'start_frame' : self.frame_num,
                'ttl' : self.ttl,
                'classes' : det['class'],
                'det_counter' : 1,
                'visual_tracker' : None,
                'tracking_id':self.tracking_id,
                })



        self.tracks_active = []
        for track in updated_tracks + new_tracks:
            if track['ttl'] == 0:
                self.tracks_extendable.append(track)
            else:
                self.tracks_active.append(track)

        return self.tracks_active


if __name__ == '__main__':
    example_data = [{'bbox': (520.0, 208.0, 645.0, 266.0),'score': 0.96,'class': 'pedestrian'},
                    {'bbox': (783.0, 162.0, 807.0, 209.0),'score': 0.88,'class': 'pedestrian'}]
    example_data2 = [{'bbox': (550.0, 228.0, 685.0, 286.0),'score': 0.94,'class': 'pedestrian'},
                    {'bbox': (788.0, 169.0, 804.0, 208.0),'score': 0.80,'class': 'pedestrian'}]
    viou_tracker = IOUTracker()

    output1 = viou_tracker.update(example_data)
    output2 = viou_tracker.update(example_data2)

    print(output1)
    print(output2)




