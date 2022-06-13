from detection_counter.DetectionHistory import DetectionHistory


class DetectionHistoryContainer:
    '''
    - detection_line : DetectionLine
    - detection_histories : DetectionHistory[]
    - frames_without_detection_expiration : int
    '''

    def __init__(self, detection_line, frames_without_detection_expiration):
        self.detection_line = detection_line
        self.detection_histories = []
        self.frames_without_detection_expiration = frames_without_detection_expiration

    def index_of_detection_history_by_id(self, id):
        '''
        Map detection identity for list index
        '''
        for idx, item in enumerate(self.detection_histories):
            if item.id == id:
                return idx
        return -1

    def add_detection_unit(self, detection_unit):
        '''
        Create new object history or adds unit to existing history
        '''
        index = self.index_of_detection_history_by_id(detection_unit.id)

        if index == -1:
            new_det_history = DetectionHistory(detection_unit)
            self.detection_histories.append(new_det_history)
        else:
            self.detection_histories[index].add_detection_unit(detection_unit)

    def increment_frames_without_detection(self):
        '''
        After each frame the counter is incremented - this is used to delete objects that appeared in the past
        '''
        for hist in self.detection_histories:
            hist.frames_without_detection = hist.frames_without_detection + 1

    def remove_expired_histories(self):
        '''
        Remove histories that have not been detected in last <frames_without_detection_expiration> frames,
        or history was marked to be deleted -> this happens after crossing a line
        '''
        for hist in self.detection_histories:
            if hist.frames_without_detection >= self.frames_without_detection_expiration or hist.to_be_deleted == True:
                self.detection_histories.remove(hist)

    def get_line_crosses(self):
        '''
        Returns DetectionHistory[] - that contains histories which objects crosses line in last frame. 
        '''
        items_that_crossed_line = []
        for idx, item in enumerate(self.detection_histories):
            last_two_detections = item.get_last_two_detections()
            if len(last_two_detections) != 2: break
            last_two_detections_center_points = [last_two_detections[0].get_center_point(
            ), last_two_detections[1].get_center_point()]

            intersect = self.detection_line.doIntersect(
                last_two_detections_center_points[0],last_two_detections_center_points[1] )

            if intersect is True:
                # add info about direction
                item.detection_direction = self.detection_line.get_side_of_detection(
                    last_two_detections_center_points)
                # append to list
                items_that_crossed_line.append(item)
                # mark as item to be deleted
                item.to_be_deleted = True

        return items_that_crossed_line
