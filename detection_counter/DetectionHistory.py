
from detection_counter.DetectionUnit import DetectionUnit
from detection_counter.DetectionLine import DetectionLine


class DetectionHistory:
    '''
    Wrapper for detection units
    - detection_units: DetectionUnit[]
    - to_be_deleted: bool
    - id: int (taken from DeepSort)
    - frames_without_detection: int
    '''

    def __init__(self, detection_unit):

        self.frames_without_detection = 0
        self.detection_units = []
        self.to_be_deleted = False
        self.id = detection_unit.id

        self.add_detection_unit(detection_unit)

    def add_detection_unit(self, detection_unit):
        self.detection_units.append(detection_unit)
        self.frames_without_detection = 0

    def get_last_two_detections(self):
        last_two = self.detection_units[-2:]
        return last_two
