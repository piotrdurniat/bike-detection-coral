from detection_counter.Point import Point
class DetectionUnit:

    def __init__(self, params, conf):

        self.left = params[0]
        self.top = params[1]
        self.right = params[2]
        self.bottom = params[3]
        self.id = params[4]
        self.cls = params[5]
        self.conf = conf

        self.centerX = self.left + (self.right - self.left) / 2
        self.centerY = self.top + (self.bottom - self.top) / 2

    def get_center_point(self):
        return Point(int(self.centerX), int(self.centerY))
