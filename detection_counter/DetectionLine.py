# Code from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# A Python3 program to find if 2 given line segments intersect or not
from detection_counter.Point import Point
class DetectionLine:

    def __init__(self, p, q):

        self.p = p
        self.q = q

        # Prevent having vertical detection line by simply moving one point 1px. Makes | as \.
        if self.p.x == self.q.x:
            self.p.x = self.p.x - 1

        self.line_equation_a, self.line_equation_b = self._get_detection_line_equation()

    # Given three collinear points p, q, r, the function checks if
    # point q lies on line segment 'pr'

    def _onSegment(self, p, q, r):
        if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False

    def _orientation(self, p, q, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise

        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
        # for details of below formula.

        val = (float(q.y - p.y) * (r.x - q.x)) - \
            (float(q.x - p.x) * (r.y - q.y))
        if (val > 0):

            # Clockwise orientation
            return 1
        elif (val < 0):

            # Counterclockwise orientation
            return 2
        else:

            # Collinear orientation
            return 0

    # The main function that returns true if
    # the line segment 'p1q1' and 'p2q2' intersect.

    def _doIntersect(self, p1, q1, p2, q2):

        # Find the 4 orientations required for
        # the general and special cases
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True

        # Special Cases

        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and self._onSegment(p1, p2, q1)):
            return True

        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and self._onSegment(p1, q2, q1)):
            return True

        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and self._onSegment(p2, p1, q2)):
            return True

        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and self._onSegment(p2, q1, q2)):
            return True

        # If none of the cases
        return False

    def doIntersect(self, p2, q2):
        return self._doIntersect(self.p, self.q, p2, q2)

    def _get_detection_line_equation(self):
        '''
        Method returns linear equation of segment with given two points.
        '''
        # a = (y2-y1)/(x2-x1)
        a = (self.q.y - self.p.y) / (self.q.x - self.p.x)
        # b = y1 - a*x1
        b = self.p.y - (a * self.p.x)

        return (a, b)

    def get_side_of_detection(self, last_two_points):
        '''
        Method used to determine - on which side on detection line the given point is now.
        Example:

        If the line (graphically) looks like that: \ or /

        <point2> \ <point1> - Detected as -1 side of line

        <point1> \ <point2> - Detected as +1 side of line

        <point2> / <point1> - Detected as +1 side of line

        <point1> / <point2> - Detected as -1 side of line

        Warning! - comparing to the output image - those ^ descriptions are inverted - because the axes on image are (x, -y) - (the 0,0 point is on the corner)
        '''
        [p1, p2] = last_two_points

        p1_to_line = self._point_to_line(p1)
        p2_to_line = self._point_to_line(p2)

        if p2_to_line == 0:
            return -p1_to_line
        else:
            return p2_to_line

    def _point_to_line(self, p):
        '''
        Returns side of point
        '''

        line_y = self.line_equation_a * p.x + self.line_equation_b
        dif_y = p.y - line_y
        if dif_y == 0:
            # point is on line
            return 0
        else:
            return abs(dif_y) / dif_y


p1 = Point(3, 3)
q1 = Point(3, 0)
p2 = Point(1, 2)
q2 = Point(3, 2)

detection_line = DetectionLine(p1, q1)


inter = detection_line.doIntersect(p2, q2)
print(inter)

side = detection_line.get_side_of_detection([q2, p2])
print(side)
