import cv2


def draw_point(annotator, point, color=(128, 128, 128)):
    cv2.circle(annotator.im, (point.x, point.y), 2, color)


def draw_line(annotator, p, q, color=(128, 128, 128)):
    cv2.line(annotator.im, (p.x,p.y), (q.x,q.y), color, 1)


def write_text(annotator, text):
    cv2.putText(annotator.im, text, (30, 30),
                cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255))
