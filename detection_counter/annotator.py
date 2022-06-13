import numpy as np
from detection_counter.DetectionLine import DetectionLine


def draw_line(draw, p, q):
    draw.line([(p.x, p.y), (q.x, q.y)], fill='black', width=3)
    draw.line([(p.x, p.y), (q.x, q.y)], fill='white', width=1)


def write_text(draw, text, pos=(30, 30)):
    draw.text((pos[0] - 1, pos[1] - 1), text, fill='black')
    draw.text(pos, text, fill='white')


def draw_objects(draw, full_dets: np.ndarray, labels):
    """Draws the bounding box and label for each object."""

    for td in full_dets:
        xmin, ymin, xmax, ymax, id, cls, conf = td

        draw.rectangle([(xmin-1, ymin-1), (xmax+1, ymax+1)], outline='black')
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='white')

        cls_name = labels.get(cls, cls)
        write_text(draw, f'{int(id)} {cls_name}\n{conf:.3f}',
                   (xmin + 10, ymin + 10))
