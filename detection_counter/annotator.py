import numpy as np
from detection_counter.DetectionLine import DetectionLine
from PIL import ImageFont

FONT_FILE = 'fonts/LiberationSans-Regular.ttf'

FONT_L = ImageFont.truetype(font=FONT_FILE, size=20)
FONT_S = ImageFont.truetype(font=FONT_FILE, size=12)


def draw_line(draw, p, q):
    draw.line([(p.x, p.y), (q.x, q.y)], fill='black', width=3)
    draw.line([(p.x, p.y), (q.x, q.y)], fill='white', width=1)


def write_text(draw, text, pos=(30, 30), font=FONT_L):
    draw.text((pos[0], pos[1]), text, font=font,  fill='white',
              stroke_width=2, stroke_fill='black')


def draw_objects(draw, full_dets: np.ndarray, labels):
    """Draws the bounding box and label for each object."""

    for td in full_dets:
        xmin, ymin, xmax, ymax, id, cls, conf = td

        draw.rectangle([(xmin-1, ymin-1), (xmax+1, ymax+1)], outline='black')
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='white')

        cls_name = labels.get(cls, cls)
        write_text(draw, f'{int(id)} {cls_name}\n{conf:.3f}',
                   (xmin + 10, ymin + 10), font=FONT_S)
