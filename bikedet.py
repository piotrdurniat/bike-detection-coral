import argparse
import time
import cv2
import numpy as np
import tempfile
import json
import subprocess
import datetime
import json

from PIL import Image
from PIL import ImageDraw
from third_party.sort_master.sort import *
from pathlib import PosixPath, Path

import detect
import tflite_runtime.interpreter as tflite
import platform

from detection_counter.DetectionLine import DetectionLine
from detection_counter.annotator import draw_line, draw_objects, write_text
from detection_counter.DetectionUnit import DetectionUnit
from detection_counter.DetectionHistoryContainer import DetectionHistoryContainer
from detection_counter.Point import Point


EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

DEFAULT_MODEL = 'models/efficientdet_lite1_384_ptq_edgetpu.tflite'
DEFAULT_LABELS = 'models/coco_labels.txt'
DEFAULT_CONFIG = 'config.json'


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
      path: path to label file.
      encoding: label file encoding.
    Returns:
      Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ])


def crop_video(in_file: str, out_file: str, detection_box: tuple):
    x, y, w = detection_box
    command = [f'bash ./crop.sh {in_file} {out_file} {x} {y} {w}']
    subprocess.run(command, shell=True, stdout=subprocess.PIPE)


def get_config(config_path: str):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

        detection_line = (
            config['detectionLine']['x1'],
            config['detectionLine']['y1'],
            config['detectionLine']['x2'],
            config['detectionLine']['y2']
        )

        detection_box = (
            config['detectionBox']["centerX"],
            config['detectionBox']["centerY"],
            config['detectionBox']["width"]
        )

    return detection_line, detection_box


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        help='File path of .tflite file.',
                        default=DEFAULT_MODEL)
    parser.add_argument('-i', '--input', required=True,
                        help='File path of image to process.')
    parser.add_argument('-l', '--labels',
                        help='File path of labels file.',
                        default=DEFAULT_LABELS)
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects.')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='output directory', required=True)
    parser.add_argument('--show-vid', action='store_true',
                        help='display video results')
    parser.add_argument('--save-vid', action='store_true',
                        help='Save output video file')
    parser.add_argument('--crop', action='store_true',
                        help='crop input based on coordinates in config.json')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help='Path to JSON file with detection line an crop box coordinates')
    args = parser.parse_args()

    detection_line, detection_box = get_config(args.config)

    input_file = args.input
    if args.crop:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
        crop_video(args.input, tmp_file.name, detection_box)
        input_file = tmp_file.name

    output_vid_path = PosixPath(args.output_dir) / os.path.basename(args.input)
    run_detector(
        args.model,
        args.labels,
        args.threshold,
        args.output_dir,
        input_file,
        args.show_vid,
        args.save_vid,
        detection_line,
        detection_box,
        output_vid_path)


def get_tracker_format(objs):
    detections = []
    for obj in objs:
        element = []
        element.append(obj.bbox.xmin)
        element.append(obj.bbox.ymin)
        element.append(obj.bbox.xmax)
        element.append(obj.bbox.ymax)
        element.append(obj.score)
        detections.append(element)

    return np.array(detections)


def combine_dets(objs, trdata):
    detections = []
    for td in trdata:
        x0, y0, x1, y1, _ = td
        overlap = 0
        for ob in objs:
            dx0, dy0, dx1, dy1 = ob.bbox.xmin, ob.bbox.ymin, ob.bbox.xmax, ob.bbox.ymax
            area = (min(dx1, x1)-max(dx0, x0)) * (min(dy1, y1)-max(dy0, y0))
            # if detection boxes overlap
            if (area > overlap):
                element = []
                element.extend(td)
                element.append(ob.id)
                element.append(ob.score)
                detections.append(element)

    return np.array(detections)


def run_detector(model, labels, threshold, output_dir, input_file, show_vid, save_vid, detection_line, detection_box, output_vid_path):
    labels = load_labels(labels) if labels else {}
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()

    left_counter = 0
    right_counter = 0
    detection_line = DetectionLine(Point(detection_line[0], detection_line[1]),
                                   Point(detection_line[2], detection_line[3]))
    detection_history_container = DetectionHistoryContainer(detection_line, 10)

    mot_tracker = Sort()

    vid_cap = cv2.VideoCapture(input_file)
    vid_writer = None

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    is_stream = input_file.startswith("/dev/video")

    while(True):

        _, frame = vid_cap.read()
        image = Image.fromarray(frame)
        objs = detect_img(image, interpreter, threshold)

        image = image.convert('RGB')
        annotator = ImageDraw.Draw(image)
        draw_line(annotator, detection_line.p, detection_line.q)
        write_text(annotator, f'Left: {left_counter}, Right: {right_counter}')

        if len(objs) > 0:
            detections = get_tracker_format(objs)
            track_bbs_ids = mot_tracker.update(detections)
            full_dets = combine_dets(objs, track_bbs_ids)

            for det in full_dets:
                detection_unit = DetectionUnit(det)
                detection_history_container.add_detection_unit(detection_unit)

            draw_objects(annotator, full_dets, labels)

        frame = np.array(image)

        detection_history_container.increment_frames_without_detection()
        detection_history_container.remove_expired_histories()
        crosses = detection_history_container.get_line_crosses()

        for cross in crosses:
            save_line_cross(cross, PosixPath(output_dir), frame)

            if cross.detection_direction == 1:
                right_counter += 1
            else:
                left_counter += 1

            # Log line cross
            print(
                f'Object id: {cross.id} direction: {cross.detection_direction}')
            print(f"right: {right_counter}, left: {left_counter}")

        if show_vid:
            cv2.imshow('frame', frame)

        if save_vid:
            if not isinstance(vid_writer, cv2.VideoWriter):
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # force *.mp4 suffix on results videos
                output_vid_path = str(
                    Path(output_vid_path).with_suffix('.mp4'))

                vid_writer = cv2.VideoWriter(
                    output_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            vid_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not is_stream and vid_cap.get(cv2.CAP_PROP_POS_FRAMES) \
                == vid_cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()


def get_timestamp():
    return datetime.datetime.utcnow().isoformat("T", "milliseconds") + 'Z'


def save_line_cross(line_cross, output_path, img):
    time_str = get_timestamp()
    detection = {}
    detection["time"] = time_str
    detection["direction"] = int(line_cross.detection_direction)

    json_filename = str(output_path / f"{time_str}.json")
    jpg_filename = str(output_path / f"{time_str}.jpg")
    json_file = json.dumps(detection, indent=4, default=str)

    with open(json_filename, 'w') as f:
        f.write(json_file)

    cv2.imwrite(jpg_filename, img)


def detect_img(image, interpreter, threshold):
    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))

    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    objs = detect.get_output(interpreter, threshold, scale)
    print('%.2f ms' % (inference_time * 1000))

    return objs


if __name__ == '__main__':
    main()
