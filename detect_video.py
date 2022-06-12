# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""

import argparse
import time
import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from third_party.sort_master.sort import *
from pathlib import PosixPath, Path


import detect
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


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


def draw_objects(draw, objs, labels, trdata):
    """Draws the bounding box and label for each object."""

    if (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = td
            overlap = 0
            for ob in objs:
                dx0, dy0, dx1, dy1 = ob.bbox.xmin, ob.bbox.ymin, ob.bbox.xmax, ob.bbox.ymax
                area = (min(dx1, x1)-max(dx0, x0)) * \
                    (min(dy1, y1)-max(dy0, y0))
                if (area > overlap):
                    overlap = area
                    obj = ob

        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%d  %s\n%.2f' % (int(trackID), labels.get(
                      obj.id, obj.id), obj.score),
                  fill='red')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file.')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of image to process.')
    parser.add_argument('-l', '--labels',
                        help='File path of labels file.')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects.')
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='output directory',
        required=True
    )
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    parser.add_argument('--show-vid', action='store_true',
                        help='display video results')
    parser.add_argument('--save-vid', action='store_true',
                        help='Save output video file')
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    mot_tracker = Sort()

    vid_cap = cv2.VideoCapture(args.input)
    vid_writer = None

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    vid_save_path = PosixPath(args.output_dir) / os.path.basename(args.input)
    print(vid_save_path)

    is_stream = args.input.startswith("/dev/video")

    while(True):

        _, frame = vid_cap.read()
        image = Image.fromarray(frame)
        objs = detect_img(image, interpreter, args.threshold)

        if len(objs) > 0:
            detections = []  # np.array([])
            for obj in objs:
                element = []  # np.array([])
                element.append(obj.bbox.xmin)
                element.append(obj.bbox.ymin)
                element.append(obj.bbox.xmax)
                element.append(obj.bbox.ymax)
                element.append(obj.score)
                detections.append(element)

            detections = np.array(detections)

            track_bbs_ids = mot_tracker.update(detections)
            print(objs)
            print(track_bbs_ids)

            image = image.convert('RGB')
            draw_objects(ImageDraw.Draw(image), objs, labels, track_bbs_ids)

        frame = np.array(image)

        if args.show_vid:
            cv2.imshow('frame', frame)

        if args.save_vid:
            if not isinstance(vid_writer, cv2.VideoWriter):
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # force *.mp4 suffix on results videos
                vid_save_path = str(Path(vid_save_path).with_suffix('.mp4'))

                vid_writer = cv2.VideoWriter(
                    vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            vid_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not is_stream and vid_cap.get(cv2.CAP_PROP_POS_FRAMES) \
                == vid_cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()


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