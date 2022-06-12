python3 detect_video.py \
    --model models/efficientdet_lite1_384_ptq_edgetpu.tflite \
    --labels models/coco_labels.txt \
    --input ~/Projects/PZ_neuro/sources/s03e03-fragment.mp4 \
    --output-dir ./run \
    --show-vid \
    --save-vid
# --model models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
# --model models/efficientdet_lite2_448_ptq_edgetpu.tflite \
# --input /dev/video0 \
# --input2 ~/Projects/PZ_neuro/sources/frames/out-098.jpg \
# --output2 images/2.jpg
