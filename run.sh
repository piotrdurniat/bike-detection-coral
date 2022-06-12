python3 detect_image.py \
    --model models/efficientdet_lite1_384_ptq_edgetpu.tflite \
    --labels models/coco_labels.txt \
    --input ~/Projects/PZ_neuro/sources/s03e03-crop.mp4 \
    --output images/1.jpg \
    --show-vid
# --model models/efficientdet_lite2_448_ptq_edgetpu.tflite \
# --model models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
# --input /dev/video0 \
# --input2 ~/Projects/PZ_neuro/sources/frames/out-098.jpg \
# --output2 images/2.jpg
