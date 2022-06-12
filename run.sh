python3 detect_image.py --model models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
    --labels models/coco_labels.txt \
    --input ~/Projects/PZ_neuro/sources/s03e03.mp4 \
    --output images/1.jpg \
    --show-vid
# --input2 ~/Projects/PZ_neuro/sources/frames/out-098.jpg \
# --output2 images/2.jpg
