python3 bikedet.py \
    --model models/efficientdet_lite1_384_ptq_edgetpu.tflite \
    --input sources/s03e04.mp4 \
    --output-dir ./runs/efficientdet_lite1/s03e04 \
    --show-vid \
    --save-vid \
    --config configs/4.json \
    --crop

# SSDLite MobileDet
# --model models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite \

# EfficientDet-Lite1
# --model models/efficientdet_lite1_384_ptq_edgetpu.tflite \

# Webcam input:
# --input /dev/video0 \
