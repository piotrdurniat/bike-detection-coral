# Bike detection on Google Coral TPU

Real-time bike detection, tracking and counting system designed for Raspberry Pi with Google Coral TPU.

The system allows user to mark a virtual line segment on video input, that is used to count the number of bicycles that pass through it and identify their direction of crossing.

# Accuracy Tests

Tests were performed for the following pre-trained models:

- Yolov5m - running on a GPU "best" (reference),
- Yolov5s - running on a GPU "smallest",
- SSDLite MobileDet - running on Google Coral TPU,
- EfficientDet-Lite1 - running on Google Coral TPU.

## Detection Accuracy Calculation

When determining the level of detection, two sources of errors are considered:

- $\varepsilon_m$ - missing a vehicle by the system (number of vehicles missed),
- $\varepsilon_f$ - detecting a non-existent vehicle by the system (number of falsely detected vehicles).

If $N$ is the real number of vehicles that passed through the measurement point, the detection level is defined by the formula:

$$
r_d = \frac{(N - \varepsilon_m - \varepsilon_f)}{N}
$$

## Confidence Interval

The actual value of the tested parameter $p$ is higher than the value $\hat p_L$ with $95\%$ probability.

$$
\hat p_L  = \max\left\lbrace 0, \frac{2N\hat p + z^2 - \left[z\sqrt{z^2 -(1/N)+4N\hat p (1-\hat p) + (4\hat p - 2)} + 1 \right ]}{2\cdot(N+z^2)}\right\rbrace
$$

- $\hat p_L$ - lower value of the symmetric confidence interval, calculated using the Wilson method,
- $\hat p$ - estimation of the given tested parameter,
- $z \approx 1.6448536$ - value resulting from the adopted confidence level (in this case 95%).

## Results of the Accuracy Tests

Test data:

1. 4 locations,
2. 5 sessions (result calculated from the sum of the number of bicycles and errors for all sessions),
3. 362 bicycle passes.

Results:

|              | **yolov5m** | **yolov5s** | **efficientdet_lite1** | **ssdlite_mobiledet** |
| ------------ | ----------- | ----------- | ---------------------- | --------------------- |
| $$\hat p_L$$ | 0.8459      | 0.7923      | 0.6878                 | 0.5328                |

## Usage

```bash
python3 bikedet.py \
    --input <input_file_path> \
    --output-dir <output_dir_path> \
    --show-vid \
    --save-vid \
    --crop
```

Optional args:

```bash
    --model <model_path> \
    --labels <labels_path> \
    --config <config_path>
```
