# EfficientDet with Intel OpenVINO

This repository demonstrates how to convert [AutoML EfficientDet](https://github.com/google/automl) to OpenVINO IR.

Follow the steps from [.github/workflows/main.yml](.github/workflows/main.yml) to convert your model.
For public models, download IRs from [GitHub Actions](https://github.com/dkurt/openvino_efficientdet/actions?query=branch%3Amaster)

## How to convert model
1. Get optimized frozen graph. If you already have frozen `.pb` graph from AutoML framework, run [scripts/opt_graph.py](scripts/opt_graph.py) specifying path to it. TensorFlow 1.x is required.

```bash
python3 scripts/opt_graph.py --input efficientdet-d4_frozen.pb --output efficientdet-d4_opt.pb
```

2. Generate `.pbtxt` which is required for conversion:
```bash
python3 scripts/tf_text_graph_efficientdet.py \
    --input efficientdet-d4_opt.pb \
    --output efficientdet-d4_opt.pbtxt \
    --width 1024 \
    --height 1024
```
find resolution of your model at https://github.com/google/automl/blob/master/efficientdet/hparams_config.py

3. Run OpenCV once to dump OpenVINO IR (OpenVINO 2020.4 is required):
```bash
export OPENCV_DNN_IE_SERIALIZE=1
python3 scripts/run_opencv.py \
    --model efficientdet-d4_opt.pb \
    --pbtxt efficientdet-d4_opt.pbtxt \
    --width 1024 \
    --height 1024
```

4. Validate model comparing accuracy with an original frozen TensorFlow graph
```bash
python3 scripts/validate.py --version d4 --width 1024 --height 1024
```

![](./images/res_d4.jpg)
