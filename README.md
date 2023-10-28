# Detection-PoseEstimation-Pipeline
物体検出にて人を検出し、その検出結果に対し姿勢推定を行うフレームワークです。

---

<img src="https://github.com/Kazuhito00/MOT-Tracking-by-Detection-Pipeline/assets/37477845/e225ac4e-5588-40e4-9c23-0e9fb1542bac" loading="lazy" width="100%">

---

https://github.com/Kazuhito00/MOT-Tracking-by-Detection-Pipeline/assets/37477845/70e22af5-078e-4548-8e52-0fea5fd46b1a


# Requirement
```
* OpenCV 4.8.1.78 or later
* onnxruntime 1.14.1 or later ※GPU推論する際は「onnxruntime-gpu」
* mediapipe 0.10.7 or later 
```

# Usage
デモの実行方法は以下です。
```bash
python main.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスと動画より優先<br>
デフォルト：指定なし
* --detector<br>
Object Detectionのモデル選択<br>
yolox, efficientdet, ssd, centernet, nanodet, light_person_detector の何れかを指定<br>
デフォルト：yolox
* --detector_config<br>
Object Detectionのコンフィグ選択<br>
デフォルト：指定なし
* --bbox_offset<br>
バウンディングボックスを姿勢推定モデルに渡す際に余分に切り抜くサイズ<br>
デフォルト：0
* --estimator<br>
姿勢推定モデルの選択<br>
movenet, mediapipe_pose, rtmpose の何れかを指定<br>
デフォルト：rtmpose
* --estimator_config<br>
Pose Estimationのコンフィグ選択<br>
デフォルト：指定なし
* --use_gpu<br>
GPU推論するか否か<br>
デフォルト：指定なし
* --use_mirror<br>
入力画像を左右反転するか否か<br>
デフォルト：指定なし

# Direcotry
```
│  main.py
│  sample.mp4
├─Detector
│  │  detector.py
│  └─xxxxxxxx
│      │  xxxxxxxx.py
│      │  config.json
│      │  LICENSE
│      └─model
│          xxxxxxxx.onnx
└─Estimator
    │  estimator.py
    └─yyyyyyyy
        │  yyyyyyyy.py
        │  config.json
        │  LICENSE
        └─model
           yyyyyyyy.onnx
```
各モデルを格納しているディレクトリには、<br>
ライセンス条項とコンフィグを同梱しています。

# Detector

| モデル名 | 取得元リポジトリ | ライセンス | 備考 |
| :--- | :--- | :--- | :--- |
| YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | [YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)にて<br>ONNX化したモデルを使用 |
| EfficientDet | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 | [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
| SSD MobileNet v2 FPNLite | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
| CenterNet | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) | Apache-2.0 |  [Object-Detection-API-TensorFlow2ONNX](https://github.com/Kazuhito00/Object-Detection-API-TensorFlow2ONNX)にて<br>ONNX化したモデルを使用 |
| NanoDet | [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet) | Apache-2.0 | [NanoDet-ONNX-Sample](https://github.com/Kazuhito00/NanoDet-ONNX-Sample)にて<br>ONNX化したモデルを使用 |
| Light Person Detector | [Person-Detection-using-RaspberryPi-CPU](https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU) | Apache-2.0 | - |

# Estimator

| アルゴリズム名 | 取得元リポジトリ | ライセンス | 備考 |
| :--- | :--- | :--- | :--- |
| movenet | [TensorFlow Hub](https://tfhub.dev/google/movenet/singlepose/lightning/4) | Apache-2.0 | ONNXファイルは [Kazuhito00/MoveNet-Python-Example](https://github.com/Kazuhito00/MoveNet-Python-Example) |
| mediapipe_pose | [google/mediapipe](https://github.com/google/mediapipe) | Apache-2.0 | - |
| rtmpose | [PINTO0309/PINTO_model_zoo/393_RTMPose_WholeBody](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/393_RTMPose_WholeBody) | Apache-2.0 | - |

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Detection-PoseEstimation-Pipeline is under [Apache 2.0 License](LICENSE).<br><br>
Detection-PoseEstimation-Pipelineのソースコード自体は[Apache 2.0 License](LICENSE)ですが、<br>
各アルゴリズムのソースコードは、それぞれのライセンスに従います。<br>
詳細は各ディレクトリ同梱のLICENSEファイルをご確認ください。

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イギリス ウースターのエルガー像](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002011239_00000)を使用しています。

