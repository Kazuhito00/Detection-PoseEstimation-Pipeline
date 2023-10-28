import copy
import json

import numpy as np


class ObjectDetector(object):
    def __init__(
        self,
        model_name='yolox',
        config_path=None,
        use_gpu=False,
    ):
        self.model_name = model_name
        self.model = None
        self.target_id = None
        self.config = None
        self.use_gpu = use_gpu

        if self.model_name == 'yolox':
            from Detector.yolox.yolox_onnx import YoloxONNX

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            if config_path is None:
                config_path = 'Detector/yolox/config.json'
            with open(config_path) as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = YoloxONNX(
                    model_path=self.config['model_path'],
                    class_score_th=self.config['class_score_th'],
                    nms_th=self.config['nms_th'],
                    nms_score_th=self.config['nms_score_th'],
                    with_p6=self.config['with_p6'],
                    providers=providers,
                )

            # 検出対象IDをPersonのみに指定
            self.target_id = [1]

        elif self.model_name == 'efficientdet':
            from Detector.efficientdet.efficientdet_onnx import EfficientDetONNX

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            if config_path is None:
                config_path = 'Detector/efficientdet/config.json'
            with open(config_path) as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = EfficientDetONNX(
                    model_path=self.config['model_path'],
                    class_score_th=self.config['class_score_th'],
                    providers=providers,
                )

            # 検出対象IDをPersonのみに指定
            self.target_id = [1]

        elif self.model_name == 'ssd':
            from Detector.ssd.ssd_onnx import SsdONNX

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            if config_path is None:
                config_path = 'Detector/ssd/config.json'
            with open(config_path) as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = SsdONNX(
                    model_path=self.config['model_path'],
                    class_score_th=self.config['class_score_th'],
                    providers=providers,
                )

            # 検出対象IDをPersonのみに指定
            self.target_id = [1]

        elif self.model_name == 'centernet':
            from Detector.centernet.centernet_onnx import CenterNetONNX

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            if config_path is None:
                config_path = 'Detector/centernet/config.json'
            with open(config_path) as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = CenterNetONNX(
                    model_path=self.config['model_path'],
                    class_score_th=self.config['class_score_th'],
                    providers=providers,
                )

            # 検出対象IDをPersonのみに指定
            self.target_id = [1]

        elif self.model_name == 'nanodet':
            from Detector.nanodet.nanodet_onnx import NanoDetONNX

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            if config_path is None:
                config_path = 'Detector/nanodet/config.json'
            with open(config_path) as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = NanoDetONNX(
                    model_path=self.config['model_path'],
                    class_score_th=self.config['class_score_th'],
                    nms_th=self.config['nms_th'],
                    providers=providers,
                )

            # 検出対象IDをPersonのみに指定
            self.target_id = [1]

        elif self.model_name == 'light_person_detector':
            from Detector.light_person_detector.light_person_detector import LightPersonDetector

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            if config_path is None:
                config_path = 'Detector/light_person_detector/config.json'
            with open(config_path) as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = LightPersonDetector(
                    model_path=self.config['model_path'],
                    score_th=self.config['score_th'],
                    nms_th=self.config['nms_th'],
                    providers=providers,
                    num_threads=self.config['num_threads'],
                )

            # 検出対象IDをPersonのみに指定
            self.target_id = [0]

        else:
            raise ValueError('Invalid Model Name')

    def __call__(self, image):
        input_image = copy.deepcopy(image)
        bboxes, scores, class_ids = None, None, None

        # 推論
        if self.model is not None:
            bboxes, scores, class_ids = self.model(input_image)
        else:
            raise ValueError('Model is None')

        # 対象のクラスIDで抽出
        if self.target_id is not None and len(bboxes) > 0:
            target_index = np.in1d(class_ids, np.array(self.target_id))
            bboxes = bboxes[target_index]
            scores = scores[target_index]
            class_ids = class_ids[target_index]

        return bboxes, scores, class_ids

    def print_info(self):
        from pprint import pprint

        print('Detector:', self.model_name)
        print('GPU:', self.use_gpu)
        pprint(self.config, indent=4)
        print()
