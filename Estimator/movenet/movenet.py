# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import onnxruntime


class MoveNet(object):
    def __init__(
        self,
        model_path=None,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        # 入力サイズ
        self.input_shape = self.input_detail.shape[1:3]

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # 前処理
        input_image = cv2.resize(
            image,
            dsize=(self.input_shape[1], self.input_shape[0]),
        )  # リサイズ
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
        input_image = input_image.reshape(
            -1,
            self.input_shape[1],
            self.input_shape[0],
            3,
        )  # リシェイプ
        input_image = input_image.astype('int32')  # int32へキャスト

        # 推論
        outputs = self.onnx_session.run(
            [self.output_name],
            {self.input_name: input_image},
        )

        keypoints_with_scores = outputs[0]
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        # キーポイント、スコア取り出し
        keypoints = []
        scores = []
        for index in range(17):
            keypoint_x = int(image_width * keypoints_with_scores[index][1])
            keypoint_y = int(image_height * keypoints_with_scores[index][0])
            score = keypoints_with_scores[index][2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        return keypoints, scores

    def draw(
        self,
        image,
        keypoints_list,
        kp_scores_list,
        keypoint_score_th=0.3,
    ):
        debug_image = copy.deepcopy(image)

        palette = [
            [55, 255, 255],
            [51, 153, 255],
            [255, 128, 0],
            [255, 153, 255],
            [0, 51, 255],
            [255, 51, 51],
        ]

        # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
        # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
        line_indices = [
            # 顔
            [(0, 1), palette[0]],
            [(0, 2), palette[0]],
            [(1, 3), palette[0]],
            [(2, 4), palette[0]],
            # 胴体
            [(5, 6), palette[1]],
            [(11, 12), palette[1]],
            [(5, 11), palette[1]],
            [(6, 12), palette[1]],
            # 左腕
            [(5, 7), palette[2]],
            [(7, 9), palette[2]],
            # 右腕
            [(6, 8), palette[3]],
            [(8, 10), palette[3]],
            # 左足
            [(11, 13), palette[4]],
            [(13, 15), palette[4]],
            # 右足
            [(12, 14), palette[5]],
            [(14, 16), palette[5]],
        ]

        for keypoints, scores in zip(keypoints_list, kp_scores_list):
            # Line
            for line_index in line_indices:
                index01, index02 = line_index[0][0], line_index[0][1]
                if scores[index01] > keypoint_score_th and scores[
                        index02] > keypoint_score_th:
                    point01 = keypoints[index01]
                    point02 = keypoints[index02]
                    cv2.line(debug_image, point01, point02, line_index[1], 2)

            # Circle：各点
            for keypoint, score in zip(keypoints, scores):
                if score > keypoint_score_th:
                    cv2.circle(debug_image, keypoint, 3, (255, 255, 255), -1)

        return debug_image
