# -*- coding: utf-8 -*-
import copy

import cv2
import mediapipe as mp


class MediapipePose(object):
    def __init__(
        self,
        model_complexity=1,  # 0,1(default),2
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=True,
        plot_z_value=False,
    ):
        mp_pose = mp.solutions.pose
        self.model = mp_pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
        )

        self.plot_z_value = plot_z_value

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.process(input_image)

        keypoints, scores = [], []
        landmarks = results.pose_landmarks
        if landmarks is not None:
            for landmark in landmarks.landmark:
                keypoints.append([
                    int(image_width * landmark.x),
                    int(image_height * landmark.y),
                    landmark.z,
                ], )
                scores.append(landmark.visibility)

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

        # 0:鼻, 1:右目(目頭), 2:右(目瞳), 3:右目(目尻), 4:左目(目頭), 5:左目(瞳), 6:左目(目尻), 7:右耳
        # 8:左耳, 9:口左端, 10:口左端, 11:右肩, 12:左肩, 13:右肘, 14:左肘, 15:右手首, 16:左手首
        # 17:右手1(外側端), 18:左手1(外側端), 19:右手2(先端), 20:左手2(先端), 21:右手3(内側端)
        # 22:左手3(内側端), 23:腰(右側), 24:腰(左側), 25:右ひざ, 26:左ひざ, 27:右足首, 28:左足首
        # 29:右かかと, 30:左かかと, 31:右つま先, 32:左つま先
        line_indices = [
            # 顔
            [(0, 1), palette[0]],
            [(1, 3), palette[0]],
            [(3, 7), palette[0]],
            [(0, 4), palette[0]],
            [(4, 6), palette[0]],
            [(6, 8), palette[0]],
            [(9, 10), palette[0]],
            # 胴体
            [(11, 12), palette[1]],
            [(11, 23), palette[1]],
            [(12, 24), palette[1]],
            [(23, 24), palette[1]],
            # 右腕
            [(11, 13), palette[2]],
            [(13, 15), palette[2]],
            [(15, 17), palette[2]],
            [(15, 21), palette[2]],
            [(17, 19), palette[2]],
            [(19, 21), palette[2]],
            # 左腕
            [(12, 14), palette[3]],
            [(14, 16), palette[3]],
            [(16, 18), palette[3]],
            [(16, 22), palette[3]],
            [(18, 20), palette[3]],
            [(20, 22), palette[3]],
            # 右足
            [(23, 25), palette[4]],
            [(25, 27), palette[4]],
            [(27, 29), palette[4]],
            [(29, 31), palette[4]],
            # 左足
            [(24, 26), palette[5]],
            [(26, 28), palette[5]],
            [(28, 30), palette[5]],
            [(30, 32), palette[5]],
        ]

        for keypoints, scores in zip(keypoints_list, kp_scores_list):
            # Line
            for line_index in line_indices:
                index01, index02 = line_index[0][0], line_index[0][1]
                if scores[index01] > keypoint_score_th and scores[
                        index02] > keypoint_score_th:
                    point01 = keypoints[index01][:2]
                    point02 = keypoints[index02][:2]
                    cv2.line(debug_image, point01, point02, line_index[1], 2)

            # Circle：各点
            for keypoint, score in zip(keypoints, scores):
                if score > keypoint_score_th:
                    cv2.circle(debug_image, keypoint[:2], 3, (255, 255, 255),
                               -1)

                    if self.plot_z_value:
                        cv2.putText(
                            debug_image,
                            "z:" + '{:.2f}'.format(keypoint[2]),
                            (keypoint[0] - 10, keypoint[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

        return debug_image
