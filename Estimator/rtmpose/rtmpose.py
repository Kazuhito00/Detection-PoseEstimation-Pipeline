# -*- coding: utf-8 -*-
import copy
from typing import Tuple

import cv2
import numpy as np
import onnxruntime


class RTMPose(object):
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
        self.input_shape = [
            self.input_detail.shape[3],
            self.input_detail.shape[2],
        ]

    def __call__(self, image):
        # preprocessing
        resized_img = self._preprocess(image, self.input_shape)

        # inference
        result = self._run_inference(
            self.onnx_session,
            resized_img,
            image,
        )
        result = np.squeeze(result)

        keypoints, scores = [], []
        for keypoint_x, keypoint_y, score in result:
            keypoints.append([
                int(keypoint_x),
                int(keypoint_y),
            ])
            scores.append(score)

        return keypoints, scores

    def _preprocess(
        self,
        img: np.ndarray,
        input_size: Tuple[int, int] = (192, 256),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            input_size (tuple): Input image size in shape (w, h).

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        # get shape of image
        img_shape = img.shape[:2]
        # get center and scale
        img_wh = np.asarray([img_shape[1], img_shape[0]], dtype=np.float32)
        center = img_wh * 0.5
        scale = img_wh * 1.25

        # do affine transformation
        resized_img, scale = self._top_down_affine(
            input_size,
            scale,
            center,
            img,
        )
        # normalize image
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        resized_img = (resized_img - mean) / std

        return resized_img

    def _top_down_affine(
        self,
        input_size: dict,
        bbox_scale: dict,
        bbox_center: dict,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bbox image as the model input by affine transform.

        Args:
            input_size (dict): The input size of the model.
            bbox_scale (dict): The bbox scale of the img.
            bbox_center (dict): The bbox center of the img.
            img (np.ndarray): The original image.

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: img after affine transform.
            - np.ndarray[float32]: bbox scale after affine transform.
        """
        w, h = input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        bbox_w = bbox_scale[0:1]
        bbox_h = bbox_scale[1:2]
        w_div_075 = bbox_w / 0.75  # 0.75 = model_input_width / model_inut_height
        h_mul_075 = bbox_h * 0.75  # 0.75 = model_input_width / model_inut_height
        w_scaled = np.maximum(h_mul_075, bbox_w)
        h_scaled = np.maximum(w_div_075, bbox_h)
        bbox_scale = np.concatenate([w_scaled, h_scaled], axis=0)

        # get the affine matrix
        center = bbox_center
        scale = bbox_scale
        rot = 0
        warp_mat = self._get_warp_matrix(center,
                                         scale,
                                         rot,
                                         output_size=(w, h))

        # do affine transform
        resized_img = cv2.warpAffine(img,
                                     warp_mat,
                                     warp_size,
                                     flags=cv2.INTER_LINEAR)

        return resized_img, bbox_scale

    def _get_warp_matrix(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        rot: float,
        output_size: Tuple[int, int],
        shift: Tuple[float, float] = (0., 0.),
        inv: bool = False,
    ) -> np.ndarray:
        """Calculate the affine transformation matrix that can warp the bbox area
        in the input image to the output size.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            shift (0-100%): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)

        Returns:
            np.ndarray: A 2x3 transformation matrix
        """
        shift = np.array(shift, dtype=np.float32)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]
        # compute transformation matrix
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(
            np.array([0., src_w * -0.5], dtype=np.float32), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5], dtype=np.float32)
        # get four corners of the src rectangle in the original image
        src_dim0 = center + scale * shift
        src_dim1 = center + src_dir + scale * shift
        src = np.concatenate(
            [
                [src_dim0],
                [src_dim1],
                [self._get_3rd_point(src_dim0, src_dim1)],
            ],
            axis=0,
            dtype=np.float32,
        )
        # get four corners of the dst rectangle in the input image
        dst_dim0 = np.asarray([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_dim1 = np.array([dst_w * 0.5, dst_h * 0.5],
                            dtype=np.float32) + dst_dir
        dst = np.concatenate(
            [
                [dst_dim0],
                [dst_dim1],
                [self._get_3rd_point(dst_dim0, dst_dim1)],
            ],
            axis=0,
            dtype=np.float32,
        )

        if inv:
            warp_mat = cv2.getAffineTransform(dst, src)
        else:
            warp_mat = cv2.getAffineTransform(src, dst)

        return warp_mat.astype(np.float32)

    def _rotate_point(self, pt: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate a point by an angle.

        Args:
            pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
            angle_rad (float): rotation angle in radian

        Returns:
            np.ndarray: Rotated point in shape (2, )
        """
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): The 1st point (x,y) in shape (2, )
            b (np.ndarray): The 2nd point (x,y) in shape (2, )

        Returns:
            np.ndarray: The 3rd point.
        """
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def _run_inference(
        self,
        sess: onnxruntime.InferenceSession,
        resized_img: np.ndarray,
        img: np.ndarray,
    ) -> np.ndarray:
        """Inference RTMPose model.

        Args:
            sess (ort.InferenceSession): ONNXRuntime session.
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input
        input = resized_img.transpose(2, 0, 1)[np.newaxis, ...]

        # build output
        sess_input = {
            sess.get_inputs()[0].name: input,
            sess.get_inputs()[1].name: [[img.shape[1], img.shape[0]]],
        }
        sess_output = []
        for out in sess.get_outputs():
            sess_output.append(out.name)

        # run model
        outputs = sess.run(sess_output, sess_input)

        return outputs

    def draw(
        self,
        image,
        keypoints_list,
        kp_scores_list,
        keypoint_score_th=0.5,
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

        line_indices = [
            # 胴体
            [(5, 6), palette[1]],
            [(5, 11), palette[1]],
            [(6, 12), palette[1]],
            [(11, 12), palette[1]],
            # 左腕
            [(5, 7), palette[2]],
            [(7, 9), palette[2]],
            # 右腕
            [(6, 8), palette[3]],
            [(8, 10), palette[3]],
            # 左足
            [(11, 13), palette[4]],
            [(13, 15), palette[4]],
            [(15, 17), palette[4]],
            [(15, 18), palette[4]],
            [(15, 19), palette[4]],
            [(17, 18), palette[4]],
            # 右足
            [(12, 14), palette[5]],
            [(14, 16), palette[5]],
            [(16, 20), palette[5]],
            [(16, 21), palette[5]],
            [(16, 22), palette[5]],
            [(20, 21), palette[5]],
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
                    cv2.circle(debug_image, keypoint, 2, (255, 255, 255), -1)

        return debug_image
