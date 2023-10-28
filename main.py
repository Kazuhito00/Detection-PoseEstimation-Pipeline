import copy
import time
import argparse

import cv2

from Detector.detector import ObjectDetector
from Estimator.estimator import PoseEstimator


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)

    parser.add_argument(
        '--detector',
        choices=[
            'yolox',
            'efficientdet',
            'ssd',
            'centernet',
            'nanodet',
            'light_person_detector',
        ],
        default='yolox',
    )
    parser.add_argument("--detector_config", type=str, default=None)

    parser.add_argument("--bbox_offset", type=int, default=0)

    parser.add_argument(
        '--estimator',
        choices=[
            'movenet',
            'mediapipe_pose',
            'rtmpose',
        ],
        default='rtmpose',
    )
    parser.add_argument("--estimator_config", type=str, default=None)

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_mirror', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    image_path = None
    if args.image is not None:
        image_path = args.image

    detector_name = args.detector
    detector_config = args.detector_config
    estimator_name = args.estimator
    estimator_config = args.estimator_config

    bbox_offset = args.bbox_offset
    use_gpu = args.use_gpu
    use_mirror = args.use_mirror

    # VideoCapture初期化
    cap = None
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)

    # Object Detection
    detector = ObjectDetector(
        detector_name,
        config_path=detector_config,
        use_gpu=use_gpu,
    )
    detector.print_info()

    # Pose Estimation
    estimator = PoseEstimator(
        estimator_name=estimator_name,
        config_path=estimator_config,
        use_gpu=use_gpu,
    )
    estimator.print_info()

    while True:
        # フレーム読み込み
        if image_path is None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cv2.imread(image_path)

        if use_mirror:
            frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Object Detection
        start_time = time.time()
        d_bboxes, d_scores, d_class_ids = detector(frame)
        d_elapsed_time = time.time() - start_time

        # Pose Estimation
        start_time = time.time()
        keypoints_list, kp_scores_list = estimator(
            frame,
            d_bboxes,
            bbox_offset,
        )
        p_elapsed_time = time.time() - start_time

        # 描画
        debug_image = draw_debug_info(
            debug_image,
            d_elapsed_time,
            p_elapsed_time,
            d_bboxes,
            d_scores,
            d_class_ids,
            bbox_offset,
            estimator.draw,
            keypoints_list,
            kp_scores_list,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('PoseEstimation TopDown Pipeline', debug_image)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def draw_debug_info(
    debug_image,
    d_elapsed_time,
    p_elapsed_time,
    bboxes,
    scores,
    class_ids,
    bbox_offset,
    func_keypoint_draw,
    keypoints_list,
    kp_scores_list,
):
    for bbox, score, _ in zip(bboxes, scores, class_ids):
        x1, y1 = int(bbox[0]) - bbox_offset, int(bbox[1]) - bbox_offset
        x2, y2 = int(bbox[2]) + bbox_offset, int(bbox[3]) + bbox_offset

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        # トラックID、スコア
        text = '%.2f' % score
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            thickness=2,
        )

    debug_image = func_keypoint_draw(
        debug_image,
        keypoints_list,
        kp_scores_list,
    )

    # 経過時間
    cv2.putText(
        debug_image,
        "Detection : " + '{:.1f}'.format(d_elapsed_time * 1000) + "ms",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug_image,
        "PoseEstimation : " + '{:.1f}'.format(p_elapsed_time * 1000) + "ms",
        (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()
