import os
import cv2
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.utils import check_random_state
import torch
from hamer.configs import CACHE_DIR_HAMER
import os
import cv2
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.utils import check_random_state
import torch
from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
import json
from typing import Dict, Optional
import glob
from natsort import natsorted
from transformations import *


# Configuration constants
CAMERA_IDS = [1, 2, 3, 4]
CALIBRATION_PATH = # PATH TO CAMERA CALIBRATION
DEFAULT_DATA_ROOT = # PATH TO HUMAN DATA
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
MANO2FRANKA = np.array(
    [
        [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        [np.sqrt(2) / 2, 0, -np.sqrt(2) / 2],
        [0, 1, 0],
    ]
) @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

MAX_GRIPPER_WIDTH = 0.08500000089406967  # robotiq 2f-85




def load_intrinsics(path="./", cam_idx_list=[1, 2, 3, 4]):
    intrinsics_list = []

    for cam_idx in cam_idx_list:
        intrinsics = np.load(f"{path}/cam{cam_idx}_intrinsics.npy")

        intrinsics_list.append(intrinsics)

    return intrinsics_list


def load_transforms(path="./", cam_idx_list=[1, 2, 3, 4]):

    transforms_list = []

    for cam_idx in cam_idx_list:
        extrinsics = np.load(f"{path}/cam{cam_idx}_extrinsics.npy")

        transforms_list.append(extrinsics)

    return transforms_list


def parse_args():
    parser = argparse.ArgumentParser(description="Full hand tracking pipeline")
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of task (e.g. grasp)"
    )
    parser.add_argument(
        "--traj_num", type=int, required=True, help="Trajectory number (e.g. 1)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing task folders",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip image extraction if already done",
    )
    parser.add_argument(
        "--skip_reconstruction",
        action="store_true",
        help="Skip 2D keypoint detection if already done",
    )
    parser.add_argument(
        "--skip_triangulation",
        action="store_true",
        help="Skip 2D keypoint detection if already done",
    )

    return parser.parse_args()


def extract_images(task_path, traj_num):
    traj_path = os.path.join(task_path, f"traj_{traj_num}")

    # 00001,00002, ...
    pkl_path = os.path.join(task_path, f"traj_{traj_num:05d}.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Trajectory pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        trajs = pickle.load(f)

    for cam_id in CAMERA_IDS:
        cam_path = os.path.join(traj_path, str(cam_id))
        os.makedirs(cam_path, exist_ok=True)

    for i, transition in enumerate(trajs[0]):
        for cam_idx in range(len(CAMERA_IDS)):
            cam_id = CAMERA_IDS[cam_idx]
            image = transition[0][f"cam{cam_id}"]
            depth = transition[0][f"cam{cam_id}_depth"]

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)

            base_path = os.path.join(traj_path, str(cam_id))
            cv2.imwrite(os.path.join(base_path, f"{i:05d}.jpg"), image)
            cv2.imwrite(os.path.join(base_path, f"{i:05d}_depth.jpg"), depth)


def run_hamer_reconstruction(cam_images_path):
    # (Include all HAMER reconstruction code from your original script here)
    # This should process each camera's images and save keypoints/MANO params
    # Make sure to organize outputs in a 'processed' directory per camera'
    img_folder = cam_images_path
    out_folder = os.path.join(img_folder, "processed")

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    cfg_path = (
        Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    os.makedirs(out_folder, exist_ok=True)

    img_paths = [img for img in Path(img_folder).glob("*.jpg")]

    # remove ones ends with depth.jpg
    img_paths = [img for img in img_paths if not str(img).endswith("depth.jpg")]

    img_paths = natsorted(img_paths)

    for img_path in img_paths:

        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=48, shuffle=False, num_workers=0
        )

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = 2 * batch["right"] - 1
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )

            pred_cam_t_full = (
                cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_length
                )
                .detach()
                .cpu()
                .numpy()
            )

            # 2d keypoints
            try:
                right_keypoints_2d = (
                    (
                        out["pred_keypoints_2d"][(right == 1)] * box_size[(right == 1)]
                        + box_center[right == 1].reshape(-1, 1, 2)
                    )
                    .cpu()
                    .numpy()
                )[
                    0:1
                ].squeeze()  # if multiple hands, just take the first one

            except:
                # fill nans for not detected frames
                right_keypoints_2d = np.full((21, 2), np.nan)


            # mano params
            right_mano_params = {
                "global_orient": out["pred_mano_params"]["global_orient"][(right == 1)]
                .squeeze()
                .cpu()
                .numpy(),
                "hand_pose": out["pred_mano_params"]["hand_pose"][(right == 1)]
                .squeeze()
                .cpu()
                .numpy(),
                "betas": out["pred_mano_params"]["betas"][(right == 1)]
                .squeeze()
                .cpu()
                .numpy(),
            }

            # Render the result
            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch["personid"][n])

                white_img = (
                    torch.ones_like(batch["img"][n]).cpu()
                    - DEFAULT_MEAN[:, None, None] / 255
                ) / (DEFAULT_STD[:, None, None] / 255)
                input_patch = batch["img"][n].cpu() * (
                    DEFAULT_STD[:, None, None] / 255
                ) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                regression_img = renderer(
                    out["pred_vertices"][n].detach().cpu().numpy(),
                    out["pred_cam_t"][n].detach().cpu().numpy(),
                    batch["img"][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )

                final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(
                    os.path.join(out_folder, f"{img_fn}_{person_id}.png"),
                    255 * final_img[:, :, ::-1],
                )

                np.save(
                    os.path.join(out_folder, f"{img_fn}_righthand_keypoints.npy"),
                    right_keypoints_2d,
                )

                np.save(
                    os.path.join(out_folder, f"{img_fn}_righthand_mano_params.npy"),
                    right_mano_params,
                )

                # Add all verts and cams to list
                verts = out["pred_vertices"][n].detach().cpu().numpy()
                is_right = batch["right"][n].cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                if is_right:
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                all_right.append(is_right)

                save_mesh = True

                # Save all meshes to disk
                if save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(
                        verts, camera_translation, LIGHT_BLUE, is_right=is_right
                    )
                    tmesh.export(os.path.join(out_folder, f"{img_fn}_{person_id}.obj"))

        full_frame = True
        # Render front view
        if full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[n],
                # is_right=all_right,
                **misc_args,
            )

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )  # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )

            # add right hand 2d keypoints as visualication
            right_keypoints_2d_int = right_keypoints_2d.astype(int)[:, ::-1]

            input_img_overlay[
                right_keypoints_2d_int[0, 0] - 3 : right_keypoints_2d_int[0, 0] + 3,
                right_keypoints_2d_int[0, 1] - 3 : right_keypoints_2d_int[0, 1] + 3,
                0,
            ] = 1.0
            input_img_overlay[
                right_keypoints_2d_int[0, 0] - 3 : right_keypoints_2d_int[0, 0] + 3,
                right_keypoints_2d_int[0, 1] - 3 : right_keypoints_2d_int[0, 1] + 3,
                1,
            ] = 0.0
            input_img_overlay[
                right_keypoints_2d_int[0, 0] - 3 : right_keypoints_2d_int[0, 0] + 3,
                right_keypoints_2d_int[0, 1] - 3 : right_keypoints_2d_int[0, 1] + 3,
                2,
            ] = 0.0

            input_img_overlay[
                right_keypoints_2d_int[4, 0] - 3 : right_keypoints_2d_int[4, 0] + 3,
                right_keypoints_2d_int[4, 1] - 3 : right_keypoints_2d_int[4, 1] + 3,
                0,
            ] = 1.0
            input_img_overlay[
                right_keypoints_2d_int[4, 0] - 3 : right_keypoints_2d_int[4, 0] + 3,
                right_keypoints_2d_int[4, 1] - 3 : right_keypoints_2d_int[4, 1] + 3,
                1,
            ] = 0.0
            input_img_overlay[
                right_keypoints_2d_int[4, 0] - 3 : right_keypoints_2d_int[4, 0] + 3,
                right_keypoints_2d_int[4, 1] - 3 : right_keypoints_2d_int[4, 1] + 3,
                2,
            ] = 0.0

            input_img_overlay[
                right_keypoints_2d_int[8, 0] - 3 : right_keypoints_2d_int[8, 0] + 3,
                right_keypoints_2d_int[8, 1] - 3 : right_keypoints_2d_int[8, 1] + 3,
                0,
            ] = 1.0
            input_img_overlay[
                right_keypoints_2d_int[8, 0] - 3 : right_keypoints_2d_int[8, 0] + 3,
                right_keypoints_2d_int[8, 1] - 3 : right_keypoints_2d_int[8, 1] + 3,
                1,
            ] = 0.0
            input_img_overlay[
                right_keypoints_2d_int[8, 0] - 3 : right_keypoints_2d_int[8, 0] + 3,
                right_keypoints_2d_int[8, 1] - 3 : right_keypoints_2d_int[8, 1] + 3,
                2,
            ] = 0.0

            input_img_overlay[
                right_keypoints_2d_int[12, 0] - 3 : right_keypoints_2d_int[12, 0] + 3,
                right_keypoints_2d_int[12, 1] - 3 : right_keypoints_2d_int[12, 1] + 3,
                0,
            ] = 1.0
            input_img_overlay[
                right_keypoints_2d_int[12, 0] - 3 : right_keypoints_2d_int[12, 0] + 3,
                right_keypoints_2d_int[12, 1] - 3 : right_keypoints_2d_int[12, 1] + 3,
                1,
            ] = 0.0
            input_img_overlay[
                right_keypoints_2d_int[12, 0] - 3 : right_keypoints_2d_int[12, 0] + 3,
                right_keypoints_2d_int[12, 1] - 3 : right_keypoints_2d_int[12, 1] + 3,
                2,
            ] = 0.0

            input_img_overlay[
                right_keypoints_2d_int[16, 0] - 3 : right_keypoints_2d_int[16, 0] + 3,
                right_keypoints_2d_int[16, 1] - 3 : right_keypoints_2d_int[16, 1] + 3,
                0,
            ] = 1.0
            input_img_overlay[
                right_keypoints_2d_int[16, 0] - 3 : right_keypoints_2d_int[16, 0] + 3,
                right_keypoints_2d_int[16, 1] - 3 : right_keypoints_2d_int[16, 1] + 3,
                1,
            ] = 0.0
            input_img_overlay[
                right_keypoints_2d_int[16, 0] - 3 : right_keypoints_2d_int[16, 0] + 3,
                right_keypoints_2d_int[16, 1] - 3 : right_keypoints_2d_int[16, 1] + 3,
                2,
            ] = 0.0

            input_img_overlay[
                right_keypoints_2d_int[20, 0] - 3 : right_keypoints_2d_int[20, 0] + 3,
                right_keypoints_2d_int[20, 1] - 3 : right_keypoints_2d_int[20, 1] + 3,
                0,
            ] = 1.0
            input_img_overlay[
                right_keypoints_2d_int[20, 0] - 3 : right_keypoints_2d_int[20, 0] + 3,
                right_keypoints_2d_int[20, 1] - 3 : right_keypoints_2d_int[20, 1] + 3,
                1,
            ] = 0.0
            input_img_overlay[
                right_keypoints_2d_int[20, 0] - 3 : right_keypoints_2d_int[20, 0] + 3,
                right_keypoints_2d_int[20, 1] - 3 : right_keypoints_2d_int[20, 1] + 3,
                2,
            ] = 0.0

            cv2.imwrite(
                os.path.join(out_folder, f"{img_fn}_all.jpg"),
                255 * input_img_overlay[:, :, ::-1],
            )


def triangulate_3d_keypoints(traj_path):
    # Process all frames using multiple camera views
    kp3d_list = []
 
    # Create output directory
    output_dir = os.path.join(traj_path, "processed_3d")
    os.makedirs(output_dir, exist_ok=True)

    # Get frame list from first camera
    sample_cam_dir = os.path.join(traj_path, str(CAMERA_IDS[1]), "processed")
    frame_files = sorted(
        [
            f
            for f in os.listdir(sample_cam_dir)
            if f.endswith("_righthand_keypoints.npy")
        ]
    )

    # Process each frame
    for frame_file in frame_files:
        frame_idx = frame_file.split("_")[0]
        keypoints_list = []

        # Load keypoints from all cameras
        for cam_id in CAMERA_IDS:
            cam_dir = os.path.join(traj_path, str(cam_id), "processed")
            kp_path = os.path.join(cam_dir, f"{frame_idx}_righthand_keypoints.npy")
            try:
                keypoints = np.load(kp_path)  # (21, 2)
            except:
                # just fill with nan if not found
                keypoints = np.full((21, 2), np.nan)
            keypoints_list.append(keypoints)

        # Triangulate with RANSAC
        kp3d, inlier_counts = triangulate_with_ransac(
            keypoints_list=keypoints_list,
            intrinsics=load_intrinsics(
                path=CALIBRATION_PATH, cam_idx_list=CAMERA_IDS
            ),  # Implement your intrinsics loader
            transforms=load_transforms(
                path=CALIBRATION_PATH, cam_idx_list=CAMERA_IDS
            ),  # Implement your transform loader
        )
        print(inlier_counts)
        kp3d_list.append(kp3d)

    # Save 3D keypoints
    output_path = os.path.join(output_dir, f"righthand_3d_keypoints.npy")
    np.save(output_path, np.array(kp3d_list))


def triangulate_with_ransac(
    keypoints_list, intrinsics, transforms, ransac_iter=100, reproj_threshold=25.0
):
    """
    :param keypoints_list: List of 4 (21,2) arrays
    :param ransac_iter: Number of RANSAC iterations
    :param reproj_threshold: Pixel error threshold for inliers
    :return: 3D points (21,3), inlier counts (21,)
    """
    # Build world-to-camera projection matrices 
    projections = []
    for i, T in enumerate(transforms):
        T_4x4 = np.eye(4)
        T_4x4[:3, :4] = T
        extrinsic = np.linalg.inv(T_4x4)[:3, :]

        projections.append(intrinsics[i] @ extrinsic)

    num_kps = keypoints_list[0].shape[0]
    kp3d = np.zeros((num_kps, 3))
    inlier_counts = np.zeros(num_kps, dtype=int)

    for kp_id in range(num_kps):
        # Collect all observations
        observations = []
        for cam_idx in range(len(intrinsics)):
            y, x = keypoints_list[cam_idx][kp_id]

            # x, y = keypoints_list[cam_idx][kp_id]
            if not np.isnan(x) and not np.isnan(y):  # Handle missing obs
                observations.append((cam_idx, x, y))

        best_inliers = []
        best_point = None

        # RANSAC loop
        for _ in range(ransac_iter):
            # Randomly sample minimal set (2 views)
            sample = check_random_state(None).choice(
                len(observations), size=min(2, len(observations)), replace=False
            )

            # Build linear system
            A = []
            for idx in sample:
                cam_idx, x, y = observations[idx]
                P = projections[cam_idx]
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])

            # Solve via SVD [1]
            _, _, V = np.linalg.svd(np.array(A))
            point_homo = V[-1, :4]
            candidate = point_homo[:3] / point_homo[3]

            # Count inliers
            inliers = []
            for obs in observations:
                cam_idx, u, v = obs
                P = projections[cam_idx]
                proj = P @ np.append(candidate, 1)
                proj = proj[:2] / proj[2]
                error = np.linalg.norm(proj - [u, v])
                if error < reproj_threshold:
                    inliers.append(obs)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_point = candidate

        # Refine with all inliers
        if best_point is not None and len(best_inliers) >= 2:
            A = []
            for obs in best_inliers:
                cam_idx, x, y = obs
                P = projections[cam_idx]
                A.append(y * P[2] - P[1])
                A.append(x * P[2] - P[0])

            _, _, V = np.linalg.svd(np.array(A))
            point_homo = V[-1, :4]
            kp3d[kp_id] = point_homo[:3] / point_homo[3]
            inlier_counts[kp_id] = len(best_inliers)

    # import ipdb; ipdb.set_trace()
    return kp3d, inlier_counts


def exponential_moving_average(data, alpha=0.1):
    """Apply EMA smoothing to 3D keypoints trajectory"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def save_eef_gripper(traj_path, cam_id=1, alpha=0.1):
    # save eef pose, gripper action.
    extrinsics = load_transforms(path=CALIBRATION_PATH, cam_idx_list=[cam_id])[0]
    extrinsics_R = extrinsics[
        :3, :3
    ]  # this is rotation of cam, represented from world frame.

    # normalize rotation
    extrinsics_R[:, 0] = extrinsics_R[:, 0] / (
        np.linalg.norm(extrinsics_R[:, 0]) + 1e-8
    )
    extrinsics_R[:, 1] = extrinsics_R[:, 1] / (
        np.linalg.norm(extrinsics_R[:, 1]) + 1e-8
    )

    # left hand to right hand frame
    extrinsics_R[:, 2] = np.cross(extrinsics_R[:, 0], extrinsics_R[:, 1])

    keypoints_path = os.path.join(traj_path, "processed_3d/righthand_3d_keypoints.npy")

    keypoints_3d_list = np.load(keypoints_path)

    # Apply EMA to each keypoint coordinate independently
    smoothed_kps = np.zeros_like(keypoints_3d_list)
    for kp_idx in range(21):  # For each keypoint
        for coord in range(3):  # For x,y,z coordinates
            smoothed_kps[:, kp_idx, coord] = exponential_moving_average(
                keypoints_3d_list[:, kp_idx, coord], alpha
            )

    keypoints_3d_list = smoothed_kps


    eef_pose_list = []
    gripper_action_list = []
    prev_orient = np.eye(3)

    for keypoints_3d in keypoints_3d_list:
        eef_pose = np.zeros(7)



        A = keypoints_3d[0]  # Wrist (smoothed)
        B = keypoints_3d[4]  # Thumb tip (smoothed)
        C = keypoints_3d[[8, 12, 16, 20]].mean(axis=0)  # Other tips average (smoothed)

        # Calculate orthogonal basis vectors
        vec_BA = B - A
        vec_CA = C - A

        # z axis is mean of the two vectors
        z_axis = vec_CA + vec_BA
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        x_axis = -np.cross(vec_CA, vec_BA)  
        x_axis /= np.linalg.norm(x_axis) + 1e-8  # Prevent division by zero

        y_axis = np.cross(z_axis, x_axis)

        ori_robot = np.column_stack(
            (x_axis.squeeze(), y_axis.squeeze(), z_axis.squeeze())
        )

        prev_orient = ori_robot

        quat = rmat_to_quat(ori_robot)

        eef_pose[:3] = keypoints_3d[0]
        eef_pose[3:] = quat

        eef_pose_list.append(eef_pose)

        gripper_width = np.linalg.norm(B - C)

        gripper_width_norm = np.clip(gripper_width / MAX_GRIPPER_WIDTH, 0, 1)

        print(gripper_width_norm)

        gripper_action = 1 - gripper_width_norm


        gripper_action_list.append(gripper_action)

    eef_pose_list = np.array(eef_pose_list)
    
    # update offset for robotiq gripper.
    for i in range(len(eef_pose_list)):
        eef_pose = eef_pose_list[i]    
        eef_rotation = quat_to_rmat(eef_pose[3:])
        eef_pose_list[i][:3] -= eef_rotation @ np.array([0,0,0.062]).T

    gripper_action_list = np.array(gripper_action_list)

    save_path = os.path.join(traj_path, "processed_3d")
    os.makedirs(save_path, exist_ok=True)

    np.save(f"{save_path}/eef_pose.npy", eef_pose_list)
    np.save(f"{save_path}/retarget_gripper_action.npy", gripper_action_list)


def main():
    args = parse_args()
    task_path = os.path.join(args.data_root, args.task_name)
    traj_path = os.path.join(task_path, f"traj_{args.traj_num}")

    if not args.skip_extraction:
        print("Extracting images from pickle...")
        extract_images(task_path, args.traj_num)

    if not args.skip_reconstruction:
        print("Running 2D hand reconstruction...")
        for cam_id in CAMERA_IDS:
            cam_images_path = os.path.join(traj_path, str(cam_id))
            run_hamer_reconstruction(cam_images_path)

    if not args.skip_triangulation:
        print("Triangulating 3D keypoints...")
        triangulate_3d_keypoints(traj_path)

    print("saving end effector pose and gripper actions..")
    save_eef_gripper(traj_path)

    print(f"Processing complete for {args.task_name}/traj_{args.traj_num}")


if __name__ == "__main__":
    main()
