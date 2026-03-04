import torch
from lightning.fabric import Fabric

import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from lib.helpers.config_helper import Config
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


class InheritedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self._base_dataset = datasets[0]  # you can use others too if needed

    def __getattr__(self, name):
        # This gets called only if the attribute isn't found on self
        return getattr(self._base_dataset, name)

    def get_calib(self, img_id):
        """
        Get calibration for a given image ID, properly handling multiple concatenated datasets.

        Args:
            img_id: The actual image ID (not concatenated index)

        Returns:
            Calibration object from the dataset that contains this image ID
        """
        # Try to find the image in each dataset
        for dataset in self.datasets:
            # Format the image ID according to the dataset's filename_format
            formatted_id = dataset.filename_format % img_id
            # Check if this image ID exists in this dataset's idx_list
            if formatted_id in dataset.idx_list:
                return dataset.get_calib(img_id)

        # If not found in any dataset, raise an error
        raise ValueError(f"Image ID {img_id} not found in any of the concatenated datasets")

    def get_label(self, img_id):
        """
        Get label for a given image ID, properly handling multiple concatenated datasets.

        Args:
            img_id: The actual image ID (not concatenated index)

        Returns:
            Label objects from the dataset that contains this image ID
        """
        # Try to find the image in each dataset
        for dataset in self.datasets:
            # Format the image ID according to the dataset's filename_format
            formatted_id = dataset.filename_format % img_id
            # Check if this image ID exists in this dataset's idx_list
            if formatted_id in dataset.idx_list:
                return dataset.get_label(img_id)

        # If not found in any dataset, raise an error
        raise ValueError(f"Image ID {img_id} not found in any of the concatenated datasets")

    def get_dataset_name(self, img_id):
        """
        Get the dataset name/source for a given image ID.

        Args:
            img_id: The actual image ID (not concatenated index)

        Returns:
            String identifier for the dataset (derived from root_dir)
        """
        # Try to find the image in each dataset
        for i, dataset in enumerate(self.datasets):
            print(f"Dataset {i}: {dataset.root_dir}")
            # Format the image ID according to the dataset's filename_format
            print(dataset.filename_format)
            formatted_id = dataset.filename_format[i] % img_id
            # Also try string version of img_id (idx_list might have strings or ints)
            str_img_id = str(img_id)

            # Check if this image ID exists in this dataset's idx_list
            if formatted_id in dataset.idx_list or str_img_id in dataset.idx_list:
                # Extract a more unique dataset name from root_dir
                import os
                root_dir = dataset.root_dir
                parts = root_dir.rstrip('/').split('/')

                # Use last 2 parts of path for better uniqueness
                # E.g., "IVECO_SIM_MONO3D_albarola/ugv_camera_fc_f_n_img"
                # or "IVECO_albarola_real/20250521_ALBAROLA_idv_truck_barra_loop_marker_econ2_correspMergedCloud"
                if len(parts) >= 2:
                    dataset_name = f"{parts[-2]}_{parts[-1]}"
                else:
                    dataset_name = parts[-1]

                # Clean up the name to be filesystem-safe
                dataset_name = dataset_name.replace('/', '_')
                return dataset_name

        # If not found in any dataset, raise an error
        raise ValueError(f"Image ID {img_id} not found in any of the concatenated datasets")

    def get_image(self, img_id):
        """
        Get image for a given image ID, properly handling multiple concatenated datasets.

        Args:
            img_id: The actual image ID (not concatenated index)

        Returns:
            Image from the dataset that contains this image ID
        """
        # Try to find the image in each dataset
        for dataset in self.datasets:
            # Format the image ID according to the dataset's filename_format
            formatted_id = dataset.filename_format % img_id
            # Check if this image ID exists in this dataset's idx_list
            if formatted_id in dataset.idx_list:
                return dataset.get_image(img_id)

        # If not found in any dataset, raise an error
        raise ValueError(f"Image ID {img_id} not found in any of the concatenated datasets")

    def eval(self, results_dir, logger, cur_img_ids=None):
        """
        Evaluate results for the specific dataset that matches the results_dir.

        Args:
            results_dir: Directory containing prediction results for a specific dataset
            logger: Logger for output messages
            cur_img_ids: Optional list of specific image IDs to evaluate (currently unused)

        Returns:
            Tuple of (3D mAP, 2D mAP) for the matched dataset
        """
        import lib.datasets.kitti.indy_eval_python.indy_common as indy
        import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
        from lib.datasets.kitti.indy_eval_python.eval import get_indy_eval_result
        from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
        from lib.helpers.rpn_util import get_MAE
        import os

        # Determine which dataset to evaluate based on results_dir
        # Extract the dataset name from the results_dir path
        results_dir_name = os.path.basename(str(results_dir).rstrip('/'))

        if logger is not None:
            logger.info(f"==> Looking for dataset matching results_dir: {results_dir_name}")
        else:
            print(f"==> Looking for dataset matching results_dir: {results_dir_name}")

        # Find the matching dataset
        target_dataset = None
        for dataset in self.datasets:
            dataset_name = os.path.basename(str(dataset.root_dir).rstrip('/'))
            if dataset_name == results_dir_name:
                target_dataset = dataset
                break

        # If no exact match, use the first dataset (fallback for single dataset case)
        if target_dataset is None:
            if logger is not None:
                logger.info(f"==> No exact dataset match found for '{results_dir_name}', using first dataset")
            else:
                print(f"==> No exact dataset match found for '{results_dir_name}', using first dataset")
            target_dataset = self.datasets[0]

        # Get dataset name for reporting
        dataset_name = os.path.basename(str(target_dataset.root_dir).rstrip('/'))

        if logger is not None:
            logger.info(f"==> Evaluating dataset: {dataset_name}")
            logger.info(f"==> Root dir: {target_dataset.root_dir}")
        else:
            print(f"==> Evaluating dataset: {dataset_name}")
            print(f"==> Root dir: {target_dataset.root_dir}")

        # Get img_ids only from this specific dataset
        img_ids = [int(id) for id in target_dataset.idx_list]

        if logger is not None:
            logger.info(f"==> Number of images in this dataset: {len(img_ids)}")
        else:
            print(f"==> Number of images in this dataset: {len(img_ids)}")

        # Check if KITTI official evaluation is enabled
        kitti_official_eval = getattr(target_dataset, 'kitti_official_eval', False)

        if kitti_official_eval:
            # ============ KITTI Official Evaluation ============
            if logger is not None:
                logger.info('='*80)
                logger.info('Starting KITTI official evaluation')
                logger.info('='*80)
            else:
                print('='*80)
                print('Starting KITTI official evaluation')
                print('='*80)

            # Load annotations using KITTI format
            dt_annos = kitti.get_label_annos(results_dir)
            gt_annos = kitti.get_label_annos(target_dataset.label_dir, img_ids)

            # Standard KITTI class IDs
            test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

            if logger is not None:
                logger.info('==> Evaluating (official KITTI) ...')
            else:
                print('==> Evaluating (official KITTI) ...')

            # Collect results for all categories
            all_results = {}
            car_moderate = 0
            mAP_2d_total = 0

            for category in target_dataset.writelist:
                if category not in test_id:
                    msg = f"Warning: Category '{category}' not in KITTI standard classes, skipping."
                    if logger is not None:
                        logger.info(msg)
                    else:
                        print(msg)
                    continue

                msg = f"\n--- Evaluating category: {category} ---"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)

                results_str, results_dict, mAP3d_R40 = get_official_eval_result(
                    gt_annos, dt_annos, test_id[category]
                )

                if category == 'Car':
                    car_moderate = mAP3d_R40
                print(results_str)

                # Extract 2D mAP (image detection) - using moderate difficulty
                # Key format: '{class_name}_image_moderate_R40'
                mAP_2d_key = f'{category}_image_moderate_R40'
                mAP_2d = results_dict.get(mAP_2d_key, 0.0)

                all_results[category] = {
                    'results_str': results_str,
                    'results_dict': results_dict,
                    'mAP3d_R40': mAP3d_R40,
                    'mAP2d_R40': mAP_2d
                }

                if logger is not None:
                    logger.info(results_str)
                else:
                    print(results_str)

                mAP_2d_total += mAP_2d

            # Average 2D mAP across all evaluated categories
            avg_2d_mAP = mAP_2d_total / len(all_results) if len(all_results) > 0 else 0

            summary = f'KITTI Official Evaluation Complete\nCar Moderate 3D mAP (R40): {car_moderate:.4f}\nAverage 2D mAP: {avg_2d_mAP:.4f}'
            if logger is not None:
                logger.info('='*80)
                logger.info(summary)
                logger.info('='*80)
            else:
                print('='*80)
                print(summary)
                print('='*80)

            return car_moderate, avg_2d_mAP

        else:
            # ============ Custom/INDY Evaluation ============
            # Load ground truth annotations only from this dataset
            gt_annos = indy.get_label_annos(target_dataset.label_dir, img_ids, filename_format=target_dataset.filename_format)

            # Load detection results only for this dataset's images
            dt_annos = indy.get_label_annos(results_dir, img_ids, filename_format=target_dataset.filename_format)

            if logger is not None:
                logger.info(f'==> GT annotations: {len(gt_annos)}, Detections: {len(dt_annos)}')
            else:
                print(f'==> GT annotations: {len(gt_annos)}, Detections: {len(dt_annos)}')

            # Compute MAE for depth estimation for this dataset
            if logger is not None:
                logger.info(f"==> Computing depth MAE for {dataset_name}...")
            else:
                print(f"==> Computing depth MAE for {dataset_name}...")

            error_avg = get_MAE(
                results_folder=results_dir,
                gt_folder=target_dataset.label_dir,
                use_logging=True,
                logger=logger,
                visualize=True
            )

            # Evaluate using this dataset's annotations
            if logger is not None:
                logger.info(f'==> Computing mAP for {dataset_name}...')
            else:
                print(f'==> Computing mAP for {dataset_name}...')

            results_str, results_dict = get_indy_eval_result(
                gt_annos,
                dt_annos,
                target_dataset.writelist,
                max_depth=target_dataset.depth_threshold,
                iou_thres=0.7
            )

            if logger is not None:
                logger.info(results_str)
            else:
                print(results_str)

            return results_dict['3d mAP'], results_dict['2d mAP']


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(fabric: Fabric, cfg: Config, workers: int =4):
    # perpare dataset
    if cfg.dataset.type == 'KITTI':
        train_set = KITTI_Dataset(split=cfg.dataset.train_split, cfg=cfg)
        test_set = KITTI_Dataset(split=cfg.dataset.test_split, cfg=cfg)
    elif cfg.dataset.type == 'INDY':
        from lib.datasets.indy.indy_dataset import INDY_Dataset
        train_sets = []
        test_sets = []
        root_dirs_train = cfg.dataset.root_dir_train
        for root_dir in root_dirs_train:
            train_sets.append(INDY_Dataset(split=cfg.dataset.train_split, cfg=cfg, root_dir=root_dir))
            print('training: ', root_dir)
        root_dirs_test = cfg.dataset.root_dir_test
        for root_dir_test in root_dirs_test:
            test_sets.append(INDY_Dataset(split=cfg.dataset.test_split, cfg=cfg, root_dir = root_dir_test))
            print('test: ', root_dir_test)

        train_set = InheritedConcatDataset(train_sets)
        train_set.max_objs = train_sets[0].max_objs
        base_dataset = train_sets[0]
        print('Number of samples in train', len(train_set))
        test_set = InheritedConcatDataset(test_sets)
        test_set.max_objs = test_sets[0].max_objs
        print('Number of samples in test', len(test_set))
    elif cfg.dataset.type == 'CUSTOM':
        from lib.datasets.custom.custom_dataset import Custom_Dataset
        train_sets = []
        test_sets = []

        # Handle root_dir_train or fallback to root_dir
        if hasattr(cfg.dataset, 'root_dir_train') and cfg.dataset.root_dir_train is not None:
            root_dirs_train = cfg.dataset.root_dir_train
        elif isinstance(cfg.dataset.root_dir, list):
            root_dirs_train = cfg.dataset.root_dir
        else:
            root_dirs_train = [cfg.dataset.root_dir]

        # Handle root_dir_test or fallback to root_dir
        if hasattr(cfg.dataset, 'root_dir_test') and cfg.dataset.root_dir_test is not None:
            root_dirs_test = cfg.dataset.root_dir_test
        elif isinstance(cfg.dataset.root_dir, list):
            root_dirs_test = cfg.dataset.root_dir
        else:
            root_dirs_test = [cfg.dataset.root_dir]

        for n, root_dir in enumerate(root_dirs_train):
            train_sets.append(Custom_Dataset(split=cfg.dataset.train_split, cfg=cfg, root_dir=root_dir, dataset_id=n ))
            print('training: ', root_dir)
        for n, root_dir_test in enumerate(root_dirs_test):
            test_sets.append(Custom_Dataset(split=cfg.dataset.test_split, cfg=cfg, root_dir = root_dir_test, dataset_id=n))
            print('test: ', root_dir_test)

        train_set = InheritedConcatDataset(train_sets)
        train_set.max_objs = train_sets[0].max_objs
        base_dataset = train_sets[0]
        print('Number of samples in train', len(train_set))
        test_set = InheritedConcatDataset(test_sets)
        test_set.max_objs = test_sets[0].max_objs
        print('Number of samples in test', len(test_set))
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg.dataset.type)

    # prepare dataloader
    train_loader = fabric.setup_dataloaders(
        DataLoader(dataset=train_set,
                              batch_size=cfg.dataset.batch_size,
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False)
    )
    test_loader = fabric.setup_dataloaders(
            DataLoader(dataset=test_set,
                             batch_size=cfg.dataset.batch_size,
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    )

    return train_loader, test_loader
