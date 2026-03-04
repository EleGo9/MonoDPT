import os,sys
import re
from easydict import EasyDict as edict
import subprocess
import glob
import torch

from lib.helpers.file_io import *
from lib.helpers.util import *


def get_MAE(results_folder, gt_folder, conf=None, use_logging=False, logger=None,
            thresholds=np.array([0, 25, 50, 1000]),
            iou_overlap_threshold=0.5,
            penalize_unmatched_gt=False,
            visualize=False, vis_bin_width=2, vis_max_error=30, vis_save_path='vis_depth_error.png'):
    num_thresholds = thresholds.shape[0]
    error_box    = []
    error_box_perc= []
    error_rotation= []
    for i in range(num_thresholds):
        error_box.append([])
        error_box_perc.append([])
    pred_box_cnt      = np.zeros(thresholds.shape[0])
    gt_box_cnt         = np.zeros(thresholds.shape[0])

    if visualize:
        all_errors = []
        gt_depths = []

    # Read all pred files
    pred_files    = sorted(glob.glob(str(results_folder) + "/*.txt"))
    num_images    = len(pred_files)
    if use_logging and logger is not None:
        logger.info("Found {} predictions in {}, GT= {}".format(num_images, results_folder, gt_folder))
    else:
        print("Found {} predictions in {}, GT= {}".format(num_images, results_folder, gt_folder))

    for i in range(num_images):
        pred_file = pred_files[i]

        basename_wo_ext = os.path.basename(pred_files[i]).split(".")[0]
        gt_file         = os.path.join(gt_folder   , basename_wo_ext + ".txt")

        # Read ground truth and filter by cars
        gt_boxes_orig = read_csv(path=gt_file, delimiter=" ", ignore_warnings=False, use_pandas=True)
        if gt_boxes_orig is None:
            continue
        # gt_boxes = filter_boxes_by_cars(gt_boxes_orig)
        # gt_boxes = filter_boxes_by_dump(gt_boxes_orig)
        gt_boxes = gt_boxes_orig[:][:, 1:]
        if gt_boxes.shape[0] == 0:
            continue

        # GT boxes data
        for j in range(1, thresholds.shape[0]):
            valid_gt_index_for_th = np.logical_and(gt_boxes[:, 12] >= thresholds[j-1], gt_boxes[:, 12] < thresholds[j])
            num_valid_gt_boxes_th = np.sum(valid_gt_index_for_th)
            gt_box_cnt[j] += num_valid_gt_boxes_th
            gt_box_cnt[0] += num_valid_gt_boxes_th

        # Read predictions and filter by cars
        pred_boxes_orig = read_csv(path= pred_file, delimiter= " ", ignore_warnings= False, use_pandas= True)
        if pred_boxes_orig is None:
            continue
        # pred_boxes      = filter_boxes_by_cars(pred_boxes_orig)
        # pred_boxes = filter_boxes_by_dump(pred_boxes_orig)
        pred_boxes = pred_boxes_orig[:][:, 1:]
        if pred_boxes.shape[0] == 0:
            continue

        # Filter by confidence threshold if provided
        if conf is not None:
            # print('Applying confidence threshold:', conf)
            confidence_mask = pred_boxes[:, 14] > conf
            pred_boxes = pred_boxes[confidence_mask]

        if pred_boxes.shape[0] == 0:
            continue

        # Now get an IoU between them
        pred_x1_y1_x2_y2, pred_z = get_x1_y1_x2_y2_z(pred_boxes)
        gt_x1_y1_x2_y2  , gt_z   = get_x1_y1_x2_y2_z(gt_boxes)
        iou_mat = iou(pred_x1_y1_x2_y2, gt_x1_y1_x2_y2)
        pred_rotation = get_ry(pred_boxes)
        gt_rotation = get_ry(gt_boxes)

        # Get max for each box (prediction -> GT matching)
        max_gt_overlap = np.max(iou_mat, axis=1)
        max_gt_index = np.argmax(iou_mat, axis=1)

        # Check if there is sufficient overlap
        valid_index = max_gt_overlap >= iou_overlap_threshold
        num_valid_boxes = np.sum(valid_index)

        # Track which GT boxes were matched (GT -> prediction matching)
        matched_gt_indices = set()
        if num_valid_boxes > 0:
            # Record which GT boxes have been matched
            matched_gt_indices = set(max_gt_index[valid_index].tolist())

        if num_valid_boxes > 0:
            pred_boxes_valid    = pred_boxes[valid_index]
            pred_z_valid        = pred_z[valid_index]
            pred_boxes_valid_gt = gt_boxes[max_gt_index][valid_index]
            gt_z_valid          = gt_z[max_gt_index][valid_index]
            pred_roty = pred_rotation[valid_index]
            gt_roty   = gt_rotation[max_gt_index][valid_index]

            error_rotation_valid = np.abs(pred_roty - gt_roty)
            error_rotation.extend(error_rotation_valid)

            error_boxes_valid     = np.abs(pred_z_valid - gt_z_valid)
            error_boxes_valid_percentage = np.abs(pred_z_valid - gt_z_valid) / (gt_z_valid + 1e-6)

            if visualize:
                all_errors.extend(error_boxes_valid)
                gt_depths.extend(gt_z_valid)


            # All boxes data
            error_box[0] = custom_append(storage= error_box[0], curr= error_boxes_valid)
            error_box_perc[0] = custom_append(storage= error_box_perc[0], curr= error_boxes_valid_percentage)
            pred_box_cnt[0]   += num_valid_boxes


            # GT binned threshold boxes data
            for j in range(1, thresholds.shape[0]):
                valid_index_for_th  = np.logical_and(gt_z_valid >= thresholds[j-1], gt_z_valid < thresholds[j])
                num_valid_boxes_th  = np.sum(valid_index_for_th)

                if num_valid_boxes_th > 0:
                    error_box[j] = custom_append(storage= error_box[j], curr= error_boxes_valid[valid_index_for_th])
                    error_box_perc[j] = custom_append(storage= error_box_perc[j], curr= error_boxes_valid_percentage[valid_index_for_th])
                    pred_box_cnt[j]   += num_valid_boxes_th

        # Handle unmatched GT boxes (penalize missing predictions if enabled)
        if penalize_unmatched_gt:
            all_gt_indices = set(range(len(gt_boxes)))
            unmatched_gt_indices = all_gt_indices - matched_gt_indices

            if len(unmatched_gt_indices) > 0:
                unmatched_gt_indices = np.array(list(unmatched_gt_indices))
                unmatched_gt_z = gt_z[unmatched_gt_indices]

                # Penalty: use GT depth as error for missing detections
                penalty_errors = unmatched_gt_z
                penalty_errors_percentage = np.ones_like(unmatched_gt_z)

                if visualize:
                    all_errors.extend(penalty_errors)
                    gt_depths.extend(unmatched_gt_z)

                # Add penalties to overall statistics
                error_box[0] = custom_append(storage=error_box[0], curr=penalty_errors)
                error_box_perc[0] = custom_append(storage=error_box_perc[0], curr=penalty_errors_percentage)
                pred_box_cnt[0] += len(unmatched_gt_indices)

                # Add penalties to binned threshold statistics
                for j in range(1, thresholds.shape[0]):
                    valid_index_for_th = np.logical_and(unmatched_gt_z >= thresholds[j-1], unmatched_gt_z < thresholds[j])
                    num_unmatched_th = np.sum(valid_index_for_th)

                    if num_unmatched_th > 0:
                        error_box[j] = custom_append(storage=error_box[j], curr=penalty_errors[valid_index_for_th])
                        error_box_perc[j] = custom_append(storage=error_box_perc[j], curr=penalty_errors_percentage[valid_index_for_th])
                        pred_box_cnt[j] += num_unmatched_th

    error_avg = np.zeros((num_thresholds, ))
    error_perc_avg = np.zeros((num_thresholds, ))
    error_med = np.zeros((num_thresholds, ))
    error_perc_med = np.zeros((num_thresholds, ))
    for i in range(num_thresholds):
        if pred_box_cnt[i] > 0:
            error_avg[i] = np.sum(error_box[i]) / (pred_box_cnt[i] + 1e-6)
            error_perc_avg[i] = np.sum(error_box_perc[i]) / (pred_box_cnt[i] + 1e-6)
            error_med[i] = np.median(error_box[i])
            error_perc_med[i] = np.median(error_box_perc[i])

    # Cicular permute the matrix
    error_avg    = np.roll(error_avg,-1)
    error_perc_avg= np.roll(error_perc_avg,-1)
    error_med    = np.roll(error_med,-1)
    error_perc_med= np.roll(error_perc_med,-1)
    pred_box_cnt = np.roll(pred_box_cnt, -1).astype(int)
    gt_box_cnt   = np.roll(gt_box_cnt   ,-1).astype(int)

    # Prepare log_str
    log_str  = 'Running MAE Statistics...\n'
    log_str += '----------------------------------------------------------------------------------\n'
    log_str += '       Avg              |         Median         |      # boxes / #total boxes\n'
    for j in range(3):
        for i in range(num_thresholds):
            if i == num_thresholds-2:
                log_str += "{:2d}+   ".format(thresholds[i])
            elif i == num_thresholds-1:
                log_str += "All   |"
            else:
                log_str += "{:2d}-{:2d} ".format(thresholds[i], thresholds[i+1])
    log_str += "\n"
    log_str += '----------------------------------------------------------------------------------\n'
    for i in range(num_thresholds):
        log_str += '{:.3f} '.format(error_avg[i])
    log_str += "| "
    for i in range(num_thresholds):
        log_str += '{:.3f} '.format(error_med[i])
    log_str += "| "
    for i in range(num_thresholds):
        log_str += '{:d}/{:d} '.format(pred_box_cnt[i], gt_box_cnt[i])
    log_str += "\n"
    log_str += "\n"
    log_str += '----------------------------------------------------------------------------------\n'
    log_str += '       Avg %           |         Median %       |      # boxes / #total boxes\n'
    for j in range(3):
        for i in range(num_thresholds):
            if i == num_thresholds-2:
                log_str += "{:2d}+   ".format(thresholds[i])
            elif i == num_thresholds-1:
                log_str += "All   |"
            else:
                log_str += "{:2d}-{:2d} ".format(thresholds[i], thresholds[i+1])    
    log_str += "\n"
    log_str += '----------------------------------------------------------------------------------\n'
    for i in range(num_thresholds):
        log_str += '{:.3f} '.format(error_perc_avg[i]*100)
    log_str += "| "
    for i in range(num_thresholds):
        log_str += '{:.3f} '.format(error_perc_med[i]*100)

    log_str += "| "
    for i in range(num_thresholds):
        log_str += '{:d}/{:d} '.format(pred_box_cnt[i], gt_box_cnt[i])
    log_str += "\n"
    log_str += '----------------------------------------------------------------------------------\n'
    log_str += '       Rotation Error   \n'
    # log_str += error_rotation[0].shape[0] * ' '
    log_str += '{}'.format(np.mean(error_rotation))
    error_rotation_degrees = np.mean(error_rotation) * 180 / np.pi
    log_str += " | "
    log_str += '{:.3f}'.format(error_rotation_degrees)
    log_str += "\n"

    if use_logging and logger is not None:
        logger.info(log_str)
    else:
        print(log_str)


    # Generate visualization if requested
    if visualize and len(all_errors) > 0:
        # Convert lists to numpy arrays
        all_errors = np.array(all_errors)
        gt_depths = np.array(gt_depths)
        
        # Import visualization libraries if needed
        # Use non-GUI backend to avoid tkinter threading issues
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator

                # Determine base path for saving the visualizations
        if vis_save_path:
            # Extract the directory and base filename without extension
            # save_dir = os.path.dirname(vis_save_path)
            # if not save_dir:  # If no directory was specified
            #     save_dir = '.'
            save_dir = os.path.dirname(str(results_folder))
                
            # Get the filename without extension
            base_filename = os.path.splitext(os.path.basename(vis_save_path))[0]
            
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Create paths for each visualization
            hist_path = os.path.join(save_dir, f"{base_filename}_histogram.png")
            scatter_path = os.path.join(save_dir, f"{base_filename}_scatter.png")
            bar_path = os.path.join(save_dir, f"{base_filename}_bar.png")
            combined_path = vis_save_path  # Keep original path for combined visualization
        else:
            hist_path = None
            scatter_path = None
            bar_path = None
            combined_path = None
        
        # Calculate common statistics used across plots
        mean_error = np.mean(all_errors)
        median_error = np.median(all_errors)
        # ------ 1. Histogram of depth errors ------
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        bins = np.arange(0, vis_max_error + vis_bin_width, vis_bin_width)
        ax1.hist(all_errors, bins=bins, alpha=0.7, color='cornflowerblue', edgecolor='black')
        ax1.set_title('Distribution of Depth Estimation Errors', fontsize=16)
        ax1.set_xlabel('Absolute Error (m)', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.grid(alpha=0.3)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add mean and median lines
        ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_error:.2f}m')
        ax1.axvline(median_error, color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {median_error:.2f}m')
        ax1.legend(fontsize=12)
        
        plt.tight_layout()
        if hist_path:
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            if use_logging and logger is not None:
                logger.info(f"Histogram saved to {hist_path}")
            else:
                print(f"Histogram saved to {hist_path}")
        else:
            plt.show()
        
        # ------ 2. Scatter plot of error vs. depth ------
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ax2.scatter(gt_depths, all_errors, alpha=0.1, s=15, color='navy')
        ax2.set_title('Depth Error vs. Ground Truth Depth', fontsize=16)
        ax2.set_xlabel('Ground Truth Depth (m)', fontsize=14)
        ax2.set_ylabel('Absolute Error (m)', fontsize=14)
        ax2.grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(gt_depths, all_errors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(gt_depths), max(gt_depths), 100)
        ax2.plot(x_trend, p(x_trend), "r--", linewidth=2, 
                 label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
        
        # Add horizontal line for overall mean error
        ax2.axhline(mean_error, color='green', linestyle='--', linewidth=2,
                    label=f'Mean Error: {mean_error:.2f}m')
        
        ax2.legend(fontsize=12)
        
        plt.tight_layout()
        if scatter_path:
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            if use_logging and logger is not None:
                logger.info(f"Scatter plot saved to {scatter_path}")
            else:
                print(f"Scatter plot saved to {scatter_path}")
        else:
            plt.show()
        
        # ------ 3. Bar chart of errors by distance range ------
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        # Define ranges based on bin_width
        max_gt_depth = max(gt_depths) if len(gt_depths) > 0 else 100
        distance_ranges = np.arange(0, max_gt_depth + vis_bin_width, vis_bin_width)
        avg_errors = []
        error_counts = []
        
        for i in range(len(distance_ranges) - 1):
            range_min = distance_ranges[i]
            range_max = distance_ranges[i + 1]
            
            # Find errors in this range
            in_range = (gt_depths >= range_min) & (gt_depths < range_max)
            errors_in_range = all_errors[in_range]
            
            if len(errors_in_range) > 0:
                avg_error = np.mean(errors_in_range)
                avg_errors.append(avg_error)
                error_counts.append(len(errors_in_range))
            else:
                avg_errors.append(0)
                error_counts.append(0)
        
        # Create labels for the x-axis
        x_labels = [f"{distance_ranges[i]:.0f}-{distance_ranges[i+1]:.0f}" 
                    for i in range(len(distance_ranges) - 1)]
        
        # Plot average errors by distance range
        bars = ax3.bar(x_labels, avg_errors, alpha=0.7, color='teal', edgecolor='black')
        
        # Add count annotations above each bar
        for i, (bar, count) in enumerate(zip(bars, error_counts)):
            if count > 0:  # Only annotate bars with data
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_title('Average Depth Error by Distance Range', fontsize=16)
        ax3.set_xlabel('Ground Truth Depth Range (m)', fontsize=14)
        ax3.set_ylabel('Average Absolute Error (m)', fontsize=14)
        ax3.grid(alpha=0.3)
        
        # Rotate x-axis labels for better readability if many bins
        if len(x_labels) > 10:
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add horizontal line for overall mean error
        ax3.axhline(mean_error, color='red', linestyle='--', linewidth=2,
                    label=f'Overall Mean: {mean_error:.2f}m')
        ax3.legend(fontsize=12)
        
        plt.tight_layout()
        if bar_path:
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            if use_logging and logger is not None:
                logger.info(f"Bar chart saved to {bar_path}")
            else:
                print(f"Bar chart saved to {bar_path}")
        else:
            plt.show()
        
        # ------ 4. Combined visualization (if original path is provided) ------
        if combined_path:
            fig4, (ax1_comb, ax2_comb, ax3_comb) = plt.subplots(3, 1, figsize=(12, 15))
            
            # Recreate histogram
            ax1_comb.hist(all_errors, bins=bins, alpha=0.7, color='cornflowerblue', edgecolor='black')
            ax1_comb.set_title('Distribution of Depth Estimation Errors', fontsize=14)
            ax1_comb.set_xlabel('Absolute Error (m)', fontsize=12)
            ax1_comb.set_ylabel('Count', fontsize=12)
            ax1_comb.grid(alpha=0.3)
            ax1_comb.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1_comb.axvline(mean_error, color='red', linestyle='--', linewidth=1.5, 
                        label=f'Mean: {mean_error:.2f}m')
            ax1_comb.axvline(median_error, color='green', linestyle='--', linewidth=1.5, 
                        label=f'Median: {median_error:.2f}m')
            ax1_comb.legend()
            
            # Recreate scatter plot
            ax2_comb.scatter(gt_depths, all_errors, alpha=0.1, s=10, color='navy')
            ax2_comb.set_title('Depth Error vs. Ground Truth Depth', fontsize=14)
            ax2_comb.set_xlabel('Ground Truth Depth (m)', fontsize=12)
            ax2_comb.set_ylabel('Absolute Error (m)', fontsize=12)
            ax2_comb.grid(alpha=0.3)
            ax2_comb.plot(x_trend, p(x_trend), "r--", linewidth=1.5, 
                     label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
            ax2_comb.legend()
            
            # Recreate bar chart
            bars = ax3_comb.bar(x_labels, avg_errors, alpha=0.7, color='teal', edgecolor='black')
            for i, (bar, count) in enumerate(zip(bars, error_counts)):
                if count > 0:
                    height = bar.get_height()
                    ax3_comb.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'n={count}', ha='center', va='bottom', fontsize=8)
            ax3_comb.set_title('Average Depth Error by Distance Range', fontsize=14)
            ax3_comb.set_xlabel('Ground Truth Depth Range (m)', fontsize=12)
            ax3_comb.set_ylabel('Average Absolute Error (m)', fontsize=12)
            ax3_comb.grid(alpha=0.3)
            if len(x_labels) > 10:
                plt.setp(ax3_comb.get_xticklabels(), rotation=45, ha='right')
            ax3_comb.axhline(mean_error, color='red', linestyle='--', linewidth=1.5,
                        label=f'Overall Mean: {mean_error:.2f}m')
            ax3_comb.legend()
            
            plt.tight_layout()
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            if use_logging and logger is not None:
                logger.info(f"Combined visualization saved to {combined_path}")
            else:
                print(f"Combined visualization saved to {combined_path}")
        
        plt.close('all')

    return np.mean(error_avg)


def custom_append(storage, curr):
    if type(storage) == np.ndarray:
        storage = np.append(storage, curr)
    else:
        storage = curr
    return storage

def safe_subprocess(command):
    """Execute subprocess with timeout protection."""
    try:
        proc = subprocess.Popen([command], preexec_fn=os.setsid, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True)
        output, error = proc.communicate()
        return output.decode('UTF-8')
    except:
        return "Error Occurred"

def get_x1_y1_x2_y2_z(boxes):
    #       0  1   2      3   4   5   6    7    8    9   10   11   12
    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
    return boxes[:, 3:7], boxes[:, 12]
def get_ry(boxes):
    #   0  1   2      3   4   5   6    7    8    9   10   11   12   13  14  15
    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
    return boxes[:, 13]

def filter_boxes_by_cars(boxes):
    mask = np.logical_or(boxes[:, 0] == "Car", boxes[:, 0] == "car")
    return boxes[mask][:, 1:]

def filter_boxes_by_dump(boxes):
    mask = np.logical_or(boxes[:, 0] == "Dump", boxes[:, 0] == "dump")
    return boxes[mask][:, 1:]

def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            # print("Using combinations in intersect function on numpy")
            # Calculates the coordinates of the overlap box
            # eg if the box x-coords is at 4 and 5, then the overlap will be minimum
            # of the two which is 4
            # np.maximum is to take two arrays and compute their element-wise maximum.
            # Here, 'compatible' means that one array can be broadcast to the other.
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))

def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0) + area_b.unsqueeze(1) - inter
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))
