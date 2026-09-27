import sys
sys.path.insert(0, ".")

import argparse
import json
import numpy as np
import os
import glob
from pathlib import Path

# Import official evaluator functions
from lib.datasets.indy.indy_eval_python import eval as kitti_eval
from lib.datasets.indy.indy_eval_python import kitti_common

def load_predictions(results_dir, image_ids):
    annos = []
    for idx in image_ids:
        path = os.path.join(results_dir, f"{idx:06d}.txt")
        if os.path.exists(path):
            annos.append(kitti_common.get_label_anno(path))
        else:
            annos.append({
                'name': np.array([]), 'truncated': np.array([]),
                'occluded': np.array([]), 'alpha': np.array([]),
                'bbox': np.zeros((0, 4)), 'dimensions': np.zeros((0, 3)),
                'location': np.zeros((0, 3)), 'rotation_y': np.array([]),
                'score': np.array([])
            })
    return annos

def compute_statistics_matched(overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes, metric, min_overlap, thresh=0, compute_fp=False, compute_aos=False):
    """
    Exact replica of official compute_statistics_jit, but augmented to return
    (gt_idx, det_idx, iou, dt_score) for matched pairs.
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1] if det_size > 0 else np.array([])
    
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    
    matched_pairs = []

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
                max_overlap = overlap
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True
                max_overlap = overlap

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
            matched_pairs.append((i, det_idx, max_overlap, dt_scores[det_idx]))

    return tp, fp, fn, similarity, thresholds[:thresh_idx], matched_pairs

def print_stats(name, values):
    if len(values) == 0:
        print(f"{name}:")
        print(f"  N/A")
        return
    print(f"{name}:")
    print(f"  mean:   {np.mean(values):.4f}")
    print(f"  median: {np.median(values):.4f}")
    print(f"  p25:    {np.percentile(values, 25):.4f}")
    print(f"  p75:    {np.percentile(values, 75):.4f}")
    print(f"  p90:    {np.percentile(values, 90):.4f}")

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--class_name', type=str, default='Car')
    parser.add_argument('--official_threshold', type=float, default=0.7)
    parser.add_argument('--diagnostic_threshold', type=float, default=1e-5)
    parser.add_argument('--max_depth', type=float, default=120.0)
    parser.add_argument('--out_json', type=str, default='eval_diagnostics_3d_errors.json')
    args = parser.parse_args()

    print(f"Loading GT from: {args.gt_dir}")
    print(f"Loading Predictions from: {args.results_dir}")
    
    gt_files = glob.glob(os.path.join(args.gt_dir, "*.txt"))
    image_ids = [int(os.path.basename(f).split('.')[0]) for f in gt_files]
    image_ids.sort()
    
    if len(image_ids) == 0:
        print("No GT files found!")
        return

    gt_annos = kitti_common.get_label_annos(args.gt_dir, image_ids)
    dt_annos = load_predictions(args.results_dir, image_ids)

    print(f"Loaded {len(gt_annos)} images from GT directory.")

    difficulty = 0 
    
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts = []
    ignored_dets = []
    dontcares = []
    
    total_valid_gt = 0
    total_valid_dt = 0
    
    for i in range(len(gt_annos)):
        rets = kitti_eval.clean_data_kitti(gt_annos[i], dt_annos[i], args.class_name, difficulty, args.max_depth)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        dontcares.append(dc_bboxes)
        
        gt_datas = np.concatenate([gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        if len(dt_annos[i]["name"]) > 0:
            dt_datas = np.concatenate([dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis], dt_annos[i]["score"][..., np.newaxis]], 1)
        else:
            dt_datas = np.zeros((0, 6))
            
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
        
        total_valid_gt += num_valid_gt
        if len(ignored_det) > 0:
            total_valid_dt += np.sum(np.array(ignored_det) == 0)

    print("Computing official 3D IoU overlaps...")
    rets = kitti_eval.calculate_iou_partly(dt_annos, gt_annos, 2, 100)
    overlaps_3d, parted_overlaps, total_dt_num, total_gt_num = rets
    
    print("\n" + "="*50)
    print("OFFICIAL-STYLE MATCHING")
    print("="*50)
    
    official_tp = 0
    official_fn = 0
    
    for i in range(len(gt_annos)):
        tp, fp, fn, _, _, _ = compute_statistics_matched(
            overlaps_3d[i], gt_datas_list[i], dt_datas_list[i],
            ignored_gts[i], ignored_dets[i], dontcares[i],
            metric=2, min_overlap=args.official_threshold, thresh=0.0, compute_fp=False
        )
        official_tp += tp
        official_fn += fn
    
    print(f"TP at IoU > {args.official_threshold}: {official_tp}")
    print(f"FN at IoU > {args.official_threshold}: {official_fn}")
    print("(Note: This validates the matching reproduction but is not the full AP calculation)")
    
    print("\n" + "="*50)
    print("BEST 3D IoU PER VALID GT (Score-Independent)")
    print("="*50)
    
    best_iou_per_gt_list = []
    for i in range(len(gt_annos)):
        for gt_idx in range(len(gt_annos[i]['name'])):
            if ignored_gts[i][gt_idx] != 0: # Only count valid target GTs
                continue
            
            valid_det_mask = (ignored_dets[i] != -1)
            if np.any(valid_det_mask) and len(dt_annos[i]['name']) > 0:
                best_iou = np.max(overlaps_3d[i][valid_det_mask, gt_idx])
            else:
                best_iou = 0.0
            best_iou_per_gt_list.append(best_iou)
            
    print(f"GT count: {len(best_iou_per_gt_list)}")
    print_stats("3D IoU", best_iou_per_gt_list)
    
    if len(best_iou_per_gt_list) > 0:
        iou_arr = np.array(best_iou_per_gt_list)
        print("\nPercentage of valid GTs with at least one prediction having:")
        print(f"  IoU > 0.30: {np.mean(iou_arr > 0.30)*100:.2f}%")
        print(f"  IoU > 0.50: {np.mean(iou_arr > 0.50)*100:.2f}%")
        print(f"  IoU > {args.official_threshold:.2f}: {np.mean(iou_arr > args.official_threshold)*100:.2f}%")

    print("\n" + "="*50)
    print("DIAGNOSTIC POSITIVE-OVERLAP MATCHES (IoU > 0)")
    print("="*50)
    
    all_matched_pairs = []
    dx_list, dy_list, dz_list = [], [], []
    rel_dz_list = []
    dh_list, dw_list, dl_list = [], [], []
    drot_list = []
    iou_list = []
    
    unmatched_gt_count = 0
    nearest_matches = []
    
    for i in range(len(gt_annos)):
        # Diagnostic matching (IoU > 0)
        tp, fp, fn, _, _, matched_pairs = compute_statistics_matched(
            overlaps_3d[i], gt_datas_list[i], dt_datas_list[i],
            ignored_gts[i], ignored_dets[i], dontcares[i],
            metric=2, min_overlap=0.0, thresh=0.0, compute_fp=False
        )
        
        gt_anno = gt_annos[i]
        dt_anno = dt_annos[i]
        
        matched_gt_indices = set([m[0] for m in matched_pairs])
        
        for (gt_idx, dt_idx, max_overlap, dt_score) in matched_pairs:
            gt_loc = gt_anno['location'][gt_idx]
            gt_dim = gt_anno['dimensions'][gt_idx]
            gt_ry = gt_anno['rotation_y'][gt_idx]
            
            dt_loc = dt_anno['location'][dt_idx]
            dt_dim = dt_anno['dimensions'][dt_idx]
            dt_ry = dt_anno['rotation_y'][dt_idx]
            
            dx = dt_loc[0] - gt_loc[0]
            dy = dt_loc[1] - gt_loc[1]
            dz = dt_loc[2] - gt_loc[2]
            
            dh = dt_dim[0] - gt_dim[0]
            dw = dt_dim[1] - gt_dim[1]
            dl = dt_dim[2] - gt_dim[2]
            
            drot = wrap_angle(dt_ry - gt_ry)
            
            rel_dz = abs(dz) / abs(gt_loc[2]) if gt_loc[2] != 0 else 0
            
            dx_list.append(abs(dx))
            dy_list.append(abs(dy))
            dz_list.append(abs(dz))
            rel_dz_list.append(rel_dz)
            
            dh_list.append(abs(dh))
            dw_list.append(abs(dw))
            dl_list.append(abs(dl))
            drot_list.append(abs(drot))
            iou_list.append(max_overlap)
            
            all_matched_pairs.append({
                "image_id": f"{image_ids[i]:06d}",
                "gt_index": int(gt_idx),
                "prediction_index": int(dt_idx),
                "gt": {
                    "position": gt_loc.tolist(),
                    "dimensions": gt_dim.tolist(),
                    "rotation_y": float(gt_ry)
                },
                "prediction": {
                    "position": dt_loc.tolist(),
                    "dimensions": dt_dim.tolist(),
                    "rotation_y": float(dt_ry),
                    "confidence": float(dt_score)
                },
                "errors": {
                    "dx": float(dx),
                    "dy": float(dy),
                    "dz": float(dz),
                    "relative_depth_error": float(rel_dz),
                    "dh": float(dh),
                    "dw": float(dw),
                    "dl": float(dl),
                    "rotation_error": float(drot)
                },
                "3d_iou": float(max_overlap)
            })

        # Process unmatched GTs for Population B
        dt_bboxes = dt_datas_list[i][:, :4]
        gt_bboxes = gt_datas_list[i][:, :4]
        if len(dt_bboxes) > 0 and len(gt_bboxes) > 0:
            overlaps_2d = kitti_eval.image_box_overlap(dt_bboxes, gt_bboxes, 0)
        else:
            overlaps_2d = np.zeros((len(dt_bboxes), len(gt_bboxes)))

        for gt_idx in range(len(gt_anno['name'])):
            if ignored_gts[i][gt_idx] != 0: # Only count valid target GTs
                continue
            if gt_idx in matched_gt_indices:
                continue
                
            unmatched_gt_count += 1
            
            best_det_idx = -1
            best_2d_iou = 0
            for det_idx in range(len(dt_anno['name'])):
                if ignored_dets[i][det_idx] == -1:
                    continue
                if overlaps_2d[det_idx, gt_idx] > best_2d_iou:
                    best_2d_iou = overlaps_2d[det_idx, gt_idx]
                    best_det_idx = det_idx
                    
            if best_det_idx != -1:
                nearest_matches.append({
                    "image_id": f"{image_ids[i]:06d}",
                    "gt_index": int(gt_idx),
                    "diagnostic_nearest_prediction_index": int(best_det_idx),
                    "diagnostic_2d_iou": float(best_2d_iou),
                    "diagnostic_3d_iou": float(overlaps_3d[i][best_det_idx, gt_idx]),
                    "note": "no positive-overlap 3D prediction from evaluator matching"
                })
    
    print(f"matched count: {len(all_matched_pairs)}")
    print("-" * 50)
    
    print_stats("3D IoU statistics", iou_list)
    print_stats("position errors |ΔX|", dx_list)
    print_stats("position errors |ΔY|", dy_list)
    print_stats("depth errors |ΔZ|", dz_list)
    print_stats("relative depth error", rel_dz_list)
    
    print_stats("dimension errors H", dh_list)
    print_stats("dimension errors W", dw_list)
    print_stats("dimension errors L", dl_list)
    print_stats("rotation errors (rad)", drot_list)
    
    print("\n" + "="*50)
    print("GTs UNMATCHED BY SCORE-PRIORITY POSITIVE-OVERLAP MATCHING")
    print("="*50)
    print(f"count: {unmatched_gt_count}")
    percentage = (unmatched_gt_count / total_valid_gt * 100) if total_valid_gt > 0 else 0
    print(f"percentage: {percentage:.2f}%")
    if len(nearest_matches) > 0:
        print(f"diagnostic_nearest_match found for: {len(nearest_matches)} objects (via max 2D IoU)")
    
    # Dump to JSON
    output_data = {
        "positive_overlap_matches": all_matched_pairs,
        "diagnostic_nearest_matches_for_unmatched_gt": nearest_matches,
        "unmatched_gt_count": unmatched_gt_count,
        "total_valid_gt": total_valid_gt
    }
    with open(args.out_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nSaved detailed diagnostic data to {args.out_json}")

if __name__ == '__main__':
    main()
