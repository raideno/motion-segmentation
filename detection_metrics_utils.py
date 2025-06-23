import json
import torch
import numpy as np
import pandas as pd

def interpolated_prec_rec(prec, rec):
    """
    Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.

    tIoU = segments_intersection / segments_union
    tIoU[np.where(segments_union <= 0)[0]] = 0
    tIoU[np.where(np.isnan(np.asarray(tIoU,dtype=np.float64)))[0]] = 0
    tIoU[np.where(np.isinf(np.asarray(tIoU,dtype=np.float64)))[0]] = 0

    return tIoU


def convert_sequence_to_detection_format(
    sequences,
    video_ids=None, 
    sequence_lengths=None,
    scores=None
):
    """
    Convert frame-wise label sequences to detection format with segments.
    
    Parameters
    ----------
    sequences : list of arrays
        List of frame-wise label sequences
    video_ids : list, optional
        List of video IDs. If None, uses indices.
    sequence_lengths : list, optional
        List of sequence lengths in time units. If None, uses frame indices.
    scores : list of arrays, optional
        List of confidence scores for each frame. If None, uses uniform scores.
    
    Returns
    -------
    detection_data : list of dicts
        Detection format data
    """
    if video_ids is None:
        video_ids = [f"video_{i}" for i in range(len(sequences))]
    
    detection_data = []
    
    for i, (sequence, video_id) in enumerate(zip(sequences, video_ids)):
        # Extract segments from sequence
        segments = []
        labels = []
        segment_scores = []
        
        prev_label = sequence[0]
        start_idx = 0
        
        for j in range(1, len(sequence)):
            if sequence[j] != prev_label:
                segments.append([start_idx, j])
                labels.append(prev_label)
                
                # Calculate segment score (average of frame scores if available)
                if scores is not None and i < len(scores):
                    segment_score = np.mean(scores[i][start_idx:j])
                else:
                    segment_score = 1.0  # Default confidence
                segment_scores.append(segment_score)
                
                start_idx = j
                prev_label = sequence[j]
        
        # Add final segment
        segments.append([start_idx, len(sequence)])
        labels.append(prev_label)
        if scores is not None and i < len(scores):
            segment_score = np.mean(scores[i][start_idx:])
        else:
            segment_score = 1.0
        segment_scores.append(segment_score)
        
        # Convert to time units if sequence_lengths provided
        if sequence_lengths is not None and i < len(sequence_lengths):
            time_factor = sequence_lengths[i] / len(sequence)
            segments = [[s[0] * time_factor, s[1] * time_factor] for s in segments]
        
        detection_data.append({
            video_id: {
                'segments': torch.tensor(segments, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'scores': torch.tensor(segment_scores, dtype=torch.float32),
                'length': torch.tensor(sequence_lengths[i] if sequence_lengths else len(sequence), dtype=torch.float32)
            }
        })
    
    return detection_data


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.1, 0.5, 5)):
    """
    Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    if ground_truth.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)

        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    for tidx in range(len(tiou_thresholds)):
        # this_tp = np.cumsum(tp[tidx, :]).astype(np.float)
        this_tp = np.cumsum(tp[tidx, :]).astype(np.float64)
        # this_fp = np.cumsum(fp[tidx, :]).astype(np.float)
        this_fp = np.cumsum(fp[tidx, :]).astype(np.float64)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap


class TemporalActionDetectionEvaluator:
    """
    Simplified version of ActionDetectionEvaluator that works with 
    frame-wise predictions converted to temporal segments.
    """
    
    def __init__(self, tiou_thresholds=np.linspace(0.1, 0.9, 9), verbose=True):
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
    
    def convert_to_dataframe(self, detection_data, is_prediction=True):
        """Convert detection format to DataFrame"""
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        score_lst = [] if is_prediction else None
        
        for data_dict in detection_data:
            for video_id, v in data_dict.items():
                if len(v['segments']) == 0:
                    continue
                    
                segments = v['segments']
                labels = v['labels']
                
                for i in range(len(segments)):
                    video_lst.append(video_id)
                    t_start_lst.append(float(segments[i][0]))
                    t_end_lst.append(float(segments[i][1]))
                    label_lst.append(int(labels[i]))
                    
                    if is_prediction and score_lst is not None:
                        score_lst.append(float(v['scores'][i]))
        
        df_data = {
            'video-id': video_lst,
            't-start': t_start_lst,
            't-end': t_end_lst,
            'label': label_lst
        }
        
        if is_prediction and score_lst is not None:
            df_data['score'] = score_lst
            
        return pd.DataFrame(df_data)
    
    def evaluate(self, groundtruth_data, prediction_data):
        """
        Evaluate temporal action detection performance.
        
        Parameters
        ----------
        groundtruth_data : list of dicts
            Ground truth in detection format
        prediction_data : list of dicts
            Predictions in detection format
            
        Returns
        -------
        evaluate_stat : dict
            Evaluation statistics including mAP scores
        """
        # Convert to DataFrames
        groundtruth_dataframe = self.convert_to_dataframe(groundtruth_data, is_prediction=False)
        prediction_dataframe = self.convert_to_dataframe(prediction_data, is_prediction=True)
        
        if groundtruth_dataframe.empty or prediction_dataframe.empty:
            if self.verbose:
                print("[WARNING] Empty ground truth or predictions for temporal detection evaluation")
            return {'mAP': 0.0}
        
        # Get unique classes
        unique_classes = sorted(list(set(groundtruth_dataframe['label'].unique()) | set(prediction_dataframe['label'].unique())))
        
        # Compute AP for each class
        ap = np.zeros((len(self.tiou_thresholds), len(unique_classes)))
        
        for i, class_id in enumerate(unique_classes):
            gt_class = groundtruth_dataframe[groundtruth_dataframe['label'] == class_id].reset_index(drop=True)
            pred_class = prediction_dataframe[prediction_dataframe['label'] == class_id].reset_index(drop=True)
            
            if len(gt_class) > 0 and len(pred_class) > 0:
                ap[:, i] = compute_average_precision_detection(
                    gt_class, pred_class, tiou_thresholds=self.tiou_thresholds
                )
        
        self.ap = ap
        self.mAP = ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()
        
        if self.verbose:
            print('[RESULTS] Performance on temporal action detection task.')
            for i, tiou in enumerate(self.tiou_thresholds):
                print(f'\tmAP@tIoU{tiou:.2f}: {100 * self.mAP[i]:.2f}%')
            print(f'\tAverage mAP: {100 * self.average_mAP:.2f}%')
        
        # Prepare evaluation statistics
        evaluate_stat = {'mAP': float(self.average_mAP)}
        for tIoU, mAP in zip(self.tiou_thresholds, self.mAP):
            evaluate_stat[f'{tIoU:.2f}_mAP'] = float(mAP)
            
        return evaluate_stat

def convert_sequence_to_detection_format_with_logits(
    sequences_logits,
    video_ids=None, 
    sequence_lengths=None
):
    """
    Convert frame-wise class logits to detection format with confidence-based segments.
    
    Parameters
    ----------
    sequences_logits : list of arrays
        List of frame-wise class logits/probabilities [seq_len, num_classes]
    video_ids : list, optional
        List of video IDs. If None, uses indices.
    sequence_lengths : list, optional
        List of sequence lengths in time units. If None, uses frame indices.
   
    Returns
    -------
    detection_data : list of dicts
        Detection format data with confidence scores
    """
    if video_ids is None:
        video_ids = [f"video_{i}" for i in range(len(sequences_logits))]
    
    detection_data = []
    
    for i, (sequence_logits, video_id) in enumerate(zip(sequences_logits, video_ids)):
        sequence_probs = torch.tensor(sequence_logits).numpy()
        
        predicted_classes = np.argmax(sequence_probs, axis=-1)
        
        segments = []
        labels = []
        segment_scores = []
        
        prev_label = predicted_classes[0]
        start_idx = 0
        
        for j in range(1, len(predicted_classes)):
            if predicted_classes[j] != prev_label:
                segment_logits = sequence_probs[start_idx:j]
                segment_confidence = calculate_segment_confidence(
                    segment_logits, prev_label
                )
                
                segments.append([start_idx, j])
                labels.append(prev_label)
                segment_scores.append(segment_confidence)
                
                start_idx = j
                prev_label = predicted_classes[j]
        
        segment_logits = sequence_probs[start_idx:]
        segment_confidence = calculate_segment_confidence(
            segment_logits, prev_label
        )
        
        segments.append([start_idx, len(predicted_classes)])
        labels.append(prev_label)
        segment_scores.append(segment_confidence)
        
        if sequence_lengths is not None and i < len(sequence_lengths):
            time_factor = sequence_lengths[i] / len(predicted_classes)
            segments = [[s[0] * time_factor, s[1] * time_factor] for s in segments]
        
        detection_data.append({
            video_id: {
                'segments': torch.tensor(segments, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'scores': torch.tensor(segment_scores, dtype=torch.float32),
                'length': torch.tensor(sequence_lengths[i] if sequence_lengths else len(predicted_classes), dtype=torch.float32)
            }
        })
    
    return detection_data


def calculate_segment_confidence(segment_logits, predicted_label):
    """
    Calculate confidence score for a segment based on its logits.
    
    Parameters
    ----------
    segment_logits : array
        Logits/probabilities for frames in the segment [segment_len, num_classes]
    predicted_label : int
        The predicted class label for this segment
  
    Returns
    -------
    float
        Confidence score between 0 and 1
    """
    if len(segment_logits) == 0:
        return 0.0
    
    # NOTE: average of maximum probabilities across framesn, apply softmax if not already probabilities
    if segment_logits.shape[-1] > 1:
        probs = torch.softmax(torch.tensor(segment_logits), dim=-1).numpy()
        max_probs = np.max(probs, axis=-1)
    else:
        max_probs = np.max(segment_logits, axis=-1)
    return float(np.mean(max_probs))