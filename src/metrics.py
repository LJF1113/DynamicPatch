"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from numpy import ndarray
from skimage import measure
from skimage.measure import regionprops
from skimage import measure
from sklearn.metrics import auc
from statistics import mean
import pandas as pd
from scipy.ndimage.measurements import label
from adeval import EvalAccumulatorCuda
import torch

def compute_adeval_pro(masks,amaps,scores,min_score,max_score,min_seg_score,max_seg_score):
    # amaps_norm = (amaps-amaps.min())/(amaps.max()-amaps.min())
    gt_sp = masks.max(axis=(1,2))
    pr_sp_max = amaps.max(axis=(1,2))
    pr_sp_mean = amaps.mean(axis=(1,2))

    # score_min = min(pr_sp_max) - 1e-7
    # # print(score_min.shape)
    # score_max = max(pr_sp_max) + 1e-7
    score_min = min(scores) - 1e-7
    # print(score_min.shape)
    score_max = max(scores) + 1e-7
    anomap_min = amaps.min()
    anomap_max = amaps.max()
    # print(min_score.shape)
    # accum = EvalAccumulatorCuda(min_score,max_score,min_seg_score,max_seg_score,skip_pixel_aupro=False,nstrips=50)
    accum = EvalAccumulatorCuda(score_min,score_max,anomap_min,anomap_max,skip_pixel_aupro=False,nstrips=1000)
    accum.add_anomap_batch(torch.tensor(amaps).cuda(non_blocking=True),torch.tensor(masks.astype(np.uint8)).cuda(non_blocking=True))

    # for i in range(torch.tensor(amaps).size(0)):
    #     accum.add_image(torch.tensor(pr_sp_max[i]),torch.tensor(gt_sp[i]))
    for i in range(torch.tensor(amaps).size(0)):
        accum.add_image(torch.tensor(scores[i]),torch.tensor(gt_sp[i]))
    metrics = accum.summary()
    print(metrics)
    return metrics

def cal_pro_score(masks,amaps,max_step=200,expect_fpr=0.3,mp=False):
    min_th, max_th = amaps.min(),amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    binary_amaps = np.zeros_like(amaps,dtype=np.bool_)
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps<=th], binary_amaps[amaps>th] = 0,1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:,0], region.coords[:,1]].sum()
                pro.append(tp_pixels/region.area)
        inverse_masks = 1-masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fpr < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs-fprs.min()) / (fprs.max()-fprs.min())
    
    fprs = fprs[0,:]
    pros = pros[idxes][0,:]
    pro_auc = auc(fprs,pros)
    return pro_auc



def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    # TODO: draw_curve
    # draw_curve(fpr, tpr, auroc)
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(f1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        
        new_raw = pd.Series({"pro": mean(pros), "fpr": fpr, "threshold": th})
        df = pd.concat([df,new_raw.to_frame().T], ignore_index=True)
        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        # print({"pro": mean(pros), "fpr": fpr, "threshold": th})

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def draw_curve(fpr, tpr, auroc):
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.4f})'.format(auroc), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    error = 0.015
    miss = 0.1
    plt.plot([error, error], [-0.05, 1.05], 'k:', lw=1)
    plt.plot([-0.05, 1.05], [1-miss, 1-miss], 'k:', lw=1)
    error_y, miss_x = 0, 1
    for i in range(len(fpr)):
        if fpr[i] <= error <= fpr[i + 1]:
            error_y = tpr[i]
        if tpr[i] <= 1-miss <= tpr[i + 1]:
            miss_x = fpr[i]
    # plt.scatter(error, error_y, c='k')
    # plt.scatter(miss_x, 1-miss, c='k')
    plt.text(error, error_y, "({0}, {1:.4f})".format(error, error_y), color='k')
    plt.text(miss_x, 1-miss, "({0:.4f}, {1})".format(miss_x, 1-miss), color='k')
    plt.show()
