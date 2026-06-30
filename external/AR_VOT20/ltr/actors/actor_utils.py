import torch
import numpy as np
import cv2


def compute_iou(pred, target):
    '''form: (x1,y1,x2,y2)'''
    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 2]
    pred_y2 = pred[:, 3]

    target_x1 = target[:, 0]
    target_y1 = target[:, 1]
    target_x2 = target[:, 2]
    target_y2 = target[:, 3]
    '''Compute seperate areas'''
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    '''Compute intersection area and iou'''
    w_inter = torch.clamp(torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1), 0)
    h_inter = torch.clamp(torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1), 0)
    area_intersect = w_inter * h_inter
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    return ious.mean()


def mask2bbox(mask_tensor, MASK_THRESHOLD=0.5):
    batch = mask_tensor.size(0)
    # result = np.zeros((batch,4))
    mask_arr = np.array(mask_tensor.squeeze().cpu())  # (b,H,W) (0,1)
    target_mask = (mask_arr > MASK_THRESHOLD)
    target_mask = target_mask.astype(np.uint8)
    valid_list = []
    result_list = []
    for i in range(batch):
        '''get contours'''
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask[i],
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask[i],
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        '''choose contour with max area, then transform it to bbox'''
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(cnt_area) != 0:
            valid_list.append(i)
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            result_list.append(np.array(cv2.boundingRect(polygon)))  # (x1,y1,w,h)
    if len(valid_list) > 0:
        result = np.array(result_list)
        result_tensor = torch.from_numpy(result.astype(np.float32)).cuda()
        result_tensor[:, 2:] += result_tensor[:, :2]  # (x1,y1,x2,y2)
        return result_tensor, valid_list
    else:
        return torch.zeros((1,)), valid_list
