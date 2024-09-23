import cv2
import torch
import numpy as np


def draw_bboxes(img: np.ndarray[np.uint8],
                bboxes: list[dict],
                win_name: str = None,
                wait_time: int = 0,
                draw_label=True):
    draw = img.copy()
    for bbox_anno in bboxes:
        label = bbox_anno['class_idx']
        x0, y0, x1, y1 = [int(i) for i in bbox_anno['bbox']]
        cv2.rectangle(draw, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if draw_label:
            cv2.putText(draw, str(label), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    if win_name:
        cv2.imshow(win_name, draw)
        cv2.waitKey(wait_time)

    return draw


def rescale_img_and_bboxes(img: np.ndarray[np.uint8], bboxes: list[dict], new_h: int, new_w: int):
    h, w = img.shape[:2]
    aspect_ratio_x = new_w / w
    aspect_ratio_y = new_h / h
    new_bboxes = []
    for box in bboxes:
        box_x, box_y, box_w, box_h = box['bbox']
        box_x *= aspect_ratio_x
        box_y *= aspect_ratio_y
        box_w *= aspect_ratio_x
        box_h *= aspect_ratio_y
        new_bboxes.append({
            'bbox': [box_x, box_y, box_w, box_h],
            'class_idx': box['class_idx']
        })

    img = cv2.resize(img, (new_w, new_h))
    return img, new_bboxes


def fit_rescale_and_pad_img_and_bboxes(img: np.ndarray[np.uint8], bboxes: list[dict], img_size: int):
    h, w = img.shape[:2]
    if h > w:
        new_h = img_size
        new_w = round(new_h * w / h)
    else:
        new_w = img_size
        new_h = round(new_w * h / w)

    img, bboxes = rescale_img_and_bboxes(img, bboxes, new_h, new_w)

    x0 = round((img_size - new_w) / 2)
    y0 = round((img_size - new_h) / 2)

    for box in bboxes:
        box['bbox'][0] += x0
        box['bbox'][1] += y0
        box['bbox'][2] += x0
        box['bbox'][3] += y0

    padded_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded_img[y0:y0 + new_h, x0:x0 + new_w, :] = img

    return padded_img, bboxes


def calculate_iou(box1, box2):
    box1_x1, box1_y1, box1_w1, box1_h1 = box1
    box2_x1, box2_y1, box2_w1, box2_h1 = box2

    area1 = box1_w1 * box1_h1
    area2 = box2_w1 * box2_h1

    x0 = max(box1_x1, box2_x1)
    y0 = max(box1_y1, box2_y1)
    x1 = min(box1_x1 + box1_w1, box2_x1 + box2_w1)
    y1 = min(box1_y1 + box1_h1, box2_y1 + box2_h1)

    if x1 < x0 or y1 < y0:  # No intersection
        return 0

    inter_area = (x1 - x0) * (y1 - y0)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def get_bboxes_from_cnn_output(cnn_target_cls: list[torch.Tensor],
                               cnn_target_bbox: list[torch.Tensor],
                               anchors_by_scale: list[torch.Tensor]):
    target_bboxes = []
    for scale_idx in range(len(cnn_target_cls)):
        h, w, n_anchors, n_labels = cnn_target_cls[scale_idx]
        flat_target_cls = cnn_target_cls[scale_idx].reshape(-1, n_labels).argmax(dim=-1)
        flat_target_cnn_box = cnn_target_bbox[scale_idx].reshape(-1, 4)
        mask = flat_target_cls > 0
        scale_target_cls = flat_target_cls[mask]

        scale_anchors = anchors_by_scale[scale_idx].reshape(-1, 4)[mask].numpy()
        scale_target_bboxes = get_target_bbox(flat_target_cnn_box[mask], scale_anchors).numpy()

        for i in range(len(scale_anchors)):
            target_bboxes.append({
                'class_idx': scale_target_cls[i].item(),
                'bbox': scale_target_bboxes[i],
            })
    return target_bboxes


def get_cnn_bbox(target_bboxes, anchor_bboxes):
    w_a = anchor_bboxes[:, 2] - anchor_bboxes[:, 0]
    h_a = anchor_bboxes[:, 3] - anchor_bboxes[:, 1]
    w = target_bboxes[:, 2] - target_bboxes[:, 0]
    h = target_bboxes[:, 3] - target_bboxes[:, 1]

    t_x = (target_bboxes[:, 0] - anchor_bboxes[:, 0]) / w_a
    t_y = (target_bboxes[:, 1] - anchor_bboxes[:, 1]) / h_a
    t_w = torch.log(w / w_a)
    t_h = torch.log(h / h_a)
    return torch.stack([t_x, t_y, t_w, t_h], dim=1)


def get_target_bbox(cnn_bbox, anchor_boxes):
    w_a = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    h_a = anchor_boxes[:, 3] - anchor_boxes[:, 1]

    x = cnn_bbox[:, 0] * w_a + anchor_boxes[:, 0]
    y = cnn_bbox[:, 1] * h_a + anchor_boxes[:, 1]
    w = torch.exp(cnn_bbox[:, 2]) * w_a
    h = torch.exp(cnn_bbox[:, 3]) * h_a
    return torch.stack([x, y, x + w, y + h], dim=1)


def generate_anchors_by_scale(strides: list[int],
                              img_size,
                              scales: list[float],
                              aspect_ratios: list[float]):
    n_anchors = len(scales) * len(aspect_ratios)

    def transform_and_clip_anchor_box(cell_y, cell_x, w, h, cell_width):
        center_x = cell_x * cell_width + cell_width / 2
        center_y = cell_y * cell_width + cell_width / 2
        x0 = max(0, center_x - w / 2)
        y0 = max(0, center_y - h / 2)
        x1 = min(img_size, center_x + w / 2)
        y1 = min(img_size, center_y + h / 2)

        return [x0, y0, x1, y1]

    def generate_cell_anchor_bboxes(cell_i, cell_j, cell_width: int):
        cell_anchors = []
        for scale in scales:
            anchor_width = cell_width * scale
            for aspect_ratio in aspect_ratios:
                anchor_height = anchor_width * aspect_ratio
                anchor_box = transform_and_clip_anchor_box(cell_i, cell_j, anchor_width, anchor_height, cell_width)
                cell_anchors.append(torch.Tensor(anchor_box))

        return torch.stack(cell_anchors)

    all_anchors_by_scale = []
    for stride in strides:
        grid_w = grid_h = int(img_size / stride)
        grid_anchors = torch.zeros(grid_h, grid_w, n_anchors, 4)
        for i in range(grid_h):
            for j in range(grid_w):
                grid_anchors[i, j, :, :] = generate_cell_anchor_bboxes(i, j, stride)

        all_anchors_by_scale.append(grid_anchors)

    return all_anchors_by_scale
