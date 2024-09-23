import os
import json
import pickle
from torch.utils.data import Dataset, DataLoader

from utils.config import *
from utils.constants import *
from utils.detection_utils import *
from torchvision.ops import box_iou


def get_label_encoder():
    with open(VAL_COCO_ANNO_PATH, 'r') as f:
        coco_json = json.load(f)

    coco_categories = coco_json['categories']
    label2idx = {0: 0}
    for i in range(len(coco_categories)):
        category = coco_categories[i]
        label2idx[category['id']] = i + 1

    return label2idx


# Restructuring of coco annotations to list of {image_id and bbox annotations}
def build_or_load_annotations(coco_anno_path: str, our_anno_path: str, label2idx: dict) -> list[dict]:
    if os.path.exists(our_anno_path):
        with open(our_anno_path, 'rb') as f:
            return pickle.load(f)

    with open(coco_anno_path, 'r') as f:
        coco_json = json.load(f)

    coco_annotations = coco_json["annotations"]
    bboxes_annotations = {}
    for item in coco_annotations:
        img_id = item["image_id"]
        x, y, w, h = item["bbox"]
        label = item["category_id"]
        bbox_anno = {'bbox': [x, y, x + w, y + h], 'class_idx': label2idx[label]}
        bboxes_annotations[img_id] = bboxes_annotations.get(img_id, []) + [bbox_anno]

    annotations = []

    img_filenames = {}
    for img_info in coco_json["images"]:
        img_filenames[img_info["id"]] = img_info["file_name"]

    for img_id in img_filenames:
        img_anno = {
            'image_id': img_id,
            'bboxes': bboxes_annotations.get(img_id, []),
            'file_name': img_filenames[img_id],
        }
        annotations.append(img_anno)

    with open(our_anno_path, 'wb') as f:
        pickle.dump(annotations, f)

    return annotations


def plus_minus_value_augment(value, plus_minus_rate: float):
    delta = (2 * np.random.rand() - 1) * plus_minus_rate * value
    return value + delta


def random_rescale_img_and_bboxes(img: np.ndarray[np.uint8], bboxes: list[dict], rescale_rate: float):
    h, w = img.shape[:2]
    new_h = int(round(plus_minus_value_augment(h, rescale_rate)))
    new_w = int(round(plus_minus_value_augment(w, rescale_rate)))
    return rescale_img_and_bboxes(img, bboxes, new_h, new_w)


def generate_efficient_det_target(all_cnn_anchors: list[torch.Tensor],
                                  targets_annotations: list[dict],
                                  n_labels: int,
                                  iou_th: float = 0.5):
    target_cnn_cls = []
    target_cnn_bbox = []
    identity_mat = torch.eye(n_labels)

    target_boxes = torch.Tensor([target_anno['bbox'] for target_anno in targets_annotations])

    if len(targets_annotations) > 0:
        target_cls = torch.stack([identity_mat[target_anno['class_idx'], :] for target_anno in targets_annotations])

        max_iou_for_target = torch.zeros(len(targets_annotations))
        best_anchor_for_target = torch.zeros(len(targets_annotations), 2)  # scale_idx, anchor_idx
        matched_targets = torch.zeros(len(targets_annotations)).bool()

    # Here at first, we find best target per each grid cell anchor based on iou and threshold it.
    # We store all assigned targets based on threshold, we will need this info later on
    # Then we calculate best anchor per target based on iou on all scales
    # Then unassigned targets will be matched to the best anchors regardless of thresholds
    for scale_idx, grid_anchors in enumerate(all_cnn_anchors):
        grid_h, grid_w, n_anchors = grid_anchors.shape[:3]

        flat_grid_anchors = grid_anchors.reshape(-1, 4)
        flat_grid_cnn_bbox = torch.zeros(flat_grid_anchors.shape[0], 4).clone()
        flat_grid_cnn_cls = torch.zeros(flat_grid_anchors.shape[0], n_labels)
        flat_grid_cnn_cls[:, 0] = 1
        if len(targets_annotations) > 0:
            ious = box_iou(flat_grid_anchors, target_boxes)

            max_ious, best_targets = ious.max(dim=-1)
            mask = (max_ious >= iou_th).unsqueeze(1)

            best_grid_cnn_cls = target_cls[best_targets]
            best_grid_cnn_bbox = get_cnn_bbox(target_boxes[best_targets], flat_grid_anchors)

            flat_grid_cnn_cls = torch.where(mask, best_grid_cnn_cls, flat_grid_cnn_cls)
            flat_grid_cnn_bbox = torch.where(mask, best_grid_cnn_bbox, flat_grid_cnn_bbox)

        target_cnn_cls.append(flat_grid_cnn_cls.reshape(grid_h, grid_w, n_anchors, n_labels))
        target_cnn_bbox.append(flat_grid_cnn_bbox.reshape(grid_h, grid_w, n_anchors, 4))

        if len(targets_annotations) == 0:
            continue
        # Here we work on unassigned targets, we need to calculate best grid cells for them in all scales
        ious = ious.t()

        max_ious, best_anchors = ious.max(dim=-1)
        mask = max_ious > max_iou_for_target
        scale_idx_per_target = torch.Tensor([scale_idx] * len(max_ious))
        matched_targets = torch.logical_or(matched_targets, max_ious >= iou_th)
        max_iou_for_target = torch.where(mask, max_ious, max_iou_for_target)

        best_anchors = torch.stack([scale_idx_per_target, best_anchors], dim=1)
        best_anchor_for_target = torch.where(mask.unsqueeze(1), best_anchors, best_anchor_for_target)

    if len(targets_annotations) == 0:
        return target_cnn_cls, target_cnn_bbox

    # In this loop we match previously unassigned targets, i.e targets that did not have high iou with any of anchors
    for i in range(len(matched_targets)):
        if not matched_targets[i]:
            scale_idx, flat_anchor_idx = best_anchor_for_target[i].cpu().detach().numpy().astype(np.int32)
            grid_anchors = all_cnn_anchors[scale_idx]
            grid_h, grid_w, n_anchors = grid_anchors.shape[:3]

            anchor_bbox = grid_anchors.reshape(-1, 4)[flat_anchor_idx, :].unsqueeze(0)
            target_bbox = target_boxes[i].unsqueeze(0)
            cnn_bbox = get_cnn_bbox(target_bbox, anchor_bbox).squeeze(0)

            cell_i, cell_j, anchor_idx = np.unravel_index(flat_anchor_idx, [grid_h, grid_w, n_anchors])
            target_cnn_cls[scale_idx][cell_i, cell_j, anchor_idx, :] = target_cls[i]
            target_cnn_bbox[scale_idx][cell_i, cell_j, anchor_idx, :] = cnn_bbox

    return target_cnn_cls, target_cnn_bbox


def preprocess_img(img: np.ndarray[np.uint8]):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = (rgb_img.astype(np.float32) / 255 - 0.5) * 2
    processed = torch.Tensor(processed.transpose([2, 0, 1]))
    return processed


class AnchorBasedDataset(Dataset):
    def __init__(self, split: str, device: torch.device):

        if split == SPLIT_TRAIN:
            coco_anno_path = TRAIN_COCO_ANNO_PATH
            anno_path = TRAIN_ANNO_PATH
            self.img_path_prefix = TRAIN_IMAGES_PREFIX
        elif split == SPLIT_VAL:
            coco_anno_path = VAL_COCO_ANNO_PATH
            anno_path = VAL_ANNO_PATH
            self.img_path_prefix = VAL_IMAGES_PREFIX
        else:
            raise NotImplementedError

        self.split = split
        self.label2idx = get_label_encoder()
        self.annotations = build_or_load_annotations(coco_anno_path, anno_path, self.label2idx)
        self.device = device

        self.img_size = INPUT_IMAGE_SIDE
        self.anchors = generate_anchors_by_scale(EFFICIENT_DET_STRIDES,
                                                 self.img_size,
                                                 ANCHOR_SCALES,
                                                 ANCHOR_ASPECT_RATIOS)
        self.n_labels = N_LABELS
        self.aug_rescale_rate = AUG_RESCALE_RATE

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]
        img_path = os.path.join(self.img_path_prefix, img_anno['file_name'])
        bboxes = img_anno['bboxes']
        img = cv2.imread(img_path)
        if self.split == SPLIT_TRAIN:
            # TODO add noise, rotation, clip and other augmentations here
            img, bboxes = random_rescale_img_and_bboxes(img, bboxes, self.aug_rescale_rate)
        img, bboxes = fit_rescale_and_pad_img_and_bboxes(img, bboxes, self.img_size)

        cnn_target_cls, cnn_target_bbox = generate_efficient_det_target(self.anchors,
                                                                        bboxes,
                                                                        self.n_labels)

        inp = preprocess_img(img).to(self.device)

        for i in range(len(cnn_target_bbox)):
            cnn_target_cls[i] = cnn_target_cls[i].to(self.device)
            cnn_target_bbox[i] = cnn_target_bbox[i].to(self.device)

        return inp, cnn_target_cls, cnn_target_bbox

    def __len__(self):
        return len(self.annotations)


def anchor_based_dataset_collate_fn(batch):
    batch_size = len(batch)
    batch_inp = []
    n_scales = len(EFFICIENT_DET_STRIDES)
    batch_targets_cnn_cls = [[] for _ in range(n_scales)]
    batch_targets_cnn_bbox = [[] for _ in range(n_scales)]

    for inp, cnn_target_cls, cnn_target_bbox in batch:
        batch_inp.append(inp)
        for i in range(n_scales):
            batch_targets_cnn_cls[i].append(cnn_target_cls[i])
            batch_targets_cnn_bbox[i].append(cnn_target_bbox[i])

    batch_inp = torch.stack(batch_inp)
    for i in range(n_scales):
        batch_targets_cnn_cls[i] = torch.stack(batch_targets_cnn_cls[i])
        batch_targets_cnn_bbox[i] = torch.stack(batch_targets_cnn_bbox[i])

    return batch_inp, batch_targets_cnn_cls, batch_targets_cnn_bbox


if __name__ == '__main__':

    for (batch_inp, batch_targets_cnn_cls, batch_targets_cnn_bbox) in dataloader:
        print(batch_inp.shape)
        for i in range(len(EFFICIENT_DET_STRIDES)):
            print(batch_targets_cnn_cls[i].shape)
            print(batch_targets_cnn_bbox[i].shape)

        break
