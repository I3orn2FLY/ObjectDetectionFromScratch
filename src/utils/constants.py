# Anchor scales are multiple coefficient to grid cell width at each scale,
# For instance, on 1/32 scale the width of cell or stride is 32 and anchor size will be 32 * anchor_scale
# Anchor aspect ratios are for each anchor size to generate anchor box with shape (aspect_ratio * anchor_size, anchor_size)

# Here, I choose aspect ratios very simple, 0.5 -> wide box, 1->square, 2-> tall box
# For scales, I choose 17 * 32 / 640 = 0.84 (large object that can take whole image will fit this box) and other scales just arbitrary

## Finally for grid cell in certain scale there will be 9 anchor boxes (3 scales * 3 ratios)
INPUT_IMAGE_SIDE = 480
EFFICIENT_DET_STRIDES = [8, 16, 32]
ANCHOR_SCALES = [17, 9, 2]
ANCHOR_ASPECT_RATIOS = [0.5, 1, 2]
N_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_ASPECT_RATIOS)

N_LABELS = 81

### Augmentations
AUG_RESCALE_RATE = 0.1



### Training
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
N_EPOCH = 10