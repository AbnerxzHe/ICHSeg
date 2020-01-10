import SimpleITK as sitk
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
import os

os.environ['OMP_NUM_THREADS'] = '1'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def dice(PD, GT):
    result = np.atleast_1d(PD.astype(np.bool))
    reference = np.atleast_1d(GT.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dice = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dice = 0.0
    return dice

def pre(PD, GT):
    result = np.atleast_1d(PD.astype(np.bool))
    reference = np.atleast_1d(GT.astype(np.bool))
    tfp = np.count_nonzero(result)
    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        pre = tp / float(tfp)
    except ZeroDivisionError:
        pre = 0.0
    return pre


def sen(PD, GT):
    result = np.atleast_1d(PD.astype(np.bool))
    reference = np.atleast_1d(GT.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        sen = tp / float(tp + fn)
    except ZeroDivisionError:
        sen = 0.0
    return sen


def spe(PD, GT):
    result = np.atleast_1d(PD.astype(np.bool))
    reference = np.atleast_1d(GT.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        spe = tn / float(tn + fp)
    except ZeroDivisionError:
        spe = 0.0
    return spe


def hd(PD, GT, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    """
    hd1 = __surface_distances(PD, GT, voxelspacing, connectivity).max()
    hd2 = __surface_distances(GT, PD, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def normal(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


# shape   [C,H,W,D]
def nonmeanstd(x):
    sr = x
    sr = sr[sr > 0]
    mean = sr.mean()
    std = sr.std()
    sr = (sr - mean) / std
    x[x > 0] = sr
    return x

def normalize(x):
   # x=nonmeanstd(x)
#     x = np.maximum(x,-5)
#     x = np.minimum(x,5)
    M = np.max(x)
    N = np.min(x)
    X = (x-N)/(M-N)
 #   X=X[a:a+128,b:b+128,c:c+128]
    return X


def aug_image(x, y):
    if (random.random() > 0.5):
        x, y = flip_sample(x, y)

    if (random.random() > 0.5):
        x, y = augment_rot90(x, y)

    if (random.random() > 0.5):
        x, y = scale_coords(x, y)

    #     if(random.random()>0.5):
    #         x,y=random_channel_shift(x, y, 0.1)
    #         print('shift')
    #         print(np.sum(x==0))
    if (random.random() > 0.5):
        x = x
        y = y

    return x, y


def transpose_channels(batch):
    if len(batch.shape) == 4:
        return np.transpose(batch, axes=[1, 2, 3, 0])
    elif len(batch.shape) == 5:
        return np.transpose(batch, axes=[0, 2, 3, 4, 1])
    else:
        raise ValueError("wrong dimensions in transpose_channel generator!")


def random_crop(crop_size, x, y):
    """
    Args:
        x: 4d array, [channel, h, w, d]
    """
    crop_size = crop_size
    height, width, depth = x.shape[-3:]
    sx = random.randint(0, height - crop_size[0] - 1)
    sy = random.randint(0, width - crop_size[1] - 1)
    sz = random.randint(0, depth - crop_size[2] - 1)
    crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
    crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

    return crop_volume, crop_seg


def center_crop(crop_size, x, y):
    crop_size = crop_size
    height, width, depth = x.shape[-3:]
    sx = (height - crop_size[0] - 1) // 2
    sy = (width - crop_size[1] - 1) // 2
    sz = (depth - crop_size[2] - 1) // 2
    crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
    crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

    return crop_volume, crop_seg


def maskcrop(crop_size, x, y):
    c = [np.nonzero(y)]
    c = [[np.min(i, 1), np.max(i, 1)] for i in c]
    f = np.array([np.min([i[0] for i in c], 0), np.max([i[1] for i in c], 0)]).T
    l10 = f[1, 0]
    l11 = f[1, 1]
    l20 = f[2, 0]
    l21 = f[2, 1]
    l30 = f[3, 0]
    l31 = f[3, 1]
    s3 = l31 - l30
    s2 = l21 - l20
    s1 = l11 - l10
    op = y.shape
    sz = crop_size / 2
    if (s3 <= 128):
        r3 = random.randint(0, 128 - s3)
    else:
        r3 = 0
    if (s2 <= 128):
        r2 = random.randint(0, 128 - s2)
    else:
        r2 = 0
    if (s1 <= 128):
        r1 = random.randint(0, 128 - s1)
    else:
        r1 = 0
    m1 = r1 + s1
    m2 = r2 + s2
    m3 = r3 + s3
    X = np.zeros((x.shape))
    Y = np.zeros((y.shape))
    Y[:, r1:m1, r2:m2, r3:m3] = y[:, l10:l11, l20:l21, l30:l31]
    X[:, r1:m1, r2:m2, r3:m3] = x[:, l10:l11, l20:l21, l30:l31]
    return X, Y


def flip_sample(volumes, mask):
    """
        Args:
            volumes: list of array, [h, w, d]
            mask: array [h, w, d], segmentation volume
        Ret: x, y: [channel, h, w, d]

    """
    x = np.stack(volumes, axis=0)  # [N, H, W, D]
    y = mask

    if random.random() < 0.5:
        x = np.flip(x, axis=1)
        y = np.flip(y, axis=1)
    if random.random() < 0.5:
        x = np.flip(x, axis=2)
        y = np.flip(y, axis=2)
    if random.random() < 0.5:
        x = np.flip(x, axis=3)
        y = np.flip(y, axis=3)
    # else:
    #     x, y = self.center_crop(x, y)

    return x, y


# similar to flip
def augment_mirroring(x, y, axes=(0, 1, 2)):
    if (len(x.shape) != 3) and (len(x.shape) != 4):
        raise Exception(
            "Invalid dimension for x and y. x and y should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        x[:, :] = x[:, ::-1]
        if y is not None:
            y[:, :] = y[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        x[:, :, :] = x[:, :, ::-1]
        if y is not None:
            y[:, :, :] = y[:, :, ::-1]
    if 2 in axes and len(x.shape) == 4:
        if np.random.uniform() < 0.5:
            x[:, :, :, :] = x[:, :, :, ::-1]
            if y is not None:
                y[:, :, :, :] = y[:, :, :, ::-1]
    return x, y


# shape   [C,H,W,D]
def augment_rot90(x, y, num_rot=(0, 1, 2), axes=(1, 2, 3)):
    """

    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    num_rot = np.random.choice(num_rot)
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    axes = [i for i in axes]
    x = np.rot90(x, num_rot, axes)
    if y is not None:
        y = np.rot90(y, num_rot, axes)
    return x, y


def scale_coords(x, y):
    scale = random.uniform(0.9, 1.1)
    return x * scale, y


def random_channel_shift(x, y, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x, y


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds
