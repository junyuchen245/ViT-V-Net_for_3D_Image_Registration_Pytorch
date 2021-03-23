import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import pystrum.pynd.ndutils as nd

def sliding_predict(model, image, tile_size, n_dims, overlap=1/2, flip=False):
    image_size = image.shape
    stride_x = math.ceil(tile_size[0] * (1 - overlap))
    stride_y = math.ceil(tile_size[1] * (1 - overlap))
    stride_z = math.ceil(tile_size[2] * (1 - overlap))
    num_rows = int(math.ceil((image_size[2] - tile_size[0]) / stride_x) + 1)
    num_cols = int(math.ceil((image_size[3] - tile_size[1]) / stride_y) + 1)
    num_slcs = int(math.ceil((image_size[4] - tile_size[2]) / stride_z) + 1)
    total_predictions = torch.zeros((1, n_dims, image_size[2], image_size[3], image_size[4])).cuda()
    count_predictions = torch.zeros((image_size[2], image_size[3], image_size[4])).cuda()
    tile_counter = 0
    print(num_rows)
    for row in range(num_rows):
        for col in range(num_cols):
            for slc in range(num_slcs):
                x_min, y_min, z_min = int(row * stride_x), int(col * stride_y), int(slc * stride_z)
                x_max = x_min + tile_size[0]
                y_max = y_min + tile_size[1]
                z_max = z_min + tile_size[2]
                if x_max > image_size[2]:
                    x_min = image_size[2] - stride_x
                    x_max = image_size[2]
                if y_max > image_size[3]:
                    y_min = image_size[3] - stride_y
                    y_max = image_size[3]
                if z_max > image_size[4]:
                    z_min = image_size[4] - stride_z
                    y_max = image_size[4]
                img = image[:, :, x_min:x_max, y_min:y_max, z_min:z_max]
                padded_img = pad_image(img, tile_size)
                #print(padded_img.shape)

                tile_counter += 1
                padded_prediction = model(padded_img)[1]
                if flip:
                    for dim in [-1, -2, -3]:
                        fliped_img = padded_img.flip(dim)
                        fliped_predictions = model(fliped_img)[1]
                        padded_prediction = (fliped_predictions.flip(dim) + padded_prediction)
                    padded_prediction = padded_prediction/4
                predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3], :img.shape[4]]
                count_predictions[x_min:x_max, y_min:y_max, z_min:z_max] += 1
                total_predictions[:, :, x_min:x_max, y_min:y_max, z_min:z_max] += predictions.cuda()#.data.cpu().numpy()
    total_predictions /= count_predictions
    return total_predictions

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

def dice_val(y_pred, y_true, num_clus):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 3D displacement field of size [nb_dims, *vol_shape]
    Returns:
        jacobian determinant (matrix)
    """

    # check inputs
    volshape = disp.shape[1:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, 0)

    # compute gradients
    [xFX, xFY, xFZ] = np.gradient(grid[0] - disp[0])
    [yFX, yFY, yFZ] = np.gradient(grid[1] - disp[1])
    [zFX, zFY, zFZ] = np.gradient(grid[2] - disp[2])

    jac_det = np.zeros(grid[0].shape)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                jac_mij = [[xFX[i, j, k], xFY[i, j, k], xFZ[i, j, k]], [yFX[i, j, k], yFY[i, j, k], yFZ[i, j, k]], [zFX[i, j, k], zFY[i, j, k], zFZ[i, j, k]]]
                jac_det[i, j, k] =  np.linalg.det(jac_mij)
    return jac_det


import re
def process_label():
    #process labeling information for FreeSurfer
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]


    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def write2csv(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

