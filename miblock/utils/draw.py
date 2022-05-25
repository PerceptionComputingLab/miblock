import os
import random

import numpy as np
import cv2
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import re
from glob import glob
import pandas as pd
import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon

# import seaborn as sns
# plt.style.use("ieee")
plt.rcParams.update({"font.size": 12})
plt.rcParams['font.family'] = 'Times New Roman'
import nibabel as nib
import cv2 as cv
import copy
import torch
from skimage.transform import resize
import torch.nn.functional as F


def resize_label(array, shape, order=1, n_class=4):
    '''
    Resize 一个多维度的array
    '''
    onehot_numpy = onehot_encoding(array=array, class_num=n_class)
    resized = resize(onehot_numpy.astype("float32"), shape, order=order)
    resized_array = np.argmax(resized, axis=-1)

    return resized_array


# one-hot encoding method(efficient)
def onehot_encoding(array, class_num):
    '''
    the function turn a regular array to a one-hot representation form
    :param array: input array
    :param class_num: number of classes
    :return: one-hot encoding of array
    '''
    label_one_hot = np.zeros(array.shape + (class_num,), dtype="int16")
    for k in range(class_num):
        label_one_hot[..., k] = (array == k)
    return label_one_hot


# ######################## 画各种不同图中利用到的一些函数区域###########################
def mat2gray(I, limits):
    '''
    Apply intensity transformation + window operation for CT value
    :param I: input data
    :param limits:  Interested Value Range
    :return: transformed data
    '''
    i = I.astype(np.float32)
    graymax = float(limits[1])
    graymin = float(limits[0])
    delta = 1 / (graymax - graymin)
    gray = delta * i - graymin * delta
    graycut = np.maximum(0, np.minimum(gray, 1))
    return graycut * 255.0


def onehot3d(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W*D
    :param class_n:
    :return:N*n_class*H*W*D
    '''
    shape = input.shape
    # onehot = torch.zeros((class_n,)+shape).cuda()
    onehot = torch.zeros((class_n,) + shape)
    for i in range(class_n):
        onehot[i, ...] = (input == i)
    onehot_trans = onehot.permute(1, 0, 2, 3, 4)
    return onehot_trans


def entropy(input, dim=0, eps=1e-3):
    '''
    Calculate the pixel-wise entropy of the input
    :param input: type torch tensor of shape N*C*d1*d2*d3....
    :param eps:
    :return: pixel-wise entropy
    '''
    # x = torch.add(input, eps)
    # flag = torch.all(torch.gt(x, 0))
    # if flag != True:
    #     a = 1
    # assert flag == True
    # logx = torch.log(x)
    # entropy = torch.sum(-x*logx, dim=dim)
    x = input + eps
    logx = np.log(x)
    entropy = np.sum(-x * logx, axis=dim)

    return entropy


def normalize(volume):
    '''
    this function translate mri intensity to gray values
    :param volume:
    :return:
    '''
    v_max = np.max(volume)
    v_min = np.min(volume)
    volume_norm = (volume - v_min) / (v_max - v_min)
    volume_norm = (volume_norm * 255).astype("int")
    return volume_norm


def test_input(dataout, WINDOW_CENTER, WINDOW_WIDTH):
    img_array = dataout["data"]
    label_array = dataout["seg"]
    label_array = label_array.astype("uint8")

    label_origin_display = np.zeros(label_array.shape[2:], dtype="uint8")
    label_origin_display = np.expand_dims(label_origin_display, axis=-1)
    label_origin_display = np.tile(label_origin_display, [1, 1, 1, 3])

    color = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [120, 0, 0], [0, 120, 0], [0, 0, 120],
             [120, 120, 0]]
    x_range = range(8)
    for j, item in enumerate(x_range):
        mask = label_array[0, 0, ...] == item
        label_origin_display[mask] = color[j]

    # # rename label
    # rename_map = (0, 205, 420, 500, 550, 600, 820, 850)
    # label_data = np.zeros(label_array.shape, dtype='int32')
    # for i in range(len(rename_map)):
    #     label_data[label_array == i] = rename_map[i]

    # add windowing operation for CT
    img_window = mat2gray(img_array, (WINDOW_CENTER - WINDOW_WIDTH / 2, WINDOW_CENTER + WINDOW_WIDTH / 2))
    # img_window = img_array/255.0
    test_display_path = "display"
    if not os.path.exists(test_display_path):
        os.mkdir(test_display_path)
    img_window = img_window[0, 0, ...].astype("uint8")
    label = label_array[0, 0, ...]
    W, H, D = img_window.shape
    for i in range(D):
        file_path_img = os.path.join(test_display_path, f"{i}_img.png")
        file_path_lab = os.path.join(test_display_path, f"{i}_label.png")
        cv2.imwrite(filename=file_path_img, img=img_window[:, :, i])
        cv2.imwrite(filename=file_path_lab, img=label_origin_display[:, :, i, :])


def LineChart(x_list, y_list):
    '''
    该函数实现多条折线的绘制，用于对比多种方法的某些指标，
    或者是两个指标之间的关系（如ROC曲线绘制、不确定性图reliability diagram等）
    x_list: x轴上的N个点对应x坐标list
    y_lixt: y轴上对应的N个y坐标list 
    '''
    # plot reliability diagram
    plt.figure(figsize=(6, 6))

    # 放置位置，要在plt.plot之前。且，plt.grid、plt.xlabel、plt.ylabel要放在最后，否则，有干扰，
    # 刻度线设置不成向内方向，依旧是向外方向
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    title_list = ['Baseline', 'Baseline+MC Dropout', 'Baseline+MCBN', 'Deep Ensemble', 'SLEX-Net(Ours)']
    color_list = ['red', 'blue', 'green', 'yellow', 'purple']
    for i in range(len(x_list)):
        ax1.plot(x_list[i], y_list[i], color=color_list[i], marker="s", linestyle="-", label=title_list[i])

    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel("Confidence")
    ax1.set_xlim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title('Uncertainty')

    plt.savefig('reliability_diagram_new.png', dpi=400, bbox_inches='tight')
    plt.show()


class Evaluation(object):
    def __init__(self):
        pass

    def dice_n_class(self, move_img, refer_img, n_class):
        # list of classes
        # c_list = np.unique(refer_img)
        c_list = np.arange(n_class)

        dice_c = []
        for c in range(len(c_list)):
            # intersection
            ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
            # sum
            sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
            dice_c.append((2.0 * ints) / sums)

        return dice_c

    def cal_metrics(self, gt_path, predict_path, n_class, rename_label):
        gt_volume = nib.load(gt_path).get_data().astype("int16")
        pre_volume = nib.load(predict_path).get_data().astype("int16")
        for i in range(len(rename_label)):
            gt_volume[gt_volume == rename_label[i]] = i
            pre_volume[pre_volume == rename_label[i]] = i
        dice = self.dice_n_class(pre_volume, gt_volume, n_class)
        return dice

    # 将一个体数据保存为逐个切片的形式进行可视化
    def save_slice_img(self, volume_path, output_path):
        file_name = os.path.basename(volume_path)
        output_dir = os.path.join(output_path, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            pass
        input_volume = nib.load(volume_path).get_data()
        # mapping to 0-1
        vol_max = np.max(input_volume)
        vol_min = np.min(input_volume)
        input_unit = (input_volume - vol_min) / (vol_max - vol_min)
        width, height, depth = input_unit.shape
        for i in range(0, depth):
            slice_path = os.path.join(output_dir, str(i) + '.png')
            img_i = input_unit[:, :, i]
            # normalize to 0-255
            img_i = (img_i * 255).astype('uint8')
            cv.imwrite(slice_path, img_i)
        return input_unit

    # 将一个体数据以及预测结果和GT保存为逐个切片叠加的形式进行可视化
    def save_slice_img_label(self, volume_path, pre_path, output_path,
                             window=None, show_mask=False, color=None, classes=None, alpha=0.5):

        img_volume = nib.load(volume_path).get_data()
        pre_volume = nib.load(pre_path).get_data()

        img_volume = img_volume.transpose(1, 0, 2)
        pre_volume = pre_volume.transpose(1, 0, 2)
        assert img_volume.shape == pre_volume.shape
        _, _, depth = img_volume.shape

        img_volume = img_volume.astype("float32")
        if window != None:
            img_volume = mat2gray(img_volume, window)

        # gray value mapping   from MRI value to pixel value(0-255)
        volume_max = np.max(img_volume)
        volume_min = np.min(img_volume)
        volum_mapped = (img_volume - volume_min) / (volume_max - volume_min)
        volum_mapped = (255 * volum_mapped).astype('uint8')
        # construct a directory for each volume to save slices
        # dir_volume = os.path.join(output_path, file_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            pass
        for i in range(depth):
            img_slice = volum_mapped[:, :, i]
            pre_slice = pre_volume[:, :, i]
            self.save_contour_label(img=img_slice, pre=pre_slice, classes=classes,
                                    save_path=output_path, colors=color, file_name=i, show_mask=show_mask, alpha=alpha)

    def apply_mask(self, image, mask, color_list, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(image.shape[-1]):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color_list[c],
                                      image[:, :, c])
        return image

    def apply_masks(self, image, mask, color_list, classes, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(image.shape[-1]):
            for i in range(1, len(color_list)):
                image[:, :, c] = np.where(mask == classes[i],
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color_list[i][c],
                                          image[:, :, c])
        return image

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def save_contour_label(self, img, pre, classes=None, save_path='', colors=None, file_name=None, show_mask=True,
                           alpha=1.0):
        # single channel to multi-channel
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
        height, width = img.shape[:2]
        _, ax = plt.subplots(1, figsize=(height, width))

        # ax.set_ylim(height + 10, -10)
        # ax.set_xlim(-10, width + 10)
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')
        # ax.set_title("volume mask")
        masked_image = img.astype(np.uint32).copy()

        if show_mask:
            masked_image = self.apply_masks(masked_image, pre, colors, classes, alpha)

        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height / 300, width / 300)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        ax.imshow(masked_image.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, file_name), dpi=300)
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def label_discrete2distribution(input_label, scale, stride, padding, n_class):
    '''
    This function tranfers the pixel-wise discrete label to that of label distributions
    :param input_label: pixel-wise discrete label(W,H,D...)
    :param scale: down-sampling scale
    :param n_class: number of the classes
    :return: The generated label distribution
    '''
    # We only consider 2-D and 3-D input data for image label and volume label
    shape = input_label.shape
    assert len(shape) == 2 or len(shape) == 3
    # First we pad the input_label so that the shape can be divided by the scale
    scale_tuple = np.ones(len(shape), dtype="uint8") * scale
    sub = np.mod(shape, scale_tuple)
    '''
    if np.any(sub):
        if len(shape)==2:
            gap = scale_tuple - sub
            rem_h = gap[0]%2
            rem_w = gap[1]%2
            pad_h = (gap[0]//2, gap[0]//2+ rem_h)
            pad_w = (gap[1]//2, gap[1]//2+ rem_w)
            n_pad = (pad_h, pad_w)
        elif len(shape)==3:
            gap = scale_tuple - sub
            rem_h = gap[0] % 2
            rem_w = gap[1] % 2
            rem_d = gap[2] % 2
            pad_h = (gap[0] // 2, gap[0] // 2 + rem_h)
            pad_w = (gap[1] // 2, gap[1] // 2 + rem_w)
            pad_d = (gap[2]//2, gap[2]//2 + rem_d)
            n_pad = (pad_h, pad_w, pad_d)
        else:
            raise Exception("Data Dimension Error!")
        padded_label = np.pad(input_label, pad_width=n_pad, mode="edge")
    else:
        padded_label = input_label    
   '''
    # Try to pad at the right side
    if np.any(sub):
        if len(shape) == 2:
            gap = scale_tuple - sub
            pad_length = np.mod(gap, scale_tuple)
            pad_h = (0, pad_length[0])
            pad_w = (0, pad_length[1])
            n_pad = (pad_h, pad_w)
        elif len(shape) == 3:
            gap = scale_tuple - sub
            pad_length = np.mod(gap, scale_tuple)
            pad_h = (0, pad_length[0])
            pad_w = (0, pad_length[1])
            pad_d = (0, pad_length[2])
            n_pad = (pad_h, pad_w, pad_d)
        else:
            raise Exception("Data Dimension Error!")
        padded_label = np.pad(input_label, pad_width=n_pad, mode="edge")
    else:
        padded_label = input_label

    # transfer numpy to tensor
    label_tensor = torch.from_numpy(padded_label)
    label_tensor = torch.unsqueeze(label_tensor, dim=0)
    if len(shape) == 2:
        pass
    elif len(shape) == 3:
        label_onehot = onehot3d(label_tensor, n_class)
        # We exchange the batch and channel axis to conv the volume channel-wise
        onehot_trans = label_onehot.permute(1, 0, 2, 3, 4)
        # kernel = torch.ones(1, 1, scale,scale,scale).cuda()
        kernel = torch.ones(1, 1, scale, scale, scale)
        result = F.conv3d(onehot_trans, kernel / (scale * scale * scale), stride=stride, padding=padding)
    else:
        raise Exception("Data Dimension Error!")

    data = torch.squeeze(result, dim=1)
    out = data.data.cpu().numpy()

    return out


def draw_entropy_gt_pre_image(image, label, predict, entropy_label_array, color, classes, output_path, zoom_factor=10):
    '''
    This function can be embedded in the main code to test whether the input is correct.
    该函数输入图像、标签、预测结果以及标签分布，输出路径等，实现逐个切片的图像输出、标签输出以及预测结果输出和边界先验输出。
    注意， 此处的标签和预测结果，应该为0-N_class之间的数值，不是特别的标签值！
    zoom_factor: 图像放大的倍数
    '''
    eval = Evaluation()
    # shape should be H*W*D
    image_numpy = image
    # shape should be H*W*D
    label_numpy = label
    # shape should be C*H*W*D
    # Trans to 0-255
    image_out_trans = normalize(image_numpy)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    origin_img_path = os.path.join(output_path, "Origin")
    label_path = os.path.join(output_path, "label")
    predict_path = os.path.join(output_path, "predict")
    entropy_path = os.path.join(output_path, "enropy")

    items = [origin_img_path, label_path, predict_path, entropy_path]
    for item in items:
        if not os.path.exists(item):
            os.mkdir(item)

    image_resized = resize(image_out_trans, entropy_label_array.shape, order=1, preserve_range=True)
    # for i in range(len(classes)):
    #     dist_argmax_label[dist_argmax_label==i] = classes[i]

    w, h, D = entropy_label_array.shape

    # resize the label
    # label_origin_numpy = resize(label_numpy, entropy_label_array.shape, order=0, cval=0, preserve_range=True)
    label_origin_numpy = resize_label(label_numpy, entropy_label_array.shape)
    label_origin_display = np.zeros(label_origin_numpy.shape, dtype="uint8")
    label_origin_display = np.expand_dims(label_origin_display, axis=-1)
    label_origin_display = np.tile(label_origin_display, [1, 1, 1, 3])

    # predict_origin_numpy = resize(predict, entropy_label_array.shape, order=0, preserve_range=True)
    predict_origin_numpy = resize_label(predict, entropy_label_array.shape)
    predict_origin_display = np.zeros(predict_origin_numpy.shape, dtype="uint8")
    predict_origin_display = np.expand_dims(predict_origin_display, axis=-1)
    predict_origin_display = np.tile(predict_origin_display, [1, 1, 1, 3])
    predicted_display = predict_origin_display

    # 标签转换
    # for i in range(len(classes)):
    #     label_origin_numpy[label_origin_numpy==i] = classes[i]
    #     predict_origin_numpy[predict_origin_numpy==i] = classes[i]
    # 为每一个标签赋上特定的颜色
    for j, item in enumerate(classes):
        label_origin_display[label_origin_numpy == j] = color[j]
        predicted_display[predict_origin_numpy == j] = color[j]

    # for k, item1 in enumerate(classes): 
    #     predicted_display[predict_origin_numpy ==item1] = color[k]

    for i in range(D):

        entropy_map = entropy_label_array[..., i]
        name = os.path.join(entropy_path, f"entropy_{i}.png")

        height, width = entropy_map.shape
        _, ax = plt.subplots(1, figsize=(height, width))
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')

        fig = plt.gcf()
        ax.imshow(
            entropy_map,
            cmap=plt.cm.jet,
            interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

        # 去除白边
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches((width / 300.0) * zoom_factor, (height / 300.0) * zoom_factor)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        # plt.axes('off')
        plt.savefig(name, dpi=300)
        plt.clf()
        plt.cla()
        plt.close()

        # label_slice = cv.cvtColor(label_origin_display[:,:,i,:], cv.COLOR_RGB2BGR)
        # predict_slice = cv.cvtColor(predicted_display[:,:,i,:], cv.COLOR_RGB2BGR)

        label_slice = label_origin_display[:, :, i, :]
        predict_slice = predicted_display[:, :, i, :]

        save_img_list = [label_slice, predict_slice]
        save_img_path_list = [os.path.join(label_path, f"label_{i}.png"),
                              os.path.join(predict_path, f"predict_{i}.png")]
        for u in range(len(save_img_list)):
            save_img = save_img_list[u]
            save_img_path = save_img_path_list[u]

            fig1, ax_i = plt.subplots(1, figsize=(height, width))
            ax_i.set_ylim(height + 0, 0)
            ax_i.set_xlim(0, width + 0)
            ax_i.axis('off')

            ax_i.imshow(
                save_img,
                interpolation='nearest')
            plt.xticks([])
            plt.yticks([])
            # 去除白边
            # 如果dpi=300，那么图像大小=height*width
            fig1.set_size_inches((width / 300.0) * zoom_factor, (height / 300.0) * zoom_factor)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.axes('off')
            plt.savefig(save_img_path, dpi=300)
            plt.clf()
            plt.cla()
            plt.close()

        image_out = cv.resize(image_resized[..., i].astype("uint8"), (640, 640))
        cv.imwrite(os.path.join(origin_img_path, f"origin_{i}.png"), image_out)
        # cv.imwrite(os.path.join(label_path, f"label_{i}.png"), label_slice)
        # cv.imwrite(os.path.join(predict_path, f"predict_{i}.png"), predict_slice)

    return 1


######################### 实际画各种不同图的主函数区域 ###########################


def display_LineChart():
    '''
    该函数实现该函数实现多条折线的绘制
    '''
    x_list = []
    y_list = []
    for _ in range(5):
        x = np.random.random(100)
        y = np.random.random(100)
        x_list.append(x)
        y_list.append(y)
    LineChart(x_list, y_list)


def display_brats_entropy_gt_pre_image():
    origin_path = "/home/lixiangyu/Dataset/Brats2018/Training/HGG"
    # path = "display/brats"
    output_path_top = "display/test_prior"
    if not os.path.exists(output_path_top):
        os.mkdir(output_path_top)
    color_brats = [[0, 0, 0], [128, 174, 128], [190, 166, 105], [111, 184, 210]]
    classes_brats = [0, 1, 2, 4]
    # origin_path = os.path.join(path, "origin")
    subjects = os.listdir(origin_path)
    for subject in subjects:
        # 该样本的文件夹
        subject_name = subject
        output_path = os.path.join(output_path_top, subject_name)
        # 保存GT
        subject_path = os.path.join(origin_path, subject)
        volume_path = os.path.join(subject_path, f"{subject}_flair.nii.gz")
        gt_path = os.path.join(subject_path, f"{subject}_seg.nii.gz")

        img_volume = nib.load(volume_path).get_data()
        pre_volume = nib.load(gt_path).get_data()

        img_volume = img_volume.transpose(1, 0, 2)
        pre_volume = pre_volume.transpose(1, 0, 2)
        assert img_volume.shape == pre_volume.shape
        label_dist = label_discrete2distribution(pre_volume, 2, 2, 0, 4)
        entropy_label_array = entropy(label_dist)
        draw_entropy_gt_pre_image(img_volume, pre_volume, pre_volume, entropy_label_array, color_brats, classes_brats,
                                  output_path)


def display_mmwhs_entropy_gt_pre_image():
    path = "display/mmwhs"
    output_path_top = "display/test_prior_mmwhs"
    if not os.path.exists(output_path_top):
        os.mkdir(output_path_top)
    # color_brats = [[255,255,255],[128,174,128],[190,166,105],[111,184,210]]
    # classes_brats = [0,1,2,4]
    # color = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[120,0,0],[0,120,0],[0,0,120], [120,120,0]]
    color = [[255, 255, 255], [255, 250, 205], [188, 143, 143], [199, 21, 133], [103, 255, 65], [135, 206, 235],
             [238, 130, 238], [253, 245, 230]]
    classes = [0, 205, 420, 500, 550, 600, 820, 850]
    # origin_path = os.path.join(path, "origin")
    origin_path = "/home/lixiangyu/Dataset/MMWHS/Train"
    images_path = os.path.join(origin_path, "image")
    labels_path = os.path.join(origin_path, "label")
    subjects = os.listdir(images_path)
    for subject in subjects:
        # 该样本的文件夹
        subject_name = subject[0:14]
        output_path = os.path.join(output_path_top, subject_name)

        # 保存GT
        volume_path = os.path.join(images_path, subject[0:14] + "image.nii.gz")
        gt_path = os.path.join(labels_path, subject[0:14] + "label.nii.gz")

        img_volume = nib.load(volume_path).get_data()
        pre_volume = nib.load(gt_path).get_data()
        pre_volume = pre_volume.astype("int32")
        # 标签转换到0- N-1
        for u in range(len(classes)):
            pre_volume[pre_volume == classes[u]] = u

        img_volume = img_volume.transpose(1, 0, 2)
        pre_volume = pre_volume.transpose(1, 0, 2)
        assert img_volume.shape == pre_volume.shape
        label_dist = label_discrete2distribution(pre_volume, 2, 2, 0, 4)
        entropy_label_array = entropy(label_dist)

        draw_entropy_gt_pre_image(img_volume, pre_volume, pre_volume, entropy_label_array, color, classes, output_path,
                                  zoom_factor=2)


def display_brats(input_path='', output_path=''):
    '''
    该函数只需要确定输入数据的文件夹，以及输出数据的文件夹，即可实现可视化的结果保存。

    '''
    eval = Evaluation()
    path = "display/brats"
    dataset_path = "/home/lixiangyu/Dataset/Brats2018/Training"
    prediction_path = os.path.join(path, "predictions")
    output_path_top = "display/brats_slices"

    color_brats = [[255, 255, 255], [128, 174, 128], [190, 166, 105], [111, 184, 210]]
    classes_brats = [0, 1, 2, 4]

    # 确定预测里面的样本
    models_result = os.listdir(prediction_path)
    subjects = os.listdir(os.path.join(prediction_path, models_result[0]))
    # subjects = os.listdir(dataset_path)
    # 确定输出结果的保存list
    subject_for_excel = []
    c2_dice_list = []

    for subject in subjects:
        # 该样本的文件夹
        subject_name = subject[0:-7]
        output_path = os.path.join(output_path_top, subject_name)
        # 保存GT与原图
        output_path_gt = os.path.join(output_path, "gt")
        output_path_image = os.path.join(output_path, "image")
        # if not os.path.exists(file_name_gt):
        #     os.mkdir(file_name_gt)

        subject_path = os.path.join(dataset_path, "HGG", subject_name)
        if not os.path.exists(subject_path):
            subject_path = os.path.join(dataset_path, "LGG", subject_name)

        volume_path = os.path.join(subject_path, f"{subject_name}_flair.nii.gz")
        gt_path = os.path.join(subject_path, f"{subject_name}_seg.nii.gz")

        # 保存原始图像
        """ eval.save_slice_img_label(volume_path, gt_path, output_path_image, show_mask=False, 
        color=color_brats, classes=classes_brats, alpha=1)

        eval.save_slice_img_label(volume_path, gt_path, output_path_gt, show_mask=True, 
        color=color_brats, classes=classes_brats, alpha=1) """

        subject_for_excel.append(subject_name)
        # 保存不同模型方法的结果
        models = os.listdir(prediction_path)
        model_for_excel = []
        c1_dice_list = []
        for model in models:
            model_path = os.path.join(prediction_path, model)
            file_path = os.path.join(model_path, f"{subject_name}.nii.gz")
            output_path_pre = os.path.join(output_path, f"{model}")

            model_for_excel.append(model)
            # 计算性能指标，不同模型，在不同数据上的指标
            dice = eval.cal_metrics(gt_path=gt_path, predict_path=file_path, n_class=4, rename_label=[0, 1, 2, 4])
            fg_mean_dice = np.mean(dice[1:])
            c1_dice_list.append(fg_mean_dice)

            # if not os.path.exists(file_name_pre):
            #     os.mkdir(file_name_pre)
            """ eval.save_slice_img_label(volume_path, file_path, output_path_pre, show_mask=True, 
                    color=color_brats, classes=classes_brats, alpha=1) """
        c2_dice_list.append(np.array(c1_dice_list))

    dice_value = np.array(c2_dice_list)
    dice_dataframe = pd.DataFrame(dice_value, index=subject_for_excel, columns=model_for_excel)
    writer = pd.ExcelWriter("dice_value.xlsx")
    dice_dataframe.to_excel(writer)
    writer.save()


def display_mmwhs(input_path='', output_path=''):
    '''
    该函数只需要确定输入数据的文件夹，以及输出数据的文件夹，即可实现可视化的结果保存。

    '''
    eval = Evaluation()
    path = "display/mmwhs"
    output_path_top = "display/mmwhs_slices"

    # color_mmwhs = [[255,255,255],[255,0,0],[0,255,0],[0,0,255],[120,0,0],[0,120,0],[0,0,120], [120,120,0]]
    color_mmwhs = [[255, 255, 255], [255, 250, 205], [188, 143, 143], [199, 21, 133], [103, 255, 65], [135, 206, 235],
                   [238, 130, 238], [253, 245, 230]]
    classed_mmwhs = [0, 205, 420, 500, 550, 600, 820, 850]
    origin_path = os.path.join(path, "origin")
    images_path = os.path.join(origin_path, "image")
    subjects = os.listdir(images_path)
    for subject in subjects:
        # 该样本的文件夹
        subject_name = subject[0:14]
        output_path = os.path.join(output_path_top, subject_name)
        # 保存GT
        output_path_gt = os.path.join(output_path, "gt")

        volume_path = os.path.join(images_path, subject)
        gt_path = os.path.join(origin_path, "label", subject[0:14] + "label.nii.gz")
        eval.save_slice_img_label(volume_path, gt_path, output_path_gt, show_mask=True, window=[-400, 800],
                                  color=color_mmwhs, classes=classed_mmwhs, alpha=1)
        # 保存不同模型方法的结果
        prediction_path = os.path.join(path, "predictions")
        models = os.listdir(prediction_path)
        for model in models:
            model_path = os.path.join(prediction_path, model)
            file_path = os.path.join(model_path, subject[0:14] + "label.nii.gz")
            output_path_pre = os.path.join(output_path, f"pre_{model}")
            # if not os.path.exists(file_name_pre):
            #     os.mkdir(file_name_pre)
            eval.save_slice_img_label(volume_path, file_path, output_path_pre, show_mask=True, window=[-400, 800],
                                      color=color_mmwhs, classes=classed_mmwhs, alpha=0.65)


########################## 画最后定性比较结果代码 ########################## 
def subplot_img(multi_img_array, label_row, label_column, output_name, output_path):
    '''
    这个函数实现多个子图拼接，用于文章中可视化对比图的绘制
    输入：multi_img_array:  大小为 Row*Col*H*W*C
          label_row : 行方向标签（不同样本名称的文本list）,大小Row
          label_column:  列方向的标签（不同方法名称的一个文本list），大小Col
          output_name: 输出图像名。
          output_path: 输出路径
    '''
    Row, Col, h, w, _ = multi_img_array.shape
    height = Row * h
    width = Col * w
    plt.figure(figsize=(width, height))

    # 设定整体背景风格
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 4})

    for i in range(Row):
        for j in range(Col):
            img = multi_img_array[i, j, ...]
            ax = plt.subplot2grid((Row, Col), (i, j))
            ax.imshow(img.astype(np.uint8))
            if i == 0:
                # 我们只需要第一行的图有标签即可
                # ax.set_title(label_column[j], fontdict={"fontsize":8})
                ax.set_title(label_column[j])
            if j == 0:
                # 我们只需要第一列的图有标签即可
                ax.text(-60, h / 2, label_row[i], rotation=90, verticalalignment='center')
            ax.axis('off')

    # reduce the blank part generated by plt and keep the original resolution
    fig = plt.gcf()
    fig.set_size_inches(width / 300, height / 300)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.02, wspace=0.02)
    plt.margins(0, 0)

    output_path_name = os.path.join(output_path, output_name)
    plt.savefig(output_path_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.cla()
    plt.close(fig)


def test_subplot_img():
    '''
    该函数为实现多个图像以子图形式摆放的主函数，该函数首先读取目标文件夹下的所有文件，每一个文件夹
    代表一个样本subject, 然后对于每一个样本，读取下一层文件夹，其中每一个文件夹代表原图、GT或者是预测结果，
    以切片形式存在，读完后应该是一个N_subject*N_models*N_slice*h*w*c的矩阵
    '''
    path = "test_subplot"
    files = glob(f"{path}/*.png")
    img_list = []
    for file in files:
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_list.append(img)
    img_array = np.array(img_list)
    img_array = img_array.reshape(2, 2, img.shape[0], img.shape[1], img.shape[2])
    xlabel = ["(a) U-Net", "(b) 3D U-Net", "(c) BDC-LSTM", "(d) SLEX-Net"]
    y_label = ["Br", "Br"]
    output_name = "paper_graph_new.png"
    output_path = "."
    subplot_img(img_array, xlabel, y_label, output_name, output_path)


def subplot_img_main():
    '''
    该函数为实现多个图像以子图形式摆放的主函数，该函数首先读取目标文件夹下的所有文件，每一个文件夹
    代表一个样本subject, 然后对于每一个样本，读取下一层文件夹，其中每一个文件夹代表原图、GT或者是预测结果，
    以切片形式存在，读完后应该是一个N_subject*N_models*N_slice*h*w*c的矩阵
    '''
    output_path = "subplot_paper_br"
    src_path = "display/brats_slices"
    # tar_subject = ["Brats18_TCIA04_111_1", "Brats18_TCIA01_221_1", "Brats18_CBICA_AVV_1", "Brats18_TCIA06_165_1"] # 对比其他方法
    tar_subject = ["Brats18_CBICA_ANG_1", "Brats18_TCIA06_409_1", "Brats18_TCIA08_278_1"]
    tar_models = ["image", "gt", "UNETLDLN1_216", "UNETLDLBest", "测试nonewnet", "UNETWITHOUTLDS", "UNETLDLWITHOUTPPM",
                  "UNETLDLBATCH1_100"]
    # xlabel = ["Image", "GT", "PAU-Net", "PAU-Net(w/o PPM)", "PAU-Net(w/o LDS)", "PAU-Net(N=1)", "PAU-Net(N=2)", "Baseline"]
    xlabel = ["Image", "GT", "Ours (PAU-Net)", "3D UNet", "V-Net", "Cascaded UNet", "Cascaded V-Net", "S3D-UNet"]

    subject_list_img = []
    for subject in tar_subject:
        tar_path = os.path.join(src_path, subject)
        model_list_img = []
        for item in range(len(tar_models)):
            model_path = os.path.join(tar_path, tar_models[item])
            # 下面，遍历每一个切片
            files = os.listdir(model_path)
            # files.sort()
            img_list = []
            for m in range(len(files)):
                img = cv.imread(os.path.join(model_path, f'{m}.png'))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_list.append(img)
            img_array = np.array(img_list)
            model_list_img.append(img_array)
        subject_list_img.append(model_list_img)

    subject_img_array = np.array(subject_list_img)
    for l in range(subject_img_array.shape[2]):
        img_data = subject_img_array[:, :, l, ...]
        output_name = f"{l}.png"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        y_label = [subject[8:] + f"_{l}" for subject in tar_subject]
        subplot_img(img_data, y_label, xlabel, output_name, output_path)


if __name__ == "__main__":
    # display_mmwhs()
    # display_brats()
    # display_mmwhs_entropy_gt_pre_image()
    subplot_img_main()
    # display_brats_entropy_gt_pre_image()
    pass
