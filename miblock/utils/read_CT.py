# -*- coding: utf-8 -*-
import pandas as pd
from imageio import volread, mvolread
import numpy as np
# import pydicom
import os
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
from matplotlib import cm
import re
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
# import pandas as pd

def get_rect(img_path):
    # 该函数输入lable图像的地址，读取图像，并从中得到区域中所有非零连通区域最小外接矩形的坐标
    # 最终返回box的list,其中box=[x_min,y_min,x_max,y_max],以及带框的图像
    image = cv.imread(img_path)
    image = image * 255
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
    # 注意 不同版本opencv 该函数有变化！！
    # hierachy, contours, offset = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours, hierachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boxs = []
    for contour in contours:
        # 求每条轮廓行列的最大，最小值
        Point_contour = contour.shape[0]
        x_min = 5000
        x_max = -200
        y_min = 5000
        y_max = -200
        for i in range(0, Point_contour):
            temp1 = contour[i][0][0]
            temp2 = contour[i][0][1]
            if x_min > temp1:
                x_min = temp1
            if x_max < temp1:
                x_max = temp1
            if y_min > temp2:
                y_min = temp2
            if y_max < temp2:
                y_max = temp2
        alpha = 10
        box = [int(x_min), int(y_min), int(x_max), int(y_max)]
        shape = image.shape
        box_large = [max(0, int(x_min-alpha)), max(0, int(y_min-alpha)), min(shape[0]-1, int(x_max+alpha)), min(shape[1]-1, int(y_max+alpha))]

        p1 = (max(0, int(x_min-alpha)), max(0, int(y_min-alpha)))
        p2 = (min(shape[0]-1, int(x_max+alpha)), min(shape[1]-1, int(y_max+alpha)))
        cv.rectangle(image, p1, p2, (0, 0, 255), 1)
        boxs.append(box_large)
    return boxs, image


def get_rect2(img_path):
    # 该函数输入lable图像的地址，读取图像，并从中得到区域中所有非零连通区域最小外接矩形的坐标
    # 最终返回box的list,其中box=[x_min,y_min,x_max,y_max],以及带框的图像
    image = cv.imread(img_path)
    shape = image.shape
    image = image * 255
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
    # 注意 不同版本opencv 该函数有变化！！
    # hierachy, contours, offset = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours, hierachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boxs = []
    mask = np.zeros(shape, dtype="uint8")
    for contour in contours:
        # 求每条轮廓行列的最大，最小值
        Point_contour = contour.shape[0]
        x_min = 5000
        x_max = -200
        y_min = 5000
        y_max = -200
        for i in range(0, Point_contour):
            temp1 = contour[i][0][0]
            temp2 = contour[i][0][1]
            if x_min > temp1:
                x_min = temp1
            if x_max < temp1:
                x_max = temp1
            if y_min > temp2:
                y_min = temp2
            if y_max < temp2:
                y_max = temp2
        alpha = 10
        box = [int(x_min), int(y_min), int(x_max), int(y_max)]
        box_large = [max(0, int(x_min-alpha)), max(0, int(y_min-alpha)), min(shape[0]-1, int(x_max+alpha)), min(shape[1]-1, int(y_max+alpha))]
        x_min_large= max(0, int(x_min-alpha))
        x_max_large = min(shape[0]-1, int(x_max+alpha))
        y_min_large = max(0, int(y_min-alpha))
        y_max_large = min(shape[1]-1, int(y_max+alpha))
        p1 = (max(0, int(x_min-alpha)), max(0, int(y_min-alpha)))
        p2 = (min(shape[0]-1, int(x_max+alpha)), min(shape[1]-1, int(y_max+alpha)))

        # 生成mask
        mask[y_min_large:y_max_large, x_min_large:x_max_large] = 1
        cv.rectangle(image, p1, p2, (0, 0, 255), 1)
        boxs.append(box_large)
    return mask, image


# 同matlab mat2gray函数
def mat2gray(I,limits):
    i = I.astype(np.float64)
    graymax = float(limits[1])
    graymin = float(limits[0])
    delta = 1 / (graymax - graymin)
    gray = delta * i - graymin * delta
    # 进行截断，对于大于最大值与小于最小值的部分，大于最大值设为1,小于最小值的设为0
    graycut = np.maximum(0, np.minimum(gray, 1))
    return graycut


def ct_read(save_path):
    # imgpath = r'E:\E\MRI-CT\label\doctor\CT\original\PNG\020'
    # files = os.listdir(imgpath)
    # for file in files:
    #     filepath = os.path.join(imgpath, file)
    #     # label = cv.imread(filepath)
    #     boxes, img = get_rect2(filepath)
    #     cv.imshow('', img)
    #     cv.imshow("boxex", boxes * 255)
    #     cv.waitKey(800)
    ct_general_width = 100
    ct_general_center = 40
    root_path = r"E:\E\MRI-CT\label\doctor\CT\original"
    dcm_path = os.path.join(root_path, "DICOM")
    png_path = os.path.join(root_path, "PNG")
    subjects = os.listdir(dcm_path)
    if not os.path.exists(save_path): os.mkdir(save_path)
    save_img_path = os.path.join(save_path, "image")
    save_label_path = os.path.join(save_path, "label")
    if not os.path.exists(save_img_path): os.mkdir(save_img_path)
    if not os.path.exists(save_label_path): os.mkdir(save_label_path)

    # 实例化保存图像并画GT的类
    eval = Evaluation()
    subjects1 = ['045', '065', '104', '186', '071']
    subjects = ['158' ]
    # subjects = ['071']
    for subject in subjects:
        path = os.path.join(dcm_path, subject)
        path2 = path.replace("\\", "/")

        # PNG label read
        path_png = os.path.join(png_path, subject)

        file_type_label = '*.png'
        sorted_dict_L = sort_path(path_png, file_type_label)

        slice_list = []

        for i in range(len(sorted_dict_L)):
            label = cv.imread(sorted_dict_L[i]["fullname"], cv.IMREAD_GRAYSCALE)
            slice_list.append(label)
        volume_label = np.array(slice_list, dtype="int16")

        label_shape = volume_label.shape
        vol = mvolread(path2, format='DICOM')
        # vol = volread(path2, format='DICOM')

        # 查看各个文件夹的维度数量,用来确定对应的DICOM文件
        if isinstance(vol, list):
            num = len(vol)
            for i in range(num):
                shape = vol[i].shape
                if shape == label_shape:
                    data_volume = vol[i]
                    break
                else:
                    pass
        else:
            data_volume =vol

        data_window = mat2gray(data_volume, [ct_general_center-ct_general_width/2, ct_general_center+ct_general_width/2])
        data_show = (data_window*255).astype("uint8")


        save_img_path_subject = os.path.join(save_img_path, subject)
        save_label_path_subject = os.path.join(save_label_path, subject)
        if not os.path.exists(save_img_path_subject):os.mkdir(save_img_path_subject)
        if not os.path.exists(save_label_path_subject):os.makedirs(save_label_path_subject)

        for i in range(data_show.shape[0]):
            if data_show[i, ...].shape !=(512,512):
                print(save_img_path_subject+"{}".format(i))
            # cv.imshow("image", data_show[i, ...])
            # cv.imshow("label", volume_label[i,...].astype("uint8")*255)
            # cv.waitKey(500)
            # 保存图像和label
            cv.imwrite('{}/{}.png'.format(save_img_path_subject, str(i)), data_show[i, ...])
            cv.imwrite('{}/{}.png'.format(save_label_path_subject, str(i)), (volume_label[i, ...]*255).astype("uint8"))
            # eval.save_contour_label(data_show[i, ...],volume_label[i,...].astype("uint8"),gt=None,save_path= save_img_path_subject,file_name=str(i))
        a = 1

class Evaluation(object):
    def __init__(self):
        pass

    def save_slice_img_label(self, img_volume, pre_volume, gt_volume,
                             output_path, file_name, show_mask=False, show_gt = False):
        assert img_volume.shape == pre_volume.shape
        if show_gt:
            assert img_volume.shape == gt_volume.shape
        width, height, depth = img_volume.shape
        # gray value mapping   from MRI value to pixel value(0-255)
        volume_max = np.max(img_volume)
        volume_min = np.min(img_volume)
        volum_mapped = (img_volume-volume_min)/(volume_max-volume_min)
        volum_mapped = (255*volum_mapped).astype('uint8')
        # construct a directory for each volume to save slices
        dir_volume = os.path.join(output_path, file_name)
        if not os.path.exists(dir_volume):
            os.makedirs(dir_volume)
        else:
            pass
        for i in range(depth):
            img_slice = volum_mapped[:, :, i]
            pre_slice = pre_volume[:, :, i]
            if show_gt:
                gt_slice = gt_volume[:, :, i]
            else:
                gt_slice = []
            self.save_contour_label(img=img_slice, pre=pre_slice, gt=gt_slice,
                                    save_path=dir_volume, file_name=i,show_mask=show_mask,show_gt=show_gt)

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(image.shape[-1]):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
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

    def save_contour_label(self, img, pre, gt=None, save_path='', file_name=None, show_mask=False, show_gt = False):
        # single channel to multi-channel
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
        height, width = img.shape[:2]
        _, ax = plt.subplots(1, figsize=(height, width))

        # Generate random colors
        # colors = self.random_colors(4)
        # Prediction result is illustrated as red and the groundtruth is illustrated as blue
        colors = [[0, 1.0, 0], [0, 0, 1.0]]
        # Show area outside image boundaries.

        # ax.set_ylim(height + 10, -10)
        # ax.set_xlim(-10, width + 10)
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')
        # ax.set_title("volume mask")
        masked_image = img.astype(np.uint32).copy()

        if show_mask:
            masked_image = self.apply_mask(masked_image, pre, colors[0])
            if show_gt:
                masked_image = self.apply_mask(masked_image, gt, colors[1])

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask_pre = np.zeros(
            (pre.shape[0] + 2, pre.shape[1] + 2), dtype=np.uint8)
        padded_mask_pre[1:-1, 1:-1] = pre
        contours = find_contours(padded_mask_pre, 0.5)
        for verts in contours:
            # reduce padding and  flipping from (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=colors[0], linewidth=1)
            ax.add_patch(p)

        if show_gt:
            padded_mask_gt = np.zeros((gt.shape[0] + 2, gt.shape[1] + 2), dtype=np.uint8)
            padded_mask_gt[1:-1, 1:-1] = gt
            contours_gt = find_contours(padded_mask_gt, 0.5)

            for contour in contours_gt:
                contour = np.fliplr(contour) -1
                p_gt = Polygon(contour, facecolor="none", edgecolor=colors[1], linewidth=1)
                ax.add_patch(p_gt)

        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height/37.5, width/37.5)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        ax.imshow(masked_image.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, file_name))
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def draw_ellipse():
    '''椭球面'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    a, b, c = 5.0, 25.0, 7.0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='b', cmap=cm.coolwarm)

    cset = ax.contourf(x, y, z, zdir='x', offset=-2 * a, cmap=cm.coolwarm)
    cset = ax.contourf(x, y, z, zdir='y', offset=1.8 * b, cmap=cm.coolwarm)
    cset = ax.contourf(x, y, z, zdir='z', offset=-2 * c, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_xlim(-2 * a, 2 * a)
    ax.set_ylabel('Y')
    ax.set_ylim(-1.8 * b, 1.8 * b)
    ax.set_zlabel('Z')
    ax.set_zlim(-2 * c, 2 * c)

    # plt.show()

def test_pandas():
    fd = [1]
    ddfd = np.squeeze(fd)

    # ct_read()
    arr = np.zeros((5, 5))
    arr1 = np.ones((5, 5))
    a1 = {"subject": 1, "data": arr1, "label": 6 * arr1}
    a2 = {"subject": 2, "data": 2 * arr1, "label": 5 * arr1}
    a3 = {"subject": 2, "data": 3 * arr1, "label": 4 * arr1}
    a4 = {"subject": 1, "data": 4 * arr1, "label": 3 * arr1}
    a5 = {"subject": 2, "data": 5 * arr1, "label": 2 * arr1}
    a6 = {"subject": 1, "data": 6 * arr1, "label": 1 * arr1}

    alist = []
    alist.append(a1)
    alist.append(a2)
    alist.append(a3)
    alist.append(a4)
    alist.append(a5)
    alist.append(a6)

    pdss = pd.DataFrame(alist)
    pre_list = []
    target_list = []
    for i in pdss.groupby("subject"):
        print(i)
        pre = i[1]["data"]
        target = i[1]["label"]
        pre_list.append(pre.values)
        target_list.append(target.values)
        a = 1


def sort_path(dir_path, file_type):
    imgpathlist = glob.glob(os.path.join(dir_path, file_type))
    list_dict = []
    pattern = r'[\_\-\(\)\.\\]+'
    for imgpath in imgpathlist:
        str_dict = str2dic(imgpath, pattern)
        list_dict.append(str_dict)
    list_dict.sort(key=lambda x: x["slice"])
    a = 1
    return list_dict

# 该函数将一个字符序列中的数字取出来，放入字典中，字典中还包括整个序列
def str2dic(pathstr,re_str):
    # 此处用到了正则表达式的切分功能
    b = re.split(re_str, os.path.basename(pathstr))
    temparr = []
    for i in range(0, len(b)):
        if b[i].isdigit() == True:
            tempp = int(b[i])
            temparr.append(tempp)
        else:
            continue
    if len(temparr) > 1:
        split_dict = {'count': temparr[0], 'slice': temparr[1], 'fullname': pathstr}
    elif len(temparr) == 1:
        split_dict = {'count': b[0], 'slice': temparr[0], 'fullname': pathstr}
    else:
        raise Exception('False Input Data!')

    return split_dict

def check_ct_png_dataset(save_path):
    ct_general_width = 100
    ct_general_center = 40
    root_path = r"D:\MICCAI 2020\data\hemorrhage\PNG\Trans_PNG"
    dcm_path = os.path.join(root_path, "brain")
    png_path = os.path.join(root_path, "label")
    subjects = os.listdir(dcm_path)
    if not os.path.exists(save_path): os.mkdir(save_path)
    save_img_path = os.path.join(save_path, "image")
    save_label_path = os.path.join(save_path, "label")
    if not os.path.exists(save_img_path): os.mkdir(save_img_path)
    if not os.path.exists(save_label_path): os.mkdir(save_label_path)

    # 实例化保存图像并画GT的类
    eval = Evaluation()

    for subject in subjects:
        path = os.path.join(dcm_path, subject)
        path2 = path.replace("\\", "/")

        # PNG label read
        path_png = os.path.join(png_path, subject)
        path_img = os.path.join(dcm_path, subject)
        file_type_label = '*.png'
        slices = os.listdir(path_img)

        slice_list = []
        img_list = []
        for i in range(len(slices)):
            img_path = os.path.join(path_img, '{}.png'.format(i))
            label_path = os.path.join(path_png, '{}.png'.format(i))
            label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            slice_list.append(label)
            img_list.append(img)

        volume_label = np.array(slice_list, dtype="int16")
        volume_image = np.array(img_list, dtype="uint8")
        label_shape = volume_label.shape

        data_show = volume_image

        save_img_path_subject = os.path.join(save_img_path, subject)
        save_label_path_subject = os.path.join(save_label_path, subject)
        if not os.path.exists(save_img_path_subject): os.mkdir(save_img_path_subject)
        if not os.path.exists(save_label_path_subject): os.makedirs(save_label_path_subject)

        for i in range(data_show.shape[0]):
            if data_show[i, ...].shape != (512, 512):
                print(save_img_path_subject + "{}".format(i))
            # cv.imshow("image", data_show[i, ...])
            # cv.imshow("label", volume_label[i,...].astype("uint8")*255)
            # cv.waitKey(500)
            # 保存图像和label
            # cv.imwrite('{}/{}.png'.format(save_img_path_subject, str(i)), data_show[i, ...])
            # cv.imwrite('{}/{}.png'.format(save_label_path_subject, str(i)),(volume_label[i, ...] * 255).astype("uint8"))
            eval.save_contour_label(data_show[i, ...],volume_label[i,...].astype("uint8"),gt=None,save_path= save_img_path_subject,file_name=str(i))
        a = 1



if __name__ == "__main__":
    # root_path = r"E:\E\MRI-CT\label\doctor\CT\original\DICOM"
    # path = r"E:\E\MRI-CT\label\doctor\CT\original\DICOM\001\201805211058\1.2.392.200036.9116.2.5.1.37.2417525831.1517281394.622013\1.2.392.200036.9116.2.5.1.37.2417525831.1517281660.308001"
    # files = os.listdir(path)
    # dcmlist = []
    # for file in files:
    #     pathall = os.path.join(path, file)
    #     dcm = pydicom.dcmread(pathall)
    #     dcmlist.append(dcm)
    #     f = 1
    save_path = "./Trans_PNG"
    check_ct_png_dataset(save_path)




