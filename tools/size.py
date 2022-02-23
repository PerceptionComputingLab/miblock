from os import listdir
import SimpleITK as sitk
import numpy as np
def loadfiles(img_dir,lab_dir):
    img = listdir(img_dir)
    img = [img_dir + '/' + i for i in img]
    lab = listdir(lab_dir)
    lab = [lab_dir + '/' + i for i in lab]
    return list(zip(sorted(img),sorted(lab)))
img_dir= "W:\\LITS17\\image"
lab_dir= "W:\\LITS17\\label"
file_info = loadfiles(img_dir,lab_dir)
for i in range(0,len(file_info)):
    path=file_info[i]
    img = sitk.ReadImage(path[0])
    img_array = sitk.GetArrayFromImage(img)
    img_array = img_array.astype(np.float32)
    lab = sitk.ReadImage(path[1])
    lab_array = sitk.GetArrayFromImage(lab)
    lab_array = lab_array.astype(np.float32)
    print(img_array.shape)
    print(type(img_array[0][0][0]))
    print(lab_array.shape)
    print(type(lab_array[0][0][0]))