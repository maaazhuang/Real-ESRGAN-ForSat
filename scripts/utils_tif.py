
import os
import math
import glob
import time
import cv2
import tifffile as tiff
import numpy as np
from osgeo import gdal

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tif_crop(in_path, out_path, crop_size, overlap_size):
    print("Start Crop!!!")
    if in_path.split(".")[-1] == "tif" or in_path.split(".")[-1] == "tiff":
        img = tiff.imread(in_path)
        if img.shape[2] != 3:
            img = img.transpose(1, 2, 0)
    else:
        img = cv2.imread(in_path, -1)

    # 方法三 用GDAL的ds读，可能会不占内存

    height, width = img.shape[0], img.shape[1]
    print(img.shape)
    print("image height width", height, width)
    rows = math.ceil(height / crop_size)  # 高度可以分成几行
    cols = math.ceil(width / crop_size)  # 宽度可以分成几列
    print("crop rows cols", rows, cols)

    crop_path = os.sep.join([out_path, "crop"])  # 传入的是列表
    os.makedirs(crop_path, exist_ok=True)

    for i in range(rows):    # 先行(x)
        for j in range(cols):    # 后列(y)

            x_coor = i * crop_size
            y_coor = j * crop_size

            # 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
            x_block = min(height - x_coor, crop_size + overlap_size)
            y_block = min(width - y_coor, crop_size + overlap_size)

            img_crop = img[x_coor:x_coor+x_block, y_coor:y_coor+y_block]

            file_path = crop_path + os.sep + str(i) + "_" + str(j) + ".png"
            tiff.imwrite(file_path, img_crop)

    print("Crop Complete!!!")
    return crop_path, rows, cols, height, width

def tif_merge(in_path, out_path, rows, cols):
    print("Start Merge!!!")

    out_path = os.sep.join([out_path, "merge"])  # 传入的是列表
    os.makedirs(out_path, exist_ok=True)

    files = os.listdir(in_path)     # 默认排序为0， 1, 10, 11...
    files.sort(key=lambda x: (int(x.split('.')[0])))

    # for file in files:
    #     print(file)

    # files2 = [file for _, _, file in os.walk(in_path2)][0]    # 相同效果
    file_paths_h = [in_path + os.sep + file for file in files]
    # 方法二
    # file_paths_h = glob.glob(os.path.join(in_path2, "*.png"))

    hstack_path = out_path + os.sep + "h"
    os.makedirs(hstack_path, exist_ok=True)

    # 方式一
    for i in range(rows):
        img_path1 = file_paths_h[i * cols]
        img_h = tiff.imread(img_path1)
        for j in range(1, cols):    # 从第二个开始
            img_path2 = file_paths_h[i * cols + j]
            img_ht = tiff.imread(img_path2)
            # print(i, j, img_path2)
            img_h = np.hstack([img_h, img_ht])

        temp_path_h = hstack_path + os.sep + str(i) + ".png"
        tiff.imwrite(temp_path_h, img_h)

    # 方式二
    # for i in range(0, rows):
    #     list = []
    #     for j in range(0, cols):
    #         img_path2 = file_paths_h[i * cols + j]
    #         img_ht = tiff.imread(img_path2)
    #         list.append(img_ht)
    #         # print(i, j)
    #         # print(img_path2)
    #         img_h = np.hstack(list)
    #
    #     temp_path_h = hstack_path + os.sep + str(i) + ".png"
    #     tiff.imwrite(temp_path_h, img_h)

    # 按列竖向拼接成一个
    vstack_path = out_path + os.sep + "v"
    os.makedirs(vstack_path, exist_ok=True)
    files_v = os.listdir(hstack_path)
    files_v.sort(key=lambda x: (int(x.split('.')[0])))
    file_paths_v = [hstack_path + os.sep + file2 for file2 in files_v]

    img_path3 = file_paths_v[0]
    img_v = tiff.imread(img_path3)
    for m in range(1, rows):
        img_path4 = file_paths_v[m]
        img_vt = tiff.imread(img_path4)
        img_v = np.vstack([img_v, img_vt])

    temp_path_v = vstack_path + os.sep + "merge.png"
    tiff.imwrite(temp_path_v, img_v)
    print("Merge Complete!!!")

    merge_path = temp_path_v
    return merge_path

def sr_crop(in_path, out_path, height, width, crop_size, scale):

    print("Start SR Crop!!!")
    out_path = os.sep.join([out_path, "crop2"])  # 传入的是列表
    os.makedirs(out_path, exist_ok=True)

    # files = os.listdir(in_path)     # 默认排序为0， 1, 10, 11...
    # files.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1])))

    rows = math.ceil(height / crop_size)  # 高度可以分成几行
    cols = math.ceil(width / crop_size)  # 宽度可以分成几列
    print(rows, cols)

    count = 0
    for i in range(rows):    # 先行(x)
        for j in range(cols):    # 后列(y)
            # print(i, j)
            count += 1
            x_coor = i * crop_size * scale
            y_coor = j * crop_size * scale
            # print(x_coor, y_coor)
    #         # 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
            x_block = min(height*scale - x_coor, crop_size*scale)
            y_block = min(width*scale - y_coor, crop_size*scale)
            # print(x_block, y_block)
            img_path = in_path + os.sep + str(i) + "_" + str(j) + ".png"
            # print(img_path)
            img = tiff.imread(img_path)
            img_crop = img[0:x_block, 0:y_block]
    #
            file_path = out_path + os.sep + str(count) + ".png"
            tiff.imwrite(file_path, img_crop)

    print("SR Crop Complete!!!")
    return out_path

def add_coor(in_path, out_path, merge_path, scale):
    print("Start Add Coordinate!!!")

    imgname, extension = os.path.splitext(os.path.basename(in_path))
    out_path = out_path + os.sep + str(imgname) + "_sr.tif"
    ds = gdal.Open(in_path)
    ds2 = gdal.Open(merge_path)
    # im_bands2 = ds2.RasterCount  # 波段数
    # im_width2 = ds2.RasterXSize  # width
    # im_height2 = ds2.RasterYSize  # height
    im_bands = ds.RasterCount  # 波段数
    im_width = ds.RasterXSize * scale # width
    im_height = ds.RasterYSize * scale # height
    im_geotrans = ds.GetGeoTransform()  # 仿射矩阵
    im_proj = ds.GetProjection()  # 地理坐标
    out_geotrans = (im_geotrans[0], im_geotrans[1]/scale, im_geotrans[2], im_geotrans[3], im_geotrans[4], im_geotrans[5]/scale)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_path, im_width, im_height, im_bands, gdal.GDT_Byte)

    out_ds.SetProjection(im_proj)
    out_ds.SetGeoTransform(out_geotrans)


    for i in range(im_bands):
        out_ds.GetRasterBand(i + 1).WriteArray(ds2.GetRasterBand(i + 1).ReadAsArray())
    out_ds.GetRasterBand(1).SetNoDataValue(0)

    del ds, ds2, out_ds

    print("Add Coordinate Complete!!!")


# if __name__ == '__main__':
#     in_path = "D:/MZ/SR/data/Aomen.tif"
#     out_path = "D:/MZ/SR/data/output"
#     crop_size = 2000
#     overlap_size = 200
#     
#     tif_crop(in_path, out_path, crop_size, overlap_size)

