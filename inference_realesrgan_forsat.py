import argparse
import shutil

import cv2
import numpy as np
import tifffile as tf
from osgeo import gdal
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from scripts.utils_tif import *

def crop_sr(crop_path, process_path, outscale, upsampler):

    # (2)按块超分
    print("Start SR!!!")
    crop_files = os.listdir(crop_path)  # 默认排序为0， 1, 10, 11...
    # crop_files.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1])))
    # files2 = [file for _, _, file in os.walk(in_path2)][0]    # 相同效果
    crop_file_paths = [crop_path + os.sep + file for file in crop_files]
    sr_path = process_path + os.sep + 'sr'
    os.makedirs(sr_path, exist_ok=True)

    for idx2, crop_file_path in enumerate(crop_file_paths):

        # imgname2, extension2 = os.path.splitext(os.path.basename(crop_file_path))
        print('Crop Processing', idx2, '/', len(crop_file_paths), crop_file_path)
        img = tf.imread(crop_file_path)  # RGB

        # print(img.shape)
        ds = gdal.Open(crop_file_path)
        im_width = ds.RasterXSize * outscale  # width
        im_height = ds.RasterYSize * outscale  # height
        im_proj = ds.GetProjection()  # 地理坐标
        im_geotrans_raw = ds.GetGeoTransform()  # 仿射矩阵
        im_geotrans_new = list(im_geotrans_raw)
        im_geotrans_new[1] = im_geotrans_new[1] / outscale
        im_geotrans_new[5] = im_geotrans_new[5] / outscale

        del ds

        # 超分, 内存不够时报错
        try:
            if np.count_nonzero(img) == 0:
                output = cv2.resize(img, fx=outscale, fy=outscale, dsize=None)
            elif np.count_nonzero(img) == img.shape[0] * img.shape[1] * img.shape[2]:
                output, _ = upsampler.enhance(img, outscale=outscale)
                output[output == 0] = 1
            else:
                output1, _ = upsampler.enhance(img, outscale=outscale)

                # 去黑边算法
                output2 = img
                output2[output2 != 0] = 1

                output2 = cv2.resize(output2, fx=outscale, fy=outscale, dsize=None)
                kernel = np.ones((3, 3), np.uint8)
                output2 = cv2.erode(output2, kernel, iterations=3)

                output1[output1 == 255] = 254
                output1 = output1 + output2

                output = np.multiply(output1, output2)

        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            save_path = os.path.join(sr_path, os.path.basename(crop_file_path))

            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(save_path, im_width, im_height, 3, gdal.GDT_Byte)
            out_ds.SetGeoTransform(im_geotrans_new)
            out_ds.SetProjection(im_proj)
            out_ds.GetRasterBand(1).WriteArray(output[:, :, 0])
            out_ds.GetRasterBand(2).WriteArray(output[:, :, 1])
            out_ds.GetRasterBand(3).WriteArray(output[:, :, 2])
            out_ds.GetRasterBand(1).SetNoDataValue(0)
            # driver.CreateCopy(save_path2, out_ds, strict=0, options=["TILED=YES", "COMPRESS=LZW"])    # 保存压缩文件
            del out_ds
    print("SR Complete!!!")
    return sr_path

def main():
    """Inference demo for Real-ESRGAN."""
    # 1、参数获取
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output folder')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus',
                        help='Model names: RealESRGAN_x4plus | RealESRNet_x4plus  | RealESRGAN_x2plus')
    parser.add_argument('-m', '--model_path', type=str, default='./weights/net_ESRNet_1000k.pth', help='[Option] Model path')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    # 文件名后缀和文件扩展名
    parser.add_argument('--suffix', type=str, default='sr', help='Suffix of the restored image')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | tif')
    # 裁剪及分块大小、tile_pad和pre_pad是为了在内部分块时并避免伪影
    parser.add_argument('-c', '--crop_size', type=int, default=512, help='Crop size for tif')
    parser.add_argument('-l', '--overlap_size', type=int, default=100, help='Overlap size for tif')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=10, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true',
                        help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('-g', '--gpu-id', type=int, default=None,
                        help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    args = parser.parse_args()

    # 2、确定模型及参数文件
    # 2.1 确定模型
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2

    # 2.2 确定模型参数文件
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            print('please input model_path')

    # 3、restorer 超分构建器(dni_weight为dni降噪参数，是另一个模型的参数, 不需要)
    upsampler = RealESRGANer(
        scale=netscale,
        model=model,
        model_path=model_path,
        dni_weight=None,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    # 4、确定其他参数、输入文件和输出路径
    # 最终输出比例
    outscale = args.outscale

    # 创建输出目录、已存在时不报异常
    os.makedirs(args.output, exist_ok=True)

    # 遍历并输出文件路径列表
    file_paths = sorted(glob.glob(os.path.join(args.input, '*')))
    print(file_paths)

    # 6、遍历影像并超分输出
    for idx, file_path in enumerate(file_paths):
        imgname, extension = os.path.splitext(os.path.basename(file_path))
        print('File Processing', idx, imgname)

        # 创建临时处理路径
        process_path = args.output + os.sep + "temp"
        os.makedirs(process_path, exist_ok=True)

        # 超分影像
        # (1)切块
        crop_size = args.crop_size
        overlap_size = args.overlap_size
        crop_path, rows, cols, height, width = tif_crop(file_path, process_path, crop_size, overlap_size)

        # (2)切块超分
        sr_path = crop_sr(crop_path, process_path, outscale, upsampler)

        # (3)超分后裁切: 规避切块之间拼接明显的问题
        crop_path2 = sr_crop(sr_path, process_path, height, width, crop_size, outscale)

        # (4)合并
        merge_path = tif_merge(crop_path2, process_path, rows, cols)

        # (5)加坐标
        add_coor(file_path, args.output, merge_path, outscale)

        # 删除临时文件
        # shutil.rmtree(process_path)

if __name__ == '__main__':
    main()
