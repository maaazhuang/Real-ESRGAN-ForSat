# Real-ESRGAN-ForSat
Real-ESRGAN-ForSat based on Real-ESRGAN for Satellite Image Restoration

### 一、概述
本项目基于Real-ESRGAN实现对卫星影像进行超分辨率

本人所学并非深度学习专业，此项目为2022年部分学习和工作内容，存档备份使用，如无工作需要不再进行代码优化和更新

文件内容上，在Real-ESRGAN的基础上重构inference_realesrgan.py为inference_realesrgan_forsat，删减了部分代码，并添加scripts/utils_tif.py以支持对卫星影像的处理操作

### 二、卫星影像超分思路
**1. 对卫星影像进行切块**

在实际卫星影像生产中, 一张影像为几G或一个城市的成果影像为几十G，现有的代码无法支持此类成果的超分，所以需要进行切块处理
    
    切块时进行了冗余、有重叠度的裁切，主要是为了避免超分拼接处出现明显拼接痕迹的情况
    
    （U-net网络可不进行冗余超分，拼接处一般也不会出现拼接痕迹，但是效果没有real-esrgan好）

**2. 对切块进行超分**

利用real-esrgan对裁切后的图片进行超分，默认超分方法对卫星影像处理时会遇到超分出空洞和黑边的情况
    
在超分后，利用原图和超分后的图像进行像素修正：
    
    （1）规避影像内部像素出现0值，8bit影像在ArcGIS这种专业软件中，0像素一般被设置为透明，内部如果有（0，100，200）这样的像素会显示成空洞效果
    
    （2）遥感影像成果可能是不规则的形状，需要规避超分后图像边缘出现低像素值的情况，如1，3，5，此类像素值显示效果近似黑边，视觉效果较差，通过滤波和图像叠加去除黑边

**3. 超分后进行裁切**

对图像超分后冗余、有重叠度的部分进行裁切，以进行有效拼接

**4. 合并超分后图片**

使用数组stack进行图像合并，先合并每一行，再将每行合并成整体
    
    此步骤吃内存，建议用集群式服务器，例如：20G影像超分2倍约为80G，服务器需要80G内存
    
    此步骤有优化空间，若想优化，建议直接创建空白栅格数据集，按每块图像向空白数据集中写入数据（目前不做优化）

**5. 写入地理坐标**

根据超分的输出比例计算像素大小和仿射变换参数，写入tif文件中。
    
**6. 注意事项**

    （1）建议只超分2倍和4倍，其它非整数倍数暂未尝试，去空洞、去黑边、写坐标的算法可能会失效；
    
    （2）建议只生成而不对抗，real-esrgan在对抗训练时，影像中海洋部分会出现大量噪声；
    
 ### 三、使用方法：
 1. 部署Real-ERSGAN, 并测试成功
 
 2. 安装卫星影像处理相关库, gdal库
 
 3. 添加inference_realesrgan_forsat.py、scripts/utils_tif.py文件
 
 4. 设置inference_realesrgan_forsat.py代码中input、output、weights，并准备对应文件
 
    [个人训练模型](https://pan.baidu.com/s/1upLhwP8PsGD8GStV2XkhQA?pwd=fhu7)（北京三号卫星0.5m分辨率影像训练模型）
 
 5. 执行
 
 ### 四、效果示例
 
 细节可通过下载asset_sat文件夹下超分前后图片对比观看
 
 <table rules="none" align="center">
	<tr>
		<td>
			<center class="half">
				<img src="assets_sat/sr_before.png" width=500 />
				<br/>
                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
				<font color="AAAAAA">sr_before.png</font>
			</center>
		</td>
		<td>
			<center class="half">
				<img src="assets_sat/sr_after.png" width=500 />
				<br/>
                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
				<font color="AAAAAA">sr_after.png</font>
			</center>
		</td>
	</tr>
</table>
