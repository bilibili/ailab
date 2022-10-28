Real Cascade U-Nets for Anime Image Super Resolution
-------------------------------------------

[中文](README.md) **|** [English](README_EN.md)


:fire: **Real-CUGAN**:fire: 是一个使用百万级动漫数据进行训练的，结构与Waifu2x兼容的通用动漫图像超分辨率模型。它支持**2x\3x\4x**倍超分辨率，其中2倍模型支持4种降噪强度与保守修复，3倍/4倍模型支持2种降噪强度与保守修复。

**Real-CUGAN** 为Windows用户打包了一个可执行环境。同时目前已有Windows-GUI与Web版本可使用。

[更新进展](https://github.com/bilibili/ailab/tree/main/Real-CUGAN#Acknowledgement)<br>
2022-02-07:[Windows-GUI版](https://github.com/Justin62628/Squirrel-RIFE/releases/tag/v0.0.3)/[Web-CPU版](https://huggingface.co/spaces/mayhug/Real-CUGAN)<br>
2022-02-09:[Colab示例代码](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/colab-demo.ipynb)<br>
2022-02-17:适用于移动端和AMD显卡的[NCNN版本](https://github.com/nihui/realcugan-ncnn-vulkan)<br>
2022-02-20:添加低显存模式(支持>1.5G显存的N卡)，以牺牲60%的速度为代价，解锁超大分辨率输入图像；下载20220220更新包或完整包使用<br>
2022-02-27:添加faster低显存模式，相比普通模式耗时仅增加25%；添加增强处理强度alpha参数（实验性参数）；下载20220227更新包或完整包使用<br>
2022-05-35: [Real-CUGAN-Pro](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/Changelog_CN.md)；下载weights_pro参数+20220535更新包或完整包使用<br>
2022-06-25: [Browser-CPU-Ver](https://github.com/hanFengSan/realcugan-ncnn-webassembly)可以用浏览器体验demo效果啦，不过因为是CPU执行的，建议图像分辨率小一些。<br>
2022-07-28:[NCNN版本](https://github.com/nihui/realcugan-ncnn-vulkan)支持pro模型了，<br>
2022-08-09:[Squirrel Anime Enhance(WIN-GUI，中文+英文)](https://github.com/Justin62628/Squirrel-RIFE/releases/tag/v3.20.1)支持RealCUGAN-Pro了，支持视频超分，导出常见视频格式。

访问这个网页体验demo，https://real-cugan.animesales.xyz/<br>
如果Real-CUGAN对您的项目有帮助，可以⭐与分享一波，感谢~

### 1. 效果对比


https://user-images.githubusercontent.com/61866546/147812864-52fdde74-602f-4f64-ac05-4d34cc58aa79.mp4


- **效果图对比**(推荐点开大图在原图分辨率下对比)
  <br>
  纹理挑战型(注意地板纹理涂抹)(图源:《侦探已死》第一集10分20秒)
  ![compare1](demos/title-compare1.png)
  线条挑战型(注意线条中心与边缘的虚实)(《东之伊甸》第四集7分30秒)
  ![compare2](demos/compare2.png)
  极致渣清型(注意画风保留、杂线、线条)(图源:Real-ESRGAN官方测试样例)
  ![compare3](demos/compare3.png)
  景深虚化型(蜡烛为后景，刻意加入了虚化特效，应该尽量保留原始版本不经过处理)(图源:《～闘志の華～戦国乙女2ボナ楽曲PV》第16秒)
  ![compare4](demos/compare4.png)
- **详细对比**

|                | Waifu2x(CUNet)                                               | Real-ESRGAN(Anime6B)                                         | Real-CUGAN                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 训练集         | 私有二次元训练集，量级与质量未知                             | 私有二次元训练集，量级与质量未知                             | 百万级高清二次元patch dataset                                |
| 推理耗时(1080P)    | Baseline                                                     | 2.2x                                                         | 1x                                                           |
| 效果(见对比图) | 无法去模糊，artifact去除不干净                               | 锐化强度最大，容易改变画风，线条可能错判，<br />虚化区域可能强行清晰化 | 更锐利的线条，更好的纹理保留，虚化区域保留                   |
| 兼容性         | 大量windows-APP使用，VapourSynth支持，<br />Caffe支持，PyTorch支持，NCNN支持 | PyTorch支持，VapourSynth支持，NCNN支持                       | 同Waifu2x，结构相同，参数不同，与Waifu2x无缝兼容             |
| 强度调整       | 仅支持多种降噪强度                                           | 不支持                                                       | 已完成4种降噪程度版本和保守版，未来将支持调节不同去模糊、<br />去JPEG伪影、锐化、降噪强度 |
| 尺度           | 仅支持1倍和2倍                                               | 仅支持4倍                                                    | 已支持2倍、3倍、4倍，1倍训练中              |

### 2. Windows玩家
修改config.py配置参数，双击go.bat运行。
- #### 超分工具：
    [百度网盘(提取码ds2a) :link:](https://pan.baidu.com/s/10NbgnusDucllKiE0sgBWvQ)｜[GithubRelease :link:](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN) | [和彩云(提取码1RHQ,手机号验证码登录,不限速无需客户端) :link:](https://caiyun.139.com/m/i?015CHx2VU4RNd)｜ [GoogleDrive :link:](https://drive.google.com/drive/folders/1UFgpV14uEAcgYvVw0fJuajzy1k7JIz6H)
- #### 系统环境：
    - :heavy_check_mark: 在win10-64bit系统下进行测试
    - :heavy_check_mark: 小包需求系统cuda >= 10.0. 【大包需求系统cuda >= 11.1】
    - :heavy_check_mark: 只支持N卡或CPU，N卡需要至少1.5G显存
    - :heavy_exclamation_mark: **注意30系列 nvidia GPU 只能用大包；<20系建议用小包**

- #### 使用config文件说明：
  #### a. 通用参数设置
    - mode: 在其中填写video或者image决定超视频还是超图像；
    - scale: 超分倍率；
    - model_path: 填写模型参数路径(目前3倍4倍超分只有3个模型，2倍有4个不同降噪强度模型和1个保守模型)；
    - device: 显卡设备号。如果有多卡超图片，建议手工将输入任务平分到不同文件夹，填写不同的卡号；
    - 超图像，需要填写输入输出文件夹；超视频，需要指定输入输出视频的路径。
    - cache_mode:根据个人N卡显存大小调节缓存模式.mode2/3可超任意大小分辨率（瓶颈不在显存）图像
        > **0:** 默认使用cache缓存必要参数 <br>
        > **1:** 使用cache缓存必要参数，对缓存进行8bit量化节省显存，带来15%延时增长，肉眼完全无法感知的有损模式 <br>
        > **2:** 不使用cache，有损模式。耗时约增加25%，仅在有景深虚化的图里有微小的误差，不影响纹理判断 <br>
        > **3:** 不使用cache，无损模式。耗时约为默认模式的2.5倍，但是显存不受输入图像分辨率限制，tile填得够大，1.5G显存可超任意分辨率 <br>
    - tile: 数字越大显存需求越低，相对地可能会小幅降低推理速度 **{0, 1, x, auto}** <br>
        > **0:** 直接使用整张图像进行推理，大显存用户或者低分辨率需求可使用 <br>
        >  **1:** 对长边平分切成两块推理<br>
        >  **x:** 宽高分别平分切成x块推理<br>
        >  **auto:** 当输入图片文件夹图片分辨率不同时，填写auto自动调节不同图片tile模式，未来将支持该模式。
    - alpha: 该值越大AI修复程度、痕迹越小，越模糊；alpha越小处理越烈，越锐化，色偏（对比度、饱和度增强）越大；默认为1不调整，推荐调整区间(0.7,1.3)；
    - half: 半精度推理，>=20系显卡直接写True开着就好
    - :heavy_exclamation_mark: 如果使用windows路径，建议在双引号前加r
  
  #### b. 超视频设置
    - nt: 每张卡的线程数，如果显存够用，建议填写>=2
    - n_gpu: 使用显卡张数；
    - encode_params: 编码参数 **{crf，preset}** 
        > **crf:** 通俗来讲，crf变低=高码率高质量 <br>
        > **preset:** 越慢代表越低编码速度越高质量+更吃CPU，CPU不够应该调低级别，比如slow，medium，fast，faster

- #### 模型分类说明：
	 - 降噪版：如果原片噪声多，压得烂，推荐使用；目前2倍模型支持了3个降噪等级；
	 - 无降噪版：如果原片噪声不多，压得还行，但是想提高分辨率/清晰度/做通用性的增强、修复处理，推荐使用；
	 - 保守版：如果你担心丢失纹理，担心画风被改变，担心颜色被增强，总之就是各种担心AI会留下浓重的处理痕迹，推荐使用该版本；但对于较模糊、渣清的视频，修复程度不会比降噪版更好。

### 3. Python玩家
环境依赖 <br>
:white_check_mark:  **torch>=1.0.0**      <br>
:white_check_mark:  **numpy**             <br>
:white_check_mark:  **opencv-python**     <br>
:white_check_mark:  **moviepy**           <br>
upcunet_v3.py:模型+图像推理包 <br>
inference_video.py:一个简单的使用Real-CUGAN推理视频的脚本

### 4. VapourSynth玩家
移步[Readme](VapourSynth/README.md)

### 5. realcugan-ncnn-vulkan
[NCNN版本](https://github.com/nihui/realcugan-ncnn-vulkan)已出现，这意味着A卡用户和移动端用户也可以使用GPU跑Real-CUGAN模型了~

### 6. Waifu2x-caffe玩家

#### 我们目前为waifu2x-caffe玩家提供了两套参数：
:fire: **Real-CUGAN2x标准版(denoise-level3)** 和 :fire: **Real-CUGAN2x无切割线版**
<br>
    [百度网盘(提取码ds2a) :link:](https://pan.baidu.com/s/10NbgnusDucllKiE0sgBWvQ)｜[GithubRelease :link:](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN)｜ [GoogleDrive :link:](https://drive.google.com/drive/folders/1UFgpV14uEAcgYvVw0fJuajzy1k7JIz6H)
    <br>
用户可以用这套参数覆盖原有model-cunet模型参数（如有需要，记得对原有参数进行备份），用原有被覆盖的预设（按当前的文件名，是2x仅超分不降噪）进行超分。<br>

:heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark: 由于waifu2x-caffe的切割机制，对于标准版，crop_size应该尽量调大，否则可能造成切割线。如果**发现出现切割线，** 请移步下载windows应用，它支持无切割线痕迹的crop(tile_mode），既能有效降低显存占用需求，crop也是无损的。或者使用我们额外提供的无切割线版，它会造成更多的纹理涂抹和虚化区域清晰化。

>开发者可以很轻松地进行适配，推荐使用整张图像作为输入。如果顾及显存问题，建议基于PyTorch版本进行开发，使用tile_mode降低显存占用需求。

### 7.:european_castle: Model Zoo

可在网盘路径下载完整包与更新参数包获取各模型参数。

<table>
	<tr>
	    <th align="center"></th>
        <th align="center">1倍</th>
	    <th align="center">2倍</th>
	    <th align="center">3倍/4倍</th>  
	</tr >
	<tr>
	    <td align="center" >降噪程度</td>
	    <td align="center">仅支持无降噪，训练中</td>
	    <td align="center">现支持无降噪/1x/2x/3x</td>
        <td align="center">现支持无降噪/3x，1x/2x训练中</td>
	</tr>
	<tr>
	    <td  align="center">保守模型</td>
	    <td  align="center">训练中</td>
	    <td  colspan="2" align="center">已支持</td>
	</tr>
	<tr>
        <td  align="center">快速模型</td>
	    <td  colspan="3" align="center">调研中</td>
	</tr>
</table>

### 8. TODO：
- [ ]  快速模型，提高推理速度
- [x]  降低显存占用需求
- [x]  可调整的增强锐度，降噪强度，去模糊强度
- [ ]  一步超到任意指定分辨率
- [x]  优化纹理保留，削减模型处理痕迹
- [x]  简单的GUI

:stuck_out_tongue_closed_eyes: 欢迎各位大佬在**issue**:innocent: 进行留言,提出各种建议和需求:thumbsup: ! 

### Acknowledgement
这里不公开训练代码，训练步骤参考了但不局限于[RealESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md). 想自行训练的请移步该仓库。<br>
模型结构魔改自Waifu2x官方[CUNet](https://github.com/nagadomi/nunif/blob/master/nunif/models/waifu2x/cunet.py).<br>
另有:star2:更新进展:star2:如下：<br>
 - [Squirrel补帧团队](https://github.com/Justin62628/Squirrel-RIFE/releases/tag/v3.20.1)基于RealCUGAN(PyTorch版本)与Waifu2x/RealESRGAN开发了一个图形界面程序（默认中文，支持英文），并免费发布；<br>
 - [mnixry](https://github.com/mnixry)制作了RealCUGAN的[Web-CPU-PyTorch版](https://huggingface.co/spaces/mayhug/Real-CUGAN),大家可以免费尝鲜测试。1080P图像的2倍尺度大约需要等待24s返回结果；<br>
 - [nihui](https://github.com/nihui) 实现了RealCUGAN的[NCNN版本](https://github.com/nihui/realcugan-ncnn-vulkan) ,AMD显卡用户和移动端用户亦可以使用； <br>
 - 网页demo版https://github.com/hanFengSan/realcugan-ncnn-webassembly
 - [第三方GUI](https://github.com/Baiyuetribe/paper2gui/blob/main/Video%20Super%20Resolution/RealCugan-GUI.md) <br>
 - [AaronFeng753](https://github.com/AaronFeng753)将RealCUGAN的Caffe版本集成进[Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI)；<br>

感谢他们的贡献！
