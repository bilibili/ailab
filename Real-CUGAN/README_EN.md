Real Cascade U-Nets for Anime Image Super Resolution
-------------------------------------------

:fire: **Real-CUGAN**:fire: is an AI super resolution model for anime images, trained in a million scale anime dataset, using the same architecture as Waifu2x-CUNet. It supports **2x\3x\4x** super resolving. For different enhancement strength, now 2x Real-CUGAN supports 5 model weights, 3x/4x Real-CUGAN supports 3 model weights.

**Real-CUGAN** packages an executable environment for windows users. GUI and web version are also supported now.<br>

[Update progress](https://github.com/bilibili/ailab/tree/main/Real-CUGAN/README_EN.md#Acknowledgement)<br>
2022-02-07:Windows-GUI/Web versions<br>
2022-02-09:Colab demo file<br>
2022-02-17:[NCNN version](https://github.com/nihui/realcugan-ncnn-vulkan):AMD graphics card users and mobile phone users can use Real-CUGAN now.<br>
2022-02-20:Low memory mode added. Now you can super resolve very large resolution images. You can download 20220220 updated packages to use it.<br>
2022-02-27:Faster low memory mode added, 25% slower than baseline mode; enhancement strength config alpha added. You can download 20220227 updated packages to use it.<br>

If you find Real-CUGAN helpful for your anime videos/projects, please help by starring :star: this repo or sharing it with your friends, thanks! <br>

### 1. Comparison


https://user-images.githubusercontent.com/61866546/152800856-45bdee20-f7c7-443d-9430-f08dc5c805b8.mp4


- **Visual effect comparison**
  <br>
  texture challenge case
  ![compare1](demos/title-compare1.png)
  line challenge case
  ![compare2](demos/compare2.png)
  heavy artifact case
  ![compare3](demos/compare3.png)
  bokeh effect case
  ![compare4](demos/compare4.png)
- **Detailed comparison**

|                | Waifu2x(CUNet)                                               | Real-ESRGAN(Anime6B)                                         | Real-CUGAN                                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| training set         | Private anime training set, unknown magnitude and quality                             | Private anime training set, unknown magnitude and quality                             | Million scale high quality patch training set                                |
| speed(times)    | Baseline                                                     | 2.2x                                                         | 1x                                                           |
| visual effect | can't deblur; artifacts are not completely removed                               | The sharpening strength is the largest. The painting style may be changed. The lines may be incorrectly reconstructed. The bokeh effect region may be forced to be clear.                  | Sharper line style than waifu2x. Bokeh effect region maintained better. Texture preserved better.
| compatibility         | numbers of existing windows-apps，<br />Caffe，PyTorch，NCNN, VapourSynth | PyTorch，VapourSynth，NCNN                       | the same as waifu2x             |
| enhancement        | supports multiple denoise strength                                           |                only support default enhancement strength                            | 5 different enhancement strength for 2x, 3 for 3x, 3 for 4x now |
| SR resolution           | 1x+2x                                               | 4x                                                    | 2x+3x+4x are supported now, and 1x model is training             |

### 2. For windows users
Modify config.py, and double click go.bat to execute Real-CUGAN.
- #### Executable file：
    [BaiduDrive(extract code:ds2a) :link:](https://pan.baidu.com/s/10NbgnusDucllKiE0sgBWvQ)｜[GithubRelease :link:](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN) ｜ [GoogleDrive :link:](https://drive.google.com/drive/folders/1UFgpV14uEAcgYvVw0fJuajzy1k7JIz6H)
- #### System environment：
    - :heavy_check_mark: Tested in windows10 64bit.
    - :heavy_check_mark: CPU with SSE4 and AVX support.
    - :heavy_check_mark: Light version: cuda >= 10.0. 【Heavy version: cuda >= 11.1】
    - :heavy_check_mark: If you use Nvidia cards, 1.5G video memory is needed.
    - :heavy_exclamation_mark: **Note that 30 series nvidia GPU only supports heavy version. Nvidia GPU (which is <2 series) users are recommended to use light version. **

- #### config file：
  #### a. common settings
    - mode: image/video;
    - model_path: the path of model weights (2x model has 4 weight files, 3x/4x model only has 1 weight file);
    - device: cuda device number. If you want to use multiple GPUs to super resolve images, it's recommended to manually divide the tasks into different folders and fill in different cuda device numbers;
    - you need fill in the input and output dir path for image task, and the input and output video file path for video task.
    - half: FP16 inference or FP32 inference. 'True' is recommended.
    - cache_mode: Default 0. Memory needed:0>1>>2=3, speed:0>1(+15%time)>2(+25%time)>3(+150%time). You can super resolve very large resolution images using mode2/3.
    - tile: The bigger the number, less video memory is needed, and lower inference speed it is.
    - alpha: The bigger the number, the enhancement strength is smaller, more blurry the output images are; the smaller the number, the enhancement strength is bigger, more sharpen image will be generated. Default 1 (don't adjust it). Recommended range: (0.75,1.3)

  #### b. settings for video task
    - nt: the threading number of each GPU. If the video memory is enough, >=2 is recommended (for faster inference).
    - n_gpu: the number of GPUs you will use.
    - encode_params: if you don't know how to use ffmpeg, you shouldn't change it.


### 3. Python environment dependencies
:white_check_mark:  **torch>=1.0.0**      <br>
:white_check_mark:  **numpy**             <br>
:white_check_mark:  **opencv-python**     <br>
:white_check_mark:  **moviepy**           <br>

upcunet_v3.py: model file and image inference script <br>
inference_video.py: a simple script for inferencing anime videos using Real-CUGAN.

### 4. For VapourSynth users

Please see [Readme](VapourSynth/README_EN.md)

### 5. realcugan-ncnn-vulkan
[NCNN version](https://github.com/nihui/realcugan-ncnn-vulkan):AMD graphics card users and mobile phone users can use Real-CUGAN now.<br>

### 6. For waifu2x-caffe users

#### We support two weights for waifu2x-caffe users now:
:fire: **Real-CUGAN2x standard version** and :fire: **Real-CUGAN2x no crop line version**
<br>
    [BaiduDrive(extract code:ds2a) :link:](https://pan.baidu.com/s/10NbgnusDucllKiE0sgBWvQ)｜[GithubRelease :link:](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN) ｜ [GoogleDrive :link:](https://drive.google.com/drive/folders/1UFgpV14uEAcgYvVw0fJuajzy1k7JIz6H)
    <br>
    Users can replace the original weights with the new ones (remember to backup the original weights if you want to reuse them), and use the original setting to super resolve images.<br>

:heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark: due to the crop mechanism of waifu2x-caffe, for standard version，big crop_size is recommanded, or the crop line artifact may be caused. If **crop line edge artifact** is found, please use our windows package or the 'no crop line' version. The tile_mode of windows package version is lossless. The 'no edge artifact' version waifu2x-caffe weights may cause more texture loss.

>For developers, it is recommended to use the whole image as input. Pytorch version (tile mode) is recommended if you want the program to require less video memory.

### 7.:european_castle: Model Zoo

You can download the weights from [netdisk links](README_EN.md#2-for-windows-users).

<table>
	<tr>
	    <th align="center"></th>
        <th align="center">1x</th>
	    <th align="center">2x</th>
	    <th align="center">3x/4x</th>  
	</tr >
	<tr>
	    <td align="center" >denoise strength</td>
	    <td align="center">only no denoise version supported, training...</td>
	    <td align="center">no denoise/1x/2x/3x are supported now</td>
        <td align="center">no denoise/3x are supported now, 1x/2x:training...</td>
	</tr>
	<tr>
	    <td  align="center">conservative version</td>
	    <td  align="center">training...</td>
	    <td  colspan="2" align="center">supported now</td>
	</tr>
	<tr>
        <td  align="center">fast model</td>
	    <td  colspan="3" align="center">under investigation</td>
	</tr>
</table>


### 8. TODO：
- [ ]  Lightweight/fast version
- [x]  Low (video card) memory mode
- [x]  Adjustable denoise, deblur, sharpening strength
- [ ]  Super resolve the image to specified resolution end to end
- [ ]  Optimize texture retention and reduce AI processing artifacts
- [x]  Simple GUI

### Acknowledgement
The training code is from but not limited to:[RealESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md).<br>
The original waifu2x-cunet architecture is from:[CUNet](https://github.com/nagadomi/nunif/blob/master/nunif/models/waifu2x/cunet.py).<br>
Update progress:
- Windows GUI (PyTorch version), [Squirrel Anime Enhance v0.0.3](https://github.com/Justin62628/Squirrel-RIFE/releases/tag/v0.0.3)<br>
- [nihui](https://github.com/nihui) achieves RealCUGAN-[NCNN Ver.](https://github.com/nihui/realcugan-ncnn-vulkan). AMD graphics card users and mobile phone users can use Real-CUGAN now.<br>
- [Web service (CPU-PyTorch version)](https://huggingface.co/spaces/mayhug/Real-CUGAN),by [mnixry](https://github.com/mnixry)

Thanks for their contribution!
