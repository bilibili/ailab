Real Cascade U-Nets for Anime Image Super Resolution
-------------------------------------------

:fire: **Real-CUGAN**:fire: is an AI super resolution model for anime images, trained in a million scale anime dataset, using the same architecture as Waifu2x-CUNet. It supports **2x\3x\4x** super resolving. For different enhancement strength, now 2x Real-CUGAN supports 5 model weights, 3x/4x Real-CUGAN supports 3 model weights.

**Real-CUGAN** package an executable environment for windows users. GUI will be supported in the future.

### 1. Comparasion




https://user-images.githubusercontent.com/61866546/148071244-90293316-21cf-43b9-a81b-e6f7abe113d4.mp4




- **visual effect comparasion**
  <br>
  texture chanllenge case
  ![compare1](demos/title-compare1.png)
  line chanllenge case
  ![compare2](demos/compare2.png)
  heavy artifact case
  ![compare3](demos/compare3.png)
  bokeh effect case
  ![compare4](demos/compare4.png)
- **Detailed comparasion**

|                | Waifu2x(CUNet)                                               | Real-ESRGAN(Anime6B)                                         | Real-CUGAN                                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| traning set         | Private anime training set, unknown magnitude and quality                             | Private anime training set, unknown magnitude and quality                             | Million scale high quality patch training set                                |
| speed(times)    | Baseline                                                     | 2.2x                                                         | 1x                                                           |
| visual effect | can't deblur; artifacts are not completely removed                               | The sharpening strength is the largest. The painting style may be changed. The lines may be incorrectly reconstructed. The bokeh effect region may be forced to be clear.                  | Sharper line style than waifu2x. Bokeh effect region maintained better. Texture preserved better.
| compatibility         | numbers of existing windows-apps，<br />Caffe，PyTorch，NCNN, VapourSynth | PyTorch，VapourSynth，NCNN                       | the same as waifu2x             |
| enhancement        | supports multiple denoise strength                                           |                only support default enhancement strength                            | 5 different enhancement strength for 2x, 3 for 3x, 3 for 4x now |
| SR resolution           | 1x+2x                                               | 4x                                                    | 2x+3x+4x are supported now, and 1x model is training             |

### 2. For windows users
modify config.py, and double click go.bat to execute Real-CUGAN.
- #### Executable file：
    [BaiduDrive(extract code:ds2a) :link:](https://pan.baidu.com/s/10NbgnusDucllKiE0sgBWvQ)｜[GithubRelease :link:](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN)｜[HecaiDrive(extract code:tX4O) :link:](https://caiyun.139.com/m/i?015CHcCjUh9SL)｜ [GoogleDrive :link:](https://drive.google.com/drive/folders/1UFgpV14uEAcgYvVw0fJuajzy1k7JIz6H)
- #### System environment：
    - :heavy_check_mark: Tested in windows10 64bit.
    - :heavy_check_mark: Light version: cuda >= 10.0. 【Heavy version: cuda >= 11.1】
    - :heavy_exclamation_mark: **Note that 30 series nvidia GPU only supports heavy version.**

- #### config file：
  #### a. common settings
    - mode: image/video;
    - model_path: the path of model weights(2x model has 4 weight files, 3x/4x model only has 1 weight file);
    - device: cuda device number. If you want to use multiple GPUs to super resolve images, it's recommanded to manually divide the tasks into different folders and fill in different cuda device numbers;
    - you need fill in the input and output dir path for image task, and the input and output video file path for video task.

  #### b. settings for video task
    - nt: the threading number of each GPU. If the video memory is enough, >=2 is recommended (for faster inference).
    - n_gpu: the number of GPUs you will use.
    - encode_params: if you don't know how to use ffmpeg, you shouldn't change it.
    - half: FP16 inference or FP32 inference. 'True' is recommended.
    - tile: 0~4 is supported. The bigger the number, the less video memory is needed, and the lower inference speed it is.

### 3. For waifu2x-caffe users

#### We support two weights for waifu2x-caffe users now:
:fire: **Real-CUGAN2x standard version** and :fire: **Real-CUGAN2x no crop line version**
<br>
    [BaiduDrive(extract code:ds2a) :link:](https://pan.baidu.com/s/10NbgnusDucllKiE0sgBWvQ)｜[GithubRelease :link:](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN)｜[HecaiDrive(extract code:tX4O) :link:](https://caiyun.139.com/m/i?015CHcCjUh9SL)｜ [GoogleDrive :link:](https://drive.google.com/drive/folders/1UFgpV14uEAcgYvVw0fJuajzy1k7JIz6H)
    <br>
    Users can replace the original weights with the new ones (remember to backup the original weights if you want to reuse them), and use the original setting to super resolve images.<br>

:heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark: due to the crop mechanism of waifu2x-caffe, for standard version，big crop_size is recommanded, or the crop line artifact may be caused. If **crop line edge artifact** is found, please use our windows package or the 'no crop line' version. The tile_mode of windows package version is lossless. The 'no edge artifact' version waifu2x-caffe weights may cause more texture loss.

>For developers, it is recommended to use the whole image as input. Pytorch version (tile mode) is recommended if you want the program to require less video memory.


### 4. Python environment dependence
:white_check_mark:  **torch>=1.0.0**      <br>
:white_check_mark:  **numpy**             <br>
:white_check_mark:  **opencv-python**     <br>
:white_check_mark:  **moviepy**           <br>

upcunet_v3.py:model file and image inference script <br>
inference_video.py: a simple script for inferencing anime videos using Real-CUGAN

### 5. For VapourSynth users

Please see [Readme](VapourSynth/README_EN.md)

### 6.:european_castle: Model Zoo

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


### 7. TODO：
- [ ]  Lightweight/fast version
- [ ]  Adjustable denoise, deblock, deblur, sharpening strength
- [ ]  Super resolve the image to specified resolution end to end
- [ ]  Optimize texture retention and reduce AI processing artifacts
- [ ]  Simple GUI

### 8. Acknowledgement
The training code is from but not limited to: :star2: [RealESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/Training.md):star2: . <br>

The original waifu2x-cunet architecture is from: :star2: [CUNet](https://github.com/nagadomi/nunif/blob/master/nunif/models/waifu2x/cunet.py):star2: .


