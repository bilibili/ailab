Real-CUGAN for VapourSynth, User Guide
-------------------------------------------

2022-07-27

You can also use the [latest vs-mlrt release](https://github.com/AmusementClub/vs-mlrt/releases/latest) to experience RealCUGAN. vs-mlrt supports OpenVINO, ONNXRuntime and TensorRT runtimes and provides better performance & reduced resource consumption compared to the PyTorth implementation. It also includes some other ML models (e.g. [DPIR](https://github.com/cszn/DPIR), [waifu2x](https://github.com/nagadomi/waifu2x)) and a user friendly [Python wrapper](https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py). vs-mlrt binary release package `vsmlrt-windows-x64-cuda` already includes everything you need to deploy those models; Alternatively, you can also use the pre-integerated [VapourSynth Portable](https://github.com/AmusementClub/tools/releases/latest).

######official pytorch version   ↓   ######

simple-upcunet.vpy is a simple demo.<br>
You should append file the root of "upcunet_v3_vs.py"/"upcunet_v20220227_vs.py" into sys.path<br>
RealWaifuUpScaler init setting：
  - scale: 2x/3x/4x;
  - weight_path：the path of the weights of model;
  - device: cuda device number;
  - tile_mode：0/1/2/3/4/5. The bigger the number, the less video memory is needed, and the lower inference speed it is.<br>
v20220227 add cache_mode and alpha config
  - cache_mode: Default 0. Memory needed:0>1>>2=3, speed:0>1(+15%time)>2(+25%time)>3(+150%time). You can super resolve very large resolution images using mode2/3 (low memory mode).
  - alpha: The smaller the number, the enhancement strength is smaller, more blurry the output images are; the bigger the number, the enhancement strength is bigger, more sharpen image will be generated. Default 1 (don't adjust it). Recommended range: (0.75,1.3)

Tested in (environment):
  - PyTorch==1.10.1+cu111
  - Python==3.9.5
  - VapourSynth==R57

You should install PyTorch using pip, the version of PyTorch should be >=1.0.0 (for 30 series N card: >=1.9.0; A card is not supported)<br>
VapourSynth version: >=R54-API4-test2
