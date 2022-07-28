VapourSynth用户使用说明
-------------------------------------------
[中文](README.md) **|** [English](README_EN.md)

2022-07-27

你也可以使用最新版的 [vs-mlrt](https://github.com/AmusementClub/vs-mlrt/releases/latest) 来体验 RealCUGAN。vs-mlrt 支持 OpenVINO，ONNXRuntime 与 TensorRT 等运行时，相较 PyTorch 提供了更优的性能与更低的资源占用；同时也提供了多种 ML 模型（如 [DPIR](https://github.com/cszn/DPIR), [waifu2x](https://github.com/nagadomi/waifu2x) 等）与用户友好的 [Python wrapper](https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py)。vs-mlrt 的二进制发布包 `vsmlrt-windows-x64-cuda` 已含全部依赖，安装简便；你也可以直接下载已经整合好的完整 [VapourSynth Portable 包](https://github.com/AmusementClub/tools/releases/latest)。


######官方pytorch版本   ↓   ######

simple-upcunet.vpy是一个简单的示例脚本

你需要将upcunet_v3_vs.py/upcunet_v20220227_vs.py文件的根目录添加进sys路径

RealWaifuUpScaler初始化参数：
  - scale: 放大倍数
  - weight_path：需要载入模型的参数的路径
  - device: 显卡设备号
  - tile_mode：0/1/x，决定了切块的尺寸，0直接将整张图进行超分，否则先将图像切块，分别超分完再拼接。数字越大,切得越碎，越省显存，速度越低。
20220227版本新增了cache_mode和alpha参数
  - cache_mode: 给低显存用户更多显存/速度权衡，数字越大越慢，越省显存。选2/3模式解锁超大分辨率。<br>
        **0:** baseline <br>
        **1:** 对缓存进行8bit量化节省显存，带来15%延时增长，肉眼完全无法感知的有损模式 <br>
        **2:** 不使用cache，有损模式。耗时约增加25%，仅在有景深虚化的图里有微小的误差，不影响纹理判断 <br>
        **3:** 不使用cache，无损模式。耗时约为默认模式的2.5倍，但是显存不受输入图像分辨率限制，tile填得够大，1.5G显存可超任意分辨率 <br>
  - alpha：修复强度调节。默认为1（不调节），推荐调整区间(0.7,1.3)。该值越小AI修复程度、痕迹越小，越模糊；alpha越大处理越烈，越锐化，色偏（对比度、饱和度增强）越大。

这个包的测试环境为：
  - PyTorch版本1.10.1+cu111
  - Python版本3.9.5
  - VapourSynth版本R57

需要先为Python环境装上PyTorch，版本最好>=1.0.0(30系显卡最好>=1.9.0，A卡不支持)<br>
VapourSynth版本最低R54-API4-test2
