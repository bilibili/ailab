VapourSynth用户使用说明
-------------------------------------------
[中文](README.md) **|** [English](README_EN.md)

simple-upcunet.vpy是一个简单的示例脚本<br>
你需要将upcunet_v3_vs.py文件的根目录添加进sys路径<br>
RealWaifuUpScaler初始化参数：
  - scale: 放大倍数
  - weight_path：需要载入模型的参数的路径
  - device: 显卡设备号
  - tile_mode：0/1/2/3/4，决定了切块的尺寸，0直接将整张图进行超分，1~4先将图像切块，分别超分完再拼接。数字越大越省显存速度越低。


这个包的测试环境为：
  - PyTorch版本1.10.1+cu111
  - Python版本3.9.5
  - VapourSynth版本R57

需要先为Python环境装上PyTorch，版本最好>=1.0.0(30系显卡最好>=1.9.0，A卡不支持)<br>
VapourSynth版本最低R54-API4-test2
