#超分倍率
scale=3

#参数路径，可更换
model_path2 = "weights_v3/up2x-latest-no-denoise.pth"
# model_path2 = "weights_v3/up2x-latest-denoise3x.pth"e
model_path3 = "weights_v3/up3x-latest-denoise3x.pth"
model_path4 = "weights_v3/up4x-latest-denoise3x.pth"

#超分模式，视频or图像文件夹
mode="image"#video#image

#早期显卡开半精度不会提速，但是开半精度可以省显存。
half=True
#tile分为0~4一共5个mode。0在推理时不对图像进行切块，最占内存，mode越提升越省显存，但是可能会降低GPU利用率，降低推理速度
tile=3

#超图像设置
device="cuda:0"#0代表卡号，多卡的话可以写不同config并行开，显存多的话一张卡也可以开多个
input_dir="to-test1"#输入图像路径
output_dir="to-test1-output3x"#超分图像输出路径

#超视频设置
inp_path="../东之伊甸4raw-clip10s.mp4"
opt_path="../东之伊甸4raw-clip10s-2x.mp4"
#线程数：6G显存<=720P可写2,6G显存+1080P写1,12G可写2，24G可写4，边缘显存量爆显存降低线程数
nt=2
#显卡数
n_gpu=1
#别乱动
p_sleep=(0.005,0.012)
decode_sleep=0.002
#编码参数，不懂别乱动;通俗来讲，crf变低=高码率高质量，slower=低编码速度高质量+更吃CPU，CPU不够应该调低级别，比如slow，medium，fast，faster
encode_params=['-crf', '18', '-preset', 'medium']
