import vapoursynth as vs
import sys
sys.path.append(r"C:\Program Files\VapourSynth\core\plugins\Scripts")
import mvsfunc as mvf
sys.path.append(r"C:\codes\pytorch-test")#it should countain upcunet_v3_vs.py/upcunet_v20220227_vs.py
from upcunet_v3_vs import RealWaifuUpScaler
from vapoursynth import core
core.num_threads = 16
core.max_cache_size = 1000

#v3
#upscaler=RealWaifuUpScaler(2,r"C:\codes\pytorch-test\pytorch-test\weights_v3\up2x-latest-no-denoise.pth",True,"cuda:0",0)#scale-weight_path-half-device-tile_mode
#v20220227
upscaler=RealWaifuUpScaler(2,r"C:\codes\pytorch-test\pytorch-test\weights_v3\up2x-latest-no-denoise.pth",True,"cuda:0",5,3,1)#scale-weight_path-half-device-tile_mode-cache_mode-alpha
sourcename = r"C:\codes\pytorch-test\东之伊甸4raw-clip10s.mp4"

#source = core.avisource.AVISource(sourcename)
#source= core.lsmas.LWLibavSource(sourcename)
source = core.ffms2.Source(sourcename)

#source = core.resize.Spline36(source, format = vs.RGBS, matrix_in_s="709")
source = mvf.ToRGB(source,depth=32)
source = upscaler(source)
source= mvf.ToYUV(source,depth=16,css="444")

source.set_output()
