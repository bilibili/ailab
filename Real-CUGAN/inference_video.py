import threading,cv2,torch,os
from random import uniform
from multiprocessing import Queue
import multiprocessing
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
from upcunet_v3 import RealWaifuUpScaler
from time import time as ttime,sleep
class UpScalerMT(threading.Thread):
    def __init__(self, inp_q, res_q, device, model,p_sleep,nt,tile,cache_mode,alpha):
        threading.Thread.__init__(self)
        self.device = device
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.cache_mode=cache_mode
        self.nt = nt
        self.p_sleep=p_sleep
        self.tile=tile
        self.alpha = alpha

    def inference(self, tmp):
        idx, np_frame = tmp
        with torch.no_grad():
            res = self.model(np_frame,self.tile,self.cache_mode,self.alpha)
        if(self.nt>1):
            sleep(uniform(self.p_sleep[0],self.p_sleep[1]))
        return (idx, res)

    def run(self):
        while (1):
            tmp = self.inp_q.get()
            if (tmp == None):
                # print("exit")
                break
            self.res_q.put(self.inference(tmp))
class VideoRealWaifuUpScaler(object):
    def __init__(self,nt,n_gpu,scale,half,tile,cache_mode,alpha,p_sleep,decode_sleep,encode_params):
        self.nt = nt
        self.n_gpu = n_gpu  # 每块GPU开nt个进程
        self.scale = scale
        self.encode_params = encode_params
        self.decode_sleep=decode_sleep

        device_base = "cuda"
        self.inp_q = Queue(self.nt * self.n_gpu * 2)  # 抽帧缓存上限帧数
        self.res_q = Queue(self.nt * self.n_gpu * 2)  # 超分帧结果缓存上限
        for i in range(self.n_gpu):
            device = device_base + ":%s" % i
            #load+device初始化好当前卡的模型
            model=RealWaifuUpScaler(self.scale, eval("model_path%s" % self.scale), half, device)
            for _ in range(self.nt):
                upscaler = UpScalerMT(self.inp_q, self.res_q, device, model,p_sleep,self.nt,tile,cache_mode,alpha)
                upscaler.start()

    def __call__(self, inp_path,opt_path):
        objVideoreader = VideoFileClip(filename=inp_path)
        w,h=objVideoreader.reader.size
        fps=objVideoreader.reader.fps
        total_frame=objVideoreader.reader.nframes
        if_audio=objVideoreader.audio
        if(if_audio):
            tmp_audio_path="%s.m4a"%inp_path
            objVideoreader.audio.write_audiofile(tmp_audio_path,codec="aac")
            writer = FFMPEG_VideoWriter(opt_path, (w * self.scale, h * self.scale), fps, ffmpeg_params=self.encode_params,audiofile=tmp_audio_path)  # slower#medium
        else:
            writer = FFMPEG_VideoWriter(opt_path, (w * self.scale, h * self.scale), fps, ffmpeg_params=self.encode_params)  # slower#medium
        now_idx = 0
        idx2res = {}
        t0 = ttime()
        for idx, frame in enumerate(objVideoreader.iter_frames()):
            # print(1,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
            if(idx%10==0):
                print("total frame:%s\tdecoded frames:%s"%(int(total_frame),idx))  ##
            self.inp_q.put((idx, frame))
            sleep(self.decode_sleep)#否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
            while (1):  # 取出处理好的所有结果
                if (self.res_q.empty()): break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            # if (idx % 100 == 0):
            while (1):  # 按照idx排序写帧
                if (now_idx not in idx2res): break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
        idx+=1
        while (1):
            # print(2,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
            # if (now_idx >= idx + 1): break  # 全部帧都写入了，跳出
            while (1):  # 取出处理好的所有结果
                if (self.res_q.empty()): break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            while (1):  # 按照idx排序写帧
                if (now_idx not in idx2res): break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
            if(self.inp_q.qsize()==0 and self.res_q.qsize()==0 and idx==now_idx):break
            sleep(0.02)
        # print(3,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
        for _ in range(self.nt * self.n_gpu):  # 全部结果拿到后，关掉模型线程
            self.inp_q.put(None)
        writer.close()
        if(if_audio):
            os.remove(tmp_audio_path)
        t1 = ttime()
        print(inp_path,"done,time cost:",t1 - t0)

if __name__ == '__main__':
    from config import half, model_path2, model_path3, model_path4, tile, scale, device, encode_params, p_sleep, decode_sleep, nt, n_gpu,cache_mode,alpha
    inp_path = "432126871-clip6s.mp4"
    opt_path = "432126871-clip6s-2x.mp4"
    video_upscaler=VideoRealWaifuUpScaler(nt,n_gpu,scale,half,tile,cache_mode,alpha,p_sleep,decode_sleep,encode_params)
    video_upscaler(inp_path,opt_path)
