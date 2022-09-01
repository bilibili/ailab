import tempfile
import argparse
from upcunet_v3 import *
import time
import cv2
from time import time as ttime

def main():
    parser = argparse.ArgumentParser(description='RealCUGAN')
    parser.add_argument('-i', type=str, default='in',
                        help='input image path (jpg/png/webp) or directory')
    parser.add_argument('-o', type=str, default='out',
                        help='output image path (jpg/png/webp) or directory')
    parser.add_argument('-n', type=int, default=-1,
                        help='denoise level (-1/0/1/2/3, default=-1)')
    parser.add_argument('-s', type=int, default=2,
                        help='upscale ratio (1/2/3/4, default=2)')
    parser.add_argument('-t', type=int, default=0,
                        help='tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu')
    parser.add_argument('-c', type=int, default=3,
                        help='cache mode')
    parser.add_argument('-x', help='enable tta mode') 
    parser.add_argument(
        '-f', help='output image format (jpg/png/webp, default=ext/png)')
    parser.add_argument(
        '-m', type=str, help='realcugan model path (default=weights_v3)', default='weights_v3')
    parser.add_argument('-d', type=str, default='cuda:0',
                        help='device (default=cuda:0)')
    parser.add_argument('--half', default=True,
                        help='Half mode, CUDA only (default=True)')
    args = parser.parse_args()

    def get_weight(scale: int, denoise: int, pro: bool, fallback=True):
        denoise = 'no-denoise' if denoise == - \
            1 else 'conservative' if denoise == 0 else f'denoise{denoise}x'
        if pro:
            result = f'pro-{denoise}-up{str(scale)}x.pth'
        else:
            result = f'up{str(scale)}x-latest-{denoise}.pth'
        if os.path.exists(os.path.join(args.m, result)):
            return os.path.join(args.m, result), scale
        elif fallback:
            return get_weight(scale, denoise, not pro, False)
        else:
            raise Exception('Model not found')

    weight_path, scale = get_weight(args.s, args.n, True)
    tile_mode = args.t
    cache_mode = args.c
    half = args.half if args.d != 'cpu' else False
    weight_name = weight_path.split("/")[-1].split(".")[0]
    print(f'weight: {weight_name}, scale: {scale}, tile: {tile_mode}, cache: {cache_mode}, half: {half}, device: {args.d}, weight_path: {weight_path}')

    upscaler2x = RealWaifuUpScaler(
        scale, weight_path, half=half, device=args.d)
    input_dir = args.i
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = tempfile.gettempdir()
    for name in os.listdir(input_dir):
        tmp = name.split(".")
        inp_path = os.path.join(input_dir, name)
        suffix = tmp[-1]
        prefix = ".".join(tmp[:-1])
        tmp_path = os.path.join(tmp_dir, "%s.%s" % (
            int(time.time() * 1000000), suffix))
        # print(inp_path, tmp_path)
        # 支持中文路径
        # os.link(inp_path, tmp_path)#win用硬链接
        os.symlink(os.path.join(root_path, inp_path), tmp_path)  # linux用软链接
        frame = cv2.imread(tmp_path)[:, :, [2, 1, 0]]
        t0 = ttime()
        result = upscaler2x(frame, tile_mode=tile_mode,
                            cache_mode=cache_mode, alpha=1)[:, :, ::-1]
        t1 = ttime()
        print(prefix, "done", t1 - t0, "tile_mode=%s" %
              tile_mode, cache_mode)
        tmp_opt_path = os.path.join(tmp_dir, "%s.%s" % (
            int(time.time() * 1000000), suffix))
        cv2.imwrite(tmp_opt_path, result)
        n = 0
        while (1):
            if (n == 0):
                suffix = "_%sx_tile%s_cache%s_alpha%s_%s.png" % (
                    scale, tile_mode, cache_mode, 1, weight_name)
            else:
                suffix = "_%sx_tile%s_cache%s_alpha%s_%s_%s.png" % (
                    scale, tile_mode, cache_mode, 1, weight_name, n)
            if (os.path.exists(os.path.join(output_dir, prefix + suffix)) == False):
                break
            else:
                n += 1
        final_opt_path = os.path.join(output_dir, prefix + suffix)
        os.rename(tmp_opt_path, final_opt_path)
        os.remove(tmp_path)


if __name__ == '__main__':
    exit(main())
