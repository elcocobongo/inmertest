import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"
    if True:
        if os.path.isdir("temp"):
            shutil.rmtree("temp")
        os.makedirs("temp")
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster inference')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to output png format')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Usar ambos GPUs
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key=lambda x: int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left:left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):
    global model
    if model.module.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model(I0, I1, (i + 1) * 1. / (n + 1), args.scale))
        return res
    else:
        middle = model(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n // 2)
        second_half = make_inference(middle, I1, n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img):
    if args.fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left:left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

# Procesamiento en lotes
batch_size = 4  # Ajusta según resolución (2 para 4K, 4 para 1080p)
frame_buffer = [lastframe]
while True:
    frame = read_buffer.get()
    if frame is None:
        break
    frame_buffer.append(frame)

    if len(frame_buffer) >= batch_size + 1 or (frame is None and len(frame_buffer) > 1):
        batch_I0 = []
        batch_I1 = []
        for i in range(len(frame_buffer) - 1):
            I0 = torch.from_numpy(np.transpose(frame_buffer[i], (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = torch.from_numpy(np.transpose(frame_buffer[i + 1], (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I0 = pad_image(I0)
            I1 = pad_image(I1)
            batch_I0.append(I0)
            batch_I1.append(I1)

        batch_I0 = torch.cat(batch_I0, dim=0)
        batch_I1 = torch.cat(batch_I1, dim=0)

        with torch.no_grad():
            if model.module.version >= 3.9:
                output_batch = []
                for i in range(args.multi - 1):
                    interp = model(batch_I0, batch_I1, (i + 1) * 1. / (args.multi), args.scale)
                    output_batch.append(interp)
            else:
                output_batch = make_inference(batch_I0, batch_I1, args.multi - 1)

        for idx in range(len(batch_I0)):
            I0_frame = frame_buffer[idx]
            if args.montage:
                write_buffer.put(np.concatenate((I0_frame, I0_frame), 1))
            else:
                write_buffer.put(I0_frame)
            for mid in [output_batch[i][idx] for i in range(len(output_batch))]:
                mid = (((mid * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w]
                if args.montage:
                    write_buffer.put(np.concatenate((I0_frame, mid), 1))
                else:
                    write_buffer.put(mid)

        frame_buffer = [frame_buffer[-1]]
        pbar.update(len(batch_I0))

if args.montage:
    write_buffer.put(np.concatenate((frame_buffer[-1], frame_buffer[-1]), 1))
else:
    write_buffer.put(frame_buffer[-1])
write_buffer.put(None)

import time
while not write_buffer.empty():
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
