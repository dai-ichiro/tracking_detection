import os
import sys
import glob

import cv2
import torch

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from gdown import download
model_folder = os.path.join('experiments', 'siamrpn_r50_l234_dwxcorr')
pth_file = os.path.join(model_folder, 'model.pth')

if not os.path.isfile(pth_file):
    download('https://drive.google.com/uc?id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH', pth_file, quiet = False)

argv_list = sys.argv
del argv_list[0]

class_num = len(sys.argv)

print('class count = %d'%class_num)

video_list = []
target_name = []

for i, each_class in enumerate(argv_list):
    if os.path.isdir(each_class):
        classname_without_ext = os.path.basename(each_class)
        print('name of class%d: %s'%(i, classname_without_ext))
        target_name.append(classname_without_ext)
        video_list.append(glob.glob(os.path.join(each_class, '*')))
    else:
        classname_without_ext = os.path.splitext(os.path.basename(each_class))[0]
        print('name of class%d: %s'%(i, classname_without_ext))
        target_name.append(classname_without_ext)
        video_list.append([each_class])

for i, video in enumerate(video_list):
    print('video of class%d: '%i + ','.join(video))

out_path = 'train_data'

config = 'experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
snapshot = 'experiments/siamrpn_r50_l234_dwxcorr/model.pth'
#=========================================================

train_images_dir = os.path.join(out_path, 'images', 'train')
train_labels_dir = os.path.join(out_path, 'labels', 'train')

os.makedirs(train_images_dir)
os.makedirs(train_labels_dir)

init_rect_list = []

for videos in video_list:
    init_rect_list_each_class = []
    for video in videos:

        cap = cv2.VideoCapture(video)
        ret, img = cap.read()
        cap.release()

        source_window = "draw_rectangle"
        cv2.namedWindow(source_window)
        rect = cv2.selectROI(source_window, img, False, False)

        init_rect_list_each_class.append(rect)
        cv2.destroyAllWindows()
    init_rect_list.append(init_rect_list_each_class)

# モデルを取得する
cfg.merge_from_file(config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
device = torch.device('cuda' if cfg.CUDA else 'cpu')
model = ModelBuilder()
model.load_state_dict(torch.load(snapshot,
    map_location=lambda storage, loc: storage.cpu()))
model.eval().to(device)
tracker = build_tracker(model)

for class_index, videos in enumerate(video_list):
    for video_index, video in enumerate(videos):
        # 映像ファイルを読み込む
        video_frames = []
        cap = cv2.VideoCapture(video)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        while(True):
            ret, img = cap.read()
            if not ret:
                break
            video_frames.append(img)
        cap.release()

        #トラッキングを実行
        for ind, frame in enumerate(video_frames):
            if ind == 0:
                tracker.init(frame, init_rect_list[class_index][video_index])
                bbox = init_rect_list[class_index][video_index]
            else:
                outputs = tracker.track(frame)
                bbox = outputs['bbox']
                
            filename = '%d_%d_%06d'%(class_index, video_index,ind)

            #画像の保存
            jpeg_filename = filename + '.jpg'
            cv2.imwrite(os.path.join(train_images_dir, jpeg_filename), frame)

            #ラベルテキストの保存
            txt_filename= filename + '.txt'
            with open(os.path.join(train_labels_dir, txt_filename), 'w') as f:
                center_x = (bbox[0] + bbox[2] / 2) / w
                center_y = (bbox[1] + bbox[3] / 2) / h
                width = bbox[2] / w
                height = bbox[3] / h
                f.write('%d %f %f %f %f'%(i, center_x, center_y, width, height))

with open('train.yaml', 'w', encoding='cp932') as f:
    f.write('path: %s'%out_path)
    f.write('\n')
    f.write('train: images/train')
    f.write('\n')
    f.write('val: images/train')
    f.write('\n')
    f.write('nc: %d'%len(target_name))
    f.write('\n')
    f.write('names: ')
    f.write('[')
    output_target_name = ['\'' + x + '\'' for x in target_name]
    f.write(', '.join(output_target_name))
    f.write(']')
