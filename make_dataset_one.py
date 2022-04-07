import os
import sys
import cv2
import torch

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from gdown import download
from torchvision.datasets.utils import download_url

model_folder = os.path.join('experiments', 'siamrpn_r50_l234_dwxcorr')
pth_file = os.path.join(model_folder, 'model.pth')

if not os.path.isfile(pth_file):
    download('https://drive.google.com/uc?id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH', pth_file, quiet = False)

video_name = sys.argv[1]

target_name = "hero"

out_path = 'train_data'

config = 'experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
snapshot = 'experiments/siamrpn_r50_l234_dwxcorr/model.pth'

train_images_dir = os.path.join(out_path, 'images', 'train')
train_labels_dir = os.path.join(out_path, 'labels', 'train')

os.makedirs(train_images_dir)
os.makedirs(train_labels_dir)

# 映像ファイルを読み込む
video_frames = []
cap = cv2.VideoCapture(video_name)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
while(True):
    ret, img = cap.read()
    if not ret:
        break
    video_frames.append(img)
cap.release()

# モデルを取得する
cfg.merge_from_file(config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
device = torch.device('cuda' if cfg.CUDA else 'cpu')
model = ModelBuilder()
model.load_state_dict(torch.load(snapshot,
    map_location=lambda storage, loc: storage.cpu()))
model.eval().to(device)
tracker = build_tracker(model)

#最初の位置を取得する
source_window = "draw_rectangle"
cv2.namedWindow(source_window)
init_rect = cv2.selectROI(source_window, video_frames[0], False, False)
cv2.destroyAllWindows()

#トラッキングを実行
jpeg_filenames_list = []

for ind, frame in enumerate(video_frames):
    if ind == 0:
        tracker.init(frame, init_rect)
        bbox = init_rect
    else:
        outputs = tracker.track(frame)
        bbox = outputs['bbox']
        
    filename = '%06d'%(ind)

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
        f.write('0 %f %f %f %f'%(center_x, center_y, width, height))

with open('train.yaml', 'w', encoding='cp932') as f:
    f.write('path: %s'%out_path)
    f.write('\n')
    f.write('train: images/train')
    f.write('\n')
    f.write('val: images/train')
    f.write('\n')
    f.write('nc: 1')
    f.write('\n')
    f.write('names: ')
    f.write('[')
    f.write('\'' + target_name + '\'')
    f.write(']')