import os
import cv2
import torch

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from gdown import download
from torchvision.datasets.utils import download_url

#=========================================================
video_list = ['target.mp4', 'green.mp4']

url_1 = 'https://github.com/dai-ichiro/robo-one/raw/main/video_1.mp4'
url_2 = 'https://github.com/dai-ichiro/robo-one/raw/main/video_2.mp4'

download_url(url_1, root = '.', filename = video_list[0])
download_url(url_2, root = '.', filename = video_list[1])

model_folder = os.path.join('experiments', 'siamrpn_r50_l234_dwxcorr')
pth_file = os.path.join(model_folder, 'model.pth')

if not os.path.isfile(pth_file):
    download('https://drive.google.com/uc?id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH', pth_file, quiet = False)

target_name = [x.split('.')[0] for x in video_list]

out_path = 'train_data'

config = 'experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
snapshot = 'experiments/siamrpn_r50_l234_dwxcorr/model.pth'
#=========================================================

train_images_dir = os.path.join(out_path, 'images', 'train')
train_labels_dir = os.path.join(out_path, 'labels', 'train')

os.makedirs(train_images_dir)
os.makedirs(train_labels_dir)

init_rect_list = []

for video in video_list:
    cap = cv2.VideoCapture(video)
    ret, img = cap.read()
    cap.release()

    source_window = "draw_rectangle"
    cv2.namedWindow(source_window)
    rect = cv2.selectROI(source_window, img, False, False)

    init_rect_list.append(rect)
    cv2.destroyAllWindows()

# モデルを取得する
cfg.merge_from_file(config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
device = torch.device('cuda' if cfg.CUDA else 'cpu')
model = ModelBuilder()
model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
model.eval().to(device)
tracker = build_tracker(model)

for i, video in enumerate(video_list):
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
    jpeg_filenames_list = []

    for ind, frame in enumerate(video_frames):
        if ind == 0:
            tracker.init(frame, init_rect_list[i])
            bbox = init_rect_list[i]
        else:
            outputs = tracker.track(frame)
            bbox = outputs['bbox']
            
        filename = '%d%06d'%((i+1),ind)

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
    f.write('nc: %d'%len(video_list))
    f.write('\n')
    f.write('names: ')
    f.write('[')
    output_target_name = ['\'' + x + '\'' for x in target_name]
    f.write(', '.join(output_target_name))
    f.write(']')

