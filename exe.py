from gdown import download

import os

model_folder = os.path.join('experiments', 'siamrpn_r50_l234_dwxcorr')
pth_file = os.path.join(model_folder, 'model.pth')

if not os.path.isfile(pth_file):
    download('https://drive.google.com/uc?id=17cwy6tNR0N2eftNZDIG2JAl79D9LUgoa', pth_file, quiet = False)

