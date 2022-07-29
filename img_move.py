import pandas as pd
import shutil
import os

save_path = 'img_flowers'
img_path = r'flowers_google'
img_path = os.path.abspath(img_path)
df = pd.read_csv('flowers_idx.csv')
cls = list(set(df['flower_cls']))
if not os.path.exists(save_path):
    os.mkdir(save_path)

for c in cls:
    imgs_t = df[df['flower_cls'] == c]['id']
    imgs_t = [str(i) for i in imgs_t]
    cls_path = os.path.join(save_path, c.strip())
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
    for img in imgs_t:
        shutil.copy(os.path.join(img_path, img + '.jpeg'), os.path.join(cls_path, img + '.jpeg'))
        print(img, 'is copied')
    print(c, 'is copied\n---------------')
