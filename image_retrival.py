import numpy as np
import scipy.signal as signal
import torch
import clip
from PIL import Image
import time
import cv2
import os
import h5py
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import math
def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.JPG')or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.ppm')]
def extract_features(dataset_dir,h5_path):
    # os.makedirs(h5_dir_path, exist_ok=True)
    start = time.perf_counter()
    print("available_models():", clip.available_models())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)# pretrained-models are saved in /home/yyf/.cache/clip
    print("Loading CLIP model done! Time:", time.perf_counter() - start)
    # dataset_dir = "../../datasets/plagiarism posters/Plagiarized_dataset/database_posters"
    imgs_path = get_imgs_list(dataset_dir)
    # print(imgs_path)
    all_features = []
    start = time.perf_counter()
    with torch.no_grad():
        for i in tqdm(range(len(imgs_path))):
            img_name = os.path.basename(imgs_path[i])
            img_data = cv2.imread(imgs_path[i])
            img_data = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
            img_data = preprocess(img_data)
            image = [img_data]
            image = torch.stack(image, dim=0).to(device)  # torch.Size([1, 3, 224, 224])
            image_feature = model.encode_image(image).squeeze().cpu()      # torch.Size([1, 512])
            all_features.append(image_feature)
    print("Extracting CLIP features done! Time:", time.perf_counter() - start)
    all_features = torch.stack(all_features, dim=0)    # torch.Size([N, 512])
    file_name = os.path.basename(dataset_dir) + "_clip_features_" + str(all_features.shape[0])+".h5"
    # h5_path = os.path.join(h5_dir_path,file_name)
    h5f = h5py.File(h5_path, 'w')
    h5f.create_dataset('clip_features', data=all_features.cpu())
    h5f.create_dataset('imgs_path', data=imgs_path)
    h5f.close()
def rank_together(list_A, list_B):
    temp_zip = zip(list_A, list_B)
    sorted_zip = sorted(temp_zip, key=lambda x: x[1], reverse=True)
    result = zip(*sorted_zip)
    sorted_A, sorted_B = [list(x) for x in result]
    return sorted_A, sorted_B
def image_retrival(h5_path,pic_dir_path):
    results={}
    # os.makedirs(retrival_dir_path, exist_ok=True)
    print("available_CLIP_models():", clip.available_models())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)  # pretrained-models are saved in /home/yyf/.cache/clip
    print("Loading CLIP model done!")
    # h5_path = "./features/database_posters_clip_features_5100.h5"
    h5f = h5py.File(h5_path, 'r')
    clip_feats = torch.tensor(h5f['clip_features'][:]).cuda()       #  torch.Size([N, 512])
    imgs_path = h5f['imgs_path'][:]     # (N,)
    h5f.close()
    top_k = 5
    for pic_path in tqdm(os.listdir(pic_dir_path)):
        print("Processing query", pic_path)
        query_path=os.path.join(pic_dir_path,pic_path)
        query_data_cv2 = cv2.imread(query_path)
        query_data = Image.fromarray(cv2.cvtColor(query_data_cv2, cv2.COLOR_BGR2RGB))
        query_data = preprocess(query_data)
        # possible other patches obtained from the query
        queries_data = [query_data]
        queries_data = torch.stack(queries_data, dim=0).to(device)  # torch.Size([B, 3, 224, 224])

        with torch.no_grad():
            query_feats = model.encode_image(queries_data)      # torch.Size([1, 512])
        similarities = torch.cosine_similarity(clip_feats, query_feats).cpu().tolist()  # torch.Size([N])
        sorted_imgs, sorted_sim = rank_together(imgs_path, similarities)

        paths_list = [each_name.decode() for each_name in sorted_imgs]
        top_k_path = paths_list[:top_k]
        top_k_sim = sorted_sim[:top_k]
        results[pic_path]=[top_k_path,top_k_sim]
        # print(results)
    return results
def save_as_json(results, json_results_path):
    with open(json_results_path, "w") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


def plot_imgs_path(query_path, sorted_top_path, sorted_top_sim, show_n_final, save_path):
    row = 1
    column = math.ceil(show_n_final / row)
    # 基于图片大小自适应调整的字体大小和厚度
    def get_font_scale_and_thickness(image):
        (height, width, _) = image.shape
        # 根据图片宽度进行缩放
        scale = width / 1000
        font_scale = min(3, max(1, 1.5*scale))  # 字体大小在0.5到2之间变化
        thickness = max(1, int(3 * scale))  # 厚度至少为1，随着图片宽度增加而增加
        return font_scale, thickness
    plt.figure(dpi=300)

    # 查询图片的路径组装和标注
    query_path = os.path.join("F:/project_data/outputs0112", query_path)
    query_img = cv2.imread(query_path)
    font_scale, thickness = get_font_scale_and_thickness(query_img)
    text = os.path.basename(query_path)
    cv2.putText(query_img, text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    plt.subplot(row, column, 1)
    plt.imshow(query_img)
    plt.xticks([])
    plt.yticks([])

    # 相似图片的路径组装和标注
    for i, (path, sim) in enumerate(zip(sorted_top_path, sorted_top_sim)):
        img = cv2.imread(path)
        font_scale, thickness = get_font_scale_and_thickness(img)
        text = os.path.basename(path) + str(round(sim, 4))
        cv2.putText(img, text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(row, column, i + 2)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
dataset_dir="F:/project_data/mask"#山海经、面具啥的图片  目前的路径就是你们重新收集的山海经图片的路径  如果还是做这个不用改
h5_path="F:/project_data/image_features_mask_new0112.h5"#随便改一个和原来不一样的名字
pic_dir_path="F:/project_data/outputs0112"#截屏分割出来的图片路径   要从上一步输出的文件夹复制过来
json_resuklts_path="F:/project_data/results_mask_new0112.json"#随便改
retrival_image_dir="F:/project_data/retrival_images_mask_new0112"#随便改  输出图片在这里
os.makedirs(retrival_image_dir,exist_ok=True)
extract_features(dataset_dir,h5_path)
results=image_retrival(h5_path,pic_dir_path)
save_as_json(results,json_resuklts_path)
with open(json_resuklts_path, 'r') as f:
    results = json.load(f)
save_paths = os.listdir(retrival_image_dir)
for query_path in results.keys():
    if query_path not in save_paths:
        print(query_path)
        # continue
        sorted_top_path = results[query_path][0]
        sorted_top_sim = results[query_path][1]
        show_n_final = 6
        save_path = os.path.join(retrival_image_dir, query_path)
        plot_imgs_path(query_path, sorted_top_path, sorted_top_sim, show_n_final, save_path)

