import argparse
import os
from tqdm import tqdm

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS


# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk
def save_cropped_objects(output_dir, image_path, masks, boxes, labels):
    """
    根据bounding box和mask信息裁剪并保存图像的每个部分。
    """
    # 打开原始图像
    image = cv2.imread(image_path)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个对象的bounding box信息
    for idx, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        # 将mask转换为uint8并应用于原始图像
        mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        cropped_image = cv2.bitwise_and(image, image, mask=mask)

        # 转换bounding box数据格式，裁剪图像
        x0, y0, x1, y1 = box.int().numpy()
        # cropped_obj = image[y0:y1, x0:x1]
        cropped_obj = cropped_image[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]
        if cropped_obj.size == 0:
            print(f"Cropped image is empty at path: {image_path}. Skipping label: {label}")
            continue
        # 保存裁剪后的图像
        name, _ = label.split('(')
        obj_filename=os.path.basename(image_path)[:-4]+"_"+name.strip()+"_"+str(x0)+"_"+str(y0)+"_"+str(x1)+"_"+str(y1)+".png"
        # obj_filename = f"{name.strip()}_{idx}.png"
        try:
            cv2.imwrite(os.path.join(output_dir, obj_filename), cropped_obj)
            cv2.imwrite(os.path.join(output_dir, obj_filename[:-4]+"_mask.png"), cropped_mask)
        except Exception as e:
            print(f"Error saving cropped object: {e}")
            continue

        # print(f"Saved: {obj_filename}")
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    # print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask': [{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--input_image_dir", type=str, required=True, help="path to image dir")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    # image_path = args.input_image
    image_dir = args.input_image_dir
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    # processed_images=os.listdir(output_dir)
    # saved_images=[]
    # for image in processed_images:
    #     if image.split("_")[0]+".jpg" not in saved_images:
    #         saved_images.append(image.split("_")[0]+".jpg")
    # print(len(saved_images))
    # processed_images=[]
    # ChatGPT or nltk is required when using tags_chineses
    # openai.api_key = openai_key
    # if openai_proxy:
    # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])

    # load model
    ram_model = ram(pretrained=ram_checkpoint,
                    image_size=384,
                    vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()

    ram_model = ram_model.to(device)
    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    # load image
    images=os.listdir(image_dir)
    n=0
    for i in tqdm(range(n,len(images))):
        # print(str(i)+"/"+str(len(images)))
        image_path=os.path.join(image_dir,images[i])
        image_pil, image = load_image(image_path)
        raw_image = image_pil.resize(
            (384, 384))
        raw_image = transform(raw_image).unsqueeze(0).to(device)

        res = inference_ram(raw_image, ram_model)

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        tags = res[0].replace(' |', ',')
        tags_chinese = res[1].replace(' |', ',')

        # print("Image Tags: ", res[0])
        # print("图像标签: ", res[1])

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, tags, box_threshold, text_threshold, device=device
        )
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        # print(f"After NMS: {boxes_filt.shape[0]} boxes")
        tags_chinese = check_tags_chinese(tags_chinese, pred_phrases)
        # print(f"Revise tags_chinese with number: {tags_chinese}")
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        try:
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )
        except RuntimeError:
            print(f"WARNING: Empty masks for image {image_path}. Skipping.")
            continue
        save_cropped_objects(output_dir, image_path, masks, boxes_filt, pred_phrases)
    # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)
    #
    # # plt.title('RAM-tags' + tags + '\n' + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
    # plt.axis('off')
    # plt.savefig(
    #     os.path.join(output_dir, "automatic_label_output.jpg"),
    #     bbox_inches="tight", dpi=300, pad_inches=0.0
    # )
    #
    # save_mask_data(output_dir, tags_chinese, masks, boxes_filt, pred_phrases)
