# AI Errors as Catalysts: Human-AI Synergy Sparks Creative Evolution
Here we share our code and dataset~
## Code
We use [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) for object detection of Vlog screenshots. According to the requirements of our project, the automatic_label_ram_demo.py has been modified accordingly，You need to move the modified version to the original project and configure the required environment.
```bash
export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image_dir "your_path" \
  --output_dir your_path" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
```
Image retrieval utilizes the CLIP model to extract features from images to be retrieved and database images, calculate cosine similarity, and return a list of similar images.
```python
python image_retrival.py
```
## Dataset
First, we constructed datasets for celestial bodies, mythical creatures, and masks based on the Taoist perspectives of Heaven, Earth, and Human.
It encompasses the origin of the universe, the relationship between humans and nature, moral and ethical concepts.  Mythical creatures dataset was generated using midjourny, and the other two datasets were collected from the network can be downloaded from [here](https://drive.google.com/drive/folders/1ptWDjc0I8999jC9eaoNphiGheYBHpxWB?usp=drive_link).

These datasets were integrated into the computational model to train AI to have different recognition capabilities under corresponding classifications.

Then, we searched YouTube for 100 daily vlogs across 10 categories, including food, travel, work, shopping, and etc. 

In order to create a dataset representing normal human daily life perspectives.

Finaly, we built a computational model that enables object detection in target images, creates analogical connections with the detection results, and forms the eye of AI.
