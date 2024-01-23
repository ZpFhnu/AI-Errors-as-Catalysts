# AI Errors as Catalysts: Human-AI Synergy Sparks Creative Evolution
Here we share our code and dataset~
## Code
We use [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) for object detection of Vlog screenshots. According to the requirements of our project, the automatic_label_ram_demo.py has been modified accordingly，You need to move the modified version to the original project and configure the required environment.
```bash
export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_demo_zpf.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image_dir "data/train2017" \
  --output_dir "output/train2017" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
```
Image retrieval utilizes the CLIP model to extract features from images to be retrieved and database images, calculate cosine similarity, and return a list of similar images.
```python
python image_retrival.py
```
