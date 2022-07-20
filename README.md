# explosion-crater-detection

## 1. Clone repo
```bash
git clone git@github.com:popryho/explosion-crater-detection.git
```

## 2. Setup an environment

```bash
pip install -r requirements.txt
```
## 3. Prepare dataset

```bash
python utils/image_cutting.py
    --image_path_input data/data_part/S2B_tile_20220702_37UDQ_cut.tif
    --mask_path_input data/data_part/tr_class_cut.tif
    --image_path_output dataset/train/images
    --mask_path_output dataset/train/masks
    --image_size 64 --step 16
```
or just decompress zip archives using

```bash
unzip filename.zip -d /path/to/directory
```

## 4. Verify coniguration in utils/config.py

## 5. Train the model

```bash
python train.py
```

## 5. Visualize prediction

```bash
python predict.py
```
