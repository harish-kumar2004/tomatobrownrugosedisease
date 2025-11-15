# Training Guide for Tomato Disease Detection Model

## Overview
This guide will help you retrain the model with all tomato disease classes from your Test Image folder.

## Prerequisites
Make sure you have all required packages installed:
```bash
pip install -r requirements.txt
```

## Steps to Train the Model

### 1. Run the Training Script
Execute the training script from the project root directory:

```bash
python train_tomato_model.py
```

### 2. What the Script Does
- Loads all images from the `Test Image` folder
- Organizes them by disease class (11 classes total)
- Splits data into training, validation, and test sets
- Trains a CNN model with enhanced architecture
- Saves the trained model as `plant_disease_model.h5`
- Creates `class_names.txt` with all class names

### 3. Training Process
The training will:
- Load images from all 11 disease folders
- Normalize images (resize to 256x256, normalize pixel values)
- Train for 50 epochs with batch size of 32
- Display progress and accuracy metrics
- Save the best model

### 4. Expected Output
After training, you should see:
- Model architecture summary
- Training progress for each epoch
- Final test accuracy (should be >90%)
- Saved model file: `plant_disease_model.h5`
- Class names file: `class_names.txt`

### 5. Run the Application
After training is complete, run the Streamlit app:

```bash
streamlit run main_app.py
```

## Disease Classes Supported
The model will be trained on these 11 tomato disease classes:
1. Bacterial_spot
2. Early_blight
3. healthy
4. Late_blight
5. Leaf_Mold
6. powdery_mildew
7. Septoria_leaf_spot
8. Spider_mites Two-spotted_spider_mite
9. Target_Spot
10. Tomato_mosaic_virus
11. Tomato_Yellow_Leaf_Curl_Virus

## Troubleshooting

### Issue: Out of Memory Error
- Reduce batch size in `train_tomato_model.py` (line with `batch_size = 32`)
- Reduce number of epochs if needed
- Close other applications to free up RAM

### Issue: Model Not Found Error
- Make sure you've run the training script first
- Check that `plant_disease_model.h5` exists in the project directory

### Issue: Low Accuracy
- Ensure you have sufficient images in each class (at least 100+ per class recommended)
- Check that images are properly labeled and organized
- You may need to train for more epochs

## Notes
- Training time depends on your hardware and dataset size
- With ~25,000 images, training may take 30-60 minutes on a modern CPU, or 10-20 minutes on GPU
- The model architecture has been optimized for 11 classes with dropout layers to prevent overfitting

