# Tumor-Classifier
Overview
This project implements a binary deep learning model for tumor detection from medical images. The system uses a modified ResNet-18 architecture fine-tuned on a dataset containing two classes: tumor and no tumor. The aim is to provide reliable, reproducible classification with fast inference.

Dataset
The dataset is organized into two categories, tumor and no tumor, separated into training and testing folders. All images are resized to 224x224 pixels and normalized using ImageNet mean and standard deviation values.

Model
Backbone: ResNet-18
Final Layer: Linear layer with two output classes
Loss Function: CrossEntropyLoss
Optimizer: Adam or SGD
Evaluation Metrics: Accuracy, precision, recall, F1-score, ROC-AUC

Training
Training is executed using the train.py script. Images are preprocessed using resizing, normalization, and optional augmentation. GPU acceleration is used if available.

Command for training:
python src/train.py --epochs 25 --bs 32 --lr 0.001

Inference
Inference is performed using inference.py. The script processes a single image and outputs a class label along with confidence.

Command for inference:
python src/inference.py --image path_to_image --model model/tumor_resnet18.pth
Results
Typical performance for this architecture ranges between 92–96 percent accuracy, F1-score between 0.90–0.95, and ROC-AUC above 0.95. Replace these values with actual metrics from your training run.

Limitations
Performance depends on dataset size and quality. The model is not clinically validated and should not be used as a standalone diagnostic tool. Generalization may be limited when applied to imaging modalities not represented in the training data.

Future Work
Potential improvements include switching to EfficientNet or Vision Transformers, adding Grad-CAM for explainability, expanding the dataset, performing automated hyperparameter tuning, and exporting the model using ONNX for deployment
