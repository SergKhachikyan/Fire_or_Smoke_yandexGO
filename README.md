# 🔥 Fire and Smoke Detection in Images 🔥

This project detects fire and smoke in static images using deep learning models, specifically Convolutional Neural Networks (CNNs) built with **TensorFlow** and **Keras**. The model is trained on labeled datasets containing images with and without fire or smoke to enable accurate predictions.

## 🚀 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fire-smoke-detection.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd fire-smoke-detection
    ```

3. **Install the dependencies:**
    Make sure Python 3.x is installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch the project:**
    Open the Jupyter Notebook interface:
    ```bash
    jupyter notebook
    ```

## 🔧 Technologies

This project uses the following technologies:
- **Python** 🐍: Programming language.
- **TensorFlow** 🔥: Deep learning framework used for building and training the CNN model.
- **Keras** 🧠: High-level API for TensorFlow to simplify deep learning model development.
- **OpenCV** 🎥: Image processing library for reading and preprocessing images.
- **Pandas** 📊: Data manipulation and analysis.
- **Matplotlib & Seaborn** 📈: Libraries for data visualization and plotting.

## 📝 How to Use

1. **Prepare the dataset:**
    - Place your dataset inside the `datasets/` folder. The dataset should have two categories: one containing images with fire/smoke, and one without.

2. **Train the model:**
    - Open Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the relevant notebook (e.g., `fire_smoke_detection_training.ipynb`) and run the cells sequentially to preprocess the data, build the model, and start training.

3. **Make predictions:**
    After training the model, you can use the inference script to predict fire/smoke presence in new images:
    ```bash
    python src/inference.py --image_path path/to/your/image.jpg
    ```

## 💡 Features

- **Image Classification** 🖼️: Classify images into "fire/smoke" or "no fire/smoke" categories.
- **Data Augmentation** 🔄: Improve generalization with image augmentation techniques.
- **Model Evaluation** 📊: Evaluate model performance using accuracy, confusion matrix, and classification reports.
- **Visualization** 🌈: Visualize training metrics like loss and accuracy over epochs.

## 🧠 Model Architecture

- **Input Layer**: Accepts image data (preprocessed to standard size).
- **Convolutional Layers**: Feature extraction from images.
- **Pooling Layers**: Dimensionality reduction.
- **Fully Connected Layers**: Decision making based on extracted features.
- **Output Layer**: Binary classification (Fire/Smoke vs No Fire/Smoke).

## 🏆 Model Performance

- **Loss Function**: Binary Crossentropy, suitable for binary classification tasks.
- **Metrics**: Model performance evaluated by accuracy and confusion matrix.

## 📊 Visualizations

- **Training Curves**: Visualize loss and accuracy over epochs.
- **Sample Predictions**: See examples of correctly and incorrectly classified images.
- **Confusion Matrix**: Evaluate model's performance by examining true positives, false positives, etc.

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the project, open issues, or submit pull requests.
