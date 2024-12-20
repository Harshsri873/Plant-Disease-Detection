# Plant Disease Detection

This project focuses on building a machine learning-based system for detecting plant diseases using images of plant leaves. By leveraging advanced computer vision and data analytics techniques, the system aims to assist farmers and agricultural experts in diagnosing plant health and taking timely corrective measures.

## Features

- **Image-Based Disease Detection**: Uses Convolutional Neural Networks (CNNs) to analyze leaf images for signs of disease.
- **User-Friendly Interface**: Designed for easy use by non-technical users.
- **Real-Time Predictions**: Provides quick and accurate feedback to assist in decision-making.
- **Streamlit Deployment**: The model is deployed as an interactive web application using Streamlit for seamless user experience.

## Project Structure

- `data/`: Contains datasets for training, validation, and testing. Includes plant images.
- `src/`: Source code for data preprocessing, feature extraction, model training, and evaluation.
  - `data_preprocessing.py`: Prepares and cleans the dataset.
  - `model_training.py`: Defines and trains the machine learning model.
  - `predict.py`: Script for running predictions on new inputs.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experiments.
- `models/`: Saved trained models and their checkpoints.
- `app/`: Code for deploying the model as a Streamlit web application.
- `README.md`: Project documentation.

## Dataset

The dataset is sourced from Kaggle and consists exclusively of high-resolution images of plant leaves, labeled with the corresponding disease or healthy status. It does not include any tabular data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle and place it in the `data/` directory.

## Usage

### Training the Model
To train the model:
```bash
python src/model_training.py
```

### Making Predictions
To run predictions on new data:
```bash
python src/predict.py --image_path path/to/image
```

### Running the Streamlit Application
To launch the Streamlit web application:
```bash
streamlit run app/app.py
```

## Technologies Used

- **Programming Language**: Python
- **Machine Learning Framework**: TensorFlow/Keras
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit (for deployment)
- **Databases**: SQLite/PostgreSQL (for storing metadata)

## Future Improvements

- Expanding the dataset to include more plant species and diseases.
- Enhancing model accuracy by experimenting with advanced architectures.
- Adding multilingual support for global usability.
- Developing a mobile application for easier access.

## Contributors

- Harsh Srivastav
- [Other Contributors' Names]

## License

This project is licensed under the [MIT License](LICENSE).

---
For more information, feel free to contact [your_email@example.com].

