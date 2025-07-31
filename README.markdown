# Real-Time Financial Forecasting Using Deep Learning

## Project Overview
This repository contains the implementation of a deep learning assignment (CI7521 - Machine Learning with Deep Neural Networks) focused on forecasting GBP/USD exchange rates using historical time series data. The project develops and compares four deep learning models: Feedforward Neural Network (FNN), CNN + LSTM, GRU, Simple RNN, and their regression variants. The notebook is implemented in Python using TensorFlow and Keras, with data preprocessing and model evaluation performed in a Google Colab environment.

## Repository Structure
```
deep-learning-financial-forecasting/
├── data/
│   └── GBPUSD_open_5year.csv      # Historical GBP/USD exchange rate data
├── notebooks/
│   └── CW_2_Deep_Learning_FINAL.ipynb  # Main Jupyter Notebook with code and analysis
├── requirements.txt                # Python dependencies for the project
├── README.md                      # Project documentation (this file)
└── LICENSE                        # License file (e.g., MIT License)
```

### Directory and File Descriptions
- **data/**: Contains the dataset `GBPUSD_open_5year.csv` used for training and testing the models.
- **notebooks/**: Stores the main Jupyter Notebook `CW_2_Deep_Learning_FINAL.ipynb`, which includes data preprocessing, model implementation, training, evaluation, and comparison of results.
- **requirements.txt**: Lists the Python libraries required to run the notebook.
- **README.md**: This file, providing an overview of the project and instructions for use.
- **LICENSE**: Specifies the licensing terms for the project (e.g., MIT License).

## Installation
To run the notebook locally or in a Google Colab environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/deep-learning-financial-forecasting.git
   cd deep-learning-financial-forecasting
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**:
   - **Locally**: Use Jupyter Notebook or JupyterLab:
     ```bash
     jupyter notebook notebooks/CW_2_Deep_Learning_FINAL.ipynb
     ```
   - **Google Colab**: Upload the notebook `CW_2_Deep_Learning_FINAL.ipynb` and the dataset `GBPUSD_open_5year.csv` to Google Colab and execute the cells.

## Dependencies
The required Python libraries are listed in `requirements.txt`. Key dependencies include:
- `tensorflow`
- `keras-tuner`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tabulate`

To generate the `requirements.txt` file, you can run:
```bash
pip freeze > requirements.txt
```
after installing the dependencies in your environment.

## Usage
1. **Dataset**: Ensure the `GBPUSD_open_5year.csv` file is placed in the `data/` directory. This file contains historical GBP/USD exchange rate data used for training and testing.
2. **Notebook**: Open `CW_2_Deep_Learning_FINAL.ipynb` in Jupyter or Colab. The notebook includes:
   - Data preprocessing steps (loading, scaling, and sequence windowing).
   - Implementation of four deep learning models (FNN, CNN + LSTM, GRU, Simple RNN) and their regression variants.
   - Model training and evaluation using metrics like MAE, MAPE, MSE, RMSE, and R² Score.
   - A comparison table summarizing the performance of all models.
3. **Execution**: Run the notebook cells sequentially. Ensure a GPU environment (e.g., Google Colab with T4 GPU) is used for faster training, as specified in the notebook's metadata.
4. **Output**: The notebook outputs a comparison table of model performance metrics and visualizations of the results.

## Results
The models were evaluated based on the following metrics:

| Model                 | MAE   | MAPE   | MSE   | RMSE  | R² Score |
|-----------------------|-------|--------|-------|-------|----------|
| 1.FNN                 | 0.022 | 3.626  | 0.0001| 0.027 | 0.845    |
| 2.CNN + LSTM          | 0.030 | 5.060  | 0.001 | 0.037 | 0.709    |
| 3.GRU                 | 0.014 | 2.382  | 0.000 | 0.018 | 0.932    |
| (3.2) Simple RNN      | 0.013 | 2.130  | 0.000 | 0.016 | 0.946    |
| 4.Regression LSTM     | 0.022 | 0.036  | 0.001 | 0.027 | 0.845    |
| (4.2) Regression GRU  | 0.016 | 0.027  | 0.000 | 0.020 | 0.917    |

The Simple RNN model achieved the best performance with the lowest MAE (0.013), MAPE (2.130), RMSE (0.016), and the highest R² Score (0.946).

## Notes
- The dataset `GBPUSD_open_5year.csv` is required to run the notebook. Due to its absence in the provided document, ensure it is available or sourced appropriately.
- The notebook uses a GPU accelerator (T4) for efficient training. If running locally, ensure a compatible GPU and TensorFlow-GPU are installed.
- Data preprocessing results in a 5% data loss due to sequence windowing (60 rows out of 1200), as noted in the notebook output.

## Team Members
- Menge Samruddhi Sanjay (K2377469)
- Rajput Shivam (K2437852)
- Anastasios Papantonopoulos (K1932447)
- Itape Tanmay Vithal (K2352164)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
This project was developed as part of the CI7521 Machine Learning with Deep Neural Networks course. Special thanks to the course instructors and team members for their contributions.