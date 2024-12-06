

# Sentiment Analysis on Mental Health Dataset

## Overview
This project is focused on performing sentiment analysis on a curated mental health dataset. The dataset contains statements tagged with their corresponding mental health statuses, enabling the development of machine learning and deep learning models for sentiment classification. This project serves as a resource for exploring the intersection of AI and mental health, with potential applications in chatbot development and mental health monitoring systems.

---

## Project Description
The project aims to preprocess raw data, clean text, and apply various machine learning and deep learning models to classify mental health statuses based on statements. The dataset has been carefully cleaned and prepared to ensure robust model training and evaluation. The deep learning implementation leverages an LSTM-based architecture to capture the sequential nature of the data, improving sentiment detection accuracy.

---

## Key Features
- Preprocessing of textual data, including tokenization and padding.
- One-hot encoding of sentiment labels for compatibility with machine learning models.
- Implementation of multiple sentiment analysis techniques:
  - Machine learning models like Logistic Regression and Random Forest.
  - Deep learning models like LSTM.
- Evaluation of model performance using metrics like accuracy.
- Dynamic handling of multiple sentiment classes.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**: 
  - Pandas, NumPy (data manipulation and analysis)
  - TensorFlow, Keras (deep learning)
  - Scikit-learn (machine learning)
- **Other Tools**: Jupyter Notebook, Google Colab

---

## Project Structure
```plaintext
.
├── data/
│   ├── Combined Data.csv       # Mental health dataset
├── notebooks/
│   ├── preprocessing.ipynb     # Data cleaning and preprocessing
│   ├── ml_models.ipynb         # Machine learning models
│   ├── lstm_model.ipynb        # LSTM deep learning implementation
├── src/
│   ├── preprocessing.py        # Script for text cleaning
│   ├── train_model.py          # Script for training models
├── results/
├── README.md                   # Project documentation
```

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/mental-health-sentiment-analysis.git
   cd mental-health-sentiment-analysis
   ``
2. Preprocess the dataset:
   - Open and run the `preprocessing.ipynb` notebook or use `src/preprocessing.py`.
3. Train the models:
   - For machine learning models, run `ml_models.ipynb`.
   - For deep learning (LSTM), run `lstm_model.ipynb`.
4. Evaluate the models and analyze the results:
   - Check metrics saved in the `results/` directory.

---

## Results
- Achieved an accuracy of **X%** using Logistic Regression and **Y%** using Random Forest.
- The LSTM model outperformed traditional machine learning models with an accuracy of **Z%**.
- Insights from sentiment classification highlight patterns in mental health-related statements.

---

## Future Improvements
- Expand the dataset with more diverse and balanced data for better generalization.
- Fine-tune the LSTM model using techniques like attention mechanisms.
- Integrate the trained model into a chatbot application for real-world mental health support.
- Explore advanced NLP techniques such as transformers (e.g., BERT).
