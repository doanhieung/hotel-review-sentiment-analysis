# 🏨 Hotel Review Sentiment Analysis

Gain insights into customer sentiment based on hotel reviews across Europe.

## 📄 Project Overview

Hotel review sentiment analysis detects customer opinions, helping businesses understand customer satisfaction and areas for improvement. By analyzing feedback, companies can identify positive and negative sentiments and uncover recurring themes. This information enables proactive responses to customer concerns, enhancements to customer experience, and strategic improvements in marketing to reduce churn and maximize profits.

This project processes and explores hotel review data, using statistical and linguistic techniques to identify and quantify sentiment strengths (positive, negative, neutral) and subjective nuances. Note that while text analysis provides valuable insights, limitations exist, such as challenges in interpreting slang, uncommon abbreviations, or sarcasm.

## 📝 Data Source

- **Dataset**: 515K Hotel Reviews in Europe
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe/data)
- **Size**: 515,000+ reviews, including attributes like hotel name, location, and review text.

## 🛠️ Installation

1. **Install dependencies**:

    ```bash
    poetry install
    ```

2. **Activate the virtual environment**:

    ```bash
    poetry shell
    ```

## 🚀 Usage

1. **Data Preprocessing**: Run the notebook `notebook/preprocess_data.ipynb` to clean and preprocess the raw review data.

2. **Exploratory Data Analysis (EDA)**: Use `notebook/eda.ipynb` to explore and visualize data patterns, preparing for sentiment analysis.
