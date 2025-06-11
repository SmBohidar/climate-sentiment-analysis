
# NASA Climate Change Sentiment Analysis Dashboard

A powerful pipeline and interactive dashboard for analyzing public sentiment, topic trends, and misinformation related to NASA’s climate change communications.

## Project Setup & Workflow

Follow the steps below to process, analyze, and visualize climate comment data.

### Step 1: Clean and Process Raw Data

Run the initial data processor to clean, transform, and engineer features from the raw dataset.

```bash
python data_processing.py
```

### Step 2: Perform Sentiment Analysis

Analyze each comment's sentiment score and categorize it using the VADER sentiment tool.

```bash
python sentiment_analysis.py
```

### Step 3: Extract Topics and Trends

Apply topic modeling (LDA or BERTopic) to extract key discussion themes from the dataset.

```bash
python topic_modeling.py
```

### Step 4 (Optional but Recommended): Detect Climate Myths

Run myth detection to flag misinformation and prepare educational counter-responses.

```bash
python process_myths.py
```

### Step 5: Launch Interactive Streamlit Dashboard

Start the interactive dashboard to explore trends, sentiments, topics, and flagged myths.

```bash
streamlit run streamlit_dashboard.py
```

### Step 6: Generate Charts and Export Reports

Run the script to generate charts, summary reports, and exportable insights.

```bash
python visualizations.py
```

## Folder Structure

```
project/
├── data/                     # Raw and processed data files
├── data_processing.py        # Data cleaning and feature engineering
├── sentiment_analysis.py     # Sentiment scoring and classification
├── topic_modeling.py         # Topic extraction via NLP
├── process_myths.py          # Myth detection pipeline
├── myth_detector.py          # Myth detection logic
├── alerts.py                 # Smart alert system logic
├── visualizations.py         # Reporting and chart generation
├── streamlit_dashboard.py    # Interactive Streamlit dashboard
```

## Setup

It is recommended to use a virtual environment. Install required packages with:

```bash
pip install -r requirements.txt
```
