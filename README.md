# Automated Anomaly Detection in 5G Network Logs Using Natural Language Processing

**Module:** COS700 Research Project[cite: 1]  
**Author:** Hendré Beyer (Student Number: u26846188)[cite: 1]  
**Supervisor:** Sheunesu Makura[cite: 1]  

## Project Overview
The rapid global deployment of 5G networks has exponentially increased the volume and complexity of network traffic, creating significant challenges for post-incident digital forensics[cite: 1]. Traditional forensic analysis models rely heavily on manual keyword filtering, which is inadequate for identifying complex, high-volume abnormal events hidden within semi-structured log data[cite: 1]. 

This project is a functional Python prototype that proposes an automated anomaly detection model utilizing Natural Language Processing (NLP)[cite: 1]. By parsing semi-structured cloud infrastructure logs, this research applies Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction and unsupervised clustering algorithms to define normal network behavior and mathematically flag structural deviations[cite: 1].

## Objectives
* Develop a TF-IDF feature extraction model to tokenize semi-structured 5G log data[cite: 1].
* Implement unsupervised clustering algorithms (e.g., K-Means/DBSCAN) to baseline network behavior and flag anomalies[cite: 1].
* Benchmark the prototype against traditional keyword filtering based on Accuracy, Precision, Recall, F1-Score, and computational efficiency[cite: 1].

## Repository Structure
This repository will be organized as follows:
* `/data` - Mock datasets and generated 5G logs (Note: Large datasets will be excluded via `.gitignore`).
* `/src` - Source code for the prototype.
  * `/preprocessing` - Regex tokenization and data cleaning scripts.
  * `/models` - TF-IDF extraction and unsupervised clustering algorithms.
* `/results` - Output metrics, evaluation reports, and benchmarking graphs.

## Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** Scikit-learn, Pandas, NumPy, re (Regular Expressions)

## Setup Instructions (Local Development)

1. **Clone the repository:**
   git clone https://github.com/[YOUR-GITHUB-USERNAME]/COS700-Log-Anomaly-Detection.git
   cd COS700-Log-Anomaly-Detection

2. **Create and activate a virtual environment:**
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt
