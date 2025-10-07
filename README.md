# Phishing-domain-detection
##  Reference Paper
**Alnemari, S., & Alshammari, M. (2023).**  
*Detecting Phishing Domains Using Machine Learning.*  
Applied Sciences, 13(8), 4649.  
DOI: [https://doi.org/10.3390/app13084649](https://doi.org/10.3390/app13084649)
##  Overview
This project replicates the 2023 research paper *Detecting Phishing Domains Using Machine Learning*,  
developed by Alnemari & Alshammari, to compare four ML models:
- **Artificial Neural Network (ANN)**
- **Support Vector Machine (SVM)**
- **Decision Tree (DT)**
- **Random Forest (RF)**

Using the **UCI Phishing Websites Dataset**, this study evaluates the models’ performance  
for phishing detection. The **Random Forest** model achieved the best accuracy (~97.3%).
## Dataset
- **Source:** [UCI Machine Learning Repository – Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
- **Dataset ID:** 327 (via `ucimlrepo`)
- **Instances:** 11,055  
- **Features:** 30 input attributes + 1 target class  

Data categories include:
- Address bar features  
- HTML & JavaScript features  
- Abnormal-based features  
- Domain-based features
## Project Workflow
1. Load dataset using `ucimlrepo`
2. Apply Min-Max normalization
3. Split data (80/20)
4. Train models (SVM, DT, RF, ANN)
5. Compute accuracy, precision, recall, and F1-score
6. Visualize feature importance and model performance
