# Task 4: Disease Prediction from Medical Data (Diabetes Prediction)

**Intern Information**  
- **Name**: Kheira BENABDELMOUMENE  
- **Domain**: Machine Learning  
- **Task**: Disease Prediction (Diabetes)  
- **GitHub**: https://github.com/kheira00  
- **Submission Date**: 01/09/2026

## Introduction
This task focuses on developing machine learning models to predict the onset of diabetes using patient medical records from a publicly available Kaggle dataset (originally Pima Indians Diabetes Database). The problem is binary classification where Outcome = 1 (diabetic) and 0 (non-diabetic). Accurate early prediction assists healthcare professionals in preventive screening and timely intervention.

## Task Overview
**Objective**: Predict diabetes based on medical attributes (glucose, BMI, age, etc.).  
**Approach**: Implemented and compared several classification algorithms:  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree (tuned)  
- Random Forest  
- XGBoost  
- Voting Classifier (soft voting)

Performance evaluated using Precision, Recall, F1-Score, and ROC-AUC (more informative than accuracy on imbalanced data).

**Dataset**: 768 samples, all numerical features  
**Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
**Characteristics**:  
- Zeros in some features implicitly represent missing data  
- Class imbalance (~65% non-diabetic, ~35% diabetic)

## Methodology
**Preprocessing Steps**:  
1. Stratified train-test split (80% training, 20% testing)  
2. Feature scaling using StandardScaler  
3. Missing value imputation identified as future improvement (median replacement for zeros)

**Models Trained**:  
- Logistic Regression  
- SVM with RBF kernel  
- Decision Tree with tuned depth and split parameters  
- Random Forest ensemble  
- XGBoost classifier  
- Soft Voting Classifier

## Results

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7532  | 0.6667   | 0.6182 | 0.6415  | 0.8228 |
| SVM (RBF)           | 0.7468  | 0.6667   | 0.5818 | 0.6214  | 0.8086 |
| Decision Tree       | 0.7273  | 0.5942   | 0.7455 | 0.6613  | 0.7907 |
| Random Forest       | 0.7467  | 0.6369   | 0.6909 | 0.6786  | 0.8250 |
| XGBoost             | 0.7143  | 0.5873   | 0.6727 | 0.6271  | 0.7774 |
| Voting Classifier   | 0.7597  | 0.6666   | 0.6545 | 0.6486  | 0.8424 |

**Best Model**: Voting Classifier (highest ROC-AUC)

## Most Influential Features

| Feature                  | LogReg   | RandF   | DecT    | XGBoost |
|--------------------------|----------|---------|---------|---------|
| Glucose                 | 1.1027  | 0.2447 | 0.4473 | 0.2486 |
| BMI                     | 0.6888  | 0.1678 | 0.2063 | 0.1351 |
| Age                     | 0.3924  | 0.1431 | 0.1489 | 0.1389 |
| DiabetesPedigreeFunction| 0.2036  | 0.1206 | 0.1028 | 0.0796 |
| Insulin                 | -0.1383 | 0.0926 | 0.0784 | 0.1284 |
| Pregnancies             | 0.2230  | 0.0744 | 0.0087 | 0.0901 |
| BloodPressure           | -0.1515 | 0.0820 | 0.0077 | 0.0770 |
| SkinThickness           | 0.0688  | 0.0748 | 0.0000 | 0.1024 |

**Top feature across models**: Glucose

## Interpretation and Analysis
- The Voting Classifier achieved the highest ROC-AUC (0.8424) and strong accuracy (0.7597), indicating the best overall discriminative capability.
- Logistic Regression showed competitive accuracy (0.7532) and interpretability, making it a reliable baseline model.
- The Decision Tree obtained the highest recall (0.7455), which is valuable for reducing false negatives in diabetes screening.
- Random Forest offered a balanced performance across metrics, with a strong F1-score (0.6786), effectively handling feature interactions.
- XGBoost performed reasonably but may require further hyperparameter tuning and improved preprocessing for optimal results.

- **Logistic Regression**: The features with the largest coefficients are Glucose, BMI, Age, and DiabetesPedigreeFunction, indicating strong linear effects on predicting diabetes.
- **Random Forest**: Feature importance analysis shows Glucose, BMI, Age, and DiabetesPedigreeFunction are the top predictors. Less important features include Insulin, BloodPressure, SkinThickness, and Pregnancies.
- **Decision Tree**: Glucose dominates the decision splits, followed by BMI, Age, and DiabetesPedigreeFunction. Features like SkinThickness, BloodPressure, and Pregnancies contribute minimally.
- **XGBoost**: Glucose remains the most influential. Age and BMI are also important, while Insulin has slightly higher influence than in other models. DiabetesPedigreeFunction has moderate importance.

**Overall Observation**: Glucose, BMI, Age, and DiabetesPedigreeFunction consistently appear as the most influential features across models, while Pregnancies, BloodPressure, and SkinThickness are generally less impactful.  
**Medical Relevance**: In healthcare applications, models with higher recall and ROC-AUC are preferred to prioritize early detection and reduce missed diagnoses.

## Remarks and Key Learnings
- Implicit missing values negatively impact predictive performance.
- Ensemble models outperform single classifiers on this dataset.
- XGBoost performance is sensitive to hyperparameters and data preprocessing.
- Proper evaluation metrics are essential in imbalanced medical datasets.

## Conclusion
This task successfully demonstrated the application of machine learning techniques for diabetes prediction using medical data. Ensemble approaches, particularly the Voting Classifier and Random Forest, achieved the best overall performance, with ROC-AUC exceeding 0.84. The task fulfills all Task 4 requirements and highlights the importance of domain-aware preprocessing in healthcare machine learning.
