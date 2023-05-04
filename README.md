# Multi-task prediction-based graph contrastive learning for inferring the relationship among lncRNAs, miRNAs, and diseases

## 1. Overview

The repository is organized as follows:

+ `data/` contains the datasets used in the paper;
+ `code/similarity_calculation.py` is the calculation and integration of lncRNA/miRNA/disease similarities;
+ `code/model.py`contains unsupervised graph contrastive learning module to extract lncRNA/miRNA/disease node embeddings;
+ `code/LDA_prediction.py` contains multiple classifiers to infer the lncRNA-disease association scores (e.g., AdaBoost, XGBoost...);
+ `code/MDA_prediction.py` contains multiple classifiers to infer the miRNA-disease association scores (e.g., AdaBoost, XGBoost...);
+ `code/LMI_prediction.py` contains multiple classifiers to infer the lncRNA-miRNA interaction scores (e.g., AdaBoost, XGBoost...);


## 2. Dependencies
* numpy == 1.24.2
* pandas == 1.4.4
* torch == 1.13.1
* sklearn == 1.2.2
* xgboost == 1.7.5
* lightgbm == 3.3.5


## 3. Quick Start
Here we provide a example to predict the association scores among lncRNAs, miRNAs, and diseases:

1. Download and upzip our data and code files
2. Run similarity_calculation.py to calculate lncRNA/miRNA/disease similarity and save them to ./dataset
3. Run model.py to generate low-dimensional embeddings of lncRNA/miRNA/disease nodes and save them to ./result
4. Run LDA_prediction.py or MDA_prediction.py or LMI_prediction.py to obtain the lncRNA-disease association scores, miRNA-disease association scores, and lncRNA-miRNA interaction scores, respectively. You can choose different classifiers, including Adaboost, XGBoost, GBDT, LightGBM, MLP, RF.

## 4. Contacts
If you have any questions, please email Nan Sheng (shengnan21@mails.jlu.edu.cn)
