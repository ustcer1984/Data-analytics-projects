## Steel Plate Defect Prediction

This is a Kaggle competition --> [Link](https://www.kaggle.com/competitions/playground-series-s4e3/overview)

Below are my works and links to my Kaggle profile:  
1. [Exploratory data analysis](https://www.kaggle.com/code/ustcer1984/steel-plate-eda-s4e3)
2. [Models (RF/XGB/LGBM) and explainability](https://www.kaggle.com/code/ustcer1984/steel-plate-rf-xgb-lgbm-model-explaination-s4e3)
3. [Original vs. Train datasets, effect of adding original dataset on model score](https://www.kaggle.com/code/ustcer1984/steel-plate-effect-of-adding-original-data-s4e3)
   - Train dataset is from a deep learning model built on original dataset.
   - Train dataset is much noisier with extreme outliers.
   - Removing these outliers negatively affect model score, indicating the noise is with pattern / structure that can be learned by model.
   - Combine original and train dataset can slightly improve model score.
