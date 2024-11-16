# Wildfire-Prediction
Predicting the burnt area caused by wildfire in Zimbabwe.

#Introduction
Wildfire Prediction Challenge (Zindi)
Each year, thousands of fires across the African continent contribute to climate change and air pollution, impacting ecosystems, health, and livelihoods. These fires, both natural and human-induced, release significant amounts of CO2 and smoke, exacerbating global warming and degrading air quality. Understanding the dynamics that influence where and when these fires occur is crucial for predicting their effects and managing their environmental impact.

#Objective
Develop a machine-learning model capable of predicting the burned area in different locations across Africa from 2014 to 2016. This model will help in understanding the dynamics of fire occurrence, enabling better prediction of future fire patterns under different climatic conditions and aiding in the development of strategies to mitigate the environmental impact of fires.
Architecture diagram
 

#ETL process
Data extraction:
Datasets (variable information,training and testing)  in CSV format were collected from the Zindi website. To facilitate efficient data handling and avoid daily upload limits, these datasets were uploaded to Google Drive and then connected to Google Colab. The Pandas library was imported into the Colab environment to enable the loading and manipulation of the CSV files.

#volume:
The dataset likely includes high-dimensional data across multiple years, geographic locations, and environmental factors. 
Transform:
For both the training and test datasets, the date column was split to separate the ID and date components. From the date, the day, month, and year were successfully extracted. Subsequently, the datetime column was dropped, along with two columns (day and climate_swe) that had only one unique value, as they would not significantly impact the model's performance.

Standardization was applied using the StandardScaler to normalize the features, ensuring they have a mean of 0 and a standard deviation of 1. This step is crucial for models like XGBoost, as it helps avoid bias that can arise from differing feature scales, thereby improving the model's performance and generalization capabilities.
Subsequently, Recursive Feature Elimination with Cross-Validation (RFECV) was utilized to identify and retain the most significant features. This process reduced the dimensionality of the dataset and further improved the model's performance by eliminating less informative features.

#Load:
Data was stored in memory as pandas DataFrames for immediate access during the modeling process.

#Data modeling:
XGBoost(eXtreme Gradient Boosting) is an enhanced and scalable iteration of the gradient boosting algorithm, focusing on effectiveness, computational speed, and model performance.

#Assumption: XGBoost uses boosting, which involves combining small decision trees (weak learners) to create a powerful learner. The main concept is to reduce the loss function by introducing additional trees that make up for the mistakes made by the prior trees.
Feature selection: RFECV was used for feature selection to keep only the most important features for the model. XGBoost was used as the predictor, and feature importance was assessed through cross-validation. This procedure decreased dimensionality and removed unnecessary or duplicated characteristics, enhancing model effectiveness by reducing noise and overfitting.

#Normalization: StandardScaler was used to normalize the feature set, ensuring all features had a mean of 0 and standard deviation of 1. This was crucial for XGBoost's performance and was applied before feature selection and after feature engineering to maintain consistent scaling.
The XGBoost regressor was selected for its effectiveness, precision, and capability to manage linear and non-linear relationships in data. It is also resistant to outliers and missing data, which are frequently seen in real-life dataset.

#Hyperparameters:
Parameter Tuning via GridSearchCV
n_estimators: This parameter controls the number of trees in the ensemble, directly impacting the model's capacity to learn complex patterns.
max_depth: Determines the maximum depth of each tree, balancing model complexity and the risk of overfitting by controlling how detailed each tree can be.
learning_rate: Adjusts the step size at each iteration, controlling how much each tree contributes to the overall model, and helping to fine-tune the balance between speed and accuracy.
subsample: Specifies the fraction of the training data used to grow each tree, which helps in preventing overfitting by ensuring that each tree is built on slightly different data.
colsample_bytree: Defines the fraction of features used for each tree, promoting diversity among the trees and reducing the chance of overfitting to specific features.

#Training:
The optimal hyperparameters were determined using GridSearchCV, which tested various combinations and selected the best based on cross-validation outcomes. A 5-fold cross-validation technique was employed to assess how well the model performed on different training data sets, guaranteeing reliable performance on new data.

#Evaluation: These metrics ,Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) provided an interpretable error measure in the same units as the target variable, while the R² Score assessed how well the model performed the variance in the burn area.
Model validation:  involved using cross-validation during hyperparameter tuning to ensure the model wasn't overfitting by training and validating on different subsets of the data. After selecting the best hyperparameters, the model was trained on the full training set, and its performance was assessed using RMSE, MSE, and R² scores. Final predictions were made on the test set, with performance indicating the model's generalization ability and submission file was generated.

#Run time:
Run time for the entire notebook 24 minutes(7 minutes for feature selection and 17 minutes in model)

#Performance Metrics:
●	The metrics of the train dataset RMSE=0.02035244294567497,MSE=0.00041422193385695486, R2_SCORE=0.5150493817938411.
●	The model demonstrates strong performance with low RMSE and MSE, but the moderate R² score indicates potential for further enhancements to improve its explanatory power and I failed to do so.
●	The best hyperparameters given by  5-fold cross-validation were  colsample_bytree: 0.8, learning_rate: 0.2, max_depth: 7, n_estimators: 200, subsample: 1.0 .
●	Current on number 9 with public score:0.018897215 and private score of 0.018659509.
●	Error analysis:I kept editing code that was producing lower results in my public rating without being aware that there is a hidden private score for administrators to see. I made a significant error during my first attempt on Zindi challenge, but I have gained knowledge from it. In the future, I will remember to organize my work by creating a folder for each competition and downloading the necessary files for each submission or code modification to prevent confusion when requested.


