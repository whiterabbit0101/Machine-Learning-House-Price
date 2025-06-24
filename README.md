This is a machine learning algorithm that takes in housing data in CSV format and trains the model 
to predict home prices based on 80 metrics. Dataset obtained from kaggle.com


Main.py -- loads and cleans dataset, shows if values are missing, 
shows number of rows and colums, and saves newly cleaned file. 

Model.py -- Runs the machine learning algorithm using the random forest regressor model. Shows data points such 
as number of features, target, training and testing data sizes, and accuracy evaluation using R2, RMSE, MAE, and MAPE.
Displays top 10 features affecting the target (sales price)

