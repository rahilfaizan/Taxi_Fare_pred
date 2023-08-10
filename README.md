---

# Predicting Taxi Fare Prices using XGBoost and Random Forest

## Introduction

Predicting taxi fare prices is a classic problem in data science and machine learning. We'll be using a dataset from Kaggle that contains various features such as trip duration, passenger counts, distance traveled, and more. Our goal is to build models that accurately predict taxi fare prices based on these features.

### Dataset Overview

The dataset can be found at [this Kaggle link](https://www.kaggle.com/datasets/raviiloveyou/predict-taxi-fare-with-a-bigquery-ml-forecasting). It includes the following columns:

- `trip_duration`
- `passenger_counts`
- `distance_traveled`
- `fare`
- `tips`
- `misc_fare`
- `total_fare`

Now, let's get started with the code!

## Data Exploration and Preprocessing

```{r}
# Load the necessary libraries
library(tidyverse)

# Load the dataset
data_org <- read.csv("path_to_your_dataset/train.csv")
test_act_df <- read.csv("path_to_your_dataset/test.csv")

# Remove rows with zero fare
df <- filter(data_org, fare != 0)

# Drop unnecessary columns
no_need <- c('tip', 'miscellaneous_fees', 'total_fare')
data_org <- select(data_org, -no_need)

# Split the data
set.seed(1234)
data_org <- data_org[sample(1:nrow(data_org)), ]
```

## Train-Test Split

To assess our models' performance, we'll split the dataset into training and testing sets. The model will be trained on the training set and evaluated on the test set to ensure it generalizes well to new, unseen data.

```{r}
# Convert data to matrices
data_mat <- as.matrix(data_org)

# Split the data
train_data <- data_mat[1:146771, ]
test_data <- data_mat[-(1:146771), ]

train_labels <- data_org$fare[1:146771]
test_labels <- data_org$fare[-(1:146771)]
```

## XGBoost Model Training

We'll start by training an XGBoost model, a powerful gradient boosting algorithm.

```{r}
# Load the XGBoost library
library(xgboost)

# Create DMatrix for training and testing data
dtrain <- xgb.DMatrix(data = train_data, label = train_labels)
dtest <- xgb.DMatrix(data = test_data, label = test_labels)

# Train the XGBoost model
model <- xgboost(data = dtrain,
                 nround = 1000,
                 objective = "reg:squarederror",
                 print_every_n = 500)
```

## Model Evaluation

Now, let's evaluate the XGBoost model's performance using the Root Mean Squared Error (RMSE) metric.

```{r}
# Make predictions
pred <- predict(model, dtest)

# Calculate RMSE
xgboost_rmse <- sqrt(mean((pred - test_labels)^2))
print(paste("XGBoost RMSE:", xgboost_rmse))
```

## Hyperparameter Tuning

To optimize the XGBoost model's performance, we can perform hyperparameter tuning.

```{r}
# Define watchlist for early stopping
watchlist <- list(train = dtrain, test = dtest)

# Tune hyperparameters
model_tuned <- xgb.train(data = dtrain,
                         max.depth = 3,
                         eta = 0.01,
                         nthread = 3,
                         nround = 5000,
                         watchlist = watchlist,
                         objective = "reg:squarederror",
                         early_stopping_rounds = 50,
                         print_every_n = 500)
```

## Random Forest Model

In addition to XGBoost, we can explore the Random Forest algorithm for our prediction task.

```{r}
# Load the Random Forest library
library(randomForest)

# Train the Random Forest model
model_rf <- randomForest(
  x = train_data,
  y = train_labels,
  ntree = 500,
  mtry = sqrt(ncol(train_data)),
  importance = TRUE
)

# Make predictions using Random Forest
pred_rf <- predict(model_rf, newdata = test_data)

# Calculate RMSE for Random Forest
rf_rmse <- sqrt(mean((pred_rf - test_labels)^2))
print(paste("Random Forest RMSE:", rf_rmse))
```

## Results Visualization

Let's visualize the performance of our models by comparing actual vs. predicted fare values.

```{r}
# Convert predictions to data frame
results_df <- data.frame(Actual = test_labels, Predicted = pred)

# Create a scatter plot for XGBoost results
ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  labs(x = "Actual Fare", y = "Predicted Fare",
       title = "XGBoost: Actual vs. Predicted Fare Prices") +
  theme_minimal()
```
![image](https://github.com/rahilfaizan/Taxi_Fare_pred/assets/51293067/407f6768-e296-4292-b8dd-14fd88601406)

```{r}
# Convert Random Forest predictions to data frame
results_df_rf <- data.frame(Actual = test_labels, Predicted = pred_rf)

# Create a scatter plot for Random Forest results
ggplot(results_df_rf, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  labs(x = "Actual Fare", y = "Predicted Fare",
       title = "Random Forest: Actual vs. Predicted Fare Prices") +
  theme_minimal()
```
![image](https://github.com/rahilfaizan/Taxi_Fare_pred/assets/51293067/6194cc2e-26b4-446c-8fde-782ad3ffb358)

## Conclusion

In this post, we embarked on a journey to predict taxi fare prices using machine learning. We explored the XGBoost and Random Forest algorithms, performed data preprocessing, trained models, and evaluated their performance. Through visualization, we compared actual and predicted fare values, gaining insights into the effectiveness of our models.

Both XGBoost and Random Forest demonstrated their capabilities in predicting taxi fares, with XGBoost achieving an RMSE of 0.999 and Random Forest achieving an RMSE of 11.575. As I continue to explore the world of machine learning, will experiment with different algorithms, hyperparameters, and preprocessing techniques to further enhance my predictive models.

Thank you for joining me on this journey, and happy coding!

---
