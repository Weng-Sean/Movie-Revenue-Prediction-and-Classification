# Movie Revenue Prediction and Classification

This project focuses on predicting movie revenue and classifying movies as profitable or not based on various features such as budget, popularity, and vote count. Three machine learning models have been implemented for this purpose:

1. **Linear Regression Model**: This regression model uses budget, popularity, and vote count as input features to predict movie revenue. It learns the relationships between these features and revenue from a training dataset and can generate revenue predictions for new movies.

2. **Logistic Regression Model**: The logistic regression model classifies movies as profitable or not. It uses budget, popularity, and vote count as input features and learns the relationship between these features and the binary outcome of profitability. It employs a logistic function to model the probability of a movie being profitable.

3. **K-Nearest Neighbors (KNN) Classifier**: The KNN model classifies movies as high-revenue or low-revenue. It considers budget, popularity, and vote count to compare a new movie's features to those of its K nearest neighbors in the training dataset. The assigned label is determined by the majority class of its K nearest neighbors.

## Data Preprocessing

- The project begins with data munging and cleaning.
- Data is loaded from CSV files, and missing or irrelevant columns are removed.
- Outliers in revenue, budget, and vote count are identified and removed.
- JSON data is parsed to extract relevant information.
- Release dates are analyzed, and features like release month, year, and day of the week are created.
- Genre popularity by year and month is visualized using heatmaps and stacked area plots.

## Data Visualization

- The project visualizes the distribution of movie releases by month and year.
- It also explores genre popularity over the years, identifying trends in movie genres.
- Heatmaps and correlation matrices help identify relationships between features and revenue.

## Modelling

### Linear Regression Model

- Budget, popularity, and vote count are used as input features.
- The model is trained and tested, and root mean squared error (RMSE) is calculated.
- The model's coefficients provide insights into feature importance.

### Logistic Regression Model

- Budget, popularity, and vote count are used to classify movies as profitable or not.
- The model is trained and evaluated using F1-score and confusion matrices.
- The RSE (Residual Standard Error) is calculated to assess performance.

### K-Nearest Neighbors (KNN) Classifier

- The KNN model classifies movies as high-revenue or low-revenue based on budget, popularity, and vote count.
- A grid search is performed to find the optimal value of K.
- Model performance is evaluated using F1-score, confusion matrices, and cross-validation.

## Conclusion

- The Linear Regression model offers decent revenue predictions with an RMSE of approximately 0.44.
- The Logistic Regression model classifies movies as profitable or not with an F1-score of approximately 0.78.
- The KNN Classifier, with K=221, achieves an F1-score of approximately 0.80.
- These models provide valuable insights into movie revenue prediction and profitability classification, aiding decision-making in the film industry.

This project demonstrates the power of machine learning in predicting and classifying movie revenue, enabling stakeholders to make informed decisions in the competitive world of filmmaking.
