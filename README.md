# Reorder prediction
E-commerce stores collect data from users (how often a user reordered an item) and item statistics (how often a specific item was reordered). By using these as our feature inputs we can:
- Get insights into customer behavior, preferences, and demographics
- Improve inventory management by predicting reorders
- Give better reccomendations
- Understand which products and marketing channels yield the highest return on investment

XGBoost is a gradient boosting technique that uses [decision tree](http://scikit-learn.org/stable/modules/tree.html) ensembles for identifying which products are likely to be reordered at a store.

## Why XGBoost

## Feature Engineering
I decided to use the [instacart opensource dataset](https://www.kaggle.com/c/instacart-market-basket-analysis) since it provides transaction level data about orders and their associated products.

I've gone in depth in the data exploration notebook <INSERT LINK>

## Objective function
I've used mean F1 score as the evaluation metric. More details are in the performance-analysis notebook <INSERT LINK>







