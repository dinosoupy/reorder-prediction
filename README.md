# Reorder prediction
E-commerce stores collect data from users (how often a user reordered an item) and item statistics (how often a specific item was reordered). By using these as our feature inputs we can:
- Get insights into customer behavior, preferences, and demographics
- Improve inventory management by predicting reorders
- Give better reccomendations
- Understand which products and marketing channels yield the highest return on investment

This repository serves conatins idealised examples of models that compute probabilities of a reorder from the [instacart opensource dataset](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2). This dataset provides transaction level data about orders and their associated products. Look through [notebooks](https://github.com/dinosoupy/reorder-prediction/tree/main/notebooks) for all model implementations and techniques.

## RNN model
No manual feature engineering - pure deep learning approach. This approach can be reformulated as a binary prediction task: Given a user, a product, and the user's prior purchase history, predict whether or not the given product will be reordered in the user's next order. In short, the approach was to fit RNN model to the prior data and use the internal representations from these models as features to second-level models.

## Data Scheme
The purpose of the models implemented here is not to optimise prediction scores on the instacart dataset, but to design a pipeline for tackling similar problems with other tabular ecommerce datasets.
    
`orders` (3.4m rows, 206k users):
* `order_id`: order identifier
* `user_id`: customer identifier
* `eval_set`: prior / train / test
* `order_number`: the order sequence number for this user (1 = first, n = nth)
* `order_dow`: the day of the week the order was placed on
* `order_hour_of_day`: the hour of the day the order was placed on
* `days_since_prior`: days since the last order, capped at 30 (with NAs for `order_number` = 1)

`products` (50k rows):
* `product_id`: product identifier
* `product_name`: name of the product
* `aisle_id`: foreign key
* `department_id`: foreign key

`aisles` (134 rows):
* `aisle_id`: aisle identifier
* `aisle`: the name of the aisle

`deptartments` (21 rows):
* `department_id`: department identifier
* `department`: the name of the department

`order_products__SET` (30m+ rows):
* `order_id`: foreign key
* `product_id`: foreign key
* `add_to_cart_order`: order in which each product was added to cart
* `reordered`: 1 if this product has been ordered by this user in the past, 0 otherwise

where `SET` is one of the four following evaluation sets (`eval_set` in `orders`):
* `"prior"`: orders prior to that users most recent order (~3.2m orders)
* `"train"`: training data (~131k orders)

I've gone in depth in the [data exploration notebook](notebooks/0.3-dinosoupy-data-exploration.ipynb)


## Objective function
I've used mean F1 score as the evaluation metric.
