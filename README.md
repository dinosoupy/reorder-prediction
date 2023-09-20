# Reorder prediction
E-commerce stores collect data from users (how often a user reordered an item) and item statistics (how often a specific item was reordered). By using these as our feature inputs we can:
- Get insights into customer behavior, preferences, and demographics
- Improve inventory management by predicting reorders
- Give better reccomendations
- Understand which products and marketing channels yield the highest return on investment

XGBoost is a gradient boosting technique that uses [decision tree](http://scikit-learn.org/stable/modules/tree.html) ensembles for identifying which products are likely to be reordered at a store.

## Why XGBoost

## Feature Engineering
I decided to use the [instacart opensource dataset](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2) since it provides transaction level data about orders and their associated products.

This is the data scheme:
* `orders` (3.4m rows, 206k users):
* `order_id`: order identifier
* `user_id`: customer identifier
* `eval_set`: which evaluation set this order belongs in (see `SET` described below)
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
* `"train"`: training data supplied to participants (~131k orders)
* `"test"`: test data reserved for machine learning competitions (~75k orders)

I've gone in depth in the data exploration notebook <INSERT LINK>

## Objective function
I've used mean F1 score as the evaluation metric. More details are in the performance-analysis notebook <INSERT LINK>







