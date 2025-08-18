# Sberbank Russian Housing Market

## Description
Housing costs demand a significant investment from both consumers and developers. And when it comes to planning a budget—whether personal or corporate—the last thing anyone needs is uncertainty about one of their biggest expenses. Sberbank, Russia’s oldest and largest bank, helps their customers by making predictions about realty prices so renters, developers, and lenders are more confident when they sign a lease or purchase a building.

Although the housing market is relatively stable in Russia, the country’s volatile economy makes forecasting prices as a function of apartment characteristics a unique challenge. Complex interactions between housing features such as number of bedrooms and location are enough to make pricing predictions complicated. Adding an unstable economy to the mix means Sberbank and their customers need more than simple regression models in their arsenal.

In this competition, Sberbank is challenging Kagglers to develop algorithms which use a broad spectrum of features to predict realty prices. Competitors will rely on a rich dataset that includes housing data and macroeconomic patterns. An accurate forecasting model will allow Sberbank to provide more certainty to their customers in an uncertain economy.

## Evaluation
Submissions are evaluated on the **Root Mean Squared Logarithmic Error (RMSLE)** between predicted prices and actual sale prices. RMSLE is calculated as:

$$
\mathrm{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl(\log(p_i + 1) - \log(a_i + 1)\bigr)^2}
$$

Where:
- \(n\) is the number of properties
- \(p_i\) is your predicted price (`price_doc`)
- \(a_i\) is the actual price

## Submission File
For each `id` in the test set, you must predict the sale price (`price_doc`). The file should contain a header and have the following format:

```
id,price_doc
27235,0
27236,0
27237,0
27238,0
```

## Dataset Description
The aim of this competition is to predict the sale price of each property. The target variable in `train.csv` is `price_doc`.

- **Training data:** August 2011 – Dec 2014
- **Test data:** Jan 2015 – June 2015

In addition to individual transaction records, the dataset includes macroeconomic indicators so you can focus on modeling property-level price variation without having to forecast the overall business cycle.

## Data Files
- **train_features.csv**, **test_features.csv**
  - Transaction records, indexed by `id`. May include multiple transactions per property.
  - Includes features about the local area and property.
- **macro.csv**
  - Russia’s macroeconomic and financial sector data, joinable on the `timestamp` column.
- **sample_submission.csv**
  - Example submission in the correct format.
- **data_dictionary.txt**
  - Explanations of all fields in the other data files.
