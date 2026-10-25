# Does demographic information improve credit-default prediction?

A small retrospective experiment comparing credit-default models **with and without age, recorded sex, and education**. The goal is to measure incremental predictive performance, examine a simplified financial decision rule, and check how hypothetical decisions differ across demographic groups.

**This is a statistical study, not a recommendation to use protected characteristics in lending decisions.**

## Research question

Once a model already knows a customer's repayment and financial history, how much additional information do age, sex and education provide? Does any predictive improvement translate into a meaningful change in an **illustrative**, rather than actual, lending return?

## Data and methods

- Data: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), 30,000 historical Taiwanese credit-card customers.
- Five logistic-regression models: financial-only; plus age; plus sex; plus education; plus all three.
- One common stratified 75/25 train-test split (`random_state=42`), with preprocessing fitted only on the training data.
- Education is handled categorically; undocumented education values are grouped as `unknown`.
- Performance: ROC-AUC (higher is better), Brier score (lower is better).
- An illustrative approval rule applies fixed assumed economics: +0.15 units if a normalized one-unit loan repays; -0.70 units if the customer defaults; zero if no loan is issued. The predicted-default cutoff is approximately **17.65%**. These are **assumptions, not observed cash flows**.
- Group audit: compare simulated approval rates for sex categories reported in the source data.

## Results

The analysis was run on the 30,000-customer source dataset (7,500 held-out observations).

| Model | ROC-AUC | Brier score | Hypothetical return per 1,000 historical customers |
|---|---:|---:|---:|
| Financial only | 0.7142 | 0.1461 | 15.09 |
| + Age | 0.7148 | 0.1459 | 15.89 |
| + Sex | 0.7149 | 0.1460 | 14.91 |
| + Education | 0.7146 | 0.1460 | 14.95 |
| + All three | **0.7155** | **0.1458** | 16.34 |

**Main finding:** The full model's ROC-AUC exceeded the financial-only model's by about **0.00132**. An approximate paired bootstrap 95% confidence interval for that difference included zero (**-0.00138 to +0.00389**), so this experiment did **not** show a statistically clear ROC-AUC improvement at the 5% level. This does *not* prove demographic features have zero predictive value.

The full model had a slightly larger hypothetical return under the fixed assumptions, but these small numerical differences should **not** be interpreted as evidence that demographic-based lending would be profitable or appropriate in practice.

### Figures

![ROC-AUC by model](results/01_auc.png)

![Brier scores](results/02_brier.png)

![Illustrative returns](results/03_simulated_returns.png)

![Simulated approval rates by source-data sex group](results/04_group_audit.png)

Detailed numerical output: [`results/model_comparison.csv`](results/model_comparison.csv).

## How to reproduce

1. Install Python and the dependencies: `pip install -r requirements.txt`.
2. Download the original Excel dataset from the UCI link above and place `default of credit card clients.xls` in this folder.
3. Run `python credit_scoring_experiment.py`.
4. View the CSV and PNG plots in `results/`.

The script can alternatively take a compatible CSV as an argument: `python credit_scoring_experiment.py path/to/data.csv`.

## Important limitations

1. **Selection bias:** The dataset contains previously issued credit-card accounts, *not all applicants*. Outcomes for rejected applicants are missing, so policy profitability cannot be identified from this dataset alone.
2. **Hypothetical economics:** The returns assume uniform loan sizes, fixed interest/recovery, and no financing, operational, or regulatory costs. These are not actual bank profits.
3. **Fairness and legality:** Differences in group approval rates should be investigated, not taken as proof of causality or unlawful discrimination. Even the financial-only features can act as demographic proxies. This experiment does not advocate using sex or other protected characteristics to make lending decisions.
4. **Statistical uncertainty:** The code's paired bootstrap resamples the test set while holding fitted models fixed, and uses one train/test split. It does not quantify the variability from retraining on other training samples. The small observed ROC-AUC difference should not be overinterpreted.
5. **External validity:** Data are from a different country and historical period, not a current UK lending population.

## Data citation

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications. Dataset: [UCI Machine Learning Repository, Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (CC BY 4.0).

## AI Usage Disclosure
I independently formulated the research question: whether including demographic variables such as age, sex, and education improves credit-default prediction and hypothetical lending profitability.

I chose to investigate this using logistic regression, a statistical method I was already familiar with through my mathematical and statistical studies.

I used ChatGPT and GitHub Copilot extensively to assist with the Python implementation and visualizations.

The research idea and motivation were my own, while the implementation and initial analysis were developed with AI assistance.
