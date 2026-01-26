Uncertainty selection scripts (refactored from FinalUnceratintySelection.ipynb)

Input expectations (per loop directory):
  SamplingLoopData/loop_{N}_data/
    labeled_train_data.csv
    labeled_test_data.csv
    unlabeled_data.csv

Output:
  UncertainPoints/loop_{N}_selected.csv

Dependencies:
  pandas, numpy, scikit-learn, scipy, xgboost
