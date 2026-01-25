Uncertainty selection scripts (refactored from FinalUnceratintySelection.ipynb)

Main entrypoint:
  python -m uncertainty_selection.run_selection --loop-number 1 --project kafka \
      --data-root SamplingLoopData --output-dir UncertainPoints \
      --n-bootstrap 20 --top-uncertain 100 --k-diverse 25 --top-k-labels 3 \
      --write-full-rows --verbose

Input expectations (per loop directory):
  SamplingLoopData/loop_{N}_data/
    {project}_labeled_train_data.csv
    {project}_labeled_test_data.csv
    {project}_unlabeled_data.csv

Output:
  UncertainPoints/loop_{N}_{project}_selected.csv
  UncertainPoints/loop_{N}_{project}_selected_full.csv   (if --write-full-rows)

Dependencies:
  pandas, numpy, scikit-learn, scipy, xgboost
