import pandas as pd
from pathlib import Path
from llm_oracle_labeling.config import (
    RISK_TYPE_LABELS
)

def extract_and_assign_labels(loop_number):
    """
    Extract labels from accepted_labels.csv and assign them to unlabeled_data.csv.
    Save labeled PRs to the next loop folder as labeled_train_data.csv
    
    Labeling convention:
    - bug, security, performance, code_quality_or_maintenability: 0 or 1
    - non_risky: 1 if all risky categories are 0, else 0
    """

    # Define paths
    sampling_dir=Path("SamplingLoopData")
    
    # accepted labels location
    accepted_labels_dir=Path("AcceptedLabels")
    accepted_labels_path = accepted_labels_dir / f"accepted_labels_{loop_number}.csv"
    
    # unlabled data location
    unlabeled_dir =sampling_dir / f"loop_{loop_number - 1}_data"
    unlabeled_path = unlabeled_dir / "unlabeled_data.csv"

    # folder to save labeled prs
    label_save_dir = sampling_dir / f"loop_{loop_number}_data"
    label_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_path = label_save_dir / "labeled_train_data.csv"

    # read accepted label data
    if accepted_labels_path.exists():
        accepted_label_df = pd.read_csv(accepted_labels_path)
        print(f"Total accepted label records: {len(accepted_label_df)}")
    else:
        print(f"WARNING: {accepted_labels_path} not found")
        accepted_label_df = pd.DataFrame()

    # take the necessary columns
    accepted_label_df = accepted_label_df[["pr_number", "agreed_labels"]]

    # read unlabled data
    if unlabeled_path.exists():
        unlabeled_df = pd.read_csv(unlabeled_path)
        print(f"Total accepted label records: {len(unlabeled_df)}")
    else:
        print(f"WARNING: {unlabeled_path} not found")
        unlabeled_df = pd.DataFrame()
     
    # extract only the prs to be assigned a label
    label_process_cand_pr = unlabeled_df.merge(
        accepted_label_df[["pr_number"]],
        on="pr_number",
        how="inner"
    )
    print(len(label_process_cand_pr), len(accepted_label_df))
    # assert check for length
    assert(len(label_process_cand_pr)==len(accepted_label_df))

    # initialize label columns in the label_process_cand_pr 
    label_process_cand_pr["bug"]=0
    label_process_cand_pr["security"]=0
    label_process_cand_pr["performance"]=0
    label_process_cand_pr["code_quality_or_maintenability"]=0
    label_process_cand_pr["non_risky"]=0

    for _, row in accepted_label_df.iterrows():
        # get the pr number 
        pr_number = row["pr_number"]

        # get labels
        agreed_labels = row["agreed_labels"].split(";")

        for label in agreed_labels:
            # bug risk case
            if label == RISK_TYPE_LABELS[0]:
                label_process_cand_pr.loc[label_process_cand_pr["pr_number"]==pr_number, "bug"] = 1
            # security risk case
            if label == RISK_TYPE_LABELS[1]:
                label_process_cand_pr.loc[label_process_cand_pr["pr_number"]==pr_number, "security"] = 1
            # performace risk case
            if label == RISK_TYPE_LABELS[2]:
                label_process_cand_pr.loc[label_process_cand_pr["pr_number"]==pr_number, "performance"] = 1
            # maintainability risk case
            if label == RISK_TYPE_LABELS[3]:
                label_process_cand_pr.loc[label_process_cand_pr["pr_number"]==pr_number, "code_quality_or_maintenability"] = 1
            # non-risky case
            if label == RISK_TYPE_LABELS[4]:
                label_process_cand_pr.loc[label_process_cand_pr["pr_number"]==pr_number, "non_risky"] = 1

    label_process_cand_pr.to_csv(label_save_path, index=False)
    print(f"Wrote .csv to: {len(unlabeled_df)}")
         

if __name__ == "__main__":
    # Extract and assign labels to unlabeled data from accepted_labels
    extract_and_assign_labels(1)
        
