from typing import Tuple
import os
import pandas as pd

from .config import (
    CONSENSUS_THRESHOLD,
    RISK_TYPE_LABELS,
    ACCEPTED_FILE_PATTERN,
    HUMAN_REVIEW_FILE_PATTERN,
    ACCEPTED_LABELS_DIR,
    HUMAN_REVIEW_DIR
)
from .data_loader import (
    load_ml_features_dataset,
    get_pr_numbers_from_csv,
    lookup_pr_details,
    save_df
)
from .llm_client import query_gemma, query_llama, query_mistral
from .model_processor import process_all_prs_with_model
from .merger import merge_model_results_and_apply_consensus

MODELS_CONFIG = [
    {'name': 'Gemma', 'query_fn': query_gemma},
    {'name': 'Llama', 'query_fn': query_llama},
    {'name': 'Mistral', 'query_fn': query_mistral}
]


def run_labeling_pipeline(
    loop_number: int,
    ml_features_csv: str,
    pr_list_csv: str,
    max_items: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for the LLM labeling pipeline.
    
    This function processes PRs from a CSV file using three LLM models (Gemma, Llama, Mistral)
    and applies consensus voting to label PRs with risk types.
    
    Args:
        loop_number: Iteration number for output file naming (e.g., 1 -> accepted_labels_1.csv)
        ml_features_csv: Path to the full ML features dataset CSV
        pr_list_csv: Path to CSV containing PR numbers to label (must have 'pr_number' column)
        max_items: Optional limit on number of PRs to process (for testing)
    
    Returns:
        Tuple of (accepted_df, human_df):
            - accepted_df: DataFrame with PRs that reached consensus
            - human_df: DataFrame with PRs that need human review
    
    Output files created:
        - accepted_labels_{loop_number}.csv
        - human_needed_{loop_number}.csv
        - Gemma_predictions_{loop_number}.csv
        - Llama_predictions_{loop_number}.csv
        - Mistral_predictions_{loop_number}.csv
    
    Example:
        from llm_oracle_labeling import run_labeling_pipeline
        
        accepted, human = run_labeling_pipeline(
            loop_number=1,
            ml_features_csv='ML_Label_Input_apache_{project_name}.csv',
            pr_list_csv='loop_1.csv'
        )
    """
    
    # Generate output filenames based on loop_number
    output_accepted_csv = os.path.join(
        ACCEPTED_LABELS_DIR,
        ACCEPTED_FILE_PATTERN.format(loop_number=loop_number)
    )
    output_human_csv = os.path.join(
        HUMAN_REVIEW_DIR,
        HUMAN_REVIEW_FILE_PATTERN.format(loop_number=loop_number)
    )
    
    print('='*80)
    print('🚀 STARTING LLM LABELING PIPELINE')
    print('='*80)
    print(f'Configuration:')
    print(f'  Loop number: {loop_number}')
    print(f'  ML features CSV: {ml_features_csv}')
    print(f'  PR list CSV: {pr_list_csv}')
    print(f'  Output accepted CSV: {output_accepted_csv}')
    print(f'  Output human review CSV: {output_human_csv}')
    print(f'  Consensus threshold: {CONSENSUS_THRESHOLD}/3 models')
    print(f'  Risk type labels: {RISK_TYPE_LABELS}')
    
    # Load PR numbers from CSV
    pr_numbers = get_pr_numbers_from_csv(pr_list_csv)
    
    # Load the full ML features dataset
    ml_features_df = load_ml_features_dataset(ml_features_csv)

    if max_items:
        pr_numbers = pr_numbers[:max_items]
        print(f'  Limited to first {max_items} PRs for testing')

    # Build PR list with details
    pr_list = []
    skipped_count = 0

    for pr_number in pr_numbers:
        pr_number = int(pr_number)
        pr_details = lookup_pr_details(pr_number, ml_features_df)
        if not pr_details.get('pr_title') or pr_details['pr_title'].endswith('(not found in dataset)'):
            print(f'WARNING: Skipping PR #{pr_number} - not found in ML features dataset')
            skipped_count += 1
            continue
        pr_list.append(pr_details)

    print(f'\n📋 Loaded {len(pr_list)} PRs for processing')
    if skipped_count > 0:
        print(f'⚠️  Skipped {skipped_count} PRs not found in dataset')

    model_results = {}

    # Process with each model sequentially
    for model_config in MODELS_CONFIG:
        model_name = model_config['name']
        query_fn = model_config['query_fn']
        model_df = process_all_prs_with_model(pr_list, model_name, query_fn, loop_number)
        model_results[model_name] = model_df

    # Merge and apply consensus
    accepted_df, human_df = merge_model_results_and_apply_consensus(
        model_results, 
        CONSENSUS_THRESHOLD
    )

    # Save results
    save_df(accepted_df, output_accepted_csv)
    save_df(human_df, output_human_csv)

    print('\n' + '='*80)
    print('✅ PIPELINE COMPLETE')
    print('='*80)
    print(f'Per-model files saved in ModelIntermediateOutputs/{{model_name}}/')
    print(f'Final consensus files:')
    print(f'  - {output_accepted_csv}')
    print(f'  - {output_human_csv}')
    
    # Print summary stats
    total = len(accepted_df) + len(human_df)
    if total > 0:
        print(f'\n📊 Summary:')
        print(f'  Total PRs processed: {total}')
        print(f'  ✅ Accepted (consensus): {len(accepted_df)} ({len(accepted_df)/total*100:.1f}%)')
        print(f'  ⚠️  Need human review: {len(human_df)} ({len(human_df)/total*100:.1f}%)')

    return accepted_df, human_df
