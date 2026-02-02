from pathlib import Path
# from uncertainty_selection.run_selection import run_uncertainty_selection
from uncertainty_with_dnn.run_unceratinty_selection import run_uncertainty_selection

from llm_oracle_labeling import run_labeling_pipeline, analyze_labeling_quality
from extract_and_assign_labels import extract_and_assign_labels

def main():
    

    num_of_loops = 2
    ml_features_csv = 'ML_Label_Input_apache_beam.csv'

    uncertain_output_dir=Path("UncertainPoint")
    max_items = None # to limit llm labeling pr counts for testing purpose

    loop_number = 1
    loop_limit = loop_number + num_of_loops
    while(loop_number < loop_limit):
        pr_list_csv = uncertain_output_dir / f"loop_{loop_number}_selected.csv" 
        
        selected = run_uncertainty_selection(
            ml_features_csv='ML_Label_Input_apache_beam.csv',
            loop_number=loop_number,
            data_root=Path("SamplingLoopData"),
            output_dir=Path("UncertainPoint"),
            model_monitor_dir=Path("ModelMonitoring"),
            n_top_uncertain=100,
            k_diverse=25,
            metric="euclidean",
            verbose=True
        )

        # extract_and_assign_labels(loop_number)
        # print('\n Run completed successfully!')
        
        try:
            accepted_df, human_df = run_labeling_pipeline(
                loop_number=loop_number,
                ml_features_csv=ml_features_csv,
                pr_list_csv=pr_list_csv,
                max_items=max_items
            )
            
            if len(accepted_df) > 0 or len(human_df) > 0:
                print('\n' + '='*80)
                stats = analyze_labeling_quality(accepted_df, human_df)
                print('\n Analysis summary:', stats)
            
            extract_and_assign_labels(loop_number)
            print('\n Run completed successfully!')
            
            

        except FileNotFoundError as e:
            print(f'\n Error: File not found - {e}')
            print('Please ensure the CSV files exist in the current directory.')
            
        except Exception as e:
            print(f'\n Fatal error during pipeline execution: {e}')
            import traceback
            traceback.print_exc()
        
        loop_number += 1


if __name__ == '__main__':
    main()
