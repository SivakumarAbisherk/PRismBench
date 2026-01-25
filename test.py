from pathlib import Path
from uncertainty_selection.run_selection import run_uncertainty_selection

selected_df = run_uncertainty_selection(
        loop_number=1,
        project="kafka",
        data_root=Path("SamplingLoopData"),
        output_dir=Path("UncertainPoint"),
        n_bootstrap=20,
        top_uncertain=100,
        k_diverse=25,
        top_k_labels=3,
        verbose=True
    )