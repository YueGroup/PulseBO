from src.validation import run_validation_analysis
from src.baseline import run_baseline_comparison
from src.ablation import run_ablation_analysis
from src.BO.bo_candidates import run_bo_candidates

def run_core_pipeline() -> dict:

    outputs = {}

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION ANALYSIS")
    print("=" * 80)
    outputs["validation"] = run_validation_analysis()

    print("\n" + "=" * 80)
    print("RUNNING BASELINE COMPARISON")
    print("=" * 80)
    outputs["baseline"] = run_baseline_comparison()

    print("\n" + "=" * 80)
    print("RUNNING ABLATION ANALYSIS")
    print("=" * 80)
    outputs["ablation"] = run_ablation_analysis()

    print("\n" + "=" * 80)
    print("RUNNING BO CANDIDATE GENERATION")
    print("=" * 80)
    outputs["bo_candidates"] = run_bo_candidates()

    print("\n" + "=" * 80)
    print("CORE PIPELINE COMPLETE")
    print("=" * 80)

    return outputs


