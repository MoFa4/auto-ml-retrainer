"""
Compare new model with current production model and decide deployment
"""
import sys
import joblib
from evaluate import load_current_model_metadata, compare_models, load_config

def main():
    """
    Main comparison logic
    Usage: python src/compare.py <new_model_r2_score>
    """
    if len(sys.argv) < 2:
        print("Usage: python src/compare.py <new_model_r2_score>")
        sys.exit(1)
    
    # Get new model metrics (passed as argument)
    new_r2 = float(sys.argv[1])
    new_metrics = {'r2_score': new_r2}
    
    # Load current model metadata
    current_metadata = load_current_model_metadata()
    
    # Load config
    config = load_config()
    
    # Compare and decide
    should_deploy, reason = compare_models(new_metrics, current_metadata, config)
    
    # Output decision
    print("\n" + "=" * 60)
    if should_deploy:
        print("🚀 DECISION: DEPLOY NEW MODEL")
        print(f"📝 Reason: {reason}")
        sys.exit(0)  # Success exit code
    else:
        print("🛑 DECISION: KEEP CURRENT MODEL")
        print(f"📝 Reason: {reason}")
        sys.exit(1)  # Failure exit code (prevents deployment)
    
if __name__ == "__main__":
    main()