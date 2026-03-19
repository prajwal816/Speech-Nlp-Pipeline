import shap
import matplotlib.pyplot as plt
import os

def save_shap_plot(shap_values, candidate_labels, output_path="experiments/shap_plot.png"):
    """
    Saves a SHAP text plot to the specified path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We will generate a standard bar plot for the first instance and the first class (highest probability usually)
    # shap_values[0, :, 0] means 1st instance, all tokens, 1st candidate label
    
    plt.figure()
    shap.plots.bar(shap_values[0, :, 0], show=False)
    
    plt.title(f"SHAP Feature Importance for Label: {candidate_labels[0]}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP feature importance plot saved to {output_path}")
