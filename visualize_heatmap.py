import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_heatmap(heatmap_grid, save_path="heatmap.png", title="Exploration Heatmap"):
    """
    Visualize and save the exploration heatmap.
    
    Args:
        heatmap_grid: 2D numpy array of visit counts
        save_path: Path to save the heatmap image
        title: Title for the plot
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap_grid.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Visit Count')
    plt.title(title)
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    # Example: Load and visualize a saved heatmap
    # This would be called from train.py or used standalone
    print("Heatmap visualization utility ready.")
