import matplotlib.pyplot as plt
import numpy as np


def create_heatmaps(results_inference, path):
    
    idx = results_inference.keys()
    
    # Extract the size of the matrix
    min_idx = min([int(i.split('_')[1]) for i in idx])
    max_idx = max([int(i.split('_')[1]) for i in idx])
    size = max_idx - min_idx + 1

    # Initialize matrices for accuracy and loss
    accuracy_matrix = np.zeros((size, size))
    loss_matrix = np.zeros((size, size))

    # Populate the matrices with data from results_inference
    for key, value in results_inference.items():
        x, y = map(int, key.split('_'))
        accuracy_matrix[x - min_idx, y - min_idx] = value["accuracy"]
        loss_matrix[x - min_idx, y - min_idx] = value["loss"]

    # Function to plot heatmap
    def plot_heatmap(matrix, title, filename):
        plt.figure(figsize=(10, 8))
        # Set vmin to 0 and vmax to 1 to fix the scale from 0 to 1
        plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.xticks(ticks=np.arange(size), labels=np.arange(min_idx, max_idx + 1))
        plt.yticks(ticks=np.arange(size), labels=np.arange(min_idx, max_idx + 1))
        plt.savefig(filename)
        plt.close()


    accuracy_filename = f"{path}/accuracy.png"
    loss_filename = f"{path}/loss.png"

    # Plot and save accuracy heatmap
    plot_heatmap(accuracy_matrix, "Accuracy Heatmap", accuracy_filename)

    # Plot and save loss heatmap
    plot_heatmap(loss_matrix, "Loss Heatmap", loss_filename)
