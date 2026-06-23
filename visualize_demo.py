import matplotlib.pyplot as plt
from matplotlib.colors import ListedCollection
import numpy as np
from arc_solver import Grid, crop_to_content, color_swap, scale_up

# Standard ARC colors (0-9)
ARC_COLORS = [
    '#000000', # 0: Black (Background)
    '#0074D9', # 1: Blue
    '#FF4136', # 2: Red
    '#2ECC40', # 3: Green
    '#FFDC00', # 4: Yellow
    '#AAAAAA', # 5: Grey
    '#F012BE', # 6: Magenta
    '#FF851B', # 7: Orange
    '#7FDBFF', # 8: Teal
    '#870C25'  # 9: Maroon
]
cmap = ListedCollection(ARC_COLORS)

def plot_grid(ax, grid, title):
    """Plot a single ARC grid on a matplotlib axis."""
    ax.imshow(grid.data, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=12, pad=10)
    
    # Draw grid lines
    h, w = grid.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

def visualize_beam_search_pipeline():
    """
    Simulates a 3-step composed rule found by Beam Search to show the demo audience 
    how complex rules are built step-by-step.
    """
    # Create a synthetic "Input" Grid that needs 3 steps to become the "Output"
    # Imagine a large black grid with a small red (2) and blue (1) shape
    input_data = np.zeros((10, 10), dtype=int)
    input_data[2:5, 3:6] = [
        [0, 2, 0],
        [2, 1, 2],
        [0, 2, 0]
    ]
    inp_grid = Grid(input_data)
    
    print("Simulating Beam Search Rule Composition...")
    print("Goal: Extract shape, swap red (2) to yellow (4), and scale up 2x.")
    
    # Step 1: Crop to Content (extract object)
    step1_grid = crop_to_content(inp_grid, bg=0)
    
    # Step 2: Color Swap (2 -> 4)
    step2_grid = color_swap(step1_grid, 2, 4)
    
    # Step 3: Scale Up by 2x
    final_grid = scale_up(step2_grid, 2)
    
    # Plotting the Pipeline
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Beam Search Discovery: Composing Rules to solve complex ARC Tasks", fontsize=16, y=1.05)
    
    plot_grid(axes[0], inp_grid, "1. Raw Input")
    plot_grid(axes[1], step1_grid, "2. Crop to Content\n(Object Extraction)")
    plot_grid(axes[2], step2_grid, "3. Color Swap\n(Red -> Yellow)")
    plot_grid(axes[3], final_grid, "4. Final Output\n(Scale Up 2x)")
    
    # Draw arrows between plots
    for i in range(3):
        axes[i].annotate('', xy=(1.15, 0.5), xycoords='axes fraction', 
                         xytext=(1.05, 0.5), textcoords='axes fraction',
                         arrowprops=dict(facecolor='black', shrink=0.05, width=2))
                         
    plt.tight_layout()
    
    # Save the figure so you can embed it in presentations/artifacts
    output_filename = "demo_visualization.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_filename}")
    
    # If running interactively, this will show the window
    plt.show()

if __name__ == "__main__":
    visualize_beam_search_pipeline()
