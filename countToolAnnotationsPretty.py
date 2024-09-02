import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_tool_counts(tool_counts_over_time, save_path):
    # Set the Seaborn theme and context
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")  # You can switch to "poster", "talk", or "notebook" depending on your needs
    
    percentages = [10 * i for i in range(1, 11)]
    tools = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    
    # Initialize the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each tool's counts with a Seaborn color palette
    palette = sns.color_palette("deep")  # You can choose different palettes like "muted", "bright", etc.
    for tool, color in zip(tools, palette):
        counts = [tool_counts.get(tool, 0) for tool_counts in tool_counts_over_time]
        plt.plot(percentages, counts, marker='o', label=tool, color=color)
    
    # Add labels and title
    plt.xlabel('Percentage of Images Selected')
    plt.ylabel('Total Count of Each Tool')
    plt.title('Tool Counts by Percentage of Images Selected')
    plt.legend()
    plt.grid(True)
    plt.xticks(percentages)
    
    # Ensure the destination folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close() 
