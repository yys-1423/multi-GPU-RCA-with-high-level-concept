#!/usr/bin/env python3
"""
Plot step times (latencies) for each rank from training metrics.
"""

import json
import glob
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    print("The script will only print statistics without generating plots.")

def load_metrics(metrics_dir='metrics'):
    """Load metrics from all rank files."""
    rank_data = {}

    # Find all training rank files
    rank_files = sorted(glob.glob(os.path.join(metrics_dir, 'training_rank*.jsonl')))

    for rank_file in rank_files:
        # Extract rank number from filename
        rank_num = int(rank_file.split('rank')[-1].split('.')[0])

        steps = []
        step_times = []

        # Read JSONL file
        with open(rank_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                steps.append(data['step'])
                step_times.append(data['step_time'])

        rank_data[rank_num] = {
            'steps': steps,
            'step_times': step_times
        }

    return rank_data

def plot_step_times(rank_data, output_file='step_times.png', annotate_threshold=1.5):
    """Plot step times for each rank.

    Args:
        rank_data: Dictionary of rank data
        output_file: Output filename for the plot
        annotate_threshold: Annotate spikes that are this many times above the mean (default 1.5 = 50% increase)
    """
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(14, 7))

        colors = plt.cm.tab10(range(len(rank_data)))

        # Plot each rank and annotate spikes
        for idx, rank_num in enumerate(sorted(rank_data.keys())):
            data = rank_data[rank_num]
            steps = data['steps']
            step_times = data['step_times']

            # Plot the line
            line = plt.plot(steps, step_times,
                    marker='o', markersize=3, linewidth=1.5,
                    label=f'Rank {rank_num}', alpha=0.7, color=colors[idx])

            # Find spikes to annotate
            mean_time = sum(step_times) / len(step_times)

            # Annotate points that are significantly above the mean
            for i, (step, step_time) in enumerate(zip(steps, step_times)):
                if step_time > mean_time * annotate_threshold:
                    # Offset annotations for different ranks to avoid overlap
                    offset = 10 + (idx * 8)
                    plt.annotate(f'Step {step}',
                               xy=(step, step_time),
                               xytext=(0, offset),
                               textcoords='offset points',
                               fontsize=9,
                               color=colors[idx],
                               bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white',
                                       edgecolor=colors[idx],
                                       alpha=0.8),
                               arrowprops=dict(arrowstyle='->',
                                             connectionstyle='arc3,rad=0',
                                             color=colors[idx],
                                             lw=1.5))

        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Step Time (seconds)', fontsize=12)
        plt.title('Training Step Times by Rank (Spikes Annotated)', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close()
    else:
        print("Skipping plot generation (matplotlib not available)")

    # Display statistics
    print("\nStep Time Statistics:")
    print("-" * 60)
    for rank_num in sorted(rank_data.keys()):
        step_times = rank_data[rank_num]['step_times']
        print(f"Rank {rank_num}:")
        print(f"  Mean: {sum(step_times)/len(step_times):.4f}s")
        print(f"  Min:  {min(step_times):.4f}s")
        print(f"  Max:  {max(step_times):.4f}s")
        print(f"  Total steps: {len(step_times)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot step times from training metrics')
    parser.add_argument('--metrics-dir', default='metrics',
                       help='Directory containing training metrics (default: metrics)')
    parser.add_argument('--output', default='step_times.png',
                       help='Output file for the plot (default: step_times.png)')
    parser.add_argument('--spike-threshold', type=float, default=1.5,
                       help='Annotate spikes that are this many times above the mean (default: 1.5 = 50%% increase)')

    args = parser.parse_args()

    # Load and plot data
    print(f"Loading metrics from {args.metrics_dir}...")
    rank_data = load_metrics(args.metrics_dir)
    print(f"Found {len(rank_data)} ranks")

    plot_step_times(rank_data, args.output, args.spike_threshold)
