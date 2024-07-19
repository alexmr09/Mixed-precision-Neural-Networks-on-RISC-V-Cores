import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from brokenaxes import brokenaxes

def find_pareto_optimal_points(latencies, mpq_accuracies, mpq_confs):
    """
    Finds the Pareto optimal points given latencies and accuracies.

    Args:
        latencies (numpy.ndarray): An array of latency values.
        accuracies (numpy.ndarray): An array of accuracy values.
        mpq_confs (numpy.ndarray): An array with the sorted weight_bit_width per layer list based on
                                    estimated latency

    Returns:
        tuple: A tuple containing the following:
            - numpy.ndarray: Array of Pareto optimal points (Latency, Accuracy).
            - numpy.ndarray: Indices of Pareto optimal accuracies in the original array.
    """
    data = {
        'Configuration': [mpq_confs[idx].astype(int) for idx in range(len(mpq_confs))],
        'Accuracy': mpq_accuracies,
        'MACC Instrs': latencies
    }

    # Combine latencies, accuracies, and indices
    combined = np.column_stack((latencies, mpq_accuracies, np.arange(len(mpq_accuracies))))

    # Initialize the Pareto front with the first point
    pareto_front = [combined[0]]
    for pair in combined[1:]:
        # Check if the accuracy of the current pair is better than the last point in the Pareto front
        if pair[1] > pareto_front[-1][1]:
            pareto_front.append(pair)

    # Extract Pareto points and indices
    pareto_points_with_indices = np.array(pareto_front)
    pareto_points = pareto_points_with_indices[:, :2]
    pareto_indices = pareto_points_with_indices[:, 2].astype(int)

    # Set print options for clarity
    np.set_printoptions(precision = 2, suppress = True)
    
        # Create DataFrame
    data = {
        'Index': pareto_indices,
        'Weights Resolutions': [mpq_confs[idx].astype(int) for idx in pareto_indices],
        'Accuracy': [point[1] for point in pareto_points],
        'MACC Instrs': [point[0] for point in pareto_points]
    }

    df = pd.DataFrame(data)

    return df, pareto_points, pareto_indices

def identify_best_solution(pareto_solutions_df, fp_accuracy, maximum_acc_loss = 1):
    threshold = fp_accuracy - maximum_acc_loss
    filtered_df = pareto_solutions_df[pareto_solutions_df['Accuracy'] >= threshold]
    filtered_df = filtered_df.sort_values(by = 'MACC Instrs', ascending=True)

    if not filtered_df.empty:
        best_config = filtered_df.iloc[0]  # Get the first row
        print(f"\nBest Configuration Details with Maximum Accuracy loss < {maximum_acc_loss}%:")
        print(f"Index: {best_config['Index']}")
        print(f"Weights Resolutions: {best_config['Weights Resolutions']}")
        print(f"Accuracy: {best_config['Accuracy']:.2f}%")
        print(f"MACC Instrs: {best_config['MACC Instrs']}")
    else:
        print("No configurations meet the specified accuracy threshold.")

    return best_config

def plot_pareto_solutions(mpq_mac, fp_mac_instr, accuracy, 
                          fp_accuracy, pareto_points, name):
    """
    Plots Pareto solutions based on MAC instructions, accuracy, and other parameters.

    Args:
        mpq_mac (numpy.ndarray): Array of MAC instructions for mixed precision layer configurations.
        fp_mac_instr (float): Original (non-optimized) model's MAC instructions.
        accuracy (numpy.ndarray): Array of accuracy values for mixed precision layer configurations.
        fp_accuracy (float): Original (non-optimized) model's accuracy.
        pareto_points (numpy.ndarray): Array of Pareto optimal points (Latency, Accuracy).

    Returns:
        None: Displays the plot and saves it as a PNG file.
    """
    # Define the x-axis and y-axis ranges

    ranges = [(0, max(mpq_mac) + 0.5 * max(mpq_mac)), 
              (fp_mac_instr - 0.15 * fp_mac_instr, fp_mac_instr + 0.15 * fp_mac_instr)]

    # Calculate the exponent for notation
    exponent = int(np.log10(fp_mac_instr))
    notation = 10 ** exponent

    # Create the figure and broken axes
    fig = plt.figure(figsize = (6, 4))
    fig.subplots_adjust(left = 0.148, right = 0.88, top = 0.95, bottom = 0.2)
    bax = brokenaxes(xlims = ranges, hspace = 0.05)

    # Scatter plot for original model, mixed precision, and Pareto points
    bax.scatter(fp_mac_instr, fp_accuracy, color = 'black', s = 9, marker = '*', 
                label = 'original model')

    bax.scatter(mpq_mac, accuracy, color = 'gray', marker = 'o', s = 9, 
                label = 'mixed configs')
    
    bax.scatter(pareto_points[:, 0], pareto_points[:, 1], color = 'green', marker = 's', s = 9,
                label = 'pareto points')

    # Customize the plot
    bax.set_ylabel('Accuracy (%)', fontsize = 10, labelpad = 35)
    bax.grid(True)

    # Set the y-axis limit if provided
    
    y_lims = [max(50, 0.8 * np.min(accuracy)), min(1.1 * fp_accuracy, 100)]
    bax.set_ylim(y_lims[0], y_lims[1])

    # Place the legend
    bax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.2), ncol = 3, fontsize = 10)

    # Format x-axis labels
    sFormatter1 = ticker.FuncFormatter(lambda x, _: '{:0.2f}'.format(x / notation))
    sFormatter2 = ticker.ScalarFormatter(useOffset = False, useMathText = True)
    sFormatter2.set_powerlimits((0, 2))
    
    bax.axs[0].xaxis.set_major_formatter(sFormatter1)
    bax.axs[1].xaxis.set_major_formatter(sFormatter2)

    bax.axs[0].tick_params(axis = 'y', labelsize = 10)
    bax.axs[0].tick_params(axis = 'x', labelsize = 10)
    bax.axs[1].tick_params(axis = 'x', labelsize = 10)

    bax.axs[1].xaxis.get_offset_text().set_position((1, 0))
    bax.axs[1].xaxis.get_offset_text().set_fontsize(10)

    # Reposition the x-axis label
    bax.set_xlabel('MAC Instructions', ha = 'center', fontsize = 10, labelpad = 25)

    plt.savefig(name + ".png")

def pareto_space(fp_accuracy, test_accuracy, weights_per_layer, macc_per_layer, total_macc_opt_sorted, name):
    df, pareto_points, _ = find_pareto_optimal_points(np.array(total_macc_opt_sorted),
                                        np.array(test_accuracy), np.array(weights_per_layer))

    optimal_config = identify_best_solution(df, np.array(fp_accuracy))
    
    plot_pareto_solutions(np.array(total_macc_opt_sorted), 
                          sum(macc_per_layer), np.array(test_accuracy), 
                          float(fp_accuracy), pareto_points, name)
    
    optimal_config = optimal_config.iloc[1]

    return optimal_config.tolist()
