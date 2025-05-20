import numpy as np
from .coloring import TerminalColor
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def print_overflow_message(values_new: np.ndarray, values_old: np.ndarray, dtype: np.dtype, temperature: float, target_value: float = np.inf):
    warning = TerminalColor.colorize("WARNING", "red", bold=True)
    print_type = str(dtype).split(".")[1].split("\"")[0]

    inf_message = "No value is close to maximum"
    neg_inf_message = "No value is close to minimum"

    if target_value in values_new:
        value = str(values_old[values_new == target_value][0])
        if target_value == np.inf:
            max_val = str(np.finfo(dtype).max)
        elif target_value == 0:
            max_val = str(np.finfo(dtype).smallest_subnormal)
        inf_message = f"{value} is very close to {TerminalColor.colorize('maximum', 'red')} allowed number using np.{print_type:<15}: {max_val}"

    if -target_value in values_new:
        value = str(values_old[values_new == -target_value][0])
        if target_value == np.inf:
            min_val = str(np.finfo(dtype).min)
        else:
            min_val = str(np.finfo(dtype).smallest_subnormal)
        neg_inf_message = f"{value} is very close to {TerminalColor.colorize('minimum', 'blue')} allowed number using np.{print_type:<15}: {min_val}"

    
    lines = [
        f"{warning}: Overflow encountered in power iteration method. Returning estimate from the previous iteration.",
        inf_message,
        neg_inf_message,
        "It is suggested to stop whatever computation is being done and revise the parameters of the model to avoid this problem.",
        f"Temperature parameter: {temperature}"
    ]

    clean_lines = [TerminalColor.strip(line) for line in lines]
    max_width = max(len(line) for line in clean_lines) + 4

    border = f"\t\t\t{TerminalColor.colorize('*' * max_width, 'grey')}"

    message = "\n".join([border] + [f"\t\t\t{TerminalColor.colorize('*', 'grey')} {line}{' ' * (max_width - len(TerminalColor.strip(line)) - 4)} {TerminalColor.colorize('*', 'grey')}" for line in lines] + [border])

    print(message)


def kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-10) -> float:
    
    Q_safe = np.clip(Q, epsilon, None)
    Q_safe = Q_safe / np.sum(Q_safe, axis=1, keepdims=True)
    
    return np.mean(np.sum(P * (np.log(P + epsilon) - np.log(Q_safe)), axis=1))


def plot_colorbar(cmap_name: str, label: str, min: float, max: float, output_dir: str, vertical: bool = False, save_fig: bool = True):
    cmap = plt.get_cmap(cmap_name)
    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots(figsize=(6, 0.1))


    norm = plt.Normalize(vmin=min, vmax=max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax, orientation="vertical" if vertical else "horizontal")
    cbar.set_label(label)
    if save_fig:
        plt.savefig(output_dir, dpi=300, bbox_inches="tight")
    else:
        plt.show()