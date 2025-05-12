import numpy as np
from .coloring import TerminalColor

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