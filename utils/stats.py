import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
from PIL import Image
from typing import Literal
from typing_extensions import Self
from custom_palette import CustomPalette
import time
from scipy.interpolate import interp1d
from .spinner import Spinner

class ModelBasedAlgsStats:
    def __init__(self, time: float, iterations: int, deltas: list[float], num_states: int, vs: list[np.ndarray], descriptor: Literal["VI", "PI"]):
        self.time = time
        self.iterations = iterations
        self.deltas = deltas
        self.num_states = num_states
        self.vs = vs
        self.descriptor = descriptor
    
    def print_statistics(self):
        print(f"Converged in {self.iterations} iterations.")
        print(f"Time taken: {self.time:.4f} seconds.")
        
    
    def _remove_directory(self, dir_name):
        for file in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, file))
        os.rmdir(dir_name)
    
    def value_fun_evolution_gif(self, out_path: str, out_name: str, other_stats: Self = None):
        custom_palette = CustomPalette()
        self_color = custom_palette[3] if self.descriptor == "PI" else custom_palette[4]
        other_color = custom_palette[3] if other_stats and other_stats.descriptor == "PI" else custom_palette[4]
        
        tmp_dir_path = os.path.join(out_path, "__tmp")
        os.mkdir(tmp_dir_path)
        spinner = None
        try:
        
            max_iterations = max(len(self.vs), len(other_stats.vs) if other_stats else 0)
            
            min_value = min(
                [min(v) for v in self.vs] +
                ([min(v) for v in other_stats.vs] if other_stats else [])
            )
            
            frame_files = []
            for i in tqdm(range(max_iterations), total=max_iterations, desc="Generating GIF Frames..."):
                fig = plt.figure(figsize=(10, 5))
                plt.rcParams.update({"text.usetex": True})
                
                v_self = self.vs[i] if i < len(self.vs) else self.vs[-1]
                plt.plot(np.arange(len(v_self)), v_self, color=self_color, label=self.descriptor)
                
                if other_stats:
                    v_other = other_stats.vs[i] if i < len(other_stats.vs) else other_stats.vs[-1]
                    plt.plot(np.arange(len(v_other)), v_other, color=other_color, label=other_stats.descriptor)
                    min_value = min(min_value, min(v_other))
                
                plt.ylim(min_value, 0)
                plt.title(f"Value Function at iteration {i}")
                plt.grid()
                plt.xlabel("States")
                plt.ylabel("Value Function")
                plt.legend()
                
                frame_file = os.path.join(tmp_dir_path, f"frame_{i}.png")
                plt.savefig(frame_file, dpi=300)
                frame_files.append(frame_file)
                plt.close(fig)
            
            spinner = Spinner("Creating GIF")
            spinner.start()
            frames = [Image.open(frame) for frame in frame_files]
            frames[0].save(os.path.join(out_path, out_name), save_all=True, append_images=frames[1:], duration=250, loop=0)
            spinner.stop()
            
            self._remove_directory(tmp_dir_path)
        
        except KeyboardInterrupt:
            if spinner and spinner.running:
                print(f"Stopping spinner threads".ljust(40))
                spinner.stop(interrupted=True)
            
            print(f"Ctrl+C detected. Removing temporary directory: {tmp_dir_path}")
            self._remove_directory(tmp_dir_path)
            exit()



class GameStats:
    """
    This class contains the statistics of a simulation of a game in a GridWorld or MiniGrid environment.
    """
    def __init__(self, moves: list[int] = [], errors: list[int] = [], deaths: list[int] = []):
        assert len(moves) == len(errors) == len(deaths), "Lenght mismatch"
        self.GAME_INFO = ""
        
        self.moves = moves # The moves for each game
        self.errors = errors # The errors for each game
        self.deaths = deaths # The number of deaths for each game (times that the agent has fallen into a cliff state)
        
        self.num_games = len(moves)
        
        self.update_game_info()
    
    
    def add_game_info(self, moves: int, errors: int, deaths: int) -> None:
        self.moves.append(moves)
        self.errors.append(errors)
        self.deaths.append(deaths)
        
        self.num_games = len(self.moves)
        
        self.update_game_info()
    
    
    def update_game_info(self):
        if len(self.deaths) == 0:
            self.GAME_INFO = "No games played yet"
        else:
            total_deaths = sum(self.deaths)
            total_actions = sum(self.moves)
            total_mistakes = sum(self.errors)
            percentage_correct = round((total_actions - total_mistakes) / total_actions * 100, 2)
            self.GAME_INFO = f"After {self.num_games} games. {total_deaths} deaths. {total_actions} total actions. {total_mistakes} mistakes. {percentage_correct}% correct actions."
        
    
    def get_proportion_errors_round(self):
        return [(error / moves) * 100 for error, moves in zip(self.errors, self.moves)]

    def get_proportion_correct_moves_round(self):
        return [(moves - error) / moves * 100 for error, moves in zip(self.errors, self.moves)]


class EmbeddingStats:
    """
    This class contains the statistics of the embedding process of a MDP into a LMDP.
    It contains the information of the linear search and the ternary search.
    """
    def __init__(
        self,
        type: Literal["vectorized", "iterative"],
        linear_search_lambdas: list[float] = [],
        ternary_search_lambdas: list[float] = [],
        linear_search_errors: list[float] = [],
        ternary_search_errors: list[float] = [],
        ternary_search_mid_1s: list[float] = [],
        ternary_search_errors_mid_1s: list[float] = [],
        ternary_search_mid_2s: list[float] = [],
        ternary_search_errors_mid_2s: list[float] = [],
        ternary_search_lefts: list[float] = [],
        ternary_search_error_lefts: list[float] = [],
        ternary_search_rights: list[float] = [],
        ternary_search_error_rights: list[float] = [],
        optimal_lambda: float = None,
        error_optimal_lambda: float = None,
        total_time: float = None
    ):
        """
        Args:
            type (str): The type of the embedding process. Can be "vectorized" or "iterative".
            linear_search_lambdas (list[float]): The lambdas used in the linear search.
            ternary_search_lambdas (list[float]): The lambdas used in the ternary search.
            linear_search_errors (list[float]): The errors of the linear search.
            ternary_search_errors (list[float]): The errors of the ternary search.
            optimal_lambda (float): The optimal lambda found.
            error_optimal_lambda (float): The error of the optimal lambda found.
            total_time (float): The total time taken for the embedding process.
        """
        assert type in ["vectorized", "iterative"], f"Wrong type {type}. Valid ones: {['vectorized', 'iterative']}"
        
        self.type = type
        
        self.linear_search_lambdas = linear_search_lambdas
        self.ternary_search_lambdas = ternary_search_lambdas
        self.linear_search_errors = linear_search_errors
        self.ternary_search_errors = ternary_search_errors
        self.ternary_search_mid_1s = ternary_search_mid_1s
        self.ternary_search_errors_mid_1s = ternary_search_errors_mid_1s
        self.ternary_search_mid_2s = ternary_search_mid_2s
        self.ternary_search_errors_mid_2s = ternary_search_errors_mid_2s
        self.ternary_search_lefts = ternary_search_lefts
        self.ternary_search_errors_lefts = ternary_search_error_lefts
        self.ternary_search_rights = ternary_search_rights
        self.ternary_search_errors_rights = ternary_search_error_rights
        
        self._optimal_lambda = optimal_lambda
        self._error_optimal_lambda = error_optimal_lambda
        self._total_time = total_time
    
    
    def add_linear_search_info(self, lmbda: float, error: float) -> None:
        """
        Add the information of the linear search to the stats.
        Args:
            lmbda (float): The lambda used in the linear search.
            error (float): The error of the linear search.
        """
        self.linear_search_lambdas.append(lmbda)
        self.linear_search_errors.append(error)
    
    
    def add_optimal_ternary_search_info(self, lmbda: float, error: float) -> None:
        """
        Add the information of a ternary search to the stats.
        Args:
            lmbda (float): The lambda used in the ternary search.
            error (float): The error of the ternary search.
        """
        self.ternary_search_lambdas.append(lmbda)
        self.ternary_search_errors.append(error)
    
    
    def add_ternary_search_info(self, m1: float, error_m1: float, m2: float, error_m2: float, left: float, error_left: float, right: float, error_right) -> None:
        """
        Add the information of a ternary search to the stats.
        Args:
            m1 (float): The first midpoint of the ternary search.
            error_m1 (float): The error of the first midpoint of the ternary search.
            m2 (float): The second midpoint of the ternary search.
            error_m2 (float): The error of the second midpoint of the ternary search.
            left (float): The left bound of the ternary search.
            error_left (float): The error of the left bound of the ternary search.
            right (float): The right bound of the ternary search.
            error_right (float): The error of the right bound of the ternary search.
        Returns:
            None
        """
        self.ternary_search_mid_1s.append(m1)
        self.ternary_search_errors_mid_1s.append(error_m1)
        self.ternary_search_mid_2s.append(m2)
        self.ternary_search_errors_mid_2s.append(error_m2)
        self.ternary_search_lefts.append(left)
        self.ternary_search_rights.append(right)
        self.ternary_search_errors_lefts.append(error_left)
        self.ternary_search_errors_rights.append(error_right)
    
    
    def start_time(self):
        """
        Start the timer for the embedding process.
        """
        self._total_time = time.time()
    
    
    def end_time(self):
        """
        End the timer for the embedding process.
        """
        self._total_time = time.time() - self._total_time
    
    
    def get_total_time(self) -> float:
        """
        Get the total time taken for the embedding process.
        """
        return self._total_time
    
    
    def set_optimal_parameter(self, lmbda: float, error: float) -> None:
        """
        Set the optimal lambda and the error of the optimal lambda.
        Args:
            lmbda (float): The optimal lambda found.
            error (float): The error of the optimal lambda found.
        """
        self._optimal_lambda = lmbda
        self._error_optimal_lambda = error
        
        
    def _interpolate_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the errors of the ternary search to get a smoother curve.
        Returns:
            tuple: A tuple containing the dense lambdas and the dense errors.
        """
        # sorted_lambdas, sorted_errors = zip(*sorted(zip(self.ternary_search_lambdas, self.ternary_search_errors)))
        sorted_lambdas, sorted_errors = zip(*sorted(zip((self.ternary_search_lambdas + self.ternary_search_lefts + self.ternary_search_rights), (self.ternary_search_errors + self.ternary_search_errors_lefts + self.ternary_search_errors_rights))))
        unique = {}
        for lmbda, error in zip(sorted_lambdas, sorted_errors):
            if lmbda not in unique: unique[lmbda] = error
        
        sorted_lambdas = list(unique.keys())
        sorted_errors = list(unique.values())
    
        interp_func = interp1d(sorted_lambdas, sorted_errors, kind="cubic")
        dense_lambdas = np.linspace(min(sorted_lambdas), max(sorted_lambdas), 300)
        dense_errors = interp_func(dense_lambdas)
        
        return dense_lambdas, dense_errors
    
    
    def plot_stats(self, save_fig: bool = True):
        """
        Plot the statistics of the embedding process.
        Args:
            save_fig (bool): Whether to save the figure or not. Defaults to True.
        """
        
        plt.rcParams.update({"text.usetex": True})
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f"Optimal temperature $\lambda$ selection in MDP-LMDP embedding. Time taken: {round(self.get_total_time(), 3)}s")
        
        fig.supxlabel("$\lambda$")
        fig.supylabel("Error")
        axes[0].set_title("Linear search")
        axes[0].plot(self.linear_search_lambdas, self.linear_search_errors, color="blue", linewidth=0.5, alpha=0.5)
        axes[0].scatter(self.linear_search_lambdas, self.linear_search_errors, color="blue", marker="x", s=20, label="Attempted $\lambda$")
        axes[0].scatter(self._optimal_lambda, self._error_optimal_lambda, color="red", zorder=3, s=40)
        
        axes[1].set_title("Refined ternary search")
        axes[1].plot(self.ternary_search_lambdas, self.ternary_search_errors, color="blue", linewidth=0.5, alpha=0.5)
        axes[1].scatter(self.ternary_search_lambdas, self.ternary_search_errors, color="blue", marker="x", s=20, label="Attempted $\lambda$")
        axes[1].scatter(self._optimal_lambda, self._error_optimal_lambda, color="red", zorder=3, s=40, label=f"Optimal $\lambda = {round(self._optimal_lambda, 3)}$")
    
        # Interpolation of expected error
        if len(self.ternary_search_lambdas) >= 2:
            dense_lambdas, dense_errors = self._interpolate_errors()
            axes[1].plot(dense_lambdas, dense_errors, linestyle="--", color="gray", label="Interpolated error")

        
        min_delta = 0.05
        last_annotated = None
        for i, (x, y) in enumerate(zip(self.ternary_search_lambdas, self.ternary_search_errors)):
            if last_annotated is None or abs(x - last_annotated) >= min_delta:
                axes[1].text(x, y, str(i + 1), fontsize=8, color="black", ha="right", va="bottom")
                last_annotated = x

        axes[1].legend()
        
        if save_fig:
            plt.savefig(f"assets/embedding_stats.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    
    def visualize_ternary_search(self, save_fig: bool = True) -> None:
        """
        Visualize the ternary search process through an animation.
        
        Args:
            save_fig (bool): Whether to save the figure or not. Defaults to True.
        Returns:
            None
        """
        plt.rcParams.update({"text.usetex": True})
        fig = plt.figure(figsize=(10, 5))
        self.dense_lambdas, self.dense_errors = self._interpolate_errors()
        
        animation_res = animation.FuncAnimation(fig, self._animate, frames=2 * len(self.ternary_search_mid_1s), interval=400, repeat=True)
        
        if save_fig:
            animation_res.save(f"assets/ternary_search_lambdas_{self._optimal_lambda}.gif", dpi=300)
        else:
            plt.show()
    
    def _animate(self, i) -> None:
        """
        Animation function for the ternary search visualization.
        
        Args:
            i (int): The current frame index.
        Returns:
            None
        """
        idx = i // 2
        plt.clf()
        plt.plot(self.dense_lambdas, self.dense_errors, color="black", label="Interpolated error", linestyle="--")
        
        legend_elements = [
            plt.Line2D([0], [0], color="red", label="$m_1$", linewidth=0.5),
            plt.Line2D([0], [0], color="blue", label="$m_2$", linewidth=0.5),
            plt.Line2D([0], [0], color="gray", label="Search region", alpha=0.2, linewidth=5),
            plt.Line2D([0], [0], color="black", label="Interpolated error", linestyle="--"),
            Line2D([0], [0], color="#79FF00", label=fr"$\lambda_{{\mathrm{{curr}}}}^* = {round(self.ternary_search_lambdas[idx], 4)}$", linestyle="None", markersize=6, marker="x")
        ]

        plt.xlabel("$\lambda$")
        plt.ylabel("Error")
        plt.title(f"Optimal $\lambda$ Finding through Ternary Search. Step [{idx + 1} / {len(self.ternary_search_errors_mid_1s)}]")
        
        plt.axvline(self.ternary_search_mid_1s[idx], color="red", linewidth=0.5)
        plt.axvline(self.ternary_search_mid_2s[idx], color="blue", linewidth=0.5)
        if i % 2 != 0:
            plt.axvline(self.ternary_search_lambdas[idx], color="#4EA300", linewidth=0.5)
            plt.scatter(self.ternary_search_lambdas[idx], self.ternary_search_errors[idx], color="#79FF00", marker="x", zorder=3)
            
        plt.axvspan(self.ternary_search_lefts[idx], self.ternary_search_rights[idx], color="gray", alpha=0.2)
        
        
        plt.legend(handles=legend_elements, loc="upper left")