import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from typing import Literal
from typing_extensions import Self
from custom_palette import CustomPalette

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
    
    def value_fun_evolution_gif(self, out_path: str, out_name: str, other_stats: Self = None):
        custom_palette = CustomPalette()
        self_color = custom_palette[3] if self.descriptor == "PI" else custom_palette[4]
        other_color = custom_palette[3] if other_stats and other_stats.descriptor == "PI" else custom_palette[4]
        
        tmp_dir_path = os.path.join(out_path, "__tmp")
        os.mkdir(tmp_dir_path)
        
        max_iterations = max(len(self.vs), len(other_stats.vs) if other_stats else 0)
        
        min_value = min(
            [min(v) for v in self.vs] +
            ([min(v) for v in other_stats.vs] if other_stats else [])
        )
        
        frame_files = []
        for i in tqdm(range(max_iterations), total=max_iterations, desc="Generating GIF..."):
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
        
        frames = [Image.open(frame) for frame in frame_files]
        frames[0].save(os.path.join(out_path, out_name), save_all=True, append_images=frames[1:], duration=250, loop=0)
        
        for frame_file in frame_files:
            os.remove(frame_file)
        os.rmdir(tmp_dir_path)



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