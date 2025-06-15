# Efficient Reinforcement Learning with Transition-Dependent LMDPs and Entropy Regularization

This repository contains the code developed for my Bachelor's thesis, *Efficient Reinforcement Learning with Transition-Dependent LMDPs and Entropy Regularization*, which explores on the advantages that LMDPs have over MDPs. The thesis demonstrates that transition-dependent LMDPs offer a more intuitive way to specify control problems while preserving the computational benefits of linear solvability. By adapting Todorov’s embedding method, standard MDPs can be transformed into this LMDP framework, and a formal equivalence is established between transition and state-dependent LMDPs, both yielding the same optimal policies. To ensure a fair comparison, entropy regularization is added to MDPs and integrated into the embedding process. 


## 1. Installation

> <span style="color:red">NOTE</span>: **The code for this project has been developed and tested using Unix-like systems (particularly Ubuntu and MacOS)**. Despite some things have been tested on Windows, and despite Python being cross-platform, it is not guaranteed that everything will work as expected on a Windows system. For this reason, the installation process and the usage of the code is described for Unix-like systems. If the user is using Windows, some of the commands may need to be adapted.

This project uses [`uv`](https://docs.astral.sh/uv/) as the package and project manager. It has been chosen for its speed and reliablity, offering a much faster alternative for managing Python projects than other traditional tools such as `pip`.

### 1.1. Prerequisites

Install `uv` in the system using the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows users, should refer to the official [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) for instructions on how to install `uv` in the system.

`uv` can also be used to install Python in the system. This is not required, but it is recommended to avoid any conflicts with other Python installations. To install Python using `uv`, run the following command:

```bash
uv python install <version>
```

Replace `<version>` with teh desired Python version. This project has been developed and tested using Python 3.11.2. If this step wants to be ommited, `uv` will automatically detect the version required by the project from the `.python-version` file in the root directory of the project and install it automatically when the project is created.

### 1.2. Setup

In order to create the project, run the following command in the root directory of the project:

```bash
uv venv
```

Once activated, install the project dependencies directly from the `pyproject.toml` and `uv.lock` files using:

```bash
uv sync
```
This will create a virtual environment in the `.venv` directory and install all the required dependencies for the project. The dependencies are listed in the `pyproject.toml` file, and `uv.lock` file is used to lock the versions of the dependencies to ensure reproducibility.


### 1.3. Additional Requirements

This project uses rendered LaTeX equations in some of the generated plots. `matplotlib` supports LaTeX rendering, which allows for mathematical expressions to be displayed in the generated plots. However, this requires a working LaTeX installation on your system. The following packages are required for LaTeX rendering inside `matplotlib`:
- `dvipng` - for rendering LaTeX equations.
- `texlive-latex-extra` - for additional useful LaTeX packages.
- `texlive-fonts-recommended` - essential fonts for LaTeX rendering.
- `cm-super` - for better font rendering.
- `texlive-latex-base` - core LaTeX classes and packages.

On Ubuntu/Debian systems, these packages can be installed using the following command:

```bash
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super texlive-latex-base
```

On MacOS, it is not possible to directly install these packages using `brew`, as they are part of the MacTeX distribution. Instead, you can install the entire MacTeX distribution, which includes all the necessary packages. The MacTeX installer can be downloaded from [here](https://www.tug.org/mactex/). Alternatively, if Homebrew is installed in the system, MacTeX can be installed using the following command:

```bash
brew install --cask mactex
```

## 2. Execution

The code can be executed using the following command:

```bash
uv run <script_name>.py
```
The repository contains tests, to have an easy way to ensure that the code is working as expected. The tests are located in the `test` directory. To run the tests, use the following command:

```bash
uv run -m test.<script_name>.py
```

## 3. Code structure

### 3.1. `models` directory

The `models` directory encapsulates the core mathematical formulations used in the project. It contains implementations of different types of decision-making models, all of which are variations of Markov Decision Processes (MDPs). Each file in this directory corresponds to a distinct model, and an additional utility file which includes a function that is common to all models.

- `MDP.py`: This file defines the standard MDP model. It contains the **Value Iteration** algorithm, a model-based algorithm that computes the optimal policy and value function for a given MDP. It also contains the embedding methods to convert the MDP into an equivalent linearly-solvable version with state-dependent rewards (LMDP), or with transition-dependent rewards (LMDP_TDR). It is possible to define an MDP with entropy regularization, which is a technique used to encourage exploration in reinforcement learning. The entropy regularization term is added to the reward function, and the value iteration algorithm is modified to account for this additional term.

- `LMDP.py`: Defines the Linearly Solvable Markov Decision Process (LMDP), a variant of MDPs introduced by Todorov in 2006, that leverages the structure of passive dynamics and exponential rewards to allow for a more efficient solution. The optimal policy is computed using the **Power Iteration** method, which exploits the linearity of the problem under a logarithmic transformation. The class also implements the embedding techniques that allow to convert an LMDP into an equivalent MDP or into an equivalent LMDP with transition-dependent rewards.

- `LMDP_TDR.py`: Contains the implementation of the Linearly Solvable MDP with Transition-Dependent Rewards (LMDP_TDR). This extends the LMDP by allowing rewards to depend on both the current state and the next state in a transition. It contains the method to convert it into an identical LMDP with state-dependent rewards, one of the main contributions of this project.
  
- `utils.py`: Includes utility functions shared across the models. The main function provided here checks whether two models—regardless of type—are equal. This is useful for debugging, testing, and verifying consistency between model implementations.


### 3.2. `domains` directory

The `domains` directory defines the environments in which the `models` from the models directory are deployed and tested. These environments simulate grid-based worlds where an agent interacts through actions and receives feedback in the form of rewards. Each file supports different configurations of grid environments, and integrates with the corresponding MDP variants.

- `grid_world.py`: Provides the full implementation of a gridworld simulation environment. Key components include:
  - The `GridWorldEnv` class, which manages the overall environment setup: agent movement, allowed actions, state transitions, episode length, and rendering.
  - Separate classes for each model variant: `GridWorldMDP`, `GridWorldLMDP`, and `GridWorldLMDP_TDR`. These extend the base functionality of the environment to work with their respective model formulations.
  - Visualization tools using matplotlib and pygame for plotting agent behavior, value functions, and policies within the grid.

- `grid.py`: Contains the `CustomGrid` class, responsible for generating grid environments from textual map representations. This class defines the state layout and movement mechanics of the agent, and can be used with both `GridWorldEnv` and `MiniGridEnv`. It abstracts the map generation and agent dynamics to support flexible and reusable grid definitions.

- `minigrid_env.py`: Extends the same features as in the `grid_world.py` file but for the MiniGrid environment. It provides a more complex and challenging environment for the agent to navigate, with a larger state space size and more complex dynamics such as picking and dropping objects and openeing and closing doors.


### 3.3. `utils` directory

The `utils` directory contain additional functions and classes that are not directly related to the core models or environments, but are useful for 

The files contianed in this directory are:
- `benchmarks.py`: This file contains the benchmark functions used to evaluate the performance of different implementations and algorithms.
- `coloring.py`: This file contains the `TerminalColor` class which is used to color the output in the terminal. It provides methods to print text in different colors, making it easier to identify important information in the output such as warnings, errors or results.
- `experiments.py`: This file contains the functions used to run the experiments and some plots from which the main results and conclusions of the project are drawn.
- `maps.py`: Contains the `Map` class which is used to define the maps used in the MiniGrid and GridWorld environments. It also contains the `Maps` class, which contains as static variables the main recurrent maps used in the project.
- `spinner.py`: This file contains the `Spinner` class which is used to show a loading spinner in the terminal. It provides a visual indication that a process is running, making it easier and more informative to identify a time-consuming task that takes a while to complete.
- `state.py`: This file contains the `Object` class which is used to define the objects used in the different domains. The current implementation only supports doors and keys. The other class present in the file is the `State` class, which is used to define each state in the state space of a moodel.
- `utils.py`: This file defines standalone utility functions that are shared across the codebase but do not fit neatly into any specific domain or model logic.

### 3.4. `test` directory

The `test` directory contains the test files that have been used to ensure that certain implementations are working as expected, and that other parts of the code executed without errors. The tests are organized into different files, each corresponding to a specific model or domain. The tests are implemented using the `unittest` framework, which is a built-in Python module for creating and running unit tests.

### 3.5. Other files

- `custom_palette.py`: This file contains the custom color palette used in the project. It defines a set of colors that are used throughout the code for consistency and visual appeal.

- `algorithms.py`: This file was initially created as part of early experimentation to gain familiarity with the reinforcement learning and MDP paradigm. It was intended to host model-free algorithms, in contrast to the model-based direction the project ultimately took. The file includes:
  - A `QLearning` class implementing the Q-learning algorithm, a model-free method that estimates the value of actions in each state through direct interaction with the environment.
  - A hyperparameter explorer to systematically test combinations of learning rates, discount factors, and exploration strategies (`QLearningHyperparameters` and `QLearningHyperparameterExplorer`).
  - A plotting utility to visualize the effects of different hyperparameter settings on learning performance (`QLearningPlotter`).

## References
Todorov, E. (2006). Linearly-solvable Markov decision problems. Advances in neural information processing systems, 19.

Todorov, E. (2009). Efficient computation of optimal actions. Proceedings of the national academy of sciences, 106(28), 11478-11483.

Dvijotham, K., & Todorov, E. (2010). Inverse optimal control with linearly-solvable MDPs. In Proceedings of the 27th International conference on machine learning (ICML-10) (pp. 335-342).

Jonsson, A., & Gómez, V. (2016). Hierarchical Linearly-Solvable Markov Decision Problems. Proceedings of the International Conference on Automated Planning and Scheduling, 26(1), 193-201. https://doi.org/10.1609/icaps.v26i1.13750

Neu, G., Jonsson, A., & Gómez, V. (2017). A unified view of entropy-regularized markov decision processes. arXiv preprint arXiv:1705.07798.

