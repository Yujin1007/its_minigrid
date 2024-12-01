from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

from toy_examples_main import examples
import os
from numpy.typing import NDArray

def load_map_from_example_dict(example_name: str) -> NDArray:
    """
    Load the map from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        map_array: NDArray
            - The map array
    """
    return examples[example_name]["map_array"]


def load_starting_pos_from_example_dict(example_name: str) -> NDArray:
    """
    Load the starting position from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        starting_pos: NDArray
            - The starting position of the agent
    """
    return examples[example_name]["starting_pos"]

def load_goal_pos_from_example_dict(example_name: str) -> NDArray:
    """
    Load the goal position from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        starting_pos: NDArray
            - The goal position of the agent
    """
    return examples[example_name]["goal_pos"]



def get_output_folder_name(data_log_dir) -> str:
    """
    Return:
        folder_name: str
            - The name of the folder that holds all the logs/outputs for the current run
            - Not a complete path
    """
    # Remove the path to the repo directory
    folder_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.replace(str(os.getcwd()), "")
    # Remove the name of the folder that holds all the outputs
    folder_name = folder_path.replace(data_log_dir, "").replace("/", "")  # TODO: a hack

    return folder_name


def get_output_path() -> str:
    """
    Return:
        output_path: str
            - The absolute path to the folder that holds all the logs/outputs for the current run
    """
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

def get_model_path():
    folder_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.replace(str(os.getcwd()), "")
    # Remove the name of the folder that holds all the outputs
    folder_name = folder_path.replace(data_log_dir, "").replace("/", "")  # TODO: a hack

def get_model_path(dir, file) -> str:
    base_path_absolute = hydra.utils.to_absolute_path(dir)
    full_path = os.path.join(base_path_absolute, file)

    return full_path