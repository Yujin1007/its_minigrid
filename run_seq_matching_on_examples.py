from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import matplotlib.pyplot as plt
import os
from loguru import logger
from tqdm import tqdm

from seq_matching_toy.toy_examples_main import examples
from seq_reward.seq_utils import get_matching_fn, plot_matrix_as_heatmap_on_ax

def prepare_seq_matching_fns(seq_matching_fn_configs, cost_fn_name, reward_vmin, reward_vmax):
    """
    Return:
        a dictionary that maps the function to its name
    """
    seq_matching_fns_dict = {}

    for fn_config in seq_matching_fn_configs:
        fn_config = dict(fn_config)
        fn_config["reward_vmin"] = reward_vmin
        fn_config["reward_vmax"] = reward_vmax

        fn, fn_name = get_matching_fn(fn_config, cost_fn_name)
        seq_matching_fns_dict[fn_name] = fn
        
    return seq_matching_fns_dict

    
def run_examples_from_config(cfg: DictConfig):
    """
    Examples are stored in dictionaries in indvidual python files  
        Each dictionary has
        - ref_seq: reference sequence, which is a list of numpy arrays
        - obs_seq: a dictionary of observed sequences, key is the id, the value is a observation sequence, which is a dictionary with
            - descriptions: a string describing the sequence
            - seq: a list of numpy arrays

    There's a main dictionary that map the name of the example to the example dictionary

    For each refernece sequence, 
        For each observation sequence, we create a main plot that shows
            for each sequence matching algorithm
                the cost matrix and the assignment matrix
    So for this plot, it will have 2 columns, and the number of rows will be the number of sequence matching algorithms
    """
    data_save_dir = HydraConfig.get().runtime.output_dir

    logger.info(f"Saving the plots to {data_save_dir}")

    example_name = cfg.example

    n_seq_matching_fns = len(cfg.seq_matching_fns)
    
    reward_vmin = examples[example_name]["plot"]["reward_vmin"]
    reward_vmax = examples[example_name]["plot"]["reward_vmax"]

    seq_matching_fns_dict = prepare_seq_matching_fns(cfg.seq_matching_fns, cfg.cost_fn, reward_vmin, reward_vmax)

    for obs_id in tqdm(examples[example_name]["obs_seqs"].keys()):
        ref_seq = np.array(examples[example_name]["ref_seq"])
        obs_seq = np.array(examples[example_name]["obs_seqs"][obs_id]["seq"])

        rolcol_size = cfg.plot.rolcol_size

        # 3 * because we have 3 figure columns
        #   In each figure columns, we have len(ref_seq) for the reference sequence/cost matrix, 1 column for the vertical stack of obs seq, and 1 column for the colorbar
        fig_width = 3 * (rolcol_size * (len(ref_seq) + 2))
        # n_seq_matching_fns * because we need a figure row for each sequence matching function
        #   In each figure row, we have len(obs_seq) for the observed sequence/cost matrix, 1 row for the horizontal stack of ref seq
        fig_height = n_seq_matching_fns * (rolcol_size * (len(obs_seq) + 1))

        # Create the figure (2 columns, and the number of rows will be the number of sequence matching algorithms)
        fig, axs = plt.subplots(n_seq_matching_fns, 3, figsize=(fig_width, fig_height))

        for fn_idx, fn_name in enumerate(seq_matching_fns_dict.keys()):
            seq_matching_fn = seq_matching_fns_dict[fn_name]
            
            reward, info = seq_matching_fn(obs_seq, ref_seq)

            # Plot the cost matrix
            ax = axs[fn_idx, 0]
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, info["cost_matrix"], f"{fn_name} C", seq_cmap="plasma", matrix_cmap="gray_r", rolcol_size=rolcol_size,)

            # Plot the assignment matrix
            ax = axs[fn_idx, 1]   
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, info["assignment"], f"{fn_name} A", seq_cmap="plasma", matrix_cmap="Greens", rolcol_size=rolcol_size, vmin=0, vmax=1)

            # Plot the reward
            ax = axs[fn_idx, 2]
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, np.expand_dims(reward,1), f"{fn_name} R (Sum {np.sum(reward):.2f})", seq_cmap="plasma", matrix_cmap="Greens", rolcol_size=rolcol_size,
                                         vmin=examples[example_name]["plot"]["reward_vmin"], vmax=examples[example_name]["plot"]["reward_vmax"])

        plt.tight_layout()
        plt.savefig(os.path.join(data_save_dir, f"{example_name}_obs_{obs_id}.png"))

        plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="run_examples_config")
def main(cfg: DictConfig):

    run_examples_from_config(cfg)

if __name__ == "__main__":
    main()