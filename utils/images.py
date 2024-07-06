import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def log_generations(logger, gen_imgs, folder, global_step, transform_mean, transform_std, verbose=False):
    n_samples = gen_imgs.shape[0]
    for i in tqdm(range(n_samples), disable=not verbose):
        fig, ax = plt.subplots()
        img = gen_imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
        ax.imshow((img - img.min()) / (img.max() - img.min()))
        ax.set_axis_off()
        fig.tight_layout()
        logger.experiment.add_figure(f"{folder}/{i:03}", fig, global_step=global_step)
        plt.close(fig)
