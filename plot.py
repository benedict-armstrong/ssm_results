import os
import numpy as np
import corner


def savefig(plot, savedir, name):
    path = os.path.join(savedir, name)
    plot.savefig(path, bbox_inches="tight")
    # print(f"Saved {path}.")
    return


def toy_plots(outputs, targets, datatype, prefix="plot", savedir="."):
    """Make all plots and save in directory.

    Args:
        outputs (numpy.ndarray of shape (num_samples, 4)): Model output.
            Four parameters are param_1, param_2, uncertainty_1, uncertainty_2
        targets (numpy.ndarray of shape (num_samples, 2)): Truth values.
            truth_1, truth_2
        datatype (str): 'dho' or 'sg'
        prefix (str): Prefix for the file names to be saved.
        savedir (str): Saving directory

    Return:
        None
    """
    preds, sigmas = outputs[:, :2], outputs[:, 2:4]
    sigmas = np.sqrt(sigmas)
    diffs = preds - targets
    z_scores = (preds - targets) / sigmas

    # Create labels
    if datatype == "dho":
        labels_diffs = [r"$\hat{\omega}_0 - \omega_0$", r"$\hat{\beta} - \beta$"]
        labels_sigmas = [r"$\hat{\sigma}_{\omega_0}$", r"$\hat{\sigma}_{\beta}$"]
    elif datatype == "sg":
        labels_diffs = [r"$\hat{f}_0 - f_0$", r"$\hat{\tau} - \tau$"]
        labels_sigmas = [r"$\hat{\sigma}_{f_0}$", r"$\hat{\sigma}_{\tau}$"]
    else:
        raise ValueError(f"Unknown {datatype=}.")
    labels_z_scores = [
        f"({labels_diffs[i]})/{labels_sigmas[i]}" for i in range(len(labels_diffs))
    ]

    # Plot
    corner_kwargs = dict(
        show_title=True,
        smooth=0.8,
        label_kwargs=dict(fontsize=14),
        labelpad=-0.13,
        title_kwargs=dict(fontsize=14),
        tick_params=dict(labelsize=14),
        quantiles=[0.1587, 0.5, 0.8413],
        levels=(
            1 - np.exp(-0.5),
            1 - np.exp(-2),
            1 - np.exp(-9 / 2.0),
        ),  # 1, 2, 3 sigmas
        plot_density=True,
        plot_datapoints=False,
        fill_contours=False,
        show_titles=True,
        title_fmt=".3f",
        max_n_ticks=3,
        range=[[-5, 5], [-5, 5]],
        verbose=False,
    )
    figure_diffs = corner.corner(
        diffs, labels=labels_diffs, color="black", **corner_kwargs
    )
    figure_sigmas = corner.corner(
        sigmas, labels=labels_sigmas, color="blue", **corner_kwargs
    )
    figure_z_scores = corner.corner(
        z_scores, labels=labels_z_scores, color="red", **corner_kwargs
    )
    # figure_diffs.suptitle(f"diffs: {datatype}", fontsize=14)
    # figure_sigmas.suptitle(f"sigmas: {datatype}", fontsize=14)
    # figure_z_scores.suptitle(f"z_scores: {datatype}", fontsize=14)

    savefig(figure_diffs, savedir, f"{prefix}_diffs.png")
    savefig(figure_sigmas, savedir, f"{prefix}_sigmas.png")
    savefig(figure_z_scores, savedir, f"{prefix}_z_scores.png")

    qs = np.quantile(diffs, [0.1587, 0.5, 0.8413], axis=0)

    param1 = np.round(qs[:, 0], 3)
    param2 = np.round(qs[:, 1], 3)

    # save to file savedir/param1.txt and savedir/param2.txt
    # the numbers should be in the format of 0.000, 0.000, 0.000
    with open(os.path.join(savedir, "param1.txt"), "w") as f:
        # f.write(f"{param1[0]:.3f}, {param1[1]:.3f}, {param1[2]:.3f}")
        s = f"\({param1[1]:.3f}^{{+{param1[2]:.3f}}}_{{{param1[0]:.3f}}}\) & \\"
        print(s)
        f.write(s)
    with open(os.path.join(savedir, "param2.txt"), "w") as f:
        # f.write(f"{param2[0]:.3f}, {param2[1]:.3f}, {param2[2]:.3f}")
        s = f"\({param2[1]:.3f}^{{+{param2[2]:.3f}}}_{{{param2[0]:.3f}}}\) &"
        print(s)
        f.write(s)


if __name__ == "__main__":
    """
    data/ contains a folder for each model with the following contents:
    - outputs.csv
    - targets.csv
    - inputs.csv

    Read in these files into numpy arrays and then run the toy_plots function.

    """
    # data_dir = "results_16_hidden_dim"
    data_dir = "results"
    for model_dir in os.listdir(data_dir):
        model_dir = os.path.join(data_dir, model_dir)
        for type in ["sine_gaussian", "sho"]:
            for loss in ["gaussian_nll", "quantile_loss"]:
                directory = os.path.join(model_dir, type, loss)
                try:
                    outputs = np.loadtxt(
                        os.path.join(directory, "outputs.csv"), delimiter=","
                    )
                    targets = np.loadtxt(
                        os.path.join(directory, "targets.csv"), delimiter=","
                    )
                    if type == "sho":
                        temp = "dho"
                    if type == "sine_gaussian":
                        temp = "sg"

                    print(directory)
                    toy_plots(outputs, targets, temp, savedir=directory)

                    # save results to a csv file with the following columns:
                    # pred_param1, truth_param1, pred_param2, truth_param2, loss_per_saample (dummy), pred_sigma1, pred_sigma2
                    # the file should be saved in the directory
                    preds, sigmas = outputs[:, :2], outputs[:, 2:4]
                    sigmas = np.sqrt(sigmas)
                    results_kevin = np.stack(
                        [
                            preds[:, 0],
                            targets[:, 0],
                            preds[:, 1],
                            targets[:, 1],
                            np.zeros_like(preds[:, 0]),
                            sigmas[:, 0],
                            sigmas[:, 1],
                        ],
                        axis=1,
                    )
                    np.savetxt(
                        os.path.join(directory, "results_kevin.csv"),
                        results_kevin,
                        delimiter=",",
                        header="pred_param1,truth_param1,pred_param2,truth_param2,loss_per_sample,pred_sigma1,pred_sigma2",
                        comments="",
                    )

                except FileNotFoundError:
                    print(f"FileNotFoundError: {directory}")
