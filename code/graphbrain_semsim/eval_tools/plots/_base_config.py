from matplotlib import pyplot as plt
import scienceplots  # noqa


def plot_base_config():
    plt.style.use(['science', 'grid'])

    # Increase the font size
    plt.rcParams.update({
        'text.usetex': False,       # Disable latex because it makes your life hard
        'font.size': 14,            # Set the global font size
        'xtick.labelsize': 14,      # Set the font size for the x-axis tick labels
        'ytick.labelsize': 14,      # Set the font size for the y-axis tick labels
    })