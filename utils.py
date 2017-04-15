import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(driving_data, bins, filename, title):
    data = list(map(lambda d: d.steering_angle, driving_data))

    n_data = len(data)
    print("Number of samples: ", n_data)

    num_bins = bins
    samples_per_bin = n_data / num_bins
    hist, bins = np.histogram(data, num_bins)

    fig = plt.figure()

    width = 0.9 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(data), np.max(data)), (samples_per_bin, samples_per_bin), 'k-')

    plt.title(title)

    fig.savefig(filename)
