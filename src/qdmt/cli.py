import argparse
import numpy as np

def plot_results():

    import qdmt.utils.graph
    import matplotlib.pyplot as plt

    # parser the command line arguments
    parser = argparse.ArgumentParser(description="Plot Lo-Schmidt echo simulation.")
    parser.add_argument(
        '--filepath',
        type=str,
        help='File containing data'
    )
    parser.add_argument(
        '--type',
        type=int,
        default=1,
        help='Plot type'
    )
    args = parser.parse_args()
    type = args.type

    try:
        try:
            data = np.load(args.filepath)
        except OSError:
            print(f"Error: unable to read file at location {args.filepath}.")

        if type == 1:
            _, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['time'], data['lo_schmidt_echo'], marker='o', markersize=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(r'$|\langle\psi|\psi\rangle|^2$')

        elif type == 2:
            _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            ax1.plot(data['time'], data['lo_schmidt_echo'], marker='o', markersize=2)
            ax1.set_ylabel(r'$|\langle\psi|\psi\rangle|^2$')
            ax2.plot(data['time'], data['cost'])
            ax2.set_yscale("log")
            ax3.plot(data['time'], data['gradient_norm'])
            ax3.set_yscale("log")
            ax3.set_xlabel('Time (s)')

        plt.show()

    except IOError as e:
        print(f"Error: Could not read file: {e}")
    pass


