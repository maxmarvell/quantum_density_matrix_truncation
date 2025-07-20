import numpy as np
import matplotlib.pyplot as plt
import argparse
from qdmt.uniform_mps import UniformMps

def compute_normalization(A: UniformMps):
    return A.normalization()

def parse():
    parser = argparse.ArgumentParser(description="Annotate time evolved data.")

    parser.add_argument(
        'filepath',
        type=str,
        help='Relative file path to time evolved data.'
    )
    return parser.parse_args()

def main():
    args = parse()

    try:
        data = np.load(args.filepath)
    except OSError:
        print(f"Error: unable to read file at location {args.filepath}.")

    normalization = np.empty_like(data['time'])

    for i in range(len(data['time'])):
        A = UniformMps(data['state'][i])
        normalization[i] = compute_normalization(A)
    print(normalization)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['time'], normalization, marker='o', markersize=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$|\langle\psi|\psi\rangle|^2$')

    plt.show()
    

if __name__ == "__main__":
    main()