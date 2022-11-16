'''
Q-CTRL Boulder Opal Tutorial 5:
Find optimal pulses with automated optimization
Use closed-loop optimization without a complete system model
'''

# Import packages.
import numpy as np
import matplotlib.pyplot as plt
import qctrlvisualizer
from qctrl import Qctrl

# Apply Q-CTRL style to plots created in pyplot.
plt.style.use(qctrlvisualizer.get_qctrl_style())

# Start a Boulder Opal session.
qctrl = Qctrl(verbosity="QUIET")  # Mute status messages.

def run_experiments(controls, duration, shot_count):
    """
    Simulates a single qubit experiment using Boulder Opal with the given piecewise-constant controls.

    Parameters
    ----------
    controls : np.ndarray
        The controls to simulate.
        A 2D NumPy array of shape (control_count, segment_count) with the per-segment
        values of each control.
    duration : float
        The duration (in nanoseconds) of the controls.
    shot_count : int
        The number of shots for which to generate measurements.

    Returns
    -------
    np.ndarray
        The qubit measurement results (either 0, 1, or 2) associated to each control
        and shot, as an array of shape (len(controls), shot_count).
    """

    # Create Boulder Opal graph.
    graph = qctrl.create_graph()

    # Define simulation parameters and operators.
    filter_cutoff_frequency = 2 * np.pi * 0.3  # GHz
    segment_count = 128
    max_drive_amplitude = 2 * np.pi * 0.1  # GHz
    delta = -0.33 * 2 * np.pi  # GHz
    big_delta = 0.01 * 2 * np.pi  # GHz
    number_operator = graph.number_operator(3)
    drive_operator = 0.5 * graph.annihilation_operator(3)
    confusion_matrix = np.array(
        [[0.99, 0.01, 0.01], [0.01, 0.98, 0.01], [0.0, 0.01, 0.98]]
    )

    # Retrieve control information.
    control_values = controls * max_drive_amplitude

    # Define initial state.
    initial_state = graph.fock_state(3, 0)

    # Construct constant Hamiltonian terms.
    frequency_term = big_delta * number_operator
    anharmonicity_term = (
        delta * (number_operator @ number_operator - number_operator) / 2
    )

    # Construct filtered drive.
    filtered_drive = graph.convolve_pwc(
        pwc=graph.pwc_signal(duration=duration, values=control_values),
        kernel=graph.sinc_convolution_kernel(filter_cutoff_frequency),
    )
    drive = graph.discretize_stf(filtered_drive, duration, segment_count)
    drive_term = graph.hermitian_part(drive * drive_operator)

    # Build Hamiltonian and calculate unitary evolution operators.
    hamiltonian = drive_term + frequency_term + anharmonicity_term
    unitary = graph.time_evolution_operators_pwc(
        hamiltonian=hamiltonian, sample_times=np.array([duration])
    )
    unitary = unitary[:, -1]

    # Evolve initial state and calculate final (normalized) populations.
    final_state = unitary @ initial_state[:, None]
    populations = graph.abs(final_state) ** 2

    # Apply the confusion matrix to the populations.
    populations = confusion_matrix @ populations
    populations.name = "populations"

    # Execute graph and retrieve the populations.
    result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["populations"]
    )
    populations = result.output["populations"]["value"].squeeze(axis=2)

    # Simulate measurements for each control.
    measurements_list = []
    for p in populations:
        # Sample and store measurements.
        measurements = np.random.choice(3, size=shot_count, p=p / np.sum(p))
        measurements_list.append(measurements)

    return np.array(measurements_list)


# Define the state object for the closed-loop optimizer.
optimizer = qctrl.closed_loop.CrossEntropy(elite_fraction=0.25, seed=1)

# Calculate cost from experiment results.
def cost_function(controls):
    """
    Accepts an array of controls and returns their associated costs.
    """
    measurements = run_experiments(controls, duration, shot_count)

    costs = []
    for shots in measurements:
        shots_in_one = np.count_nonzero(shots == 1)
        fidelity = shots_in_one / len(shots)
        costs.append(1 - fidelity)

    return costs

# Number of segments in each control.
segment_count = 10

# Duration of each control.
duration = 100  # ns

# Number of controls to retrieve from the optimizer each run.
test_point_count = 32

# Number of projective measurements to take after the control is applied.
shot_count = 1000

# Define parameters as a set of controls with piecewise-constant segments.
rng = np.random.default_rng(seed=1)
initial_controls = rng.uniform(0.0, 1.0, size=(test_point_count, segment_count))

bounds = np.repeat([[0.0, 1.0]], segment_count, axis=0)

max_iteration_count = 20
target_cost = 0.02

# Run the optimization loop until the cost (infidelity) is sufficiently small.
results = qctrl.closed_loop.optimize(
    cost_function=cost_function,
    initial_test_parameters=initial_controls,
    optimizer=optimizer,
    bounds=bounds,
    target_cost=target_cost,
    max_iteration_count=max_iteration_count,
)

# Print final best cost.
print(f"Best cost reached: {results['best_cost']:.3f}")

# Print and plot controls that correspond to the best cost.
print(f"Best control values: {np.round(results['best_parameters'], 3)}")

qctrlvisualizer.plot_controls(
    {
        r"$\Omega(t)$": qctrl.utils.pwc_arrays_to_pairs(
            duration, results["best_parameters"]
        )
    },
    two_pi_factor=False,
    unit_symbol=" a.u.",
)
plt.suptitle("Best control")
plt.show()

qctrlvisualizer.plot_cost_history(results["best_cost_history"])

# Obtain a set of initial experimental results.
measurements = run_experiments(
    controls=initial_controls, duration=duration, shot_count=shot_count
)

# Find the best initial (random) control.
best_initial_control = np.argmax(np.count_nonzero(measurements == 1, axis=1))
initial_best_counts = np.unique(
    measurements[best_initial_control], return_counts=True, axis=0
)[1]
initial_best_probability = initial_best_counts / shot_count
print(f"Best initial probabilities: {initial_best_probability}")


# Obtain a set of converged experimental results.
measurements = run_experiments(
    controls=np.array([results["best_parameters"]]),
    duration=duration,
    shot_count=shot_count,
)
optimized_counts = np.unique(measurements, return_counts=True)[1]
optimized_probability = optimized_counts / shot_count
print(f"Optimized probabilities: {optimized_probability}")

# Plot distribution of measurements for the initial & converged sets of pulses.
fig, ax = plt.subplots()
ax.bar(np.arange(3) - 0.1, optimized_probability, width=0.2, label="Optimized controls")
ax.bar(
    np.arange(3) + 0.1,
    initial_best_probability,
    width=0.2,
    label="Best initial controls",
)
ax.legend()
ax.set_ylabel("Probability")
ax.set_xlabel("Measurement results")
ax.set_xticks(np.arange(3))
ax.set_xticklabels([rf"$|{k}\rangle$" for k in np.arange(3)])

plt.show()
