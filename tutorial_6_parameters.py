'''
Q-CTRL Boulder Opal Tutorial 6:
Estimate parameters of a single-qubit Hamiltonian
Performing system identification with Boulder Opal
'''

# Import packages.
import numpy as np
import matplotlib.pyplot as plt
import qctrlvisualizer
from qctrl import Qctrl

# Apply Q-CTRL style to plots created in pyplot.
plt.style.use(qctrlvisualizer.get_qctrl_style())

# Start a Boulder Opal session.
qctrl = Qctrl()

# Correct values of the parameters that you want to determine.
# The `run_experiments` function will use them to produce simulated experimental results.
actual_parameters = [3.54e5, 7.91e5, -5e5]  # Hz
parameter_names = ["Ω_x", "Ω_y", "Ω_z"]

def run_experiments(initial_states, observables, wait_times, standard_deviation):
    """
    A black-box function representing a quantum system whose parameters
    you want to determine.

    Parameters
    ----------
    initial_states : np.ndarray
        An array containing the initial states of each experiment setup.
        Its shape must be ``(N, 2, 1)``, where ``N`` is the number of
        experiment setups
    observables : np.ndarray
        An array containing the observables that you want to measure at
        the end of each experiment setup. Its shape must be ``(N, 2, 2)``,
        where ``N`` is the number of experiment setups.
    wait_times : np.ndarray
        The list of times when the experiment ends. Its shape must be
        ``(T,)``, where ``T`` is the number of points in time where the
        experiment ends.

    Returns
    -------
    np.ndarray
        A batch of simulated experimental results corresponding to the
        setups and times that you provided. It will have shape ``(N, T)``,
        where ``N`` is the number of experimental setups and ``T`` is the
        number of wait times.
    """

    (alpha_x, alpha_y, alpha_z) = actual_parameters

    graph = qctrl.create_graph()

    hamiltonian = graph.constant_pwc(
        constant=(
            alpha_x * graph.pauli_matrix("X")
            + alpha_y * graph.pauli_matrix("Y")
            + alpha_z * graph.pauli_matrix("Z")
        )
        * 0.5,
        duration=np.max(wait_times),
    )
    unitary = graph.time_evolution_operators_pwc(
        hamiltonian=hamiltonian, sample_times=wait_times
    )
    final_states = unitary @ initial_states[:, None]

    expectation_values = graph.real(
        graph.expectation_value(final_states[..., 0], observables[:, None])
    )

    errors = graph.random_normal(
        shape=expectation_values.shape, mean=0, standard_deviation=standard_deviation
    )
    measurement_results = graph.add(
        expectation_values, errors, name="measurement_results"
    )

    result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["measurement_results"]
    )

    return result.output["measurement_results"]["value"]


# Define initial states of the experiments.
state_zp = np.array([[1], [0]], dtype=complex)  # |z+⟩, also known as |0⟩
state_xp = np.array([[1], [1]], dtype=complex) / np.sqrt(2)  # |x+⟩, also known as |+⟩
state_yp = np.array([[1], [1j]], dtype=complex) / np.sqrt(2)  # |y+⟩
initial_states = np.array([state_zp, state_xp, state_yp])
initial_states_names = [r"$| z+ \rangle$", r"$| x+ \rangle$", r"$| y+ \rangle$"]

# Define observables measured in the experiments.
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
observables = np.array([sigma_x, sigma_y, sigma_z])
observables_names = [
    r"$\langle\sigma_x\rangle$",
    r"$\langle\sigma_y\rangle$",
    r"$\langle\sigma_z\rangle$",
]

# Define wait times of the experiments.
duration = 10e-6  # s
wait_times = np.linspace(0, duration, 40)
standard_deviation = 0.3

black_box_data = run_experiments(initial_states, observables, wait_times, standard_deviation)

fig, axes = plt.subplots(1, len(black_box_data), figsize=(20, 5))
fig.suptitle("Measured data from the experiments")

for index, axis in enumerate(axes):
    axis.set_title(f"Initial state {initial_states_names[index]}")
    axis.set_ylabel(observables_names[index], labelpad=0)
    axis.plot(wait_times * 1e6, black_box_data[index], "o")
    axis.set_ylim(-1.1, 1.1)
    axis.set_xlabel("Wait time $t$ (µs)")
plt.show()


# Define your model and obtain estimates of the parameters.
graph = qctrl.create_graph()

omega_fit = graph.optimization_variable(
    count=3, lower_bound=-1e6, upper_bound=1e6, name="omega_fit"
)
omega_x_fit = omega_fit[0]
omega_y_fit = omega_fit[1]
omega_z_fit = omega_fit[2]

# Define Hamiltonian of the model.
hamiltonian = graph.constant_pwc(
    constant=0.5
    * (omega_x_fit * sigma_x + omega_y_fit * sigma_y + omega_z_fit * sigma_z),
    duration=duration,
)

# Calculate the evolution according to the Hamiltonian.
unitaries = graph.time_evolution_operators_pwc(
    hamiltonian=hamiltonian, sample_times=wait_times
)

# Obtain expectation values at the requested times.
final_states = unitaries @ initial_states[:, None]

# Obtain expectation values.
expectation_values = graph.real(
    graph.expectation_value(final_states[..., 0], observables[:, None]),
    name="expectation_values",
)

# Obtain the residual sum of squares (RSS), to be used as cost.
residuals = expectation_values - black_box_data
rss = graph.sum(residuals**2, name="rss")

# Obtain the Hessian, to be used for statistical analysis later.
fit_parameters = [omega_x_fit, omega_y_fit, omega_z_fit]
hessian = graph.hessian(rss, fit_parameters, name="hessian")

result = qctrl.functions.calculate_optimization(
    graph=graph,
    cost_node_name="rss",
    output_node_names=["expectation_values", "omega_fit", "hessian"],
)

fit_values = result.output["omega_fit"]["value"]

print("\t Estimated value\t Correct value")
for name, fit, actual_value in zip(parameter_names, fit_values, actual_parameters):
    print(f"{name}: \t {fit/1e6:.3f} MHz \t\t {actual_value/1e6:.3f} MHz")

    model_data = result.output["expectation_values"]["value"]

fig, axes = plt.subplots(1, len(model_data), figsize=(20, 5))

fig.suptitle("Measured data from the experiments and predictions from the model")

for index, axis in enumerate(axes):
    axis.set_title(f"Initial state {initial_states_names[index]}")
    axis.set_ylabel(observables_names[index], labelpad=0)
    axis.plot(wait_times * 1e6, black_box_data[index], "o", label="Black-box data")
    axis.plot(wait_times * 1e6, model_data[index], "--", label="Model data")
    axis.set_ylim(-1.05, 1.05)
    axis.legend()
    axis.set_xlabel("Wait time $t$ (µs)")
plt.show()

# Plot 95%-confidence ellipses.
confidence_region = qctrl.utils.confidence_ellipse_matrix(
    hessian=result.output["hessian"]["value"],
    cost=result.cost,
    measurement_count=3 * len(wait_times),
    confidence_fraction=0.95,
)

qctrlvisualizer.plot_confidence_ellipses(
    confidence_region,
    estimated_parameters=fit_values,
    actual_parameters=actual_parameters,
    parameter_names=[r"$\Omega_x$", r"$\Omega_y$", r"$\Omega_z$"],
)
plt.suptitle("Estimated parameters (with 95% confidence region)", y=1.05)
plt.show()
