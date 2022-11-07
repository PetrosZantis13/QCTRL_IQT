'''
Q-CTRL Boulder Opal Tutorial 1:
Simulate the dynamics of a single qubit using computational graphs
Simulate and visualize quantum system dynamics in Boulder Opal
'''

# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from qctrl import Qctrl
import qctrlvisualizer

# Apply Q-CTRL style to plots created in pyplot.
plt.style.use(qctrlvisualizer.get_qctrl_style())

# Start a Boulder Opal session.
qctrl = Qctrl()


def constant_H(omega=1.0e6, delta=0.4e6):
    '''
    Simulating single-qubit dynamics with a constant Hamiltonian.
    omega : Rabi drive in Hz
    delta : Detuning in Hz
    '''

    graph = qctrl.create_graph()

    # (Constant) Hamiltonian parameters.
    omega = 2*np.pi*omega  # convert from Hz to rad
    delta = 2*np.pi*delta  # convert from Hz to rad

    # Total duration of the simulation.
    duration = 2e-6  # s

    # Hamiltonian term coefficients.
    omega_signal = graph.constant_pwc(constant=omega, duration=duration)
    delta_signal = graph.constant_pwc(constant=delta, duration=duration)

    # Pauli matrices σ- and σz.
    sigma_m = np.array([[0, 1], [0, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Total Hamiltonian, [Ω σ- + H.c.]/2 + δ σz.
    hamiltonian = (
        graph.hermitian_part(omega_signal * sigma_m) +
        delta_signal * sigma_z)

    # Times at which to sample the simulation.
    sample_times = np.linspace(0.0, duration, 100)

    # Time-evolution operators, U(t).
    unitaries = graph.time_evolution_operators_pwc(
        hamiltonian=hamiltonian, sample_times=sample_times, name="unitaries"
    )

    # Initial state of the qubit, |0⟩.
    initial_state = np.array([[1], [0]])

    # Evolved states, |ψ(t)⟩ = U(t) |0⟩
    evolved_states = unitaries @ initial_state  # @ => matrix multiplication
    evolved_states.name = "states"

    result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["unitaries", "states"]
    )

    unitaries = result.output["unitaries"]["value"]
    print(f"Shape of calculated unitaries: {unitaries.shape}")

    states = result.output["states"]["value"]
    print(f"Shape of calculated evolved states: {states.shape}")

    # Calculate qubit populations |⟨ψ|0⟩|².
    qubit_populations = np.abs(states.squeeze()) ** 2

    # Plot populations.

    plt.figure(figsize=(10, 5))
    plt.plot(sample_times / 1e-6, qubit_populations)
    plt.xlabel("Time (µs)")
    plt.ylabel("Population")
    plt.legend(labels=[rf"$|{k}\rangle$" for k in [0, 1]], title="State")
    plt.show()

    # qctrlvisualizer.plot_populations(
    #     sample_times, {rf"$|{k}\rangle$": qubit_populations[:, k] for k in [0, 1]}
    # )

    # Interactive visualisation on the Bloch Sphere;
    # ^ only works with JuPyter? ASK THE TEAM
    # qctrlvisualizer.display_bloch_sphere(states.squeeze())


constant_H(1.0e6, 0e6)   # uncomment to run


def time_dep_H(omega_max=1.0e6, delta=0.4e6):
    '''
    Simulating a π/2 gate in a single qubit with a time-dependent Hamiltonian
    omega_max : Peak of Gaussian Rabi drive in Hz
    delta : Detuning in Hz
    '''

    # Gaussian pulse parameters.
    omega_max = 2.0*np.pi*omega_max  # convert from Hz to rad
    segment_count = 50
    times = np.linspace(-3, 3, segment_count)
    omega_values = -1j * omega_max * np.exp(-(times**2))  # -i ω(t)

    # Total duration of the pulse to achieve a π/2 gate.
    # (Equation derivation explained in the tutorial)
    pulse_duration = 0.5 * segment_count * np.pi / np.sum(np.abs(omega_values))

    # Duration of each piecewise-constant segment.
    segment_duration = pulse_duration / segment_count

    # Plot Gaussian pulse.
    qctrlvisualizer.plot_controls(
        {"$\Omega$": [{"value": v, "duration": segment_duration} for v in omega_values]},
        polar=False,
    )
    plt.show()

    graph = qctrl.create_graph()

    # Times at which to sample the simulation.
    sample_times = np.linspace(0.0, pulse_duration, 100)

    # Time-dependent Hamiltonian term coefficient.
    omega_signal = graph.pwc_signal(values=omega_values, duration=pulse_duration)

    # Pauli matrix σ-
    sigma_m = np.array([[0, 1], [0, 0]])

    # Total Hamiltonian, [Ω σ- + H.c.]/2
    hamiltonian = graph.hermitian_part(omega_signal * sigma_m)

    # Time-evolution operators, U(t).
    unitaries = graph.time_evolution_operators_pwc(
        hamiltonian=hamiltonian, sample_times=sample_times, name="unitaries"
    )

    # Initial state of the qubit, |0⟩.
    initial_state = np.array([[1], [0]])

    # Evolved states, |ψ(t)⟩ = U(t) |0⟩
    evolved_states = unitaries @ initial_state
    evolved_states.name = "states"

    # Execute the graph.
    result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["unitaries", "states"]
    )

    # Retrieve values of the calculation
    unitaries = result.output["unitaries"]["value"]
    states = result.output["states"]["value"]

    print("Unitary gate implemented by the Gaussian pulse:")
    print(unitaries[-1])
    print()
    print("Final state after the gate:")
    print(states[-1])

    # Calculate qubit populations |⟨ψ|0⟩|².
    qubit_populations = np.abs(states.squeeze()) ** 2

    # Plot populations.
    plt.figure(figsize=(10, 5))
    plt.plot(sample_times / 1e-6, qubit_populations)
    plt.xlabel("Time (µs)")
    plt.ylabel("Population")
    plt.legend(labels=[rf"$|{k}\rangle$" for k in [0, 1]], title="State")
    plt.show()


# time_dep_H()   # uncomment to run
