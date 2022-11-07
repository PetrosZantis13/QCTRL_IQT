'''
Q-CTRL Boulder Opal Tutorial 2:
Design robust single-qubit gates using computational graphs
Generate and test robust controls in Boulder Opal
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


def robust_controls():
    '''
    Obtain and test robust controls which implement a Y gate
    in a noisy single-qubit system.
    '''

    graph = qctrl.create_graph()

    # Pulse parameters.
    segment_count = 50
    duration = 10e-6  # s

    # Maximum value for α(t).
    alpha_max = 2 * np.pi * 0.25e6  # Hz

    # Real PWC signal representing α(t).
    alpha = graph.utils.real_optimizable_pwc_signal(
        segment_count=segment_count,
        duration=duration,
        minimum=-alpha_max,
        maximum=alpha_max,
        name="$\\alpha$",
    )

    # Maximum value for |γ(t)|.
    gamma_max = 2 * np.pi * 0.5e6  # Hz

    # Complex PWC signal representing γ(t)
    gamma = graph.utils.complex_optimizable_pwc_signal(
        segment_count=segment_count, duration=duration, maximum=gamma_max, name="$\\gamma$"
    )

    # Detuning δ.
    delta = 2 * np.pi * 0.25e6  # Hz

    # Pauli matrices σ- and σz.
    sigma_m = np.array([[0, 1], [0, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Total Hamiltonian.
    hamiltonian = (
        alpha * sigma_z
        + graph.hermitian_part(gamma * sigma_m)
        + delta * sigma_z
    )

    # Target operation node.
    target = graph.target(operator=graph.pauli_matrix("Y"))

    # Dephasing noise amplitude.
    beta = 2 * np.pi * 20e3  # Hz

    # (Constant) dephasing noise term.
    dephasing = beta * sigma_z

    # Robust infidelity.
    robust_infidelity = graph.infidelity_pwc(
        hamiltonian=hamiltonian,
        noise_operators=[dephasing],
        target=target,
        name="robust_infidelity",
    )

    optimization_result = qctrl.functions.calculate_optimization(
        graph=graph,
        cost_node_name="robust_infidelity",
        output_node_names=["$\\alpha$", "$\\gamma$"],
    )

    print(f"Optimized robust cost: {optimization_result.cost:.3e}")

    qctrlvisualizer.plot_controls(controls=optimization_result.output)
    plt.show()

    # Retrieve values of the robust PWC controls α(t) and γ(t).
    # alpha_values = np.array(
    #     [segment["value"] for segment in optimization_result.output["$\\alpha$"]]
    # )
    # gamma_values = np.array(
    #     [segment["value"] for segment in optimization_result.output["$\\gamma$"]]
    # )

    # Retrieve values of the robust PWC controls α(t) and γ(t).
    _, alpha_values, _ = qctrl.utils.pwc_pairs_to_arrays(
        optimization_result.output["$\\alpha$"]
    )
    _, gamma_values, _ = qctrl.utils.pwc_pairs_to_arrays(
        optimization_result.output["$\\gamma$"]
    )

    graph = qctrl.create_graph()

    # Create a real PWC signal representing α(t).
    alpha = graph.pwc_signal(values=alpha_values, duration=duration)

    # Create a complex PWC signal representing γ(t).
    gamma = graph.pwc_signal(values=gamma_values, duration=duration)

    # Values of β to scan over.
    beta_scan = np.linspace(-beta, beta, 100)

    # 1D batch of constant PWC
    dephasing_amplitude = graph.constant_pwc(
        beta_scan, duration=duration, batch_dimension_count=1
    )

    # Total Hamiltonian.
    hamiltonian = (
        alpha * sigma_z
        + graph.hermitian_part(gamma * sigma_m)
        + delta * sigma_z
        + dephasing_amplitude * sigma_z
    )

    # Target operation node.
    target = graph.target(operator=graph.pauli_matrix("Y"))

    # Quasi-static scan infidelity.
    infidelity = graph.infidelity_pwc(
        hamiltonian=hamiltonian, target=target, name="infidelity"
    )

    quasi_static_scan_result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["infidelity"]
    )

    # Array with the scanned infidelities.
    infidelities = quasi_static_scan_result.output["infidelity"]["value"]

    # Create plot with the infidelity scan.
    fig, ax = plt.subplots()
    ax.plot(beta_scan / (1e6 * 2 * np.pi), infidelities)
    ax.set_xlabel(r"$\beta$ (MHz)")
    ax.set_ylabel("Infidelity")

    plt.show()


robust_controls()  # uncomment to run
