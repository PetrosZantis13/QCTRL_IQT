'''
Q-CTRL Boulder Opal Tutorial 3:
Understand Boulder Opal graphs and nodes by smoothing a piecewise-constant pulse
'''

# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from qctrl import Qctrl
import qctrlvisualizer

# Apply Q-CTRL style to plots created in pyplot.
plt.style.use(qctrlvisualizer.get_qctrl_style())

# Start a Boulder Opal session.
qctrl = Qctrl()


def smooth_pwc():
    '''
    Smooth out a piecewise-constant signal and plot it.
    '''

    graph = qctrl.create_graph()

    pulse_values_in_mhz = 2 * np.pi * np.array([0, 2, -3, 3, 2, 4, -1, 0])  # MHz
    pulse_values_in_hz = graph.multiply(pulse_values_in_mhz, 1e6)

    print(pulse_values_in_hz)
    pulse_values_in_hz.name = "signal values"
    print(pulse_values_in_hz)

    signal_duration = 1e-6  # seconds

    original_signal = graph.pwc_signal(
        values=pulse_values_in_hz, duration=signal_duration, name="original signal"
    )
    print(original_signal)

    print("durations:", original_signal.durations)
    print("values:", original_signal.values)

    result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["signal values", "original signal"]
    )

    print(f"Node results available: {list(result.output.keys())}")

    print(result.output["signal values"])

    # Using the pprint package for an easy-to-read output.
    pprint(result.output["original signal"])

    qctrlvisualizer.plot_controls(
        plt.figure(), {"original signal": result.output["original signal"]}
    )

    convolution_kernel = graph.gaussian_convolution_kernel(std=3e-8)
    print(convolution_kernel)

    filtered_signal = graph.convolve_pwc(pwc=original_signal, kernel=convolution_kernel)
    print(filtered_signal)

    discretized_signal = graph.discretize_stf(
        stf=filtered_signal, duration=1e-6, segment_count=500, name="filtered signal"
    )
    print(discretized_signal)

    result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=["filtered signal"]
    )

    qctrlvisualizer.plot_controls(plt.figure(), result.output, smooth=True)

    plt.show()


# smooth_pwc() # Uncomment to run
