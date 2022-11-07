'''
Q-CTRL Boulder Opal Tutorial 4:
Optimize Mølmer–Sørensen gates for trapped ions
Creating optimal operations with the trapped ions toolkit
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

# Number of ions in the system.
ion_count = 2

# Calculate Lamb–Dicke parameters and relative detunings.
lamb_dicke_parameters, relative_detunings = qctrl.ions.obtain_ion_chain_properties(
    atomic_mass=171,  # Yb ions
    ion_count=ion_count,
    center_of_mass_frequencies=[1.6e6, 1.5e6, 0.3e6],  # Hz
    wave_numbers=[(2 * np.pi) / 355e-9, (2 * np.pi) / 355e-9, 0.0],  # rad/m
    laser_detuning=1.6e6 + 4.7e3,  # Hz
)

# Optimizable drive characteristics.
maximum_rabi_rate = 2 * np.pi * 100e3  # rad/s
segment_count = 64

drive = qctrl.ions.ComplexOptimizableDrive(
    count=segment_count, maximum_rabi_rate=maximum_rabi_rate, addressing=(0, 1)
)

# Gate duration.
duration = 2e-4  # s

# Target phases.
target_phase = np.pi / 4
target_phases = np.array([[0, 0], [target_phase, 0]])

result = qctrl.ions.ms_optimize(
    drives=[drive],
    lamb_dicke_parameters=lamb_dicke_parameters,
    relative_detunings=relative_detunings,
    duration=duration,
    target_phases=target_phases,
)

print(f"Infidelity reached: {result['infidelities'][-1]:.3e}")

drive = qctrl.utils.pwc_arrays_to_pairs(duration, result["optimized_values"]["drive"])
qctrlvisualizer.plot_controls(controls={r"$\gamma$": drive})

sample_times = result["sample_times"] * 1e3
plt.plot(sample_times, result["phases"][:, 1, 0])
plt.plot([0, sample_times[-1]], [target_phase, target_phase], "k--")
plt.yticks([0, target_phase], ["0", r"$\frac{\pi}{4}$"])
plt.xlabel("Time (ms)")
plt.ylabel("Relative phase")
plt.show()

total_displacement = np.sum(result["displacements"], axis=-1)

fig, axs = plt.subplots(1, 2)
plot_range = 1.05 * np.max(np.abs(total_displacement))
fig.suptitle("Phase space trajectories")

for k in range(2):
    for mode in range(ion_count):
        axs[k].plot(
            np.real(total_displacement[:, k, mode]),
            np.imag(total_displacement[:, k, mode]),
            label=f"mode {mode % ion_count}",
        )
        axs[k].plot(
            np.real(total_displacement[-1, k, mode]),
            np.imag(total_displacement[-1, k, mode]),
            "k*",
            markersize=10,
        )

    axs[k].set_xlim(-plot_range, plot_range)
    axs[k].set_ylim(-plot_range, plot_range)
    axs[k].set_aspect("equal")
    axs[k].set_xlabel("q")

axs[0].set_title("$x$-axis")
axs[0].set_ylabel("p")
axs[1].set_title("$y$-axis")
axs[1].yaxis.set_ticklabels([])

hs, ls = axs[0].get_legend_handles_labels()
axs[1].legend(handles=hs, labels=ls, loc="upper left", bbox_to_anchor=(1, 1))

plt.show()
