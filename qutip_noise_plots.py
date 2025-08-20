import numpy as np
import matplotlib.pyplot as plt
from qutip_qip.device import Processor
from qutip_qip.noise import RandomNoise
from qutip import sigmaz, sigmax, basis, mesolve
try:
    from qutip_qip.pulse import Pulse
except Exception:
    from qutip.qip.pulse import Pulse

# Build processor with T1/T2
proc = Processor(N=1, t1=10.0, t2=6.0, spline_kind="step_func")

# Step pulse (Z control)
tlist = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=float)
coeff = np.array([0.3, 0.5, 0.0], dtype=float)
p = Pulse(sigmaz(), tlist=tlist, coeff=coeff, targets=0, label="Z")
proc.add_pulse(p)

# ORIGINAL control amplitude -> PDF (no title)
fig, ax = plt.subplots(figsize=(5, 3))
ax.step(tlist, np.r_[coeff, coeff[-1]], where="post", linewidth=2)
ax.set_xlabel("Time"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("original_control.pdf", format="pdf"); plt.close(fig)

# NOISY control amplitude by sampling H(t) -> PDF (no title)
gauss = RandomNoise(dt=0.01, rand_gen=np.random.normal, loc=0.0, scale=0.02)
original_noises = list(getattr(proc, "noises", []))
proc.add_noise(gauss)
try:
    out = proc.get_qobjevo(noisy=True)
    if isinstance(out, tuple):
        qobjevo_eff, _ = out
    else:
        qobjevo_eff = out
finally:
    proc.noises = original_noises

Hz = sigmaz()
def coeff_from_H(H):
    return 0.5 * (H * Hz).tr().real

t_end = float(tlist[-1])
tgrid = np.linspace(0.0, t_end, 1000)
noisy_coeff_samples = np.array([coeff_from_H(qobjevo_eff(t)) for t in tgrid])
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.plot(tgrid, noisy_coeff_samples, linewidth=2)
ax2.set_xlabel("Time"); ax2.set_ylabel("Amplitude"); ax2.grid(True, alpha=0.3)
fig2.tight_layout(); fig2.savefig("noisy_control.pdf", format="pdf"); plt.close(fig2)

# X expectation with and without T1/T2 -> PDF (no title)
def piecewise(knots, steps, t):
    y = np.zeros_like(t)
    for i in range(len(steps)):
        L, R = knots[i], knots[i+1]
        mask = (t >= L) & (t < R if i < len(steps)-1 else t <= R)
        y[mask] = steps[i]
    return y

tdense = np.linspace(0.0, t_end, 1000)
c_dense = piecewise(tlist, coeff, tdense)
H_td = [sigmaz(), c_dense]

psi0 = (basis(2, 0) + basis(2, 1)).unit()
obs = sigmax()

try:
    out = proc.get_qobjevo(noisy=True, device_noise=True)
except TypeError:
    out = proc.get_qobjevo(noisy=True)
if isinstance(out, tuple):
    _Hdev, c_ops_t12 = out
else:
    c_ops_t12 = []

sol_id  = mesolve(H_td, psi0, tdense, c_ops=[],        e_ops=[obs])
sol_t12 = mesolve(H_td, psi0, tdense, c_ops=c_ops_t12, e_ops=[obs])

x_id  = np.real(sol_id.expect[0])
x_t12 = np.real(sol_t12.expect[0])

fig3, ax3 = plt.subplots(figsize=(5, 3))
ax3.plot(tdense, x_id,  lw=1.6, label="Ideal ⟨X⟩")
ax3.plot(tdense, x_t12, lw=1.6, label=f"Lindblad (T1={proc.t1}, T2={proc.t2})")
ax3.set_xlabel("Time"); ax3.set_ylabel("⟨X⟩"); ax3.grid(True, alpha=0.3); ax3.legend()
fig3.tight_layout(); fig3.savefig("x_expectation.pdf", format="pdf"); plt.close(fig3)
