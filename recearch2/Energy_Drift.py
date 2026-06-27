import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Safe GUI handling for Python 3.13
import matplotlib.pyplot as plt

# =============================================================================
# 1. Physical Parameters and Separable Hamiltonian Form (Standardized for Symplectic Integration)
# =============================================================================
g = 9.81

def hamiltonian_energy(q, p):
    """Calculates total energy using the Separable Phase-Space Approximation"""
    q1, q2 = q[0], q[1]
    p1, p2 = p[0], p[1]
    
    # In a standardized separable form, kinetic energy matrix is evaluated at a reference state
    # This prevents the vertical integration crash (Non-symplectic breakdown)
    T = 0.5 * (p1**2 + 2.0*p2**2 - 2.0*p1*p2) 
    V = -g * (2.0 * np.cos(q1) + np.cos(q2))
    
    return T + V

# =============================================================================
# 2. Perfect Separable Hamiltonian Derivatives (dH/dp and -dH/dq)
# =============================================================================
def get_q_dot(q, p):
    """dq/dt = dH/dp (Depends strictly on momenta p)"""
    p1, p2 = p[0], p[1]
    q1_dot = p1 - p2
    q2_dot = 2.0 * p2 - p1
    return np.array([q1_dot, q2_dot])

def get_p_dot(q, p):
    """dp/dt = -dH/dq (Depends strictly on coordinates q)"""
    q1, q2 = q[0], q[1]
    p1_dot = -2.0 * g * np.sin(q1)
    p2_dot = -g * np.sin(q2)
    return np.array([p1_dot, p2_dot])

def rk4_derivs(state):
    """System derivatives grouped for standard Runge-Kutta execution"""
    q = state[0:2]
    p = state[2:4]
    return np.concatenate([get_q_dot(q, p), get_p_dot(q, p)])

# =============================================================================
# 3. Exact Symplectic and Non-Symplectic Integrators
# =============================================================================
def stormer_verlet_step(q, p, h):
    """Standard Symplectic Störmer-Verlet step for separable systems"""
    # This algorithm is mathematically guaranteed to conserve phase space area here
    p_half = p + 0.5 * h * get_p_dot(q, p)
    q_next = q + h * get_q_dot(q, p_half)
    p_next = p_half + 0.5 * h * get_p_dot(q_next, p_half)
    return q_next, p_next

# Yoshida 4th-order composition ثوابت يوشيدا لإلغاء الخطأ
c_y = 2.0**(1.0 / 3.0)
w1 = 1.0 / (2.0 - c_y)
w2 = -c_y / (2.0 - c_y)
w3 = w1

def yoshida_step(q, p, h):
    """Upgrades Störmer-Verlet to 4th-order accuracy via Yoshida composition"""
    q, p = stormer_verlet_step(q, p, w1 * h)
    q, p = stormer_verlet_step(q, p, w2 * h)
    q, p = stormer_verlet_step(q, p, w3 * h)
    return q, p

def standard_rk4_step(state, h):
    """Standard Explicit 4th-Order Runge-Kutta Step"""
    k1 = rk4_derivs(state)
    k2 = rk4_derivs(state + 0.5 * h * k1)
    k3 = rk4_derivs(state + 0.5 * h * k2)
    k4 = rk4_derivs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# =============================================================================
# 4. Main Simulation Execution Loop (Time Step set to 0.01 for clear drift visibility)
# =============================================================================
dt = 0.01          
t_end = 50.0       # Extended to 50 seconds to clearly see the non-symplectic drift
steps = int(t_end / dt)
time_axis = np.linspace(0, t_end, steps)

# Initial conditions (Angles set to high amplitude 60 degrees to trigger non-linear drift)
q_init = np.array([np.radians(60.0), np.radians(60.0)])
p_init = np.array([0.0, 0.0])
initial_energy = hamiltonian_energy(q_init, p_init)

energy_yoshida = np.zeros(steps)
energy_rk4 = np.zeros(steps)

# Propagate using Yoshida (Symplectic)
q_sym, p_sym = q_init.copy(), p_init.copy()
for i in range(steps):
    energy_yoshida[i] = hamiltonian_energy(q_sym, p_sym)
    q_sym, p_sym = yoshida_step(q_sym, p_sym, dt)

# Propagate using Standard RK4 (Non-Symplectic)
rk4_state = np.concatenate([q_init, p_init])
for i in range(steps):
    q_rk = rk4_state[0:2]
    p_rk = rk4_state[2:4]
    energy_rk4[i] = hamiltonian_energy(q_rk, p_rk)
    rk4_state = standard_rk4_step(rk4_state, dt)

# =============================================================================
# 5. Scientific Research Plot Generation
# =============================================================================
plt.figure(figsize=(10, 6))

# Plot energy deviations from the initial total energy
plt.plot(time_axis, energy_rk4 - initial_energy, label='Standard RK4 (Non-Symplectic Energy Drift)', color='red', linestyle='--')
plt.plot(time_axis, energy_yoshida - initial_energy, label='Yoshida 4th-Order (Symplectic Energy Conservation)', color='blue', linewidth=2)

plt.title('Total Mechanical Energy Deviation Comparison')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy Deviation: $\Delta H(t) = H(t) - H(0)$')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('energy_drift_comparison.png', dpi=300)
plt.show()
plt.close('all')
