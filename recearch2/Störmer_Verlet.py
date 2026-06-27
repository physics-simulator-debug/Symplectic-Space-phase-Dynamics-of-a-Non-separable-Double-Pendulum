import numpy as np
import matplotlib.pyplot as plt




# =============================================================================
# 1. Physical Constants (Masses and lengths are set to 1 for simplification)
# =============================================================================
g = 9.81  # Acceleration due to gravity






# =============================================================================
# 2. Hamiltonian Equations of Motion (Partial Derivatives of H)
# =============================================================================

def get_q_dot(q, p):
    """Calculates dq/dt (partial derivatives of Hamiltonian w.r.t momentum)"""
    q1, q2 = q[0], q[1]
    p1, p2 = p[0], p[1]
    
    delta = q1 - q2
    denom = 1.0 + np.sin(delta)**2
    
    # Equations as formulated in the paper (Page 4)
    q1_dot = (p1 - p2 * np.cos(delta)) / denom
    q2_dot = (2.0 * p2 - p1 * np.cos(delta)) / denom
    
    return np.array([q1_dot, q2_dot])

def get_p_dot(q, p):
    """Calculates dp/dt (partial derivatives of Hamiltonian w.r.t coordinates)"""
    q1, q2 = q[0], q[1]
    p1, p2 = p[0], p[1]
    
    delta = q1 - q2
    denom = 1.0 + np.sin(delta)**2
    
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)
    
    # Pre-calculating recurring algebraic expressions
    A1 = p1 * p2 * sin_d / denom
    A2 = (p1**2 + 2*p2**2 - 2*p1*p2*cos_d) * np.sin(2*delta) / (2 * denom**2)
    
    p1_dot = -g * 2.0 * np.sin(q1) - A1 + A2
    p2_dot = -g * np.sin(q2) + A1 - A2
    
    return np.array([p1_dot, p2_dot])





# =============================================================================
# 3. Störmer-Verlet Integrator Algorithm (Basic Symplectic Step)
# =============================================================================
def stormer_verlet_step(q, p, h):
    """Single step of the implicit 3-step geometric symplectic integrator"""
    
    # Step 1: Update generalized momenta by a semi-implicit half-step
    p_half = p + 0.5 * h * get_p_dot(q, p)
    
    # Step 2: Update coordinates by a full step using the mid-point momentum
    q_next = q + h * get_q_dot(q, p_half)
    
    # Step 3: Complete the momentum update by a final explicit half-step
    p_next = p_half + 0.5 * h * get_p_dot(q_next, p_half)
    
    return q_next, p_next






# =============================================================================
# 4. Yoshida's Fourth-Order Composition Method
# =============================================================================
# Mathematical constants chosen to symmetrically cancel lower-order errors
c_y = 2.0**(1.0 / 3.0)
w1 = 1.0 / (2.0 - c_y)
w2 = -c_y / (2.0 - c_y)
w3 = w1

def yoshida_step(q, p, h):
    """Splits time step h into 3 sub-steps to elevate accuracy to O(h^4)"""
    h1 = w1 * h
    h2 = w2 * h
    h3 = w3 * h
    
    # Sequential execution of Störmer-Verlet over the sub-intervals
    q, p = stormer_verlet_step(q, p, h1)
    q, p = stormer_verlet_step(q, p, h2)
    q, p = stormer_verlet_step(q, p, h3)
    
    return q, p





# =============================================================================
# 5. Simulation Configuration and Execution Loop
# =============================================================================
dt = 0.01          # Time step size
t_end = 30.0       # Total simulation duration (30 seconds)
steps = int(t_end / dt)

# Initial conditions (Angles set to 90 degrees, starting from rest)
theta1_init = np.radians(90.0)
theta2_init = np.radians(90.0)

# State vectors initialization (Initial angular velocities = 0 implies p = 0)
q = np.array([theta1_init, theta2_init])
p = np.array([0.0, 0.0])

# Arrays to store state history for plotting verification
history_q1 = np.zeros(steps)
history_q2 = np.zeros(steps)
time_axis = np.linspace(0, t_end, steps)

# Main simulation loop
for i in range(steps):
    history_q1[i] = q[0]
    history_q2[i] = q[1]
    
    # Propagate the system using Yoshida Symplectic Integrator
    q, p = yoshida_step(q, p, dt)



# =============================================================================
# 6. Basic Plotting for Developer Verification (Sanity Check)
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(time_axis, history_q1, label='Theta 1 (Upper Rod)', color='blue')
plt.plot(time_axis, history_q2, label='Theta 2 (Lower Rod)', color='red', linestyle='--')
plt.title('Double Pendulum Simulation using Yoshida 4th-Order Symplectic Integrator')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (radians)')
plt.grid(True)
plt.legend()
plt.show()
