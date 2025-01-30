import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


#======================================================
# 1. Black–Scholes Settings
#======================================================
# Example: European Call (dividend d=0, etc.)

r = 0.02         # Risk-free interest rate
sigma = 0.4      # Volatility
K = 10.0         # Strike price
T = 1.0          # Maturity
Smin = 0.4
Smax = 40.0      # Spatial range (adjust appropriately without taking it too large)
d = 0.0          # Dividend
is_call = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#======================================================
# 2. Definition of the Neural Network
#    (Input is [t, S], output is V(t,S))
#======================================================
class Net(nn.Module):
    def __init__(self, n_hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, t, s):
        # Assuming both t and s have shape=(batch_size,1)
        x = torch.cat([t, s], dim=1)  # (batch_size, 2)
        return self.mlp(x)           # (batch_size, 1)


#======================================================
# 3. Function to Compute the PDE Residual
#    (Automatically differentiates V to obtain ∂V/∂t, ∂V/∂S, ∂²V/∂S²)
#======================================================
def pde_residual(model, t, s):
    # V(t, s)
    V = model(t, s)  # (batch_size,1)
    
    # First-order derivatives
    dV_dt = torch.autograd.grad(
        V, t,
        grad_outputs=torch.ones_like(V),
        retain_graph=True,
        create_graph=True
    )[0]
    dV_ds = torch.autograd.grad(
        V, s,
        grad_outputs=torch.ones_like(V),
        retain_graph=True,
        create_graph=True
    )[0]
    
    # Second-order derivative
    d2V_ds2 = torch.autograd.grad(
        dV_ds, s,
        grad_outputs=torch.ones_like(dV_ds),
        retain_graph=True,
        create_graph=True
    )[0]
    
    # Black-Scholes PDE:
    # dV/dt + 0.5 * sigma^2 * S^2 * d²V/dS² + (r - d)* S * dV/dS - r * V = 0
    # Here, d=0 is assumed but it's okay to keep it
    # Note: t in [0,T], S>0
    # Keep in mind that in PINNs, time is often treated as "forward" from t=0 to T (as explained later)
    # PDE Residual:
    res = dV_dt + 0.5 * sigma**2 * s**2 * d2V_ds2 \
          + (r - d) * s * dV_ds - r * V
    return res


#======================================================
# 4. Loss Functions to Impose Boundary and Terminal Conditions
#======================================================
def interior_pde_loss(model, batch_size_int=128):
    t_i = torch.rand(batch_size_int, 1, device=device) * T
    s_i = torch.rand(batch_size_int, 1, device=device) * Smax
    t_i.requires_grad = True
    s_i.requires_grad = True
    
    res = pde_residual(model, t_i, s_i)
    loss_int = torch.mean(res**2)
    return loss_int

def boundary_final_loss(model, batch_size_bc=128):
    # t in [0,T], s=0
    t_bc_0 = torch.rand(batch_size_bc, 1, device=device) * T
    s_bc_0 = torch.zeros_like(t_bc_0, device=device)
    V_bc_0 = model(t_bc_0, s_bc_0)
    loss_bc_0 = torch.mean((V_bc_0 - 0.0)**2)
    
    # t in [0,T], s=Smax
    t_bc_max = torch.rand(batch_size_bc, 1, device=device) * T
    s_bc_max = torch.full_like(t_bc_max, Smax, device=device)
    target_bc_max = s_bc_max - K*torch.exp(-r*(T - t_bc_max))
    V_bc_max = model(t_bc_max, s_bc_max)
    loss_bc_max = torch.mean((V_bc_max - target_bc_max)**2)
    
    # t=T, s in [0,Smax]
    s_bc_T = torch.rand(batch_size_bc, 1, device=device) * Smax
    t_bc_T = torch.ones_like(s_bc_T, device=device) * T
    V_bc_T = model(t_bc_T, s_bc_T)
    payoff = torch.relu(s_bc_T - K)
    loss_bc_T = torch.mean((V_bc_T - payoff)**2)
    
    return loss_bc_0 + loss_bc_max + loss_bc_T



#======================================================
# 6. Training Loop
#======================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(n_hidden=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    # Loss for interior PDE points
    loss_pde = interior_pde_loss(model, batch_size_int=128)
    # Loss for boundary and terminal conditions
    loss_bc = boundary_final_loss(model, batch_size_bc=128)
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.6f}, PDE = {loss_pde.item():.6f}, BC = {loss_bc.item():.6f}")


#======================================================
# 7. Post-Training Testing and 3D Evaluation
#    Example of visualizing the price surface V(t, S)
#======================================================
model.eval()

# Import necessary libraries for 3D plotting
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Matplotlib not available, skipping plot.")
    exit()

# Generate a grid of t and S values
num_t = 100  # Number of points in t-axis
num_S = 100  # Number of points in S-axis
t_test = np.linspace(0, T, num_t)
s_test = np.linspace(Smin, Smax, num_S)
T_grid, S_grid = np.meshgrid(t_test, s_test)

# Flatten the grid to create input pairs
t_flat = T_grid.flatten()[:, None]
s_flat = S_grid.flatten()[:, None]

# Convert to torch tensors
ts_torch = torch.FloatTensor(t_flat).to(device)
Ss_torch = torch.FloatTensor(s_flat).to(device)

# Predict V(t, S) using the trained model
with torch.no_grad():
    V_pred = model(ts_torch, Ss_torch).cpu().numpy().flatten()

# Reshape the predictions to match the grid
V_grid = V_pred.reshape(S_grid.shape)

# Analytical Black-Scholes solution for comparison (optional)
def bs_call_price(S, K, r, sigma, T):
    from math import log, sqrt, exp
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    c = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return c

# Compute analytical prices (optional)
V_bs = bs_call_price(S_grid, K, r, sigma, T)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the PINN approximation
surf = ax.plot_surface(T_grid, S_grid, V_grid, cmap='viridis', alpha=0.8, label='PINN Approximation')

# Optionally, plot the Black-Scholes analytical solution
# Uncomment the following lines if you want to include the analytical solution in the plot
# surf_bs = ax.plot_surface(T_grid, S_grid, V_bs, cmap='plasma', alpha=0.6, label='BS Analytical Solution')

# Customize the plot
ax.set_xlabel('Time t')
ax.set_ylabel('Asset Price S')
ax.set_zlabel('Option Value V(t, S)')
ax.set_title('European Call Option Value Surface via PINN')

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
