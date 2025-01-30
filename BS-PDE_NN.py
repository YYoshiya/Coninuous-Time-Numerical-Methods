import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


#======================================================
# 1. ブラック–ショールズの設定
#======================================================
# 例: 欧州コール (配当 d=0, etc.)

r = 0.02         # 無リスク金利
sigma = 0.4      # ボラティリティ
K = 10.0         # ストライク
T = 1.0          # 満期
Smin = 0.4
Smax = 40.0      # 空間範囲(大きめに取りすぎず適宜調整)
d = 0.0          # 配当
is_call = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#======================================================
# 2. ニューラルネットの定義
#    (入力は [t, S], 出力は V(t,S))
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
        # t, s ともに shape=(batch_size,1) を想定
        x = torch.cat([t, s], dim=1)  # (batch_size, 2)
        return self.mlp(x)           # (batch_size, 1)


#======================================================
# 3. PDE残差を計算する関数
#    (Vに対して自動微分して ∂V/∂t, ∂V/∂S, ∂²V/∂S² を取得)
#======================================================
def pde_residual(model, t, s):
    # V(t, s)
    V = model(t, s)  # (batch_size,1)
    
    # 1階微分
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
    
    # 2階微分
    d2V_ds2 = torch.autograd.grad(
        dV_ds, s,
        grad_outputs=torch.ones_like(dV_ds),
        retain_graph=True,
        create_graph=True
    )[0]
    
    # Black-Scholes PDE:
    # dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS^2 + (r - d)* S * dV/dS - r * V = 0
    # ここでは d=0 としているが残しておいてもOK
    # 注意: t in [0,T], S>0
    # PINN上では t=0->T を「順方向」の時間として扱うことが多い点に留意 (後述)
    # PDE 残差:
    res = dV_dt + 0.5 * sigma**2 * s**2 * d2V_ds2 \
          + (r - d) * s * dV_ds - r * V
    return res


#======================================================
# 4. 境界条件、終端条件を課すための Loss
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
# 6. 学習ループ
#======================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(n_hidden=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    # PDE内部点のロス
    loss_pde = interior_pde_loss(model, batch_size_int=128)
    # 境界・終端条件ロス
    loss_bc = boundary_final_loss(model, batch_size_bc=128)
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.6f}, PDE = {loss_pde.item():.6f}, BC = {loss_bc.item():.6f}")


#======================================================
# 7. 学習後のテスト・簡易評価
#    t=0 上での価格曲線などを可視化例
#======================================================
model.eval()

# 例: t=0 で S を細かく変化させて予測
S_test = np.linspace(0.4, Smax, 1000)[:,None]
t_zero = np.zeros_like(S_test)

ts_torch = torch.FloatTensor(t_zero).to(device)
Ss_torch = torch.FloatTensor(S_test).to(device)
with torch.no_grad():
    V_pred = model(ts_torch, Ss_torch).cpu().numpy().flatten()

# 参考解(ブラック–ショールズ理論解)と比較してみる (欧州コール, d=0)
# Analytical closed-form (BS formula) for T=1, r=0.02, sigma=0.2
def bs_call_price(S, K, r, sigma, T):
    import math
    from math import log, sqrt, exp
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    c = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return c

V_bs = bs_call_price(S_test, K, r, sigma, T)

# 簡単に matplotlib でプロット (Jupyter等で実行想定)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(S_test, V_pred, label='PINN Approx (t=0)')
    plt.plot(S_test, V_bs, 'r--', label='BS Formula (t=0)')
    plt.xlabel('S')
    plt.ylabel('Option Value at t=0')
    plt.legend()
    plt.title("European Call Option via PINN vs. Black-Scholes Formula")
    plt.show()
except ImportError:
    print("Matplotlib not available, skipping plot.")
