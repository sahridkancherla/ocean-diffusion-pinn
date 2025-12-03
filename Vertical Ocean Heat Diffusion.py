import torch
import torch.nn as nn
#from torchmetrics import Accuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import xarray as xr
import tarfile
import gsw
from datetime import datetime
import gzip
import netCDF4
import xarray as xr
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import griddata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
# Faster, but less precise
torch.set_float32_matmul_precision("high")

# %%
def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pt"):
    torch.save({
    "model": pinn.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
}, filename)
# %%
# 1. Data Preparation

print('Temp Data')
ds_T = xr.open_dataset("Data/temp.cdf.gz", engine="netcdf4")
print(ds_T)
print('\nPos Data')
ds_pos = xr.open_dataset("Data/pos.cdf.gz", engine="netcdf4")
print(ds_pos)
print('\nSST Data')
ds_sst = xr.open_dataset("Data/sst.cdf.gz", engine="netcdf4")
print(ds_sst)


ST = ds_T['ST_6020'].to_numpy().reshape(282979, 21)
QT = ds_T['QT_5020'].to_numpy().reshape(282979, 21)
T = ds_T['T_20'].to_numpy().reshape((282979, 21))
for n in range(21):
    print(QT[0, n], ST[0, n], T[0, n])


SP = ds_T['SP_6001'].to_numpy().reshape(282979, 21)
QP = ds_T['QP_5001'].to_numpy().reshape(282979, 21)
P = ds_T['P_1'].to_numpy().reshape((282979, 21))
for n in range(21):
    print(QP[0, n], SP[0, n], P[0, n])


SSST = ds_sst['ST_6025'].to_numpy().reshape(282979)
QSST = ds_sst['QT_5025'].to_numpy().reshape(282979)
SST = ds_sst['T_25'].to_numpy().reshape(282979)
for n in range(21):
    print(QSST[n], SSST[n], SST[n])

# %%
# Removing invalid/non-existent datapoints
P[:,0] = 1
P = np.delete(P, 5, 1)
P = np.delete(P, 11, 1)

T = np.delete(T, 5 ,1)
T = np.delete(T, 11, 1)

tot_T = 0
for i in enumerate(T):
    for j in enumerate(T[i[0]]):
        if T[i[0], j[0]] > 1e10:
            T[i[0], j[0]] = T[i[0]-1, j[0]]
            
for i in enumerate(T):
    for j in enumerate(T[i[0]]):
        if T[i[0], j[0]] > 1e10:
            tot_T += 1
            
print(tot_T)

tot_P = 0
for i in enumerate(P):
    for j in enumerate(P[i[0]]):
        if P[i[0], j[0]] > 1e10:
            P[i[0], j[0]] = P[i[0]-1, j[0]]
            
for i in enumerate(P):
    for j in enumerate(P[i[0]]):
        if P[i[0], j[0]] > 1e10:
            tot_P += 1
            
print(tot_P)

tot_SST = 0
for i in enumerate(SST):
    if SST[i[0]] > 1e10:
        SST[i[0]] = SST[i[0]-1]
            
for i in enumerate(SST):
    if SST[i[0]] > 1e10:
        tot_SST += 1
            
print(tot_SST)
# %%
# Pressure -> Depth and time in seconds
z = gsw.z_from_p(P, 32.3)

time = ds_T['time'].to_numpy().reshape((282979))
t0 = time[0]
t = (time - t0) / np.timedelta64(1, 's')

# %%
# Torch tensor
T = np.transpose(T)
z = np.transpose(z)
# Target temps T
T = torch.from_numpy(T).flatten().to(dtype=torch.float32).to(device)
z = torch.from_numpy(z).flatten().to(dtype=torch.float32).to(device)
t = torch.from_numpy(t).repeat(19).flatten().to(dtype=torch.float32).to(device)
# BCs
SST = torch.from_numpy(SST).repeat(19).flatten().to(dtype=torch.float32).to(device)


# %%
# Normalise

T_mean = T.mean().to(device)
T_std = T.std().to(device)
T_norm = (T - T_mean) / T_std
print(T_norm)

SST_mean = SST.mean()
SST_std = SST.std()
SST_norm = (SST - SST_mean) / SST_std
print(SST_norm)
# min-max norm for X, because linearly increasing data
min_t = t.min(dim=0)[0]
max_t = t.max(dim=0)[0]
t_norm = (t - min_t) / (max_t - min_t)
t_norm
print(t_norm)

min_z = z.min(dim=0)[0]
max_z = z.max(dim=0)[0]
z_norm = (z - min_z) / (max_z - min_z)
print(z_norm)

# %%
# Actual ML starts here, finally
torch.manual_seed(2025)

class network(nn.Module):
    def __init__(self, N_in):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_in, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
            )
        self.temp_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
            )
        self.D_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
    def forward(self, x):
        encoded = self.encoder(x)
        temp = self.temp_head(encoded)
        diffusivity = self.D_head(encoded)
        return temp, diffusivity

# %%
# Initialise the PINN, set requires grad to True
torch.manual_seed(2025)
pinn = network(2).to(device)

t_norm = t_norm.requires_grad_().to(device)
z_norm = z_norm.requires_grad_().to(device)
SST_norm = SST_norm.requires_grad_().to(device)

# boundaries
t_boundary = t_norm.to(device)
print(t_boundary)
z0_boundary = torch.zeros_like(t_boundary).requires_grad_().to(device)
zH_boundary = torch.ones_like(t_boundary).to(device)
zH_boundary = zH_boundary.requires_grad_().to(device)
print(zH_boundary)

X0_boundary = torch.transpose(torch.stack([z0_boundary, t_boundary]), 0, 1).to(device)
XH_boundary = torch.transpose(torch.stack([zH_boundary, t_boundary]), 0, 1).to(device)
# collocation points
z_c = torch.rand(5376601).to(device)
t_c = torch.rand(5376601).to(device)
X_c = torch.transpose(torch.stack([z_c, t_c]), 0, 1).to(device)
# combined tensor

X = torch.transpose(torch.stack([z_norm, t_norm]), 0, 1).to(device)


# %%
# Training set up
torch.manual_seed(2025)
num_epochs = 5
optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-6)
batch_size = 4096

dataset = TensorDataset(X[:2998271,:], T_norm[:2998271], X0_boundary[:2998271,:], XH_boundary[:2998271,:], X_c[:2998271,:])

train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False
    )

# regulariser params
lambda_bc_sst, lambda_bc_dt, lambda_phys, lambda_data, lambda_D = 0.1, 0.1, 0.1, 10, 0

# %%
# Training loop
torch.manual_seed(2025)

loss = 0

#try:
for epoch in range(num_epochs):
    epoch_loss = 0
    
    pinn.train()
    torch.autograd.set_detect_anomaly(False)
    for batch_idx, (batch_X, batch_T_norm, batch_X0_boundary, batch_XH_boundary,batch_X_c) in enumerate(train_loader):
        batch_X, batch_T_norm, batch_X0_boundary, batch_XH_boundary, batch_X_c = batch_X.clone().detach().requires_grad_(), batch_T_norm.clone().detach().requires_grad_(), batch_X0_boundary.clone().detach().requires_grad_(), batch_XH_boundary.clone().detach().requires_grad_(), batch_X_c.clone().detach().requires_grad_()
        
        #loss_D = 0
        loss_bc_sst = 0
        loss_bc_dt = 0
        loss_phys = 0 
        loss_data = 0
        optimiser.zero_grad()
        
        
        # smoothing penalty on D
        D = pinn(batch_X)[1]
        #dD_dz = torch.autograd.grad(D, batch_X, torch.ones_like(D), create_graph=True)[0][:, 0]
        #loss_D = torch.mean(torch.trapz(y=dD_dz**2, x=batch_X[:, 0]))
        
        # boundary losses
        T_0t = pinn(batch_X0_boundary)[0]
        loss_bc_sst = torch.mean((T_0t)**2)
        
        T_Ht = pinn(batch_XH_boundary)[0]
        dT_dz_boundary = torch.autograd.grad(T_Ht, batch_XH_boundary, torch.ones_like(T_Ht), create_graph=True)[0][:, 0]
        loss_bc_dt = torch.mean((dT_dz_boundary)**2) 
        
        # physics loss
        D_c = pinn(batch_X_c)[1]
        T_phys = pinn(batch_X_c)[0]
        dT_dz_phys = torch.reshape(torch.autograd.grad(T_phys, batch_X_c, torch.ones_like(T_phys), create_graph=True)[0][:, 0], (len(batch_X), 1))
        dT_dt_phys = torch.reshape(torch.autograd.grad(T_phys, batch_X_c, torch.ones_like(T_phys), create_graph=True)[0][:, 1], (len(batch_X), 1))
        dT_dz_phys2 = torch.autograd.grad(D_c * dT_dz_phys, batch_X_c, torch.ones_like(T_phys), create_graph=True)[0][:, 0]
        loss_phys = torch.mean((dT_dz_phys2 - dT_dt_phys)**2)
        
        # data loss
        
        T_pred = pinn(batch_X)[0]
        loss_data = torch.mean((batch_T_norm - T_pred)**2)
        
        # total loss
        loss = (lambda_bc_sst * loss_bc_sst) + (lambda_bc_dt * loss_bc_dt) + (lambda_phys * loss_phys) + (lambda_data * loss_data) #+ (lambda_D * loss_D)
        
        epoch_loss = epoch_loss + (lambda_bc_sst * loss_bc_sst) + (lambda_bc_dt * loss_bc_dt) + (lambda_phys * loss_phys) + (lambda_data * loss_data) #+ (lambda_D * loss_D)
        
        # backprop, optimiser step
        
        loss.backward()
        optimiser.step()
        
        # tracking losses
        if batch_idx % 10 == 0:
            batch_loss = loss 
            print(f" Batch: {batch_idx} - batch loss: {batch_loss:.10f}")        
        if batch_idx % 100 == 0:
            save_checkpoint(pinn, optimiser, batch_idx)
    epoch_loss = loss / (batch_idx + 1)
    print(f" Epoch: {epoch} - epoch loss: {epoch_loss:.10f}")
#except Exception as e:
    
   # print("CRASH! Saving emergency checkpoint.")
    #save_checkpoint(pinn, optimiser, batch_idx)
# %%
# Evaluation 

dataset_eval = TensorDataset(X[2998271:,:], T_norm[2998271:], X0_boundary[2998271:,:], XH_boundary[2998271:,:], X_c[2998271:,:])

test_loader = DataLoader(
    dataset=dataset_eval,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False
    )
    
test_loss = 0

pinn.eval()

   
for batch_idx_eval, (batch_X_eval, batch_T_norm_eval, batch_X0_boundary_eval, batch_XH_boundary_eval, batch_X_c_eval) in enumerate(test_loader):
    batch_X_eval, batch_T_norm_eval, batch_X0_boundary_eval, batch_XH_boundary_eval, batch_X_c_eval = batch_X_eval.clone().detach().requires_grad_(), batch_T_norm_eval.clone().detach().requires_grad_(), batch_X0_boundary_eval.clone().detach().requires_grad_(), batch_XH_boundary_eval.clone().detach().requires_grad_(), batch_X_c_eval.clone().detach().requires_grad_()
    
    #test_loss_D = 0
    test_loss_bc_sst = 0
    test_loss_bc_dt = 0
    test_loss_phys = 0 
    test_loss_data = 0
    
    D_test = pinn(batch_X_eval)[1]
    #dD_dz_test = torch.autograd.grad(D_test, batch_X_eval, torch.ones_like(D_test), create_graph=True)[0][:, 0]
    #loss_D_test = torch.mean(torch.trapz(y=dD_dz_test**2, x=batch_X_eval[:, 0]))
    
    # boundary losses
    T_0t_test = pinn(batch_X0_boundary_eval)[0]
    loss_bc_sst_test = torch.mean((T_0t_test)**2)
    
    T_Ht_test = pinn(batch_XH_boundary_eval)[0]
    dT_dz_boundary_test = torch.autograd.grad(T_Ht_test, batch_XH_boundary_eval, torch.ones_like(T_Ht_test), create_graph=True)[0][:, 0]
    loss_bc_dt_test = torch.mean((dT_dz_boundary_test)**2)
    
    # physics loss
    D_c_test = pinn(batch_X_c_eval)[1]
    T_phys_test = pinn(batch_X_c_eval)[0]
    dT_dz_phys_test = torch.reshape(torch.autograd.grad(T_phys_test, batch_X_c_eval, torch.ones_like(T_phys_test), create_graph=True)[0][:, 0], (len(batch_X_eval), 1))
    dT_dt_phys_test = torch.reshape(torch.autograd.grad(T_phys_test, batch_X_c_eval, torch.ones_like(T_phys_test), create_graph=True)[0][:, 1], (len(batch_X_eval), 1))
    dT_dz_phys2_test = torch.autograd.grad(D_c_test * dT_dz_phys_test, batch_X_c_eval, torch.ones_like(T_phys_test), create_graph=True)[0][:, 0]
    loss_phys_test = torch.mean((dT_dz_phys2_test - dT_dt_phys_test)**2)
    
    # data loss
    T_pred_test = pinn(batch_X_eval)[0]
    loss_data_test = torch.mean((batch_T_norm_eval - T_pred_test)**2)
    
    # total loss
    test_loss = test_loss + (lambda_bc_sst * loss_bc_sst) + (lambda_bc_dt * loss_bc_dt) + (lambda_phys * loss_phys) + (lambda_data * loss_data)# + (lambda_D * loss_D)
    
    if batch_idx_eval % 10 == 0:
        print(batch_idx_eval)
        #print(f'Smoothing penalty on D: {loss_D_test}')
        print(f'Boundary losses: z=0 - {loss_bc_sst_test}, z=H - {loss_bc_dt_test}')
        print(f'Physics loss: {loss_phys_test}')
        print(f'Data loss: {loss_data_test}')
    
print(f'Total test loss: {test_loss}')   

# %%
# Sorting out plotting variables

T_plot = (T_norm * T_std + T_mean).cpu().numpy()
z_plot = (X[:, 0] * (max_z - min_z) + min_z).detach().cpu().numpy()
t_plot = (X[:, 1] * (max_t - min_t) + min_t).detach().cpu().numpy()

# %%
# Inferenece of all datapoints
pinn.eval()

dataset_total = TensorDataset(X)

loader = DataLoader(
    dataset=dataset_total,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
    )

pred_temps = []

for batch in loader:
    batch = batch[0].to(device)
    temp = pinn(batch)[0]
    pred_temps.append(temp.cpu())

pred_temps = torch.cat(pred_temps, dim=0)

pred_temps = (pred_temps.flatten() * T_std.cpu() + T_mean.cpu()).detach().cpu().numpy()


# %%
# Plot

vmin = min(T_plot.min(), pred_temps.min())
vmax = max(T_plot.max(), pred_temps.max())
cmap = "jet"  

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

h1 = ax[0].hexbin(t_plot, z_plot, C=T_plot, gridsize=200, reduce_C_function=np.mean,
                   cmap=cmap, vmin=vmin, vmax=vmax)
ax[0].invert_yaxis()   # optional: depth increases downward
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Depth")
ax[0].set_title("T (True)")

# 2. Predicted temperature
h2 = ax[1].hexbin(t_plot, z_plot, C=pred_temps, gridsize=200, reduce_C_function=np.mean,
                   cmap=cmap, vmin=vmin, vmax=vmax)
ax[1].invert_yaxis()
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Depth")
ax[1].set_title("T_pred")

# Shared colorbar
fig.colorbar(h2, ax=ax.ravel().tolist(), label="Temperature")

plt.show()

# %%

#