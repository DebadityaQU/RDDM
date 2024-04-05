import torch
import numpy as np
import torch.nn as nn
from statistics import mean
from math import floor
from functools import reduce
from typing import Dict, Tuple
import neurokit2 as nk
import torch.nn.functional as F
from model import DiffusionUNetCrossAttention, ConditionNet

def ddpm_schedule(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedule for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "beta_t": beta_t
    }

class NaiveDDPM(nn.Module):
    def __init__(
        self,
        eps_model,
        betas,
        n_T,
        criterion = nn.MSELoss(),
    ):
        super(NaiveDDPM, self).__init__()
        self.eps_model = eps_model
        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

        self.criterion = criterion

    def forward(self, x=None, cond=None, mode="train", window_size=128*4):

        if mode == "train":
            
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
                x.device
            ) 

            eps = torch.randn_like(x)
            
            x_t = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * eps
            )  

            return self.criterion(eps, self.eps_model(x_t, cond, _ts / self.n_T))

        elif mode == "sample":

            n_sample = cond["down_conditions"][-1].shape[0]
            device = cond["down_conditions"][-1].device
            
            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                
                z = torch.randn(n_sample, 1, window_size).to(device) if i > 1 else 0

                eps = self.eps_model(x_i, cond, torch.tensor(i / self.n_T).to(device).repeat(n_sample))
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i

class RDDM(nn.Module):
    def __init__(
        self,
        eps_model,
        region_model,
        betas,
        n_T,
        criterion = nn.MSELoss(),
    ):
        super(RDDM, self).__init__()
        self.eps_model = eps_model
        self.region_model = region_model

        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

        self.criterion = criterion

    def create_noise_in_regions(self, patch_labels):

        patch_roi = torch.round(patch_labels)

        mask = patch_roi == 1
        random_noise = torch.randn_like(patch_roi)
        masked_noise = random_noise * mask.float()
        
        return masked_noise, random_noise

    def forward(self, x=None, cond1=None, cond2=None, mode="train", patch_labels=None, window_size=128*4):

        if mode == "train":
            
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
                x.device
            ) 

            eps, unmasked_eps = self.create_noise_in_regions(patch_labels)
            
            x_t = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * eps
            )  

            x_t_unmasked = (
                self.sqrtab[_ts, None, None] * x
                + self.sqrtmab[_ts, None, None] * unmasked_eps
            )

            pred_x_t = self.region_model(x_t_unmasked, cond2, _ts / self.n_T)

            pred_masked_eps = self.eps_model(x_t, cond1, _ts / self.n_T)

            ddpm_loss = self.criterion(eps, pred_masked_eps)

            region_loss = self.criterion(pred_x_t, x_t)

            return ddpm_loss, region_loss

        elif mode == "sample":

            n_sample = cond1["down_conditions"][-1].shape[0]
            device = cond1["down_conditions"][-1].device
            
            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                
                if i > 1:
                    z = torch.randn(n_sample, 1, window_size).to(device)

                else:
                    z = 0
            
                # rho_phi estimates the trajectory from Gaussian manifold to Masked Gaussian manifold
                x_i = self.region_model(x_i, cond2, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                # epsilon_theta predicts the noise that needs to be removed to move from Masked Gaussian manifold to ECG manifold
                eps = self.eps_model(x_i, cond1, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i

def freeze_model(model):

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    
    return model

def load_pretrained_DPM(PATH, nT, type="RDDM", device="cuda"):

    if type == "RDDM":

        dpm = RDDM(
            eps_model=DiffusionUNetCrossAttention(512, 1, device),
            region_model=DiffusionUNetCrossAttention(512, 1, device),
            betas=(1e-4, 0.2), 
            n_T=nT
        ).to(device)
        
        dpm.load_state_dict(torch.load(PATH + "rddm_main_network.pth"))

        dpm = freeze_model(dpm)

        Conditioning_network1 = ConditionNet().to(device)
        Conditioning_network1.load_state_dict(torch.load(PATH + "rddm_condition_encoder_1.pth"))
        Conditioning_network1 = freeze_model(Conditioning_network1)

        Conditioning_network2 = ConditionNet().to(device)
        Conditioning_network2.load_state_dict(torch.load(PATH + "rddm_condition_encoder_2.pth"))
        Conditioning_network2 = freeze_model(Conditioning_network2)
 
        return dpm, Conditioning_network1, Conditioning_network2
    
    else: # Naive DDPM

        dpm = NaiveDDPM(
            eps_model=DiffusionUNetCrossAttention(512, 1, device),
            betas=(1e-4, 0.2), 
            n_T=nT
        ).to(device)

        dpm.load_state_dict(torch.load(PATH + f"ddpm_main_network_{nT}.pth"))
        dpm = freeze_model(dpm)

        Conditioning_network = ConditionNet().to(device)
        Conditioning_network.load_state_dict(torch.load(PATH + f"ddpm_condition_encoder_{nT}.pth"))
        Conditioning_network = freeze_model(Conditioning_network)
        
        return dpm, Conditioning_network, None