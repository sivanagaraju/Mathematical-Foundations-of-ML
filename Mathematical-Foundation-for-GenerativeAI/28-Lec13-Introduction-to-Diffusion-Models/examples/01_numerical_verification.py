import torch

def numerical_verification_diffusion():
    # Set seed for deterministic verification
    torch.manual_seed(42)
    B, D, T = 10000, 1, 50 # Shape: (B, D)
    
    x0 = torch.ones(B, D) * 2.0 # Clean data: mean=2.0, var=0.0
    
    betas = torch.linspace(0.01, 0.05, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # 1. Step-by-step Markov forward process
    xt = x0.clone()
    for t in range(T):
        eps = torch.randn_like(xt) # Shape: (B, D)
        xt = torch.sqrt(alphas[t]) * xt + torch.sqrt(betas[t]) * eps
        
    # 2. Closed-form jump kernel at t=T-1
    t_target = T - 1
    eps_jump = torch.randn_like(x0) # Shape: (B, D)
    xt_jump = torch.sqrt(alpha_bars[t_target]) * x0 + torch.sqrt(1.0 - alpha_bars[t_target]) * eps_jump
    
    # Check theoretical mean and variance
    expected_mean = (torch.sqrt(alpha_bars[t_target]) * 2.0).item()
    expected_var = (1.0 - alpha_bars[t_target]).item()
    
    emp_mean_step = xt.mean().item()
    emp_var_step = xt.var().item()
    emp_mean_jump = xt_jump.mean().item()
    emp_var_jump = xt_jump.var().item()
    
    print(f"Theoretical Mean: {expected_mean:.4f}, Emp Step: {emp_mean_step:.4f}, Emp Jump: {emp_mean_jump:.4f}")
    print(f"Theoretical Var:  {expected_var:.4f}, Emp Step: {emp_var_step:.4f}, Emp Jump: {emp_var_jump:.4f}")
    
    assert abs(emp_mean_step - expected_mean) < 0.05
    assert abs(emp_mean_jump - expected_mean) < 0.05
    assert abs(emp_var_step - expected_var) < 0.05
    assert abs(emp_var_jump - expected_var) < 0.05
    print("Numerical verification PASSED.")

if __name__ == "__main__":
    numerical_verification_diffusion()
