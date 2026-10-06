"""
Simulation: Knowledge Distillation (Hinton Formulation)
Focus: Teacher-Student Architecture, Temperature Scaling, and Tau-Squared Gradient Invariance

This script implements:
1. An overparameterized Teacher network and a compact Student network.
2. Temperature-scaled softmax with dark knowledge extraction.
3. Proof of tau-squared gradient scaling invariance.
4. Combined distillation loss optimization.
5. Strict ASCII-only prints and numerical assertions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def run_distillation_simulation():
    print("=" * 60)
    print("RUNNING KNOWLEDGE DISTILLATION (HINTON) SIMULATION")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    batch_size = 32
    in_features = 20
    num_classes = 5
    
    # -------------------------------------------------------------
    # 1. Define Teacher (Wide/Deep) and Student (Compact) Architectures
    # -------------------------------------------------------------
    class TeacherNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, num_classes)
            
        def forward(self, x):
            # Input: [B, 20] -> Hidden: [B, 64] -> Logits: [B, 5]
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            logits = self.fc3(h)
            return logits

    class StudentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 16)
            self.fc2 = nn.Linear(16, num_classes)
            
        def forward(self, x):
            # Input: [B, 20] -> Hidden: [B, 16] -> Logits: [B, 5]
            h = F.relu(self.fc1(x))
            logits = self.fc2(h)
            return logits

    teacher = TeacherNet()
    student = StudentNet()
    
    # Freeze teacher parameters (deterministic oracle)
    for p in teacher.parameters():
        p.requires_grad = False
        
    # Generate synthetic dataset
    x_train = torch.randn(batch_size, in_features)  # Shape: [32, 20]
    y_train = torch.randint(0, num_classes, (batch_size,))  # Shape: [32]
    
    # -------------------------------------------------------------
    # 2. Temperature Scaling & Dark Knowledge Inspection
    # -------------------------------------------------------------
    with torch.no_grad():
        z_teacher = teacher(x_train)  # Shape: [32, 5]
        
    sample_logits = z_teacher[0]
    p_sharp = F.softmax(sample_logits / 1.0, dim=0)
    p_soft = F.softmax(sample_logits / 4.0, dim=0)
    
    print("\n[Analysis 1] Temperature Scaling on Single Sample Logits:")
    print(f"Logits: {[round(v, 2) for v in sample_logits.tolist()]}")
    print(f"Probabilities (tau = 1.0): {[round(v, 4) for v in p_sharp.tolist()]}")
    print(f"Probabilities (tau = 4.0): {[round(v, 4) for v in p_soft.tolist()]}")
    
    # Verify dark knowledge exposure: entropy of soft distribution is strictly higher
    entropy_sharp = -(p_sharp * torch.log(p_sharp + 1e-12)).sum()
    entropy_soft = -(p_soft * torch.log(p_soft + 1e-12)).sum()
    assert entropy_soft > entropy_sharp, "Temperature scaling must strictly increase distribution entropy"
    
    # -------------------------------------------------------------
    # 3. Demonstration of Tau-Squared (tau^2) Gradient Scaling
    # -------------------------------------------------------------
    print("\n[Analysis 2] Verifying tau^2 Gradient Normalization Multiplier:")
    
    z_student_test = torch.zeros(1, num_classes, requires_grad=True)
    z_teacher_test = torch.tensor([[3.0, 1.0, -1.0, -2.0, 0.0]])
    
    grad_norms = []
    temperatures = [1.0, 2.0, 5.0, 10.0]
    
    for tau in temperatures:
        p_T = F.softmax(z_teacher_test / tau, dim=-1)
        log_p_S = F.log_softmax(z_student_test / tau, dim=-1)
        
        # Unscaled KL divergence
        loss_unscaled = F.kl_div(log_p_S, p_T, reduction='batchmean')
        
        # Scaled by tau^2
        loss_scaled = loss_unscaled * (tau ** 2)
        
        # Calculate gradients
        grad_unscaled = torch.autograd.grad(loss_unscaled, z_student_test, retain_graph=True)[0]
        grad_scaled = torch.autograd.grad(loss_scaled, z_student_test, retain_graph=True)[0]
        
        print(f"tau={tau:4.1f} | Unscaled Grad Norm: {grad_unscaled.norm().item():.6f} | Scaled Grad Norm: {grad_scaled.norm().item():.6f}")
        grad_norms.append(grad_scaled.norm().item())
        
    # Assert that scaled gradient norm remains approximately invariant across high temperatures
    ratio_high_temp = grad_norms[-1] / grad_norms[-2]  # Between tau=10 and tau=5
    assert 0.95 <= ratio_high_temp <= 1.05, f"Scaled gradient norm should stabilize at high tau, got ratio {ratio_high_temp:.4f}"
    print("PASS: tau^2 multiplier successfully prevents gradient vanishing as temperature increases.")
    
    # -------------------------------------------------------------
    # 4. End-to-End Distillation Training Loop
    # -------------------------------------------------------------
    print("\n[Training] Optimizing Student with Combined Distillation Loss...")
    
    tau = 3.0
    alpha = 0.6  # 60% distillation, 40% ground truth
    optimizer = optim.Adam(student.parameters(), lr=0.01)
    
    initial_loss = 0.0
    final_loss = 0.0
    
    for epoch in range(30):
        optimizer.zero_grad()
        
        # Forward pass
        z_S = student(x_train)  # Shape: [32, 5]
        
        # Soft distillation loss
        p_T = F.softmax(z_teacher / tau, dim=-1)
        log_p_S = F.log_softmax(z_S / tau, dim=-1)
        loss_soft = F.kl_div(log_p_S, p_T, reduction='batchmean') * (tau ** 2)
        
        # Hard classification loss
        loss_hard = F.cross_entropy(z_S, y_train)
        
        # Combined Hinton objective
        loss_total = (1.0 - alpha) * loss_hard + alpha * loss_soft
        
        if epoch == 0:
            initial_loss = loss_total.item()
            
        loss_total.backward()
        optimizer.step()
        
    final_loss = loss_total.item()
    print(f"Distillation Training Loss: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Combined distillation loss must decrease over optimization epochs"
    
    print("=" * 60)
    print("KNOWLEDGE DISTILLATION SIMULATION COMPLETED WITH 100% PASS RATE")
    print("=" * 60)

if __name__ == '__main__':
    run_distillation_simulation()
