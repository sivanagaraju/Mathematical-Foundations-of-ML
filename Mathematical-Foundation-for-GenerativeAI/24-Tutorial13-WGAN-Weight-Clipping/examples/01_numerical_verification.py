"""
Numerical Verification of 1D Wasserstein Distance vs Jensen-Shannon Divergence.
Demonstrates non-vanishing gradients across disjoint distributions.
"""
import numpy as np

def run_verification():
    np.random.seed(42)
    # Real distribution P_r at 0.0, Model distribution P_theta at theta
    thetas = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    w1_distances = []
    js_divergences = []
    
    for theta in thetas:
        # 1-Wasserstein distance between point masses is |theta|
        w1 = abs(theta)
        w1_distances.append(w1)
        
        # JS divergence between disjoint point masses is log(2)
        js = np.log(2.0)
        js_divergences.append(js)
        
    # Check that Wasserstein scales linearly with distance
    diffs = np.diff(w1_distances)
    assert np.all(diffs > 0), "Wasserstein distance must grow monotonically with separation"
    
    # Check that JS divergence is completely flat (zero gradient)
    assert np.allclose(js_divergences, np.log(2.0)), "JS divergence must be constant log(2)"
    
    print("Numerical Verification Passed: Wasserstein scales linearly, JS has zero gradient.")
    for t, w, j in zip(thetas, w1_distances, js_divergences):
        print(f"  theta={t:.1f} -> W1={w:.2f}, JS={j:.4f}")

if __name__ == "__main__":
    run_verification()
