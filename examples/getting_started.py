# --- START OF FILE examples/getting_started.py ---
import torch
import time
import sys
from pathlib import Path

# --- Boilerplate to ensure the compiled module can be found ---
# This assumes the script is run from the root of the HELIOS_EMBED project.
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Step 1: Import the Helios.Embed Engine ---
try:
    # We import both the stateless function and the stateful class
    from helios_embed._core import compute_rkhs_embedding, IncrementalNystromEngine
    print("✅ Successfully imported Helios.Embed engine.")
except ImportError:
    print("❌ FAILED TO IMPORT HELIOS.EMBED")
    print("Please ensure you have compiled the module by running:")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)

# --- Step 2: Prepare Sample Data ---
print("\n--- Preparing Sample Data ---")

# Check for a CUDA device
if not torch.cuda.is_available():
    print("❌ CUDA device not found. This example requires a GPU.")
    sys.exit(1)
device = torch.device("cuda")
print(f"   - Using device: {torch.cuda.get_device_name(device)}")

# Define our problem dimensions
N_initial = 1024 # Number of initial vectors
N_update = 128   # Number of new vectors to stream in
D = 384          # Embedding dimension (like a small sentence transformer)
m = 128          # Number of landmarks for the Nyström approximation

# Create random tensors to simulate embeddings
torch.manual_seed(42)
X_initial = torch.randn(N_initial, D, device=device)
X_update = torch.randn(N_update, D, device=device)
# For a realistic scenario, landmarks are a subset of the initial data
landmarks = X_initial[torch.randperm(N_initial, device=device)[:m]]

# Define kernel parameters
gamma = 0.1
ridge = 1e-6

print(f"   - Initial data: {N_initial} vectors of dimension {D}")
print(f"   - Update data: {N_update} vectors of dimension {D}")
print(f"   - Landmarks: {m} vectors")


# --- Step 3: Using the Stateless Engine (for one-shot computation) ---
print("\n--- Example 1: Stateless 'compute_rkhs_embedding' ---")

# The stateless function is ideal for when you have all your data at once.
start_time = time.perf_counter()
# This one line performs the entire Nyström feature embedding
features_stateless = compute_rkhs_embedding(X_initial, landmarks, gamma, ridge)
torch.cuda.synchronize()
duration = time.perf_counter() - start_time

print(f"   - Successfully computed features in {duration*1000:.2f} ms.")
print(f"   - Output shape: {features_stateless.shape}")
assert features_stateless.shape == (N_initial, m)


# --- Step 4: Using the Stateful Engine (for streaming data) ---
print("\n--- Example 2: Stateful 'IncrementalNystromEngine' ---")

# The stateful engine is ideal for streaming workloads.
# First, we initialize the engine. This pre-computes and caches the expensive part.
print("   - Initializing the engine (pre-computing K_mm^-1/2)...")
start_time = time.perf_counter()
streaming_engine = IncrementalNystromEngine(landmarks, gamma, ridge)
torch.cuda.synchronize()
duration = time.perf_counter() - start_time
print(f"   - Engine initialized in {duration*1000:.2f} ms.")

# Next, we build the features for our initial batch of data.
print("   - Building initial feature set...")
start_time = time.perf_counter()
features_old = streaming_engine.build(X_initial)
torch.cuda.synchronize()
duration = time.perf_counter() - start_time
print(f"   - 'build()' completed in {duration*1000:.2f} ms.")
assert features_old.shape == (N_initial, m)

# Now, we simulate a new batch of data arriving and update our feature set.
# This is much faster than recomputing from scratch.
print("   - Streaming in new data with 'update()'... ")
start_time = time.perf_counter()
features_new = streaming_engine.update(X_update, features_old)
torch.cuda.synchronize()
duration = time.perf_counter() - start_time
print(f"   - 'update()' completed in {duration*1000:.2f} ms.")
assert features_new.shape == (N_initial + N_update, m)

# --- Step 5: Verification (Proof of Correctness) ---
print("\n--- Verifying Correctness ---")
# To prove the streaming engine is correct, we compare its final output
# to a full re-computation on the combined dataset.
X_combined = torch.cat([X_initial, X_update], dim=0)
features_ground_truth = compute_rkhs_embedding(X_combined, landmarks, gamma, ridge)

# Check for bit-perfect accuracy
rel_mse = torch.mean((features_ground_truth - features_new)**2) / (torch.mean(features_ground_truth**2) + 1e-20)
accuracy_threshold = 1e-7 # Our official float32 standard

print(f"   - Relative MSE between streaming result and ground truth: {rel_mse.item():.2e}")
if rel_mse.item() <= accuracy_threshold:
    print("   - ✅ SUCCESS: The stateful engine is bit-perfectly accurate.")
else:
    print("   - ❌ FAILURE: The stateful engine produced an incorrect result.")

print("\n" + "="*50)
print("✅ Getting Started Example Complete.")
print("="*50)
# --- END OF FILE examples/getting_started.py ---