# --- START OF FILE docs/threading_model.md ---
# Helios.Embed v1.0.0 - Concurrency & Threading Model

This document outlines the thread-safety guarantees and concurrency model for the `Helios.Embed` module.

---

## 1. Guiding Principle: Simplicity and Predictability

The concurrency model for `Helios.Embed` is designed to be simple, predictable, and safe. We prioritize clear, explicit behavior over complex, implicit parallelism to prevent common concurrency bugs like race conditions and deadlocks.

## 2. Stateless Function: `compute_rkhs_embedding`

*   **Guarantee:** The `compute_rkhs_embedding` function is **fully thread-safe and reentrant.**
*   **Description:** This function is a pure, stateless computation. It accepts tensors as input and produces a new tensor as output, without modifying any shared global state.
*   **Usage:** You can safely call this function from multiple Python threads simultaneously, provided each call operates on its own independent tensor data. The underlying PyTorch ATen library and CUDA kernels are themselves thread-safe for operations on distinct data.
*   **Example:**
    ```python
    # This is a safe and valid concurrent pattern
    import threading
    
    def worker(X, landmarks):
        # Each thread works on its own data
        result = compute_rkhs_embedding(X, landmarks, 0.1, 1e-6)
        # ... do something with result ...

    # Create and run multiple threads
    threads = [threading.Thread(target=worker, args=(X_i, L_i)) for X_i, L_i in data_chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ```

## 3. Stateful Class: `IncrementalNystromEngine`

*   **Guarantee:** Instances of the `IncrementalNystromEngine` class are **NOT thread-safe.**
*   **Description:** This class is stateful. It maintains an internal, cached `K_mm_inv_sqrt_` tensor. The `build()` and `update()` methods are designed to be called sequentially on a single instance.
*   **Usage:** A single instance of `IncrementalNystromEngine` should only be accessed and modified by **one thread at a time.** If you need to use the engine in a multi-threaded environment, you must either:
    1.  **Instantiate one engine per thread:** This is the safest and recommended approach. Each thread gets its own independent engine and state.
    2.  **Use External Locking:** If multiple threads must share a single engine instance, you are responsible for wrapping all calls to its methods (`build`, `update`) with a lock (e.g., `threading.Lock`).

*   **Unsafe Example (DO NOT DO THIS):**
    ```python
    # UNSAFE: Multiple threads calling .update() on the same engine instance
    shared_engine = IncrementalNystromEngine(...)
    
    def unsafe_worker(X_new, Phi_old):
        # This will lead to race conditions and corrupted state
        shared_engine.update(X_new, Phi_old)
    ```

*   **Safe Example (One Engine Per Thread):**
    ```python
    def safe_worker_per_thread(initial_data, update_chunks):
        # Each thread creates its own private engine
        engine = IncrementalNystromEngine(...)
        phi = engine.build(initial_data)
        for chunk in update_chunks:
            phi = engine.update(chunk, phi)
    ```

## 4. CUDA Stream and Device Handling

*   **Behavior:** The engine operates on the **current default CUDA stream** as managed by PyTorch. It does not create its own streams.
*   **Synchronization:** All operations within a single function call (e.g., `compute_rkhs_embedding`) are internally synchronized with the CUDA device before returning control to the Python interpreter. The functions are blocking from the host's perspective.
*   **Advanced Usage:** For advanced users integrating with custom CUDA streams, you can use PyTorch's `with torch.cuda.stream(s):` context manager to direct the engine's operations to a specific stream. The engine will respect this context.

# --- END OF FILE docs/threading_model.md ---