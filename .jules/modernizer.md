## 2024-05-18 - Replace np.dot with @ operator
**Learning:** `np.dot` is used in a few places in the codebase instead of the modern, native Python matrix multiplication operator `@` (introduced in PEP 465, Python 3.5).
**Action:** Replace `np.dot(A, B)` with `A @ B` to improve readability and embrace modern Python syntax. This reduces verbosity while maintaining the exact same functionality, fitting perfectly with the Modernizer philosophy.
