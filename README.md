# ⚡ Phaadu

> **Phaadu** is a modern systems programming language built for high-performance mathematical computing, deep learning primitives (autodiff, self & cross attention, matrix algebra), and integrated graph/data visualization. Built on **Rust** and compiled via **LLVM**, Phaadu aims for **C/C++ level execution speed**, providing a lightning-fast alternative to traditional Python-based ML frameworks (PyTorch, NumPy, Seaborn, MATLAB).

---

## 🏗️ Project Architecture: The 8 Core Components

To achieve maximum performance, modularity, and compiler scalability, **Phaadu** is structured into 8 core modules:

1. **`phaadu-lexer` — Lexer & Tokenizer**
   - Scans Phaadu source code and generates a stream of strongly-typed tokens.
   - Recognizes matrix/tensor syntax, mathematical symbols (`@` for matrix multiplication, `∇` for gradients), strings, numbers, and language keywords (e.g., `likh`, `mat`, `tensor`, `attn`, `plot`).

2. **`phaadu-parser` — Parser & Abstract Syntax Tree (AST)**
   - Converts token streams into a structured, hierarchical Abstract Syntax Tree (AST).
   - Enforces mathematical operator precedence, syntax rules for neural network modules, attention blocks, conditional logic, and inline visualization primitives.

3. **`phaadu-analyzer` — Semantic Analyzer & Tensor Shape Engine**
   - Performs compile-time static type checking, memory safety validation, and shape inference for N-dimensional tensors.
   - Catches dimension mismatches (e.g., matrix multiplication shape errors $(m \times n) \times (p \times q)$) at compile time before code execution.

4. **`phaadu-autodiff` — Automatic Differentiation & Backprop Engine**
   - Implements dynamic and static tape-based reverse-mode and forward-mode automatic differentiation.
   - Automatically constructs computation graphs to derive exact backward pass equations ($\frac{\partial L}{\partial W}$) with zero runtime runtime-reflection overhead.

5. **`phaadu-kernels` — High-Performance Matrix & Attention Engine**
   - Features low-level assembly and Rust kernels for GEMM (General Matrix Multiplication), SIMD-vectorized operations, Self-Attention ($\text{Softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$), Cross-Attention, and FlashAttention routines.
   - Supports multi-threaded CPU execution and hardware acceleration backends (CUDA / Metal / ROCm).

6. **`phaadu-codegen` — LLVM IR & Native Code Generator**
   - Translates AST nodes and autodiff graphs into LLVM Intermediate Representation (LLVM IR).
   - Invokes LLVM optimization passes (vectorization, loop unrolling, instruction selection) to compile directly into native standalone binaries (`.exe` on Windows, ELF on Linux).

7. **`phaadu-runtime` — Zero-Cost Memory & Execution Manager**
   - Provides zero-overhead memory management via stack allocations, arena allocators, and deterministic tensor lifetimes (no runtime garbage collector).
   - Manages asynchronous compute dispatch, standard library I/O (`likh`), and OS-level threads.

8. **`phaadu-viz` — Visualization & Plotting Suite**
   - Native MATLAB & Seaborn-style plotting library integrated directly into the language.
   - Enables native rendering of loss charts, matrix heatmaps, activation distributions, scatter plots, and neural topology graphs without external Python dependencies.

---

## 💡 Syntax & Feature Highlights

### 1. Basic Syntax & Printing
```phaadu
// Phaadu standard output
likh("Hello from Phaadu Compiler!");

let x: f64 = 42.0;
likh("Value of x: " + x);
```

### 2. Matrix & Tensor Operations (C/C++ Speed)
```phaadu
// Declare matrices natively with shape checking
let A: Matrix[2, 3] = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
];

let B: Matrix[3, 2] = [
    [7.0, 8.0],
    [9.0, 1.0],
    [2.0, 3.0]
];

// Matrix multiplication using '@' operator (compiled to SIMD GEMM kernels)
let C = A @ B; 
likh("Resulting Matrix C [2, 2]:");
likh(C);
```

### 3. Self & Cross Attention Primitives
```phaadu
// Native Self-Attention Block
let Q: Tensor[1, 8, 64] = randn();
let K: Tensor[1, 8, 64] = randn();
let V: Tensor[1, 8, 64] = randn();

// Computes Softmax(Q @ K.T / sqrt(d_k)) @ V
let attention_out = attn::self_attention(Q, K, V);

// Cross-Attention between encoder & decoder states
let encoder_out: Tensor[1, 16, 64] = randn();
let cross_attn_out = attn::cross_attention(Q, encoder_out, encoder_out);
```

### 4. Backpropagation & Automatic Differentiation
```phaadu
// Define weights and input
let W = tensor::randn([64, 32], requires_grad=true);
let X = tensor::randn([1, 64]);
let Y_target = tensor::ones([1, 32]);

// Forward Pass
let Y_pred = X @ W;
let loss = mean((Y_pred - Y_target) ^ 2);

// Backward Pass (Generates gradients via compile-time autodiff engine)
loss.backward();

likh("Gradients for W:");
likh(W.grad);
```

### 5. Integrated Graph & Data Visualization (MATLAB / Seaborn Style)
```phaadu
// Plot loss curve over training epochs
let epochs = range(1, 100);
let loss_history = load_tensor("loss.dat");

viz::figure("Training Progress");
viz::plot(epochs, loss_history, label="Loss", color="cyan");
viz::xlabel("Epochs");
viz::ylabel("MSE Loss");
viz::show();

// Render Matrix Heatmap
viz::heatmap(C, cmap="viridis", annotate=true);
```

---

## ⚙️ Compilation Pipeline

```
 Source Code (.pha)
        │
        ▼
   [ 1. Lexer ] ─────► Token Stream
        │
        ▼
  [ 2. Parser ] ─────► Abstract Syntax Tree (AST)
        │
        ▼
 [ 3. Analyzer ] ────► Type & Tensor Shape Verification
        │
        ▼
 [ 4. Autodiff ] ────► Forward & Backward Computation Graph
        │
        ▼
 [ 5. Kernels ] ─────► SIMD / GPU Kernel Ingestion
        │
        ▼
 [ 6. Codegen ] ─────► LLVM IR Generation
        │
        ▼
  [ LLVM Backend ] ──► Code Optimization & Machine Pass
        │
        ▼
  Standalone Executable (.exe / binary)
```

---

## 📁 File Extensions

Phaadu source files use the following officially recognized extensions:
- **`.pha`** *(Recommended Standard)* — e.g., `main.pha`
- **`.phaadu`** *(Full Extension)* — e.g., `model.phaadu`
- **`.ph`** *(Short Extension)* — e.g., `script.ph`

---

## 🚀 Quick Start & Development

### Prerequisites
- [Rust](https://www.rust-lang.org/) (MSRV 1.75+)
- [LLVM](https://llvm.org/) (v16 or higher)

### Building & Running the Compiler
```bash
# Clone the repository
git clone https://github.com/ankitghoshthecreator/Phaadu.git
cd Phaadu

# Build using Cargo
cargo build --release

# Run the Phaadu compiler on a source file (.pha / .phaadu / .ph)
cargo run -- main.pha
```

---

## 📌 Project Status & Roadmap

- [x] **Part 1: `phaadu-lexer` — Lexer & Tokenizer** (Full token set, line/col tracking, matrix `@` / `^` / `'` ops, comments, escape sequences)
- [x] **Part 2: `phaadu-parser` — Parser & Abstract Syntax Tree (AST)** (Pratt expression precedence, matrix algebra, autodiff statements, attention blocks, control flow, functions)
- [x] Initial Runtime prototype (`src/runtime.rs`)
- [ ] Part 3: `phaadu-analyzer` — Semantic Analyzer & Tensor Shape Engine
- [ ] Part 4: `phaadu-autodiff` — Reverse-Mode Autodiff Engine
- [ ] Part 5: `phaadu-kernels` — Self & Cross Attention Optimized Kernels
- [ ] Part 6: `phaadu-codegen` — LLVM IR Codegen Engine
- [ ] Part 7: `phaadu-runtime` — Zero-Cost Runtime & Memory Manager
- [ ] Part 8: `phaadu-viz` — Integrated Plotting/Visualization Engine

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
