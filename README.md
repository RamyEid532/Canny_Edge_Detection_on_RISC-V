# Canny Edge Detection on RISC-V

[![View on GitHub](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=github)](https://github.com/RamyEid532/Canny_Edge_Detection_on_RISC-V)

A custom C++ implementation of the complete Canny Edge Detection pipeline optimized for the RISC-V architecture. 

## 🛠️ Pipeline Architecture

The image processing pipeline is divided into clear, modular phases:

* **Gaussian Blur (5x5):** Smooths the input image to reduce noise using integer-only convolution coefficients.

* **Sobel Gradient Expression:** Applies dual 3x3 derivative kernels to map directional gradients, utilizing a Structure of Arrays (SoA) layout.

* **Gradient Magnitude Evaluation:** Computes edge intensity using both the fast integer-only L1 norm and the mathematically absolute L2 norm.

* **Gradient Direction Quantization:** Categorizes angles into legal directions using integer cross-multiplication.

* **Non-Maximum Suppression (NMS):** Sharpens wide edge regions into thin, single-pixel lines.

## 🚀 Building and Running

### Host Vector Validation (Native)
You can execute the GoogleTest framework natively on the host machine to ensure basic architectural correctness.

### RISC-V Cross-Compilation
Use the build system to cross-compile the source code explicitly for the RISC-V vector architecture.

### Emulation on QEMU
Execute the pipeline on sample graphics inputs through the emulator.

## 📊 Profiling & Optimization Strategy

The implementation adheres strictly to **Amdahl's Law**.

### Core Optimization Highlights

* **Strip-Mining & Vector-Length Agnosticism:** Uses dynamic `vsetvl` configurations to safely loop through image lines.
* **Data Widening Mechanics:** Uses `vwmul` to scale 8-bit pixels into 32-bit registers, preventing overflow.
* **Fixed-Point Division Approximations:** Replaces the expensive Gaussian kernel divide-by-273 operation with a fast fixed-point calculation.

## 🎓 Acknowledgments & References

* **Project Director:** Dr. Omar Ahmed Nasr
* **Design Partner:** Claude (Anthropic AI)
