<!---
Compile to PDF
> pandoc report.md -o report.pdf -Vcolorlinks=true
Compile to .tex
> pandoc report.md -o report.tex
-->
---
title: Alternative Implementation of Hotspot3D using CUDA Matrix Libraries
author: [Aritra Bhakat, Ludwig Kristoffersson, Saga Palmér]
date: \today
documentclass: scrartcl
---

The repo containing our implementation can be found on GitHub: [hotspot3d-matrix](https://github.com/arrebarritra/hotspot3d-matrix)


<!---
A summary of each group member's contributions
-->
## Group Member Contributions

### Aritra Bhakat


### Ludwig Kristoffersson


### Saga Palmér

<!---
Introduction. Provide some background on the performed problem
-->
# Introduction

Hotspot3d tries to solve the heat equation, given by the following PDE:

\begin{align}
    \frac{\partial T}{\partial t} = \alpha \nabla^2 T(\mathbf{x}) + \beta P(\mathbf{x}) + \alpha \left[ T_{\text{amb}} - T(\mathbf{x}) \right] \\
    = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2} \right) + \beta P(\mathbf{x}) + \alpha \left[ T_{\text{amb}} - T(\mathbf{x}) \right]
\end{align}

where $T$ is the temperature field, $P$ is the power field, $T_{\text{amb}}$ is the ambient temperature, and $\mathbf{\alpha}, \beta$ are arbitrary constants which have to do with chip properties.

The PDE is discretised using forward difference for the time step and second order central differences for the Laplace operator:

\begin{align}
    T^{t+1}_{ijk} =\\
    T^t_{ijk}
    + \left[ c_w T^t_{i-1,jk} - (c_w + c_e) T^t_{ijk} + c_e T^t_{i+1,jk} \right]\\
    + \left[ c_t T^t_{i,j-1,k} - (c_t + c_s) T^t_{ijk} + c_s T^t_{i,j+1,k} \right]\\
    + \left[ c_b T^t_{ij,k-1} - (c_b + c_t) T^t_{ijk} + c_t T^t_{ij,k+1} \right]\\
    + \text{sdc} \cdot P_{ijk} \\
    + c_t [T^t_{\text{amb}} - T^t_{ijk}] \\
    = c_c T^t_{ijk} \\
    + c_w T^t_{i-1,jk} + c_e T^t_{i+1,jk} \\
    + c_t T^t_{i,j-1,k} + c_s T^t_{i,j+1,k} \\
    + c_b T^t_{ij,k-1} + c_t T^t_{ij,k+1} \\
    + \text{sdc} \cdot P_{ijk} + c_t T^t_{\text{amb}}
\end{align}

Here, the constants $c_x$ are combinations of the constant $\alpha_x$, along with the timestep $dt$, size of the chip in the given dimension and amount rows/cols/layers for the given dimension. The letters $w,e,n,s,b,t$ stand for the compass dimensions and bottom and top, each pair of irections corresponding the $xyz$ axes. The values are symmetrical, meaning $c_w = c_e \ldots$ and so on. To simplify the computation, the $T^t_{ijk}$ terms are gathered into the constant $c_c = 1 - (2 \cdot c_w + 2 \cdot c_n + 3 \cdot c_t)$. The constant $\text{sdc}$ equals $dt / \text{capacitance}$. The temperature $T$ and the power $P$ both have dimensions $n_x \times n_y \times n_z$.

<!---
Methodology. Explain all the different steps in your implementation 
-->
# Methodology

We observed that the discretised PDE corresponds well to sparse matrix multiplication. Our approach is to use a sparse matrix for each dimension to calculate the second order central difference part of the discretised equation. The rest of the equation can be calculated as trivial (vector) addition.

Our data has the $n_x \times n_y \times n_z$. Our differentiation matrices are sparse square matrices with dimensions $n_x$, $n_y$ and $n_z$ respectively:

\begin{align}
    \partial X =
    \begin{bmatrix}
    c_w + c_c   &c_e    &&&&&&0 \\
    c_w     &c_c   &c_e \\
    &c_w    &c_c    &c_e \\
    &&&\ldots \\
    &&&&&c_w    &c_c   &c_e \\
    0&&&&&&c_w   &c_c + c_e
    \end{bmatrix}
\end{align}

\begin{align}
    \partial Y =
    \begin{bmatrix}
    c_n   &c_s    &&&&&&0 \\
    c_n     &0   &c_s \\
    &c_n    &0    &c_s \\
    &&&\ldots \\
    &&&&&c_n    &0   &c_s \\
    0&&&&&&c_n   &c_s
    \end{bmatrix}
\end{align}

\begin{align}
    \partial Y =
    \begin{bmatrix}
    c_b   &c_t    &&&&&&0 \\
    c_b     &0   &c_t \\
    &c_b    &0    &c_t \\
    &&&\ldots \\
    &&&&&c_b    &0   &c_t \\
    0&&&&&&c_b   &c_t
    \end{bmatrix}
\end{align}

Note that $\partial Y$ and $\partial Z$ are missing $c_c$ in the diagonal as the term is already accounted for in $\partial X$. These matrices operate on the 3D temperature data.

Since we can only work with matrices and not directly with 3D tensors within the cuSPARSE library, we need to configure the matrix layouts for the different computations to give us the correct result.

We apply $\partial X$ and $\partial Y$ on $n_z$ batches of $n_x \times n_y$ matrices.

\begin{align}
    T_{XY} =
    \begin{bmatrix}
    T_{111} &\ldots &T_{1,n_x,1} \\
    &\ldots \\
    T_{n_y,11} &\ldots & T_{n_y,n_x,1}
    \end{bmatrix}
    \quad
    \ldots
    \quad
    \begin{bmatrix}
    T_{11,n_z} &\ldots &T_{1,n_x,n_z} \\
    &\ldots \\
    T_{n_y,1,n_z} &\ldots & T_{n_y,n_x,n_z}
    \end{bmatrix}
\end{align}

We call this layout $T_{XY}$.

When we apply $\partial X$, we need to take the transpose of $T_{XY}$ so that the $x$-dimension is lined up with the column. We then need to transpose the result so the $x$ dimension is lined up with the row again. We perform the following calculation:

$$
\left( \partial X \cdot T^{\intercal}_{XY} \right)^{\intercal} =  T_{XY} \cdot\partial X^{\intercal}
$$

For $\partial Y$ the calculation is straightforward:

$$
\partial Y \cdot T_{XY}
$$

For $\partial Z$ we need to change the layout of the $T$. We flatten the data into one big $n_z \times (n_x \cdot n_y)$ matrix, so that values adjacent on the $z$-axis are lined up in one column:

\begin{align}
    \begin{bmatrix}
    T_{111} &\ldots &T_{n_y,n_x,1} \\
    T_{112} &\ldots &T_{n_y,n_x,2} \\
    &\ldots \\
    T_{11,n_z} &\ldots & T_{n_y,n_x,n_z}
    \end{bmatrix}
\end{align}

We call this layout $T_Z$. The calculation we perform now is also straightforward:

$$
\partial Z \cdot T_{Z}
$$

Putting this together, we can rewrite the discretised equation as:

$$
T^{t+1} = T^{t}_{XY} \cdot \partial X^{\intercal} + \partial Y \cdot T^{t}_{XY} + \partial Z \cdot T^t_Z + \text{sdc} \cdot P + c_t T_{\text{amb}} I
$$

<!---

TODO: Write how we implement this in practice
- Ping pong buffers
- Matrix layouts
- Using column major format to rewrite T_row * dX_row^T -> dX_col * T_col (see cusparseSpMM docs)
- cublasSaxpy for adding sdc * P
- kernel for adding ct * T_amb
- Write about implementation oriented things that might affect performance, ex. unified memory, synchronisation, pinned memory etc.

-->

<!---
Experimental Setup. Describe the GPU platform (both software and hardware) you used for your project and the tools you used.
-->
# Experimental Setup


<!---
Results. Present the validation tests, together with performance/profiling results.
-->
# Results


<!---
Discussion and Conclusion. Discuss the performance results critically. Describe the challenges and limitations of your work and future works if you are given more time. If optimizations are included, discuss the optimization and their performance result.
-->
# Discussion


<!---
References. Provide a list of articles and books you consulted for preparing the project.
-->
# References

