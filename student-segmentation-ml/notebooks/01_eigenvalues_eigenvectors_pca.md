# Eigenvalues, Eigenvectors, and Their Role in PCA

## 1. Introduction

Eigenvalues and eigenvectors are central ideas in linear algebra. They help us understand how a matrix transforms data. In machine learning and data science, they are especially important in **Principal Component Analysis (PCA)**, which is widely used for dimensionality reduction.

## 2. Definition

For a square matrix `A`, a non-zero vector `v` is called an **eigenvector** if applying `A` to `v` changes only its magnitude and not its direction.

Mathematically:

```text
Av = λv
```

Where:

- `A` is a square matrix
- `v` is the eigenvector
- `λ` is the eigenvalue

The eigenvalue tells us how much the eigenvector is stretched or compressed.

## 3. Geometric Meaning

Most matrix transformations can rotate, stretch, compress, or reflect vectors.

Eigenvectors are special because:

- their direction stays the same after the transformation
- only their length changes

The eigenvalue explains what happens:

- `λ > 1`: vector stretches
- `0 < λ < 1`: vector shrinks
- `λ < 0`: vector reverses direction and scales
- `λ = 0`: vector collapses to the origin

## 4. How to Find Eigenvalues and Eigenvectors

### Step 1: Find Eigenvalues

Solve the characteristic equation:

```text
det(A - λI) = 0
```

Where:

- `det` means determinant
- `I` is the identity matrix

### Step 2: Find Eigenvectors

For each eigenvalue `λ`, solve:

```text
(A - λI)v = 0
```

## 5. Example 1

Let:

```text
A = [2  0
     0  3]
```

### Eigenvalues

```text
det(A - λI) = (2 - λ)(3 - λ) = 0
```

So:

- `λ1 = 2`
- `λ2 = 3`

### Eigenvectors

For `λ = 2`:

```text
[0  0
 0  1] v = 0
```

An eigenvector is:

```text
v1 = [1, 0]
```

For `λ = 3`:

```text
[1  0
 0  0] v = 0
```

An eigenvector is:

```text
v2 = [0, 1]
```

### Interpretation

- vectors along the x-axis are scaled by 2
- vectors along the y-axis are scaled by 3

## 6. Example 2

Let:

```text
A = [4  1
     2  3]
```

Characteristic equation:

```text
det(A - λI) = (4 - λ)(3 - λ) - 2 = λ^2 - 7λ + 10 = 0
```

So:

- `λ1 = 5`
- `λ2 = 2`

For `λ = 5`, solve:

```text
(A - 5I)v = [-1  1
              2 -2]v = 0
```

This gives:

```text
v1 = [1, 1]
```

For `λ = 2`, solve:

```text
(A - 2I)v = [2  1
             2  1]v = 0
```

This gives one valid eigenvector:

```text
v2 = [1, -2]
```

## 7. Important Properties

- An `n x n` matrix has `n` eigenvalues counting multiplicity.
- Eigenvectors corresponding to different eigenvalues are linearly independent.
- The sum of eigenvalues equals the trace of the matrix.
- The product of eigenvalues equals the determinant of the matrix.
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors.

The last point is very important because covariance matrices in PCA are symmetric.

## 8. Real-World Uses

Eigenvalues and eigenvectors appear in many applications:

- PCA in machine learning
- PageRank in search engines
- face recognition in computer vision
- vibration analysis in engineering
- portfolio risk analysis in finance
- signal denoising in communication systems

## 9. Role in PCA

### What PCA Does

Principal Component Analysis reduces the number of features in a dataset while preserving as much information as possible.

The information PCA tries to preserve is **variance**.

### PCA Workflow

#### Step 1: Standardize the data

Features may be measured on different scales, so we standardize them first.

#### Step 2: Compute the covariance matrix

The covariance matrix shows how features vary together:

```text
C = (1 / n) X^T X
```

for standardized data `X`.

#### Step 3: Compute eigenvalues and eigenvectors of the covariance matrix

- **Eigenvectors** give the directions of the new feature axes.
- **Eigenvalues** tell how much variance lies along each direction.

#### Step 4: Sort by eigenvalue

The eigenvector with the largest eigenvalue becomes the **first principal component**.

The eigenvector with the second-largest eigenvalue becomes the **second principal component**, and so on.

#### Step 5: Keep the top components

We select the first `k` eigenvectors and project the data onto them:

```text
Z = XW
```

Where:

- `X` is the standardized data
- `W` is the matrix of selected eigenvectors
- `Z` is the transformed lower-dimensional data

## 10. Intuition of PCA Using Eigenvectors

Suppose we have student data with:

- age
- number of friends
- sports interest scores
- music interest scores

Some features may be correlated. For example:

- students with more friends may also be more active socially
- related interests may move together

PCA finds the directions where the dataset spreads out the most.

These best directions are the eigenvectors of the covariance matrix.

The amount of spread along each direction is measured by the corresponding eigenvalue.

So in PCA:

- eigenvectors define the new axes
- eigenvalues rank the importance of those axes

## 11. Why Eigenvalues Matter in PCA

Large eigenvalue:

- more variance captured
- more information retained
- more important principal component

Small eigenvalue:

- less variance captured
- less useful component
- often removable with limited information loss

This is why PCA helps with:

- dimensionality reduction
- noise removal
- faster model training
- visualization in 2D or 3D

## 12. Simple PCA Example

Imagine a dataset with:

- height
- weight
- age

If height and weight are strongly correlated, PCA may combine them into one component. Instead of keeping both original variables, we can keep a single principal component that captures most of their shared variation.

That means:

- fewer dimensions
- less redundancy
- easier visualization

## 13. Advantages of PCA

- reduces dimensionality
- removes multicollinearity
- speeds up algorithms
- can improve downstream model performance
- useful for visualization

## 14. Limitations of PCA

- principal components are less interpretable than original variables
- PCA assumes linear relationships
- results are sensitive to scaling
- variance is not always equal to importance for every business problem

## 15. Conclusion

Eigenvalues and eigenvectors explain how matrices transform vectors. They identify special directions that remain unchanged except for scaling. In PCA, these ideas become extremely useful:

- eigenvectors define principal component directions
- eigenvalues measure the variance captured by each component

Because of this, eigenvalues and eigenvectors form the mathematical foundation of PCA and many other machine learning techniques.
