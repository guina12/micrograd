# Minimal Automatic Differentiation Framework

## Overview

This project implements a lightweight automatic differentiation framework, inspired by Andrej Karpathy's Micrograd. The framework demonstrates the core principles of automatic differentiation, showcasing how computational graphs enable efficient gradient computation in deep learning.

## Key Features

- **Computational Graph Implementation**: Build and traverse computational graphs
- **Automatic Gradient Calculation**: Automate partial derivative computation
- **Gradient Descent Support**: Enable optimization algorithms
- **Minimal and Educational Design**: Focus on core principles of automatic differentiation

## How Automatic Differentiation Works

Automatic differentiation simplifies the process of computing gradients by:
- Constructing a computational graph of operations
- Tracking dependencies between variables
- Efficiently computing gradients during the backward pass

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy (optional, depending on implementation)

### Installation
```bash
git clone https://github.com/yourusername/micrograd-framework.git
cd micrograd-framework
```



## Example Usage

```python
from micrograd.value import Value

# Create a simple computational graph
x = Value(2.0)
y = Value(3.0)
z = x * y + Value(2)
z.backward()  # Compute gradients

print(x.grad)  # Automatically computed gradient
print(y.grad)  # Automatically computed gradient
```

## Computacional Graph

![image](https://github.com/user-attachments/assets/ee79a41b-8862-45b4-9f40-24dae41cb9e5)

## References

- [Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd)
- Deep Learning : Backpropagation 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Future Improvements

- Add more activation functions
- Implement additional optimization algorithms
- Expand test coverage
- Create more comprehensive documentation

