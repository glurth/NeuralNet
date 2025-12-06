# NeuralNet

NeuralNet is a C# library for building and training artificial neural networks with support for both CPU and GPU-accelerated computation in Unity via compute shaders. The system provides flexible architectures for experimenting with feedforward and evolving neural networks, and includes mechanisms for efficient backpropagation, mutation, serialization, and visualization.

## Features

- **NetLayer Architecture:** Modular layers where each neuron uses the same activation function; supports common activations (ReLU, Sigmoid, Tanh, None).
- **GPU-Accelerated Layers:** `ComputeShaderLayer` enables fast neural processing and backpropagation on the GPU using Unity's compute shaders.
- **Flexible Neural Net Structure:** Easily build and mutate deep networks, clone structures for evolutionary algorithms, and serialize/deserialize networks.
- **ConnectionNet (Experimental):** Supports more flexible, graph-like neural architectures (beyond simple feedforward topology).
- **Async and UniTask Support:** All compute and backpropagation operations can be run asynchronously, making it suitable for real-time or large-scale simulation.
- **Serialization:** Save and load network state in binary or JSON formats.

## Getting Started

### Prerequisites

- Unity3D (recommended 2020 or newer)
- [Cysharp UniTask](https://github.com/Cysharp/UniTask) for async support
- Basic understanding of Unity Compute Shaders if using GPU acceleration

### Installation

Install this package in your Unity project using the Package Manager:

   - Open the Package Manager window (Packages > Manage Packages).
   - Click on the + button in the top left corner and select Add package from git URL.
   - Paste the following URL into the address field and click Install: https://github.com/glurth/NeuralNet.git


### Usage

#### Creating a Simple Neural Network

```csharp
using EyE.NNET;

// Define a network with 4 inputs and 2 outputs
NeuralNet net = new NeuralNet(4, 2);
// Populate the net with random layers (hidden layers and neurons auto-chosen)
net.PopulateLayersRandomly(ActivationFunction.ReLU);

// Compute output asynchronously
float[] inputs = new float[4] { 1f, 0.5f, -0.3f, 0.8f };
var output = await net.Think(inputs);

// Training: Backpropagate errors
float[] errors = new float[2] { 0.1f, -0.05f };
await net.Backpropagate(errors, learningRate: 0.01f);
```

#### Using Compute Shader Accelerated Nets

```csharp
using EyE.NNET;

ComputeShader myShader = Resources.Load<ComputeShader>("MyNeuralNetShader");
NeuralNetComputeShader gpuNet = new NeuralNetComputeShader(4, 2, myShader, computeShaderIsSingleThreaded: true);

// Populate with layers that use the GPU
gpuNet.PopulateLayersRandomly(ActivationFunction.ReLU);

// Run inference using the GPU
float[] input = new float[] { 1f, 0.5f, 0f, 1f };
float[] output = await gpuNet.GPUThink(input);

// GPU backpropagation
float[] errors = new float[2] { 0.2f, -0.1f };
await gpuNet.GPUBackpropagate(errors, learningRate: 0.01f);

// Serialize to disk
gpuNet.SaveBinary("Assets/NetworkData.bin");
```

#### Mutating Networks (Evolutionary Algorithms)

```csharp
NeuralNet mutatedNet = originalNet.CloneAndMutateLayers(
    addLayerMutationPerLayerChance: 0.05f,
    activationFunctionChangeChance: 0.02f,
    biasMutationChance: 0.05f,
    biasMutationAmount: 0.1f,
    numNeuronsMutationChance: 0.05f,
    numNeuronsMutationAmount: 0.2f,
    weightsMutationChance: 0.03f,
    weightsMutationAmount: 0.15f
);
```

## API Overview

- `NeuralNet`: Classic feedforward network with list of `NetLayer`.
- `NetLayer`: Contains neuron count, weights, biases, activation, supports CPU compute/backprop.
- `NeuralNetComputeShader`: Drop-in replacement to use GPU layers for performance.
- `ComputeShaderLayer`: Implements all neural logic (forward and backward pass) via Unity ComputeShaders.
- `ActivationFunction`: Enum for layer activations and extension methods for activation/derivative.
- Serialization: `.SaveBinary`, `.LoadBinary`, `.SaveJson`, `.LoadJson`.

## Example Compute Shader Kernel Names 

If you want to write your own compute shaders, match these kernel names:

- `ComputeLayerNone`
- `ComputeLayerReLU`
- `ComputeLayerSigmoid`
- `ComputeLayerTanh01`
- `(Backprop and multi-pass variants also required; see ComputeShaderLayer.cs for details)`

## Contributing

Contributions, issues, and feature requests are welcome! Please submit them via the GitHub repository. Note: Due to licensing, contributions can only be included with explicit written permission from the copyright holder.

## License

This package is licensed under the EyE Dual-Licensing Agreement.

It provides free, perpetual use for indie developers and non-commercial projects whose teams had Total Gross Receipts under $100,000 USD in the previous fiscal year.

Organizations exceeding this threshold must obtain a Perpetual Commercial License (PCL) for each named commercial project.

Please review the full terms in [LICENSE.md](LICENSE.md) before commercial use.