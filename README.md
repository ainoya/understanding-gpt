# Transformer (GPT Style) Implementation for Understanding LLMs

This repository is created for understanding Transformer models. It is based on the implementation of [examples/min-gpt](https://github.com/LaurentMazare/tch-rs/blob/main/examples/min-gpt/README.md) from tch-rs. The original code has been modified with added comments for better understanding.

> [!NOTE]
> This repository contains code that is still under development and may contain wrong information.

## How to Run

```bash
export TORCH_CUDA_VERSION=cu121 # If using CUDA.
cargo run -- train # Training
cargo run -- predict gpt1000.ot "Warwick: What is your name?" # Run inference
```

- Training for 1000 steps takes about 10 minutes using `g4dn.xlarge`.

## Architecture Overview

### Overview

```mermaid
graph LR
    subgraph Input
        A[Input Text] --> B(Tokenization)
    end

    subgraph Embedding
        B --> C{Token Embeddings}
        D{Positional Embeddings} --> E(+)
        C --> E
    end

    subgraph Transformer Blocks
        E --> F{Block 0}
        F --> G{Block 1}
        G --> H{Block ...}
        H --> I{Block N-1}
    end
```

### Transformer Block

```mermaid
graph LR
    subgraph Block
        subgraph Layer Normalization 1
            J[Input] --> K(Layer Norm)
        end
        
        subgraph Multi-Head Attention
            K --> L(Multi-Head Attention)
        end

        subgraph Add & Norm
            L --> M(+)
            J --> M
            M --> N(Layer Norm)
        end

        subgraph Feed Forward Network
            N --> O(Linear)
            O --> P(GeLU)
            P --> Q(Linear)
        end

        subgraph Add & Output
            Q --> R(+)
            N --> R
            R --> S[Output]
        end
    end
```

### Attention Mechanism

```mermaid
graph LR
    subgraph Input
        A["Input (Batch Size, Sequence Length, Embedding Dim)"] --> B(Linear - Key)
        A --> C(Linear - Query)
        A --> D(Linear - Value)
    end

    subgraph Split into Heads
        B --> E{"Split Heads (Key)"}
        C --> F{"Split Heads (Query)"}
        D --> G{"Split Heads (Value)"}
    end

    subgraph "Scaled Dot-Product Attention (Per Head)"
        F --> H(Matmul)
        E --> H
        H --> I["/ Sqrt(d_k)"]
        I --> J(Causal Mask)
        J --> K(Softmax)
        K --> L(Matmul)
        G --> L
    end

    subgraph Concatenate Heads
        L --> M{Concatenate Heads}
    end

    subgraph Output
        M --> N(Linear - Output)
        N --> O["Output (Batch Size, Sequence Length, Embedding Dim)"]
    end
```

## References

- [LaurentMazare/tch-rs: Rust bindings for the C++ api of PyTorch.](https://github.com/LaurentMazare/tch-rs)
- [karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training](https://github.com/karpathy/minGPT)
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [A Comprehensive Guide to Building a Transformer Model with PyTorch | DataCamp](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
