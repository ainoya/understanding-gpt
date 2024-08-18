/* This example uses the tinyshakespeare dataset which can be downloaded at:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

This is mostly a rust port of https://github.com/karpathy/minGPT

https://github.com/LaurentMazare/tch-rs/blob/a4e9362e4acbbde54ab9503ab9e37a10835e7547/examples/min-gpt/main.rs
*/

use anyhow::{bail, Result};
use tch::data::TextData;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, IndexOp, Kind, Tensor};

const LEARNING_RATE: f64 = 0.0003;
const BLOCK_SIZE: i64 = 128; // The maximum context length for predictions
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 4096; // Length of the generated sample text

#[derive(Debug, Copy, Clone)]
struct Config {
    vocab_size: i64,  // Size of the vocabulary
    n_embd: i64,      // Embedding dimension
    n_head: i64,      // Number of attention heads
    n_layer: i64,     // Number of transformer blocks/layers
    block_size: i64,  // Maximum context length
    attn_pdrop: f64,  // Dropout probability for attention weights
    resid_pdrop: f64, // Dropout probability for residual connections
    embd_pdrop: f64,  // Dropout probability for embeddings
}

// Weight decay only applies to the weight matrixes in the linear layers
const NO_WEIGHT_DECAY_GROUP: usize = 0;
const WEIGHT_DECAY_GROUP: usize = 1;

// Custom linear layer so that different groups can be used for weight
// and biases. This allows applying weight decay only to weights and not biases.
#[derive(Debug)]
struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

impl nn::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}

// Creates a linear layer with weight decay applied to the weights
fn linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> Linear {
    let wd = vs.set_group(WEIGHT_DECAY_GROUP);
    let no_wd = vs.set_group(NO_WEIGHT_DECAY_GROUP);
    Linear {
        ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02), // Weights initialized with random normal distribution
        bs: no_wd.zeros("bias", &[out_dim]),                   // Biases initialized to zero
    }
}

// Creates a linear layer without bias and weight decay applied to the weights
fn linear_no_bias(vs: nn::Path, in_dim: i64, out_dim: i64) -> Linear {
    let wd = vs.set_group(WEIGHT_DECAY_GROUP);
    let no_wd = vs.set_group(NO_WEIGHT_DECAY_GROUP);
    Linear {
        ws: wd.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
        bs: no_wd.zeros_no_train("bias", &[out_dim]), // Biases are not trainable
    }
}

// Implements causal self-attention, the core of the transformer architecture.
// "Causal" means that the attention mechanism only attends to past tokens,
// preventing information leakage from future tokens.
// Causal Self-Attention Example:
// Consider the following sequence: A B C D
// In causal self-attention, the token D can pay attention to A, B, and C,
// but A cannot pay attention to B, C, or D. This ensures that future information
// does not influence the past.
fn causal_self_attention(p: &nn::Path, cfg: Config) -> impl ModuleT {
    let key = linear(p / "key", cfg.n_embd, cfg.n_embd); // Linear transformation for keys
    let query = linear(p / "query", cfg.n_embd, cfg.n_embd); // Linear transformation for queries
    let value = linear(p / "value", cfg.n_embd, cfg.n_embd); // Linear transformation for values
    let proj = linear(p / "proj", cfg.n_embd, cfg.n_embd); // Linear transformation for output projection

    // Create a mask to prevent attending to future tokens
    let mask_init =
        Tensor::ones([cfg.block_size, cfg.block_size], (Kind::Float, p.device())).tril(0);
    let mask_init = mask_init.view([1, 1, cfg.block_size, cfg.block_size]);
    let mask = mask_init; // The mask is not trainable

    nn::func_t(move |xs, train| {
        let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
        let sizes = [sz_b, sz_t, cfg.n_head, sz_c / cfg.n_head];

        // Calculate keys, queries, and values
        let k = xs.apply(&key).view(sizes).transpose(1, 2);
        let q = xs.apply(&query).view(sizes).transpose(1, 2);
        let v = xs.apply(&value).view(sizes).transpose(1, 2);

        // Calculate attention weights
        let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));

        // Apply the mask to prevent attending to future tokens
        let att = att.masked_fill(&mask.i((.., .., ..sz_t, ..sz_t)).eq(0.), f64::NEG_INFINITY);

        // Apply softmax and dropout
        let att = att.softmax(-1, Kind::Float).dropout(cfg.attn_pdrop, train);

        // Calculate the weighted sum of values
        let ys = att
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([sz_b, sz_t, sz_c]);

        // Apply output projection and dropout
        ys.apply(&proj).dropout(cfg.resid_pdrop, train)
    })
}

// Implements a single transformer block, consisting of a self-attention layer,
// layer normalization, and a feedforward network.
fn block(p: &nn::Path, cfg: Config) -> impl ModuleT {
    let ln1 = nn::layer_norm(p / "ln1", vec![cfg.n_embd], Default::default()); // Layer normalization
    let ln2 = nn::layer_norm(p / "ln2", vec![cfg.n_embd], Default::default()); // Layer normalization
    let attn = causal_self_attention(p, cfg); // Self-attention layer
    let lin1 = linear(p / "lin1", cfg.n_embd, 4 * cfg.n_embd); // Linear transformation in feedforward network
    let lin2 = linear(p / "lin2", 4 * cfg.n_embd, cfg.n_embd); // Linear transformation in feedforward network

    nn::func_t(move |xs, train| {
        // Apply self-attention and residual connection
        let xs = xs + xs.apply(&ln1).apply_t(&attn, train);

        // Apply feedforward network, GELU activation, and residual connection
        let ys = xs
            .apply(&ln2)
            .apply(&lin1)
            .gelu("none")
            .apply(&lin2)
            .dropout(cfg.resid_pdrop, train);
        xs + ys
    })
}

// Implements the complete GPT model, consisting of an embedding layer,
// positional encoding, multiple transformer blocks, layer normalization,
// and a final linear layer for outputting logits.
fn gpt(p: nn::Path, cfg: Config) -> impl ModuleT {
    let p = &p.set_group(NO_WEIGHT_DECAY_GROUP);
    let tok_emb = nn::embedding(
        p / "tok_emb",
        cfg.vocab_size,
        cfg.n_embd,
        Default::default(),
    ); // Embedding layer for tokens
    let pos_emb = p.zeros("pos_emb", &[1, cfg.block_size, cfg.n_embd]); // Positional encoding
    let ln_f = nn::layer_norm(p / "ln_f", vec![cfg.n_embd], Default::default()); // Layer normalization
    let head = linear_no_bias(p / "head", cfg.n_embd, cfg.vocab_size); // Final linear layer

    // Create multiple transformer blocks
    let mut blocks = nn::seq_t();
    for block_idx in 0..cfg.n_layer {
        blocks = blocks.add(block(&(p / block_idx), cfg));
    }

    nn::func_t(move |xs, train| {
        let (_sz_b, sz_t) = xs.size2().unwrap();

        // Apply embedding and positional encoding
        let tok_emb = xs.apply(&tok_emb);
        let pos_emb = pos_emb.i((.., ..sz_t, ..));

        // Apply dropout, transformer blocks, layer normalization, and final linear layer
        (tok_emb + pos_emb)
            .dropout(cfg.embd_pdrop, train)
            .apply_t(&blocks, train)
            .apply(&ln_f)
            .apply(&head)
    })
}

/// Generates some sample string using the GPT model.
fn sample(data: &TextData, gpt: &impl ModuleT, input: Tensor) -> String {
    let mut input = input;
    let mut result = String::new();
    for _index in 0..SAMPLING_LEN {
        // Get the logits for the next token
        let logits = input.apply_t(gpt, false).i((0, -1, ..));

        // Sample the next token using a multinomial distribution
        let sampled_y = logits.softmax(-1, Kind::Float).multinomial(1, true);

        // Convert the sampled token to a character
        let last_label = i64::try_from(&sampled_y).unwrap();
        result.push(data.label_to_char(last_label));

        // Update the input with the sampled token
        input = Tensor::cat(&[input, sampled_y.view([1, 1])], 1).narrow(1, 1, BLOCK_SIZE);
    }
    result
}

pub fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    println!("Running on device {device:?}");
    let mut vs = nn::VarStore::new(device);
    let data = TextData::new("data/input.txt")?;
    let labels = data.labels();
    println!("Dataset loaded, {labels} labels.");
    let cfg = Config {
        vocab_size: labels,
        n_embd: 512,
        n_head: 8,
        n_layer: 8,
        block_size: BLOCK_SIZE,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
    };
    let gpt = gpt(vs.root() / "gpt", cfg);
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: main (train|predict weights.ot seqstart)")
    }
    match args[1].as_str() {
        "train" => {
            // Create an AdamW optimizer
            let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE)?;
            opt.set_weight_decay_group(NO_WEIGHT_DECAY_GROUP, 0.0);
            opt.set_weight_decay_group(WEIGHT_DECAY_GROUP, 0.1);

            let mut idx = 0;
            for epoch in 1..(1 + EPOCHS) {
                println!("Epoch: {}", epoch);
                let mut sum_loss = 0.;
                let mut cnt_loss = 0.;
                // Iterate over the data in batches
                for batch in data.iter_shuffle(BLOCK_SIZE + 1, BATCH_SIZE) {
                    println!("Batch: {idx}");
                    // Prepare the input and target tensors
                    let xs = batch
                        .narrow(1, 0, BLOCK_SIZE)
                        .to_kind(Kind::Int64)
                        .to_device(device);
                    let ys = batch
                        .narrow(1, 1, BLOCK_SIZE)
                        .to_kind(Kind::Int64)
                        .to_device(device);

                    // Calculate the logits and loss
                    let logits = xs.apply_t(&gpt, true);
                    let loss = logits
                        .view([BATCH_SIZE * BLOCK_SIZE, labels])
                        .cross_entropy_for_logits(&ys.view([BATCH_SIZE * BLOCK_SIZE]));

                    // Perform backpropagation and update the model parameters
                    opt.backward_step_clip(&loss, 0.5);

                    // Accumulate the loss for logging
                    sum_loss += f64::try_from(loss)?;
                    cnt_loss += 1.0;
                    idx += 1;

                    // Print the loss and generate a sample every 1000 iterations
                    if idx % 1000 == 0 {
                        println!("Epoch: {}   loss: {:5.3}", epoch, sum_loss / cnt_loss);
                        let input = Tensor::zeros([1, BLOCK_SIZE], (Kind::Int64, device));
                        println!("Sample: {}", sample(&data, &gpt, input));

                        // Save the model weights
                        if let Err(err) = vs.save(format!("gpt{idx}.ot")) {
                            println!("error while saving {err}");
                        } else {
                            println!("model saved to gpt{idx}.ot");
                        }

                        // Reset the loss accumulators
                        sum_loss = 0.;
                        cnt_loss = 0.;
                    }
                }
            }
            println!("Training completed.");
        }
        "predict" => {
            // Load the model weights
            vs.load(args[2].as_str())?;

            // Prepare the input tensor based on the provided starting sequence
            let seqstart = args[3].as_str();
            let input = Tensor::zeros([1, BLOCK_SIZE], (Kind::Int64, device));
            for (idx, c) in seqstart.chars().rev().enumerate() {
                let idx = idx as i64;
                if idx >= BLOCK_SIZE {
                    break;
                }
                let _filled = input
                    .i((0, BLOCK_SIZE - 1 - idx))
                    .fill_(data.char_to_label(c)? as i64);
            }

            // Generate a sample using the loaded model
            println!("Sample: {}", sample(&data, &gpt, input));
        }
        _ => bail!("usage: main (train|predict weights.ot seqstart)"),
    };

    Ok(())
}
