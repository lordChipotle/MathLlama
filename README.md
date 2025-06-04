# MathsLlama - Llama 3.1-8B Mathematical Reasoning (GRPO)
---
datasets:
- openai/gsm8k
base_model:
- meta-llama/Llama-3.1-8B-Instruct
pipeline_tag: reinforcement-learning
---

## Model Description

This model is a fine-tuned version of [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) trained using **Group Relative Policy Optimization (GRPO)** to enable autonomous mathematical reasoning capabilities. Unlike traditional supervised fine-tuning approaches, this model learned to reason through reinforcement learning without requiring pre-labeled reasoning examples or teacher model distillation.

## Key Features

- **Structured Reasoning**: Generates step-by-step mathematical reasoning enclosed in `<thinking>` tags
- **Formatted Answers**: Provides final numerical answers in `<answer>` tags  
- **Autonomous Learning**: Trained using reward functions rather than human-annotated reasoning examples
- **Mathematical Focus**: Optimized for grade school math word problems requiring multi-step reasoning

## Training Details
### Training Pipeline
# GRPO Trainer Workflow

While setting up a GRPO config is straightforward, understanding what each hyperparameter means and how the underlying training logic works is essential. This guide walks through the GRPO training loop, highlighting the key code paths in trl/trainer/grpo_trainer.py.
## GRPO Trainer in Context

GRPO is built on top of the TRL Trainer class, but it customizes key parts of the training pipeline by overriding several critical methods:

- `compute_loss`
- `prepare_input`

These functions are the heart of GRPO's training logic.

## Training Workflow

### train_step Process

In each `train_step`:

1. `Trainer.train_step()` in the parent class first calls `prepare_input()` in the GRPO trainer
2. Then it calls `compute_loss()` with the output of `prepare_input()`

## Input Preparation Logic

### High-Level Behavior

- Called once per `self.num_iterations` to generate fresh completions
- Buffering is used to reuse completions across multiple gradient updates

**Example Configuration:**
- `num_iterations = 4`: generate completions every 4 global steps. Each global step can consist of k grad accumulation steps
- `gradient_accumulation_steps = 5`: buffer stores 5 sets of completions
- If `num_iterations = 1`, completions are not reused, and buffering may be unnecessary

### Core Logic

```python
@profiling_decorator
def _prepare_inputs(self, inputs: dict) -> dict:
    mode = "eval" if self.control.should_evaluate else "train"
    if mode == "train":
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
    else:
        inputs = self._generate_and_score_completions(inputs)
    return inputs
```

## Prompt Construction

Conversations are stored in the "prompt" field. The `apply_chat_template()` function is used to convert this into a generation-ready string:

```python
prompt = tokenizer.apply_chat_template(
    example["prompt"], 
    tools=tools, 
    tokenize=False, 
    add_generation_prompt=True
)
```

## Inference and Completion Generation

### Non-vLLM Path

For the non-vLLM implementation path:

1. Each rank tokenizes its prompts and truncates them by `max_prompt_length`
2. Each rank sends its text prompts to the on-device LLM

**Example Output:**
The last 5 tokens for 6 generations made on GPU0. With batched inference, we append EOS (128009) for early completed answers:

```python
prompt_completion_ids[:,-5:]
# Output tensor:
tensor([[128009, 128009, 128009, 128009, 128009],
        [128009, 128009, 128009, 128009, 128009],
        [128009, 128009, 128009, 128009, 128009],
        [   279,   1176,    220,    605,  14741],
        [   220,    430,    568,  11021,    220],
        [128009, 128009, 128009, 128009, 128009]], device='cuda:0')
```

> **Note:** You can check the code for vLLM path which is more efficient.

## Completion Masking

Completion masks identify valid (non-EOS) completions. Each rank computes completion masks based on the first occurrence of EOS.

**Example Mask:**
The following is the mask for the completions shown above:

```python
completion_mask[:,-5:]
# Output tensor:
tensor([[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.int32)
```

## Reward Computation and Normalization

Each rank performs the following operations:

1. **Decodes completions**
2. **Computes reward** for (prompt, completion) pairs
3. **Gathers rewards** from other ranks (because it's possible for a given prompt to have its replica across GPUs)
4. **Normalizes rewards** by mean/std ⟹ This gives us advantages $A(s,a)$
5. **Discards completions** for prompts it doesn't own (called alien prompts)

### Concrete Example: Multi-GPU Setup

**Setup:**
- 3 GPUs (Ranks)
- Batch size per GPU: 8
- Number of generations per prompt: 6

**Total Prompts Processed per Iteration:**
- Total batch size = 3 GPUs × 8 prompts = 24 prompts
- Since each prompt needs 6 completions, we can only afford to have: 24 / 6 = 4 unique prompts
- Each prompt is replicated 6 times across all ranks

**Prompt Distribution Example on GPU 0 (Rank 0):**
- 8 total prompts:
  - 6 replicas of Prompt #1
  - 2 replicas of Prompt #2

> **Note:** Other ranks may hold the remaining replicas of Prompt #2 and additional prompts.

**Reward Normalization Process:**

- **For Prompt #1:**
  - All 6 completions are on GPU 0
  - ✅ Rank 0 can compute mean and std of rewards locally

- **For Prompt #2:**
  - Rank 0 has only 2 out of 6 completions
  - ❌ It cannot compute accurate reward statistics alone
  - ✅ Needs to gather the remaining 4 rewards from other ranks

> **Important:** That's why all-gather is needed so each rank has access to the required replicas.

## Logprob Computation

- Rollouts (prompts + completions) are stored per grad accumulation step (`step % G`) during the first iteration
- These buffered inputs are reused for the remaining iterations within that `num_iterations` window
- Buffering occurs only during the first `num_iterations` step (i.e., at `global_step % num_iterations == 0`)

**Key Points:**
- **Logprobs for old policy and ref policy** are computed during the first iteration only (when completions are freshly generated)
- **Logprobs for current policy** are computed in every gradient accumulation step, including when using buffered inputs

> **Note:** Old and current policy logprobs will be identical when `num_iterations = 1`.

## Loss Computation

### GRPO Loss Formula

$$\text{GRPO} = -\mathbb{E}_{(s,a)}\left[\frac{\pi(a|s)}{\pi_{\text{old}}(a|s)} A(s,a)\right] + \beta \cdot \text{KL}[\pi(\cdot|s) \| \pi_{\text{ref}}(\cdot|s)]$$

### compute_loss() Breakdown

**Steps:**

1. **Concatenate** `prompt_ids + completion_ids`

2. **Run forward pass through old policy** to compute $\pi_{\text{old}}(a|s)$
   - This actually happens only once at the first iteration when we create the rollout

3. **Run forward pass through ref policy** to compute $\pi_{\text{ref}}(a|s)$
   - This actually happens only once at the first iteration when we create the rollout
   - Ref model is the original model without LoRA adapters

4. **Run forward pass through current policy** to compute $\pi(a|s)$
   - Needed only if `num_iterations > 1`; otherwise the same as old policy

5. **Compute KL loss** between $\pi(a|s)$ and $\pi_{\text{ref}}(a|s)$

6. **Compute advantage-weighted logprobs:** $\frac{\pi(a|s)}{\pi_{\text{old}}(a|s)} \times A(s,a)$

## Workflow Summary

| Component | What It Does | Why It Matters |
|-----------|--------------|----------------|
| **GRPO Trainer Class** | Extends the TRL Trainer, overrides `prepare_input` and `compute_loss` | Customizes the loss and input preparation for rollout reuse and reward learning |
| **train_step** | Calls `prepare_input` then `compute_loss` | Controls how and when rollouts and gradients are processed |
| **prepare_input()** | Generates and buffers rollouts once every `num_iterations` steps | Allows reuse of expensive rollouts across multiple updates |
| **Prompt Construction** | Uses `apply_chat_template` to create generation-ready input | Ensures the model sees correctly formatted conversational prompts |
| **Inference** | Uses on-device or vLLM backend to generate completions | Provides actions for which logprobs and rewards are computed |
| **Completion Masking** | Identifies valid (non-EOS) completions | Ensures reward/logprob computation only applies to meaningful tokens |
| **Reward Normalization** | Aggregates and normalizes rewards across GPUs for each prompt | Yields correct advantage estimates across distributed setup |
| **Logprob Computation** | Computes logprobs of old, ref, and current policy | Needed for KL and advantage-weighted loss; reused if `num_iterations > 1` |
| **compute_loss()** | Combines KL divergence and advantage-weighted logprob ratio | Drives the optimization update direction for the policy |

---

* Up to here, we are only walking through the key code paths in `trl/trainer/grpo_trainer.py` to help understand the GRPO training workflow.*
### Training Method
- **Algorithm**: Group Relative Policy Optimization (GRPO)
- **Base Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Fine-tuning**: LoRA with NF4 quantization
- **Dataset**: GSM8K (Grade School Math 8K)

### Training Configuration
- **LoRA rank**: 8
- **LoRA alpha**: 32
- **Learning rate**: 2e-4
- **Batch size**: 32 (effective)
- **Generations per prompt**: 2-4
- **Max completion length**: 128-256 tokens
- **Temperature**: 0.7

### Reward Functions
The model was trained using multiple custom reward functions:
1. **Format compliance**: Ensures proper `<thinking>` and `<answer>` tag structure
2. **Answer correctness**: Validates numerical accuracy against ground truth
3. **Integer validation**: Confirms answers are properly formatted integers
4. **Output cleanliness**: Penalizes extraneous text after answer tags

## Usage

### Required Format
The model expects a specific system prompt to maintain the structured output format:

```python
SYSTEM_PROMPT = """
Respond in the following format and make sure the entire response is wrapped in <reasoning> and <answer> tags with no other text outside of these tags:
<thinking>
your reasoning goes here and do not use newlines so that the entire reasoning becomes a single paragraph
</thinking>
<answer>
your answer goes here
</answer>
"""
```

### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/llama3-grpo-math-reasoning")
tokenizer = AutoTokenizer.from_pretrained("your-username/llama3-grpo-math-reasoning")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Mark is looking to buy a total of 12 pieces of fruit at the store. He has already chosen 3 apples. He has also selected a bunch of bananas containing 4 bananas. How many oranges does he need to pick out to have 12 total pieces of fruit?"}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**Expected Output:**
```
<thinking>
Mark wants 12 total pieces of fruit. He already has 3 apples and 4 bananas, which gives him 3 + 4 = 7 pieces of fruit. To reach 12 total pieces, he needs 12 - 7 = 5 more pieces of fruit, which would be oranges.
</thinking>
<answer>
5
</answer>
```

## Performance

- **Dataset**: GSM8K test split (1,319 examples)
- **Evaluation Metric**: Exact match accuracy on final numerical answers
- **Performance**: [Insert actual accuracy from your evaluation]

## Technical Details

### Model Architecture
- **Base**: Llama 3.1-8B-Instruct
- **Parameters**: ~8B total, ~21M trainable (LoRA)
- **Quantization**: NF4 with double quantization
- **Precision**: bfloat16

### GRPO Specifics
- **Group size**: 2-4 completions per prompt
- **KL coefficient (β)**: 0.02
- **Advantage normalization**: Group-based mean/std normalization
- **Policy updates**: Advantage-weighted importance sampling

## Limitations

- **Domain**: Primarily trained on grade school mathematics; may not generalize to advanced mathematical concepts
- **Format dependency**: Requires specific system prompt for optimal performance
- **Language**: English only
- **Reasoning scope**: Optimized for multi-step arithmetic rather than complex mathematical proofs

## Ethical Considerations

- **Educational use**: Designed to assist with mathematical learning, not replace mathematical education
- **Accuracy**: While trained for correctness, users should verify mathematical solutions for critical applications
- **Bias**: Inherits potential biases from the base Llama model and GSM8K dataset

## Citation

If you use this model, please cite:

```bibtex
@misc{llama3-grpo-math-reasoning,
  title={Llama 3.1-8B Mathematical Reasoning via Group Relative Policy Optimization},
  author={[Your Name]},
  year={2025},
  howpublished={\\url{https://huggingface.co/your-username/llama3-grpo-math-reasoning}}
}
```

## Acknowledgments

- Meta AI for the Llama 3.1 base model
- OpenAI for the GSM8K dataset
- Hugging Face TRL library for GRPO implementation
- The reinforcement learning and mathematical reasoning research communities

## License

This model inherits the license from meta-llama/Meta-Llama-3.1-8B-Instruct. Please refer to the [original model's license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) for usage terms.
