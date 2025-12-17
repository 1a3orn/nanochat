Running procedure:
```
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync --extra gpu
source .venv/bin/activate
wandb login
# bash speedrun.sh gated_sigmoid
```



One-sentence overview of plan: Change the model files (nanochat/gpt.py) and all other necessary Python files so that one can run `bash speedrun.sh` with an additional argument, "standard," "gated_sigmoid," or "gated_softplus," which executes speedrun.sh either (1) as per normal, (2) with sigmoid gated attention, or (3) softplus gated attenton.

Long Form Explanation:

*Gated attention* was introduced by Qwen3-next.

In the simplest form, it involves takes the concatenated outputs all the heads of attention, and feeding them through a gate before putting it through the output projection.  The values of these gates are learned, from a linear n x n matrix from before the attention, where n is just the residual dimension.

This adds a (resdiual dimension * residual dimension) number of parameters. To keep things equal, usually people even this out by reducing the MLP expansion from 4.0 to 3.5, which keeps the number of parameters even.

To be clear about what this gate is, then:
- It is calculated *per token* and *per channel*, i.e., there is a separate gate for every dimension coming out of the attention.
- For simplicity, let's have the gate matrix be initialized with no bias at all.
- So we have gate_fn(gate_matrix(input_to_attention)) * attention_output, where * is an elementwise multiplication.

The configuration option should let us us sigmoid or softplus as the activation function, with gated_sigmoid or gated_softplus.