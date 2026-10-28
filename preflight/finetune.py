"""Fine-tune Muse Glimmer on Modal to say whether a Reel's opening beats its account's usual.

    uv run modal run -m preflight.finetune --run smoke                       # the whole path on a tiny model
    uv run modal run -m preflight.finetune --run practice-luna --describer codex:gpt-6-luna --model glimmer

**What it learns.** One Reel at a time: its written-down opening (and, with
`--captions`, its caption) in; "Yes" or "No" out: did it beat its own
account's median views. Two openings are compared by comparing those answers,
which is how it takes the pair test (`preflight.pairs`), and how the demo will
rank a creator's candidates.

**Held out.** A quarter of the accounts (`preflight.pairs.held_out`) never
train, so the fine-tune is scored only on accounts it never saw, on the very
pairs the zero-shot judges take.

**On Modal.** The weights (60 GB) are fetched once into a Volume by a CPU
container. QLoRA follows Meta's post-training guide for Glimmer (4-bit NF4,
r=16, alpha=32, the attention projections of the language model; about 28 GB
at peak, so one L40S). The answer is trained without reasoning, and scoring
reads P(Yes) off the first answer token, so it needs no generation.
"""
import json
import time

import modal

app = modal.App("preflight")
volume = modal.Volume.from_name("preflight", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "transformers==5.21.0", "peft==0.21.2", "trl==1.16.0",
                      "bitsandbytes==0.50.2", "accelerate==1.15.0", "datasets")
         .env({"HF_HOME": "/vol/hf"})
         .add_local_python_source("preflight"))

MODELS = {"glimmer": ("meta-models/Muse-Glimmer-30B", "L40S"),
          # Same path at a hundredth of the size, to check the plumbing for cents.
          "tiny": ("Qwen/Qwen2.5-0.5B-Instruct", "L4")}
#: The template's own default is "Reasoning strength: high"; naming it here
#: keeps that directive out, and the answer is trained to come without reasoning.
SYSTEM = ("You read the opening seconds of an Instagram Reel, written down second by second, and say whether the "
          "Reel got more views than its account's usual. Answer Yes or No. Reasoning strength: low.")
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
ADAPTERS = "/vol/adapters"


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Both bfloat16 settings: without them Glimmer's vision path loads in byte dtype (Meta's guide).
    quantized = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantized, dtype=torch.bfloat16,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=2 * 3600)
def fetch(model_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model_id)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=6 * 3600)
def train(run: str, model_id: str, examples: list[dict], epochs: float) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    started = time.time()
    model, tokenizer = load(model_id)
    targets = sorted({name for name, _ in model.named_modules()
                      if name.split(".")[-1] in PROJECTIONS and "vision" not in name})
    data = Dataset.from_list([{"prompt": messages(example["text"]),
                               "completion": [{"role": "assistant", "content": example["answer"]}]}
                              for example in examples])
    config = SFTConfig(output_dir="/tmp/out", num_train_epochs=epochs, per_device_train_batch_size=4,
                       gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
                       warmup_steps=10, bf16=True, gradient_checkpointing=True,
                       gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=2048,
                       completion_only_loss=True, logging_steps=10, save_strategy="no", report_to="none")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer, peft_config=lora)
    result = trainer.train()
    trainer.save_model(f"{ADAPTERS}/{run}")
    volume.commit()
    return {"train_loss": result.training_loss, "steps": result.global_step, "examples": len(examples),
            "minutes": round((time.time() - started) / 60, 1),
            "log": [entry for entry in trainer.state.log_history if "loss" in entry]}


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=2 * 3600)
def score(run: str | None, model_id: str, texts: list[str]) -> list[float]:
    """P(Yes) for each opening, from the logits of the first answer token."""
    import torch
    from peft import PeftModel

    model, tokenizer = load(model_id)
    if run:
        model = PeftModel.from_pretrained(model, f"{ADAPTERS}/{run}")
    model.eval()
    tokenizer.padding_side = "left"
    # Where the two answers part, right after the template's header for the reply.
    sample = messages("")
    yes, no = (tokenizer.apply_chat_template(sample + [{"role": "assistant", "content": answer}], tokenize=False)
               for answer in ("Yes", "No"))
    split = next(index for index, (a, b) in enumerate(zip(yes, no)) if a != b)
    header = yes[len(tokenizer.apply_chat_template(sample, tokenize=False)):split]
    yes_id, no_id = (tokenizer(answer, add_special_tokens=False).input_ids[0] for answer in ("Yes", "No"))
    prompts = [tokenizer.apply_chat_template(messages(text), tokenize=False) + header for text in texts]
    scores = []
    with torch.no_grad():
        for start in range(0, len(prompts), 8):
            batch = tokenizer(prompts[start:start + 8], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(model.device)
            logits = model(**batch).logits[:, -1, [yes_id, no_id]].float()
            scores += torch.softmax(logits, dim=-1)[:, 0].tolist()
    return scores


# Local side: the data in, the results out ------------------------------------------------------------------------------
def examples(describer: str, captions: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """Training Reels, held-out Reels, and the pair test, from local files."""
    from preflight.dataset import OUT as DATASET
    from preflight.describe import as_text, load_descriptions
    from preflight.pairs import held_out, load_pairs

    forms = load_descriptions(describer)
    rows = {row["id"]: row for row in map(json.loads, DATASET.open())}
    pairs = load_pairs("held-out", rows)
    train_rows, test_rows = [], []
    for row in rows.values():
        if row["id"] not in forms:
            continue
        text = as_text(forms[row["id"]])
        if captions:
            text += f"\nCaption: {row['caption'] or '(none)'}"
        example = {"id": row["id"], "account": row["username"], "text": text,
                   "answer": "Yes" if row["beats_median"] else "No"}
        (test_rows if held_out(row["username"]) else train_rows).append(example)
    return train_rows, test_rows, [pair for pair in pairs if pair["a"] in forms and pair["b"] in forms]


@app.local_entrypoint()
def main(run: str = "smoke", describer: str = "codex:gpt-6-luna", model: str = "tiny", captions: bool = False,
         epochs: float = 1.0, limit: int = 0):
    from sklearn.metrics import roc_auc_score

    from preflight.dataset import OUT as DATASET
    from preflight.pairs import wilson

    model_id, gpu = MODELS[model]
    train_rows, test_rows, pairs = examples(describer, captions)
    if run == "smoke" or limit:
        train_rows, test_rows, pairs = train_rows[:limit or 64], test_rows[:limit or 32], pairs[:16]
    print(f"{len(train_rows)} training Reels, {len(test_rows)} held out from {len({r['account'] for r in test_rows})} "
          f"accounts, {len(pairs)} pairs; {model_id} on {gpu}")
    fetch.remote(model_id)
    trained = train.with_options(gpu=gpu).remote(run, model_id, train_rows, epochs)
    print(f"trained: loss {trained['train_loss']:.3f}, {trained['steps']} steps, {trained['minutes']} min")

    needed = {row["id"]: row for row in test_rows}
    texts = {row["id"]: row["text"] for row in examples(describer, captions)[1]}  # every held-out Reel
    ids = sorted(needed | {pair[side]: None for pair in pairs for side in ("a", "b")})
    probability = dict(zip(ids, score.with_options(gpu=gpu).remote(run, model_id, [texts[i] for i in ids])))

    truth = {row["id"]: row["answer"] == "Yes" for row in test_rows}
    auc = roc_auc_score([truth[i] for i in truth], [probability[i] for i in truth]) if len(set(truth.values())) > 1 \
        else None
    right = sum((probability[pair["a"]] > probability[pair["b"]]) == (pair["better"] == "A") for pair in pairs)
    low, high = wilson(right, len(pairs))
    result = {"run": run, "model": model_id, "describer": describer, "captions": captions, "train": trained,
              "held_out_auc": auc, "pairs_right": right, "pairs": len(pairs), "pairs_interval": [low, high],
              "probabilities": probability}
    out = DATASET.parent / "finetune" / f"{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    auc_text = f"{auc:.3f}" if auc is not None else "n/a"
    print(f"held-out accounts, beats its usual: ROC AUC {auc_text}; pair test: {right}/{len(pairs)} = "
          f"{right / max(len(pairs), 1):.0%} (95%: {low:.0%}-{high:.0%}); written to {out}")
