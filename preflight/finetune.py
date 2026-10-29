"""Fine-tune Muse Glimmer on Modal to tell which of two openings did better.

    uv run modal run -m preflight.finetune --run smoke                       # the whole path on a tiny model
    uv run modal run -m preflight.finetune --run pairs-luna --describer codex:gpt-6-luna --model glimmer
    uv run modal run -m preflight.finetune --run pairs-luna-3ep --describer codex:gpt-6-luna --model glimmer --epochs 3 --no-baseline

**What it learns.** By default (`--task pairwise`), the pair test's own
question: two Reels from one account, written down second by second, in; "A"
or "B" out: which one got more views against the account's usual. Training
pairs come from the training accounts the way the test pairs are drawn (one
Reel beat the usual, the other did not, at least 2x apart), each shown in both
orders. `--task pointwise` asks of one Reel at a time "did it beat its usual?"
instead; on Luna's descriptions it learned nothing (2026-10-29: every held-out
Reel got exactly 0.5; the answer token stayed at ln 2 from step 20 on).

**Held out.** A quarter of the accounts (`preflight.pairs.held_out`) never
train, so the fine-tune is scored only on accounts it never saw, on the very
pairs the zero-shot judges take. Glimmer without the adapter takes them too,
for reference.

**On Modal.** The weights (60 GB) are fetched once into a Volume by a CPU
container. QLoRA follows Meta's post-training guide for Glimmer (4-bit NF4,
r=16, alpha=32, the attention projections of the language model; about 28 GB
at peak, so one L40S). The answer is trained without reasoning, and scoring
reads the first answer token's probability, so it needs no generation. It is
worked out from the raw logits, in float32: Glimmer soft-caps its logits at 20
(`final_logit_softcapping`), and once training lifted "Yes" and "No" both near
the cap, bfloat16 rounded every Reel's answer to exactly 0.5.
"""
import json
import random
import time
from collections import defaultdict

import modal

app = modal.App("preflight")
volume = modal.Volume.from_name("preflight", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "transformers==5.21.0", "peft==0.21.2", "trl==1.16.0",
                      "bitsandbytes==0.50.2", "accelerate==1.15.0", "datasets")
         .env({"HF_HOME": "/vol/hf"})
         .add_local_python_source("preflight"))

MODELS = {"glimmer": "meta-models/Muse-Glimmer-30B",
          # Same path at a hundredth of the size, to check the plumbing for cents.
          "tiny": "Qwen/Qwen2.5-0.5B-Instruct"}
#: GPUs a job may run on, cheapest first; Modal takes the first with capacity
#: (L4s were all busy on 2026-10-29, and a job just waited). Glimmer's QLoRA
#: needs about 28 GB. A list is only allowed here, not in `with_options`.
GPUS = ["L40S", "A100-40GB", "A100-80GB"]
#: The template's own default is "Reasoning strength: high"; naming it here
#: keeps that directive out, and the answer is trained to come without reasoning.
SYSTEMS = {
    "pointwise": "You read the opening seconds of an Instagram Reel, written down second by second, and say whether "
                 "the Reel got more views than its account's usual. Answer Yes or No. Reasoning strength: low.",
    "pairwise": "You read the opening seconds of two Instagram Reels from the same account, written down second by "
                "second, and say which one got more views compared with how the account's Reels usually do. Answer "
                "A or B. Reasoning strength: low.",
}
ANSWERS = {"pointwise": ("Yes", "No"), "pairwise": ("A", "B")}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
ADAPTERS = "/vol/adapters"


def messages(task: str, text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEMS[task]}, {"role": "user", "content": text}]


def load(model_id: str):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig)
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    # Both bfloat16 settings: without them Glimmer's vision path loads in byte dtype (Meta's guide).
    quantized = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
    # transformers 5.21 knows Glimmer only as image-text-to-text (checked 2026-10-29); it reads text alone fine.
    causal = AutoConfig.from_pretrained(model_id).model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    loader = AutoModelForCausalLM if causal else AutoModelForImageTextToText
    model = loader.from_pretrained(model_id, quantization_config=quantized, dtype=torch.bfloat16, device_map="auto")
    # A soft cap too high to bind: with Glimmer's own (20), training lifted both answers to the cap, where they
    # tie at 0.5 and the gradient vanishes (the pointwise run on 2026-10-29 learned nothing after step 20). The
    # config only takes a float, so the cap is raised rather than removed.
    for config in (model.config, getattr(model.config, "text_config", None)):
        if getattr(config, "final_logit_softcapping", None):
            config.final_logit_softcapping = 1e4
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=2 * 3600)
def fetch(model_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model_id)
    volume.commit()


def chances(model, tokenizer, task: str, texts: list[str]) -> list[float]:
    """The probability of the first answer ("Yes", or "A") for each text, from
    its first token. Leaves the model training if it was."""
    import torch

    training, padding = model.training, tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = "left"
    # Where the two answers part, right after the template's header for the reply.
    first, second = ANSWERS[task]
    sample = messages(task, "")
    one, other = (tokenizer.apply_chat_template(sample + [{"role": "assistant", "content": answer}], tokenize=False)
                  for answer in (first, second))
    split = next(index for index, (a, b) in enumerate(zip(one, other)) if a != b)
    header = one[len(tokenizer.apply_chat_template(sample, tokenize=False)):split]
    ids = [tokenizer(answer, add_special_tokens=False).input_ids[0] for answer in (first, second)]
    # The two answers' rows of the output layer, applied by hand to what goes into it: raw logits, before the cap.
    output_layer = model.get_output_embeddings()
    head, seen = output_layer.weight[ids].float(), {}
    hook = output_layer.register_forward_hook(lambda _, inputs, __: seen.update(last=inputs[0][:, -1, :].float()))
    prompts = [tokenizer.apply_chat_template(messages(task, text), tokenize=False) + header for text in texts]
    scores = []
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), 8):
                batch = tokenizer(prompts[start:start + 8], return_tensors="pt", padding=True,
                                  add_special_tokens=False).to(model.device)
                model(**batch, logits_to_keep=1)
                scores += torch.softmax(seen["last"] @ head.T, dim=-1)[:, 0].tolist()
    finally:
        hook.remove()
        tokenizer.padding_side = padding
        model.train(training)
    return scores


@app.function(image=image, volumes={"/vol": volume}, gpu=GPUS, memory=32768, timeout=6 * 3600)
def train(run: str, model_id: str, task: str, examples: list[dict], epochs: float, learning_rate: float,
          checks: list[str] | None = None) -> dict:
    """With `checks`, the adapter so far is kept after every epoch (as
    `<run>-epoch<n>`) and scores those texts, so one run shows how many epochs
    are worth it."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    started = time.time()
    model, tokenizer = load(model_id)
    each_epoch = []

    class EveryEpoch(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = round(state.epoch)
            model.save_pretrained(f"{ADAPTERS}/{run}-epoch{epoch}")
            volume.commit()
            each_epoch.append({"epoch": epoch, "step": state.global_step,
                               "probabilities": chances(model, tokenizer, task, checks)})
            print(f"epoch {epoch} scored, {(time.time() - started) / 60:.0f} min in", flush=True)

    targets = sorted({name for name, _ in model.named_modules()
                      if name.split(".")[-1] in PROJECTIONS and "vision" not in name})
    data = Dataset.from_list([{"prompt": messages(task, example["text"]),
                               "completion": [{"role": "assistant", "content": example["answer"]}]}
                              for example in examples])
    config = SFTConfig(output_dir="/tmp/out", num_train_epochs=epochs, per_device_train_batch_size=4,
                       gradient_accumulation_steps=4, learning_rate=learning_rate, lr_scheduler_type="cosine",
                       warmup_steps=10, bf16=True, gradient_checkpointing=True,
                       gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=3072,
                       completion_only_loss=True, logging_steps=10, save_strategy="no", report_to="none")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer, peft_config=lora,
                         callbacks=[EveryEpoch()] if checks else None)
    result = trainer.train()
    trainer.save_model(f"{ADAPTERS}/{run}")
    volume.commit()
    return {"train_loss": result.training_loss, "steps": result.global_step, "examples": len(examples),
            "minutes": round((time.time() - started) / 60, 1),
            "log": [entry for entry in trainer.state.log_history if "loss" in entry], "each_epoch": each_epoch}


@app.function(image=image, volumes={"/vol": volume}, gpu=GPUS, memory=32768, timeout=2 * 3600)
def score(run: str | None, model_id: str, task: str, texts: list[str]) -> list[float]:
    """`chances` for each text; without `run`, Glimmer as it comes."""
    from peft import PeftModel

    model, tokenizer = load(model_id)
    if run:
        model = PeftModel.from_pretrained(model, f"{ADAPTERS}/{run}")
    return chances(model, tokenizer, task, texts)


@app.cls(image=image, volumes={"/vol": volume}, gpu=GPUS, memory=32768, timeout=30 * 60, scaledown_window=10 * 60)
class Judge:
    """A pairwise adapter kept loaded, for the demo (`preflight.demo`), once
    deployed (`modal deploy -m preflight.finetune`). It goes cold after ten idle
    minutes, and the first call after that waits minutes for Glimmer to load."""
    run: str = modal.parameter()

    @modal.enter()
    def start(self):
        from peft import PeftModel

        model, self.tokenizer = load(MODELS["glimmer"])
        self.model = PeftModel.from_pretrained(model, f"{ADAPTERS}/{self.run}")

    @modal.method()
    def rate(self, texts: list[str]) -> list[float]:
        return chances(self.model, self.tokenizer, "pairwise", texts)


# Local side: the data in, the results out ------------------------------------------------------------------------------
def reel_texts(describer: str, captions: bool) -> tuple[dict[str, str], dict[str, dict]]:
    """Every described Reel's text as the model reads it, and the dataset rows."""
    from preflight.dataset import OUT as DATASET
    from preflight.describe import as_text, load_descriptions

    forms = load_descriptions(describer)
    rows = {row["id"]: row for row in map(json.loads, DATASET.open()) if row["id"] in forms}
    texts = {}
    for media_id, row in rows.items():
        texts[media_id] = as_text(forms[media_id]) + (f"\nCaption: {row['caption'] or '(none)'}" if captions else "")
    return texts, rows


def pair_text(texts: dict[str, str], a: str, b: str) -> str:
    return f"Reel A\n{texts[a]}\n\nReel B\n{texts[b]}"


def training_pairs(rows: dict[str, dict], per_account: int) -> list[tuple[str, str]]:
    """(better, worse) from the training accounts, drawn like the test pairs."""
    from preflight.pairs import GAP, SEED, held_out

    rng, by_account = random.Random(SEED), defaultdict(list)
    for row in rows.values():
        if not held_out(row["username"]):
            by_account[row["username"]].append(row)
    pairs = []
    for account in sorted(by_account):
        reels = by_account[account]
        candidates = [(hit["id"], miss["id"]) for hit in reels if hit["beats_median"] for miss in reels
                      if not miss["beats_median"] and hit["ratio"] >= GAP * miss["ratio"]]
        rng.shuffle(candidates)
        pairs += candidates[:per_account]
    return pairs


@app.local_entrypoint()
def main(run: str = "smoke", describer: str = "codex:gpt-6-luna", model: str = "tiny", task: str = "pairwise",
         captions: bool = False, epochs: float = 1.0, learning_rate: float = 1e-4, per_account: int = 15,
         baseline: bool = True, retrain: bool = True, limit: int = 0):
    from sklearn.metrics import roc_auc_score

    from preflight.dataset import OUT as DATASET
    from preflight.pairs import held_out, load_pairs, wilson

    model_id = MODELS[model]
    texts, rows = reel_texts(describer, captions)
    test_pairs = [pair for pair in load_pairs("held-out", rows) if pair["a"] in texts and pair["b"] in texts]
    smoke = run == "smoke" or limit
    if task == "pairwise":
        train_examples = [example for better, worse in training_pairs(rows, per_account)
                          for example in ({"text": pair_text(texts, better, worse), "answer": "A"},
                                          {"text": pair_text(texts, worse, better), "answer": "B"})]
        # Each test pair both ways round, so a lean towards A or B cancels out.
        test_texts = [pair_text(texts, pair[x], pair[y]) for pair in test_pairs for x, y in (("a", "b"), ("b", "a"))]
    else:
        train_examples = [{"text": texts[media_id], "answer": "Yes" if row["beats_median"] else "No"}
                          for media_id, row in rows.items() if not held_out(row["username"])]
        test_ids = sorted({media_id for media_id, row in rows.items() if held_out(row["username"])})
        test_texts = [texts[media_id] for media_id in test_ids]
    random.Random(0).shuffle(train_examples)
    if smoke:
        train_examples, test_pairs = train_examples[:limit or 64], test_pairs[:16]
        if task == "pairwise":
            test_texts = test_texts[:2 * len(test_pairs)]
    print(f"{len(train_examples)} training examples ({task}); {len(test_pairs)} held-out pairs from "
          f"{len({pair['account'] for pair in test_pairs})} accounts; {model_id}")
    trained = None
    if retrain:  # --no-retrain scores the adapter already saved under this run's name
        fetch.remote(model_id)
        # Past one epoch, the held-out texts are scored after each, inside the training run.
        trained = train.remote(run, model_id, task, train_examples, epochs, learning_rate,
                               test_texts if epochs > 1 else None)
        print(f"trained: loss {trained['train_loss']:.3f}, {trained['steps']} steps, {trained['minutes']} min")

    def judged(probabilities: list[float]) -> dict:
        if task == "pairwise":
            # P(the first-shown Reel is better), averaged over both orders: (P(A | a,b) + P(B | b,a)) / 2.
            chances = [(probabilities[2 * i] + 1 - probabilities[2 * i + 1]) / 2 for i in range(len(test_pairs))]
            right = sum((chance > 0.5) == (pair["better"] == "A") for chance, pair in zip(chances, test_pairs))
            one_order = sum((probabilities[2 * i] > 0.5) == (pair["better"] == "A")
                            for i, pair in enumerate(test_pairs))
            return {"pairs_right": right, "one_order_right": one_order, "pairs": len(test_pairs),
                    "interval": wilson(right, len(test_pairs)), "spread": [min(probabilities), max(probabilities)],
                    "chances": chances}
        by_id = dict(zip(test_ids, probabilities))
        right = sum((by_id[pair["a"]] > by_id[pair["b"]]) == (pair["better"] == "A") for pair in test_pairs)
        truth = [rows[media_id]["beats_median"] for media_id in test_ids]
        return {"pairs_right": right, "pairs": len(test_pairs), "interval": wilson(right, len(test_pairs)),
                "held_out_auc": roc_auc_score(truth, probabilities) if len(set(truth)) > 1 else None,
                "spread": [min(probabilities), max(probabilities)], "probabilities": by_id}

    results = {}
    for seen in (trained or {}).get("each_epoch", []):
        results[f"epoch {seen['epoch']}"] = judged(seen.pop("probabilities"))
    results["tuned"] = (list(results.values())[-1] if results
                        else judged(score.remote(run, model_id, task, test_texts)))
    if baseline:
        results["as it comes"] = judged(score.remote(None, model_id, task, test_texts))
    out = DATASET.parent / "finetune" / f"{run}{'' if retrain else '-rescored'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": run, "model": model_id, "task": task, "describer": describer,
                               "captions": captions, "train": trained, **results}, indent=1))
    for name, result in results.items():
        low, high = result["interval"]
        extra = f"; held-out AUC {result['held_out_auc']:.3f}" if result.get("held_out_auc") is not None else ""
        print(f"{name}: pair test {result['pairs_right']}/{result['pairs']} = "
              f"{result['pairs_right'] / max(result['pairs'], 1):.0%} (95%: {low:.0%}-{high:.0%}); "
              f"probabilities {result['spread'][0]:.3f}-{result['spread'][1]:.3f}{extra}")
    print(f"written to {out}")
