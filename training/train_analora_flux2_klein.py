import copy
import os
import sys
from pathlib import Path

# WandB is disabled when WANDB_MODE=disabled or WANDB_DISABLED=true.
# Some GPU nodes have no wandb-core installed, so we must skip
# all wandb calls entirely — not just put the run in offline mode.
_WANDB_MODE = os.environ.get("WANDB_MODE", "online").lower()
_USE_WANDB = (
    _WANDB_MODE not in ("disabled", "dryrun")
    and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
)
if _USE_WANDB:
    import wandb

# Ensure the sia package (parent of this training/ dir) is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors_file
from peft import LoraConfig
from accelerate.utils import set_seed
from accelerate.utils import DataLoaderConfiguration
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import math
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.models import Flux2Transformer2DModel
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse

from sia.data import DATASET, AnalogyPairDataset, collate_pairs
# Importing the ExLoRA trainer also applies its accelerate compatibility shim.
from training.train_exlora_flux2_klein import (
    compute_empirical_mu,
    compute_text_embeddings,
    _native_wh,
    encode_single_image,
    get_sigmas,
    load_models,
)

ANALORA_ADAPTER_NAME = "visual-analogy-analora"


@torch.no_grad()
def prepare_analogy_latents(pipe, batch):
    """Encode A, A', B, B' through VAE and prepare latents + IDs.

    Returns:
        target_latents, condition_latents, target_ids, condition_ids
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(batch['A'], batch['A_prime'], batch['B'], batch['B_prime']):
        lat_a = encode_single_image(pipe, a)
        lat_a_prime = encode_single_image(pipe, a_prime)
        lat_b = encode_single_image(pipe, b)
        lat_b_prime = encode_single_image(pipe, b_prime)

        # B' is target, A/A'/B are separate conditions
        target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
        cond_a = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
        cond_b = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
        cond_arr.append(
            torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])

    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr, dim=0),
        target_ids,
        condition_ids,
    )


def add_lora(transformer, args, lora_target_modules):
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
    )
    transformer.add_adapter(transformer_lora_config,
                            adapter_name=ANALORA_ADAPTER_NAME)


def get_save_model_hook(accelerator):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                print("Type of model in save_model_hook:", type(model))
                if isinstance(accelerator.unwrap_model(model), Flux2Transformer2DModel):
                    Flux2KleinPipeline.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=get_peft_model_state_dict(
                            accelerator.unwrap_model(model), adapter_name=ANALORA_ADAPTER_NAME),
                    )
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                weights.pop()  # make sure to pop weight so that corresponding model is not saved again

    return save_model_hook


def get_load_model_hook(accelerator):
    def load_model_hook(models, input_dir):
        lora_weights_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        while len(models) > 0:
            model = models.pop()
            if isinstance(accelerator.unwrap_model(model), Flux2Transformer2DModel):
                if not os.path.exists(lora_weights_path):
                    raise FileNotFoundError(
                        f"LoRA checkpoint not found at {lora_weights_path}. "
                        "Expected 'pytorch_lora_weights.safetensors' saved by the save hook."
                    )
                # The file uses diffusers key format: "transformer.<peft_key>"
                # Strip the leading "transformer." to recover peft-compatible keys.
                raw_state_dict = load_safetensors_file(lora_weights_path)
                peft_state_dict = {
                    k[len("transformer."):]: v
                    for k, v in raw_state_dict.items()
                    if k.startswith("transformer.")
                }
                unwrapped = accelerator.unwrap_model(model)
                incompatible = set_peft_model_state_dict(
                    unwrapped, peft_state_dict, adapter_name=ANALORA_ADAPTER_NAME
                )
                # set_peft_model_state_dict loads with strict=False, so every
                # frozen base-model weight (not present in the LoRA checkpoint)
                # is reported as "missing". That is expected and harmless. Only
                # genuinely missing LoRA tensors or unexpected keys are problems.
                missing_lora_keys = [
                    k for k in getattr(incompatible, "missing_keys", [])
                    if "lora_" in k
                ]
                unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
                if missing_lora_keys or unexpected_keys:
                    print(
                        "[load_model_hook] Problem loading LoRA weights — "
                        f"missing LoRA keys: {missing_lora_keys}, "
                        f"unexpected keys: {unexpected_keys}"
                    )
                else:
                    n_loaded = sum(1 for k in peft_state_dict if "lora_" in k)
                    print(
                        f"[load_model_hook] Restored {n_loaded} LoRA tensors from "
                        f"{lora_weights_path}"
                    )
            else:
                raise ValueError(f"Unexpected model type in load_model_hook: {type(model)=}.")

    return load_model_hook


@torch.no_grad()
def log_analogy_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with LoRA scale=1 and log a comparison strip to wandb."""
    print(f"Running analogy validation at step {global_step}...")

    pipe = pipeline
    device = accelerator.device
    dtype = torch.bfloat16

    # Pick a fresh random sample from the multi-edit dataset.
    val_meta = train_dataset.sample_val_pair()
    combined_prompt = val_meta['prompt']
    edit_name       = val_meta['edit_name']

    val_sample = {
        'A':         val_meta['a'],
        'A_prime':   val_meta['a_prime'],
        'B':         val_meta['b'],
        'B_prime':   val_meta['b_prime'],
        'edit_name': edit_name,
        'prompt':    combined_prompt,
    }

    # Encode images through VAE
    lat_a = encode_single_image(pipe, val_sample['A'])
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'])
    lat_b = encode_single_image(pipe, val_sample['B'])

    # Text embeddings
    prompt_embeds, text_ids = compute_text_embeddings(
        [val_sample['prompt']], tokenizer, text_encoder
    )
    # Klein: no guidance embeds, pass guidance=None
    guidance = None

    def decode_latents(packed, h, w):
        """Decode packed latents (1, N, C) → PIL image, given patchified spatial dims h, w."""
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    pipe.transformer.set_adapters([ANALORA_ADAPTER_NAME], [1])
    num_inference_steps = args.num_val_inference_steps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    # Denoise B' from noise
    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    cond_a = Flux2KleinPipeline._pack_latents(lat_a)
    cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
    cond_b = Flux2KleinPipeline._pack_latents(lat_b)
    condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

    gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
    gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b]).to(device)
    combined_ids = torch.cat([target_ids, condition_ids], dim=1)

    image_seq_len = gen_latents.shape[1]
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

    for t in timesteps:
        latent_model_input = torch.cat(
            [gen_latents, condition_latents], dim=1)
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(1).to(dtype) / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_ids,
            return_dict=False,
        )[0][:, :gen_latents.shape[1]]
        gen_latents = pipe.scheduler.step(
            noise_pred, t, gen_latents, return_dict=False)[0]

    pred_b_prime = decode_latents(gen_latents, patch_h, patch_w)

    # Condition images from val_sample (shown once, not repeated)
    cell_w, cell_h = _native_wh() or (512, 512)
    a_img = val_sample['A'].resize((cell_w, cell_h), Image.LANCZOS)
    a_prime_img = val_sample['A_prime'].resize(
        (cell_w, cell_h), Image.LANCZOS)
    b_img = val_sample['B'].resize((cell_w, cell_h), Image.LANCZOS)
    b_prime_gt_img = val_sample['B_prime'].resize(
        (cell_w, cell_h), Image.LANCZOS)

    # Log to wandb: individual images + full strip
    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = [a_img, a_prime_img, b_img, b_prime_gt_img, pred_b_prime]
        labels = ["A", "A'", "B", "B' GT", "B' pred"]

        strip_w = cell_w * len(all_imgs)
        strip_h = cell_h + 30
        strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        for i, (img, label) in enumerate(zip(all_imgs, labels)):
            strip.paste(img.resize((cell_w, cell_h)), (i * cell_w, 30))
            draw.text((i * cell_w + 4, 4), label, fill=(0, 0, 0))

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip to {strip_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(print_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )

    args = OmegaConf.load(parser.parse_args().config)

    if print_args:
        print("="*40, "Arguments", "="*40)
        for arg in args:
            print(f"{arg}: {getattr(args, arg)}")
        print("="*91)

    return args


def main():
    args = parse_args(print_args=True)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=True)],
        dataloader_config=dataloader_config,
    )

    if accelerator.is_main_process and _USE_WANDB:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(
            project=args.tracker_name,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    set_seed(args.seed)

    tokenizer, text_encoder, vae, transformer, noise_scheduler = load_models(
        args)

    # Only train the LoRA layers
    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)

    weight_dtype = torch.bfloat16
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=weight_dtype)

    transformer.enable_gradient_checkpointing()

    LORA_TARGET_MODULES = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    add_lora(transformer, args, LORA_TARGET_MODULES)

    accelerator.register_save_state_pre_hook(
        get_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(
        get_load_model_hook(accelerator))

    # Make sure the trainable params are in float32. [only upcast trainable parameters (LoRA) into fp32]
    cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(  # TODO: Change to prodigy
        [{"params": transformer_lora_parameters, "lr": args.learning_rate}],
        betas=(0.9, 0.999),
        weight_decay=1e-04,
        eps=1e-08,
    )

    train_dataset = AnalogyPairDataset(args.get("dataset", DATASET), args.get("dataset_revision"))
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_pairs, num_workers=2, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant",  # TODO: make it argument
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    len_train_dataset = len(train_dataset)
    len_train_dataloader = math.ceil(len_train_dataset / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(
        len_train_dataloader / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running training *****")
    print(f"  Num examples = {len_train_dataset}")
    print(f"  Num batches each epoch = {len_train_dataloader}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = 0
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        # Get latest checkpoint
        # A fresh run has no output_dir yet; treat that as "nothing to resume".
        dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"No checkpoint found in {args.output_dir}. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    pipeline = Flux2KleinPipeline(
        # Own copy: validation calls retrieve_timesteps() on pipe.scheduler, which would
        # otherwise shrink noise_scheduler.timesteps to the 28 inference steps and
        # collapse training timestep sampling onto the lowest-noise one.
        scheduler=copy.deepcopy(noise_scheduler),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(
            transformer, keep_fp32_wrapper=False),
    )

    transformer.train()

    for epoch in range(first_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                prompts = batch["prompts"]

                target_latents, condition_latents, target_ids, condition_ids = prepare_analogy_latents(
                    pipeline, batch
                )

                device = target_latents.device
                target_ids = target_ids.to(device)

                bsz = target_latents.shape[0]

                # Expand IDs to batch size
                target_ids = target_ids.expand(bsz, -1, -1)

                # Target (T=0) + conditions (T=10/20/30)
                condition_ids = condition_ids.to(
                    device).expand(bsz, -1, -1)
                combined_ids = torch.cat(
                    [target_ids, condition_ids], dim=1)

                prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, tokenizer, text_encoder)

                noise = torch.randn_like(target_latents)

                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none", batch_size=bsz)
                indices = (u * (args.max_train_timesteps -
                           args.min_train_timesteps) + args.min_train_timesteps).long()
                indices = indices.clamp(0, len(noise_scheduler.timesteps) - 1)
                timesteps = noise_scheduler.timesteps[indices].to(
                    device=target_latents.device)

                # Add noise via flow matching
                sigmas = get_sigmas(noise_scheduler, timesteps, device=target_latents.device,
                                    n_dim=target_latents.ndim, dtype=target_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * \
                    target_latents + sigmas * noise

                # Klein: no guidance embeds, always guidance=None
                guidance = None

                # Append condition tokens
                latent_model_input = torch.cat(
                    [noisy_model_input, condition_latents], dim=1)

                transformer.set_adapters([ANALORA_ADAPTER_NAME], [1])
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, :noisy_model_input.shape[1]]

                # Flow matching loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas)
                flow_target = noise - target_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() -
                     flow_target.float()) ** 2).reshape(bsz, -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and _USE_WANDB:
                    wandb.log({
                        "train/loss":  loss.detach().item(),
                        "train/lr":    lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step":  global_step,
                    })

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        pipeline.transformer = accelerator.unwrap_model(
                            transformer, keep_fp32_wrapper=False)
                        log_analogy_validation(
                            pipeline, train_dataset,
                            tokenizer, text_encoder,
                            args, accelerator, global_step,
                        )
                        transformer.train()

            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process and _USE_WANDB:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
