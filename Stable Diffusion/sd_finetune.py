# This script aims at fine-tuning a Stable Diffusion model on the given dataset
# uses excerpts from diffusers example train_text_to_image.py
# to know more about SD, read this explainer-
# https://poloclub.github.io/diffusion-explainer/


import math
import logging
import accelerate
import datasets
from accelerate.logging import get_logger
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.torch_utils import is_compiled_module
import bitsandbytes as bnb
import os
import random

logger = get_logger(__name__, log_level="INFO")

def main(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", # name of the model on huggingface
        use_8bit_adam=True, # quantised optimizer to save space
        dataset_name="Babypotatotang/logo-splitted", # name of dataset on HF
        train_data_dir=None, # in case using custom dataset, set this value
        resolution=(256,256), # resolution of output image
        train_batch_size=2, # batch-size
        dataloader_num_workers=20, # number of parallel jobs to run while loading dataset (depends upon num_threads present)
        gradient_accumulation_steps=1, # in case of low memory, set this value to aggregate forward pass from multiple steps  
        num_train_epochs=2, # num epochs
        checkpointing_steps=500, # save model chkpt after N steps
        output_dir="./" # dir to save the model to
        ):
    
    # instantiate accelerate object
    accelerator = Accelerator(
        mixed_precision="bf16", # used mixed precision to save space
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Load scheduler, tokenizer and models.
    # this section loads multiple parts of Stable Diffusion, read explainer
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Initialize the optimizer
    if use_8bit_adam:
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # using default parameters
    optimizer = optimizer_cls(
        unet.parameters(),
    )

    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            dataset_name,
        )
    else:
        data_files = {}
        if train_data_dir is not None:
            data_files["train"] = os.path.join(train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # Get the column names for input/target.
    # will depend upon the dataset
    dataset_columns = ("image","text")

    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids    
    
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            # transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["image"] = [train_transforms(image) for image in images]
        examples["text"] = tokenize_captions(examples)
        return examples
    
    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        image = torch.Tensor(examples[0]['image'])
        image = torch.stack([example["image"] for example in examples])
        image = image.to(memory_format=torch.contiguous_format).float()
        text = torch.stack([example["text"] for example in examples])
        return {"pixel_values": image, "text": text}
    
    # DataLoaders creation:
    # takes care of preprocessing and tokenization
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # what does this do?? haha
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # get learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["text"], return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
            unet = unwrap_model(unet)

            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
            )
            pipeline.save_pretrained(output_dir)

            # Run a final round of inference.
            images = []
            
    accelerator.end_training()

if __name__ == "__main__":
    main()
