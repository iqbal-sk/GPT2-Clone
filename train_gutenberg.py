import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch
import math
import wandb

from GPTModel import GPTModel

from utils.generation import generate_and_print_sample
from utils.loss import calc_loss_batch, evaluate_model, plot_losses
from utils.training import create_dataloaders

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer, initial_lr=3e-05,
                       min_lr=1e-6,batch_size=1024, train_ratio=0.90, global_step=-1,
                       trained_epochs_so_far=0, warmup_steps=None, total_training_steps=None):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen = 0

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]
    wandb.watch(model, log="all", log_freq=eval_freq)

    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over the books in the training corpus
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # Initialize new data loaders for each book
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_355M["context_length"],
                    stride=GPT_CONFIG_355M["context_length"],
                    num_workers=0
                )

                if warmup_steps is None:
                    # Calculate the total number of iterations in the training process
                    total_training_steps = len(train_loader) * n_epochs * len(all_files)
                    warmup_steps = int(total_training_steps * 0.15)

                    wandb.config.update({"warmup steps": warmup_steps})
                    wandb.config.update({"total_training_steps": total_training_steps})

                    # Calculate the learning rate increment during the warmup phase
                    lr_increment = (peak_lr - initial_lr) / warmup_steps

                print("Training ...")
                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    global_step += 1

                    # Adjust the learning rate based on the current phase (warmup or cosine annealing)
                    if global_step < warmup_steps:
                        # Linear warmup
                        lr = initial_lr + global_step * lr_increment
                    else:
                        # Cosine annealing after warmup
                        progress = ((global_step - warmup_steps) /
                                    (total_training_steps - warmup_steps))
                        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

                    # Apply the calculated learning rate to the optimizer
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    track_lrs.append(lr)  # Store the current learning rate

                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()

                    if global_step >= warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


                    optimizer.step()
                    tokens_seen += input_batch.numel()

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {trained_epochs_so_far+epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                        wandb.log({
                            "global_step": global_step,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "learning_rate": lr,
                            "tokens_seen": tokens_seen
                        })

                    # Generate text passage
                    if global_step % print_sample_iter == 0:
                        sample_output = generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )
                        wandb.log({"sample_output": sample_output, "global_step": global_step})

                if global_step % save_ckpt_freq:

                    checkpoint = {
                        "global_step": global_step,
                        "warmup_steps": warmup_steps,
                        "total_training_steps": total_training_steps,
                        "epoch": trained_epochs_so_far+epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    ckpt_path = output_dir / f"checkpoint_{global_step:06d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    print(f"Checkpoint saved at {ckpt_path}")

                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        checkpoint = {
            "global_step": global_step,
            "warmup_steps": warmup_steps,
            "total_training_steps": total_training_steps,
            "epoch": trained_epochs_so_far+epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, file_name)
        print(f"Checkpoint saved at {file_name}")

    # Save final checkpoint (model and optimizer).
    final_checkpoint = {
        "global_step": global_step,
        "warmup_steps": warmup_steps,
        "total_training_steps": total_training_steps,
        "trained_epochs_so_far": trained_epochs_so_far + n_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(final_checkpoint, output_dir / "final_checkpoint.pt")

    return train_losses, val_losses, track_tokens_seen, track_lrs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--data_dir', type=str, default='data/sample_train_data',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='Path to a saved model checkpoint to load')
    parser.add_argument('--load_optimizer_path', type=str, default=None,
                        help='Path to a saved optimizer checkpoint to load')

    args = parser.parse_args()

    # Initialize Weights & Biases logging.
    wandb.init(project="GPT-Training", config={
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "eval_freq": args.eval_freq,
        "save_ckpt_freq": args.save_ckpt_freq
    })
    config = wandb.config

    if args.debug:
        GPT_CONFIG_355M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,    # Context length
            "emb_dim": 768,           # Embedding dimension
            "n_heads": 12,            # Number of attention heads
            "n_layers": 12,           # Number of layers
            "drop_rate": 0.1,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_355M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 1024,          # Embedding dimension
            "n_heads": 16,           # Number of attention heads
            "n_layers": 24,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    device = get_device()
    print(f'Using {device} for training..')

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_355M)
    model.to(device)

    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if param.ndim >= 2:  # Apply weight decay only to parameters with dimension >= 2
            params_with_decay.append(param)
        else:
            params_without_decay.append(param)

    param_groups = [{'params': params_with_decay, 'weight_decay': 0.1},
                    {'params': params_without_decay, 'weight_decay': 0.0}]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    tokenizer = tiktoken.get_encoding("gpt2")

    global_step = -1
    trained_epochs_so_far = 0
    warmup_steps = None
    total_training_steps = None

    # Optionally load existing checkpoints.
    if args.load_model_path is not None:
        if os.path.exists(args.load_model_path):
            model_state = torch.load(args.load_model_path, map_location=device)
            # In case the checkpoint is a dict containing additional metadata:
            if isinstance(model_state, dict) and "model_state_dict" in model_state:
                model.load_state_dict(model_state["model_state_dict"])
                global_step = model_state["global_step"]
                trained_epochs_so_far = model_state['trained_epochs_so_far']
                warmup_steps = model_state['warmup_steps']
                total_training_steps = model_state['total_training_steps']
            else:
                model.load_state_dict(model_state)
            print(f"Loaded model state from {args.load_model_path}")
        else:
            print(f"Model checkpoint path {args.load_model_path} does not exist.")

    if args.load_optimizer_path is not None:
        if os.path.exists(args.load_optimizer_path):
            opt_state = torch.load(args.load_optimizer_path, map_location=device)
            # In case the checkpoint is a dict containing additional metadata:
            if isinstance(opt_state, dict) and "optimizer_state_dict" in opt_state:
                optimizer.load_state_dict(opt_state["optimizer_state_dict"])
            else:
                optimizer.load_state_dict(opt_state)
            print(f"Loaded optimizer state from {args.load_optimizer_path}")
        else:
            print(f"Optimizer checkpoint path {args.load_optimizer_path} does not exist.")


    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    train_losses, val_losses, tokens_seen, lrs = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        global_step=global_step,
        trained_epochs_so_far=trained_epochs_so_far,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps
    )

    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)


    print(f"Final checkpoint saved in {output_dir}")

    # Finalize the W&B run.
    wandb.finish()