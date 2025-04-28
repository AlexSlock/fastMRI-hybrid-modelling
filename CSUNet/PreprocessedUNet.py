import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from preprocessed_transforms import UnetDataTransform
from modified_unet_module import UnetModule
from preprocessed_data_module import FastMriDataModule


# python PreprocessedUNet.py --mode train --challenge multicoil --mask_type equispaced --center_fractions 0.08 0.04 --accelerations 4 8 --num_workers 8
# always change num of gpus to actual reservated amount!!
def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    ################################################################################
    # Modified to be compatible with preprocessed CS data in other directory #
    ################################################################################
    train_transform = UnetDataTransform(args.challenge)
    val_transform = UnetDataTransform(args.challenge)
    test_transform = UnetDataTransform(args.challenge)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        bart_path=args.bart_path, # added
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    if args.mode == "train":
        model = UnetModule(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
        )
        trainer.fit(model, datamodule=data_module)
        print("Training complete.")
    elif args.mode == "test":
        model = UnetModule.load_from_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
        )
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("fastmri_dirs.yaml")
    num_gpus = 3
    backend = "ddp"
    batch_size = 1 #if backend == "ddp_cpu" else num_gpus  #(just always set to 1 for now)

    # set defaults based on optional directory config  
    data_path = fetch_dir("data_path", path_config) # /path/to/NYU_fastMRI
    bart_path = fetch_dir("bart_path", path_config) # ADDED
    default_root_dir = fetch_dir("log_path", path_config)

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser)
    # CHANGED
    parser.set_defaults(data_path=data_path, bart_path=bart_path, batch_size=batch_size, test_path=None)

    # module config
    # this is where Unet architecture variables are defined (passed to unet_module then!)
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net (input = magnitude image)
        out_chans=1,  # number of output chanenls to U-Net (output = RSS image)
        chans=32,  # number of top-level U-Net channels (start of # filters, dubbels every pooling)
        num_pool_layers=4,  # number of U-Net pooling layers (depth of network)
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0005,  # weight decay (L2) regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = default_root_dir
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=default_root_dir,
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()


