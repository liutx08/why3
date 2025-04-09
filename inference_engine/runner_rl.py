# runner_rl.py

import argparse
import pickle
import os, sys
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from datetime import datetime, date
from CovaGen_rl_guide.guided_diffusion.resample import create_named_schedule_sampler
from CovaGen_rl_guide.guided_diffusion import dist_util, logger
from CovaGen_rl_guide.guided_diffusion.script_util import (
	NUM_CLASSES,
	model_and_diffusion_defaults,
	classifier_defaults,
	create_model_and_diffusion,
	create_classifier,
	add_dict_to_argparser,
	args_to_dict,
)
from CovaGen_rl_guide.guided_diffusion.qed_models import Net, Net2, condtimeNet, reg_Net,reg_Net_newt,deeper_cls_net
from CovaGen_rl_guide.optdiffusion.model import Dynamics_samp, Dynamics_t_samp, Dynamics_t_samp2, Dynamics_t,simple_reg_Net_newt,Dynamics_t_esm_deeper
from CovaGen_rl_guide.guided_diffusion.pl_datasets import load_data_smi,load_data_esm
from CovaGen_rl_guide.guided_diffusion.train_util import TrainLoopRL
sys.path.append(os.path.dirname(sys.path[0]))

def run_rl_sample(arg_dict):
    args = argparse.Namespace(**arg_dict)

    dist_util.setup_dist()
    logger.configure(dir='./logs/')
    logger.log("creating model and diffusion...")

    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    _, prior_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    agent_model = Dynamics_t_esm_deeper(
        condition_dim=28, target_dim=128, hid_dim=64,
        condition_layer=3, n_heads=2,
        condition_time=True, sampling=True
    )
    prior_model = Dynamics_t_esm_deeper(
        condition_dim=28, target_dim=128, hid_dim=64,
        condition_layer=3, n_heads=2,
        condition_time=True, sampling=True
    )

    agent_model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    prior_model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    agent_model.to(dist_util.dev())
    prior_model.to(dist_util.dev())
    agent_model.train()
    prior_model.eval()

    classifier = deeper_cls_net()
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    dat = load_data_esm(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_dir=args.vae_dir,
        dataset_save_path=args.dataset_save_path,
        data_state="train",
        sequence=args.protein_seq
    )

    TLLOOP = TrainLoopRL(
        model=agent_model,
        prior=prior_model,
        diffusion=diffusion,
        data=dat,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        scoring_model=classifier
    )

    diffusion.insert_rlloop(TLLOOP)

    for batch in dat:
        model_kwargs = {"y": th.tensor(np.ones((100,), dtype=np.int64)).to(dist_util.dev())}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            agent_model,
            (100, 128),
            batch=batch,
            cond_fn=cond_fn,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True,
            saving_path=args.save_path,
        )

        with open(args.save_path, "wb") as f:
            pickle.dump(sample, f)

        return sample  # 返回 sample 给 app.py 用

