import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch.nn as nn
from concurrent import futures
import time
from multiprocessing import current_process
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    executor = futures.ThreadPoolExecutor(max_workers=2)

    from utils.rewards import jpeg_compressibility
    reward_fn = jpeg_compressibility()

    if opt['phase'] == 'train':
        optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            diffusion.netG.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )

        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break

                samples = []

                diffusion.netG.train()
                if isinstance(diffusion.netG, nn.DataParallel):
                    diffusion.feed_data(train_data)
                    latents, log_probs, timesteps = diffusion.netG.module.p_sample_loop_with_logprob(diffusion.data['SR'], continous=True)
                    print(diffusion.data['SR'].size())
                    images = latents[-1]  # the final images
                    latents = torch.stack(
                        latents, dim=1
                    )  # (batch_size, num_steps + 1, channels, width, height)
                    log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
                    timesteps = torch.stack(timesteps, dim=1)  # (batch_size, num_steps)

                    # compute rewards asynchronously
                    rewards = executor.submit(reward_fn, images, None, None)
                    # yield to to make sure reward computation starts
                    time.sleep(0)

                    samples.append(
                        {
                            "timesteps": timesteps,
                            "next_timesteps": torch.cat(
                                (timesteps[:, 1:], torch.full((timesteps.shape[0], 1), -1).to(diffusion.device)),
                                dim=1),
                            "latents": latents[
                                       :, :-1
                                       ],  # each entry is the latent before timestep t
                            "next_latents": latents[
                                            :, 1:
                                            ],  # each entry is the latent after timestep t
                            "log_probs": log_probs,
                            "rewards": rewards,
                        }
                    )

                    # wait for all rewards to be computed
                    for sample in tqdm(
                            samples,
                            desc="Waiting for rewards",
                            disable=not rewards.done(),
                            position=0,
                    ):
                        rewards, reward_metadata = sample["rewards"].result()
                        sample["rewards"] = torch.as_tensor(rewards, device=diffusion.device)

                    rewards_mean = rewards.mean()
                    rewards_std = rewards.std()

                    # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
                    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

                    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    samples["advantages"] = torch.as_tensor(advantages).to(diffusion.device)

                    total_batch_size, num_timesteps = samples["timesteps"].shape

                    for e in range(10):
                        # shuffle samples along batch dimension
                        perm = torch.randperm(total_batch_size, device=diffusion.device)
                        samples = {k: v[perm] for k, v in samples.items()}

                        # shuffle along time dimension independently for each sample
                        perms = torch.stack(
                            [
                                torch.randperm(num_timesteps, device=diffusion.device)
                                for _ in range(total_batch_size)
                            ]
                        )
                        for key in ["timesteps", "next_timesteps", "latents", "next_latents", "log_probs"]:
                            samples[key] = samples[key][
                                torch.arange(total_batch_size, device=diffusion.device)[:, None],
                                perms,
                            ]

                        # rebatch for training
                        samples_batched = {
                            k: v.reshape(-1, opt['datasets']['train']['batch_size'], *v.shape[1:])
                            for k, v in samples.items()
                        }

                        # dict of lists -> list of dicts for easier iteration
                        samples_batched = [
                            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
                        ]

                        num_train_timesteps = num_timesteps

                        total_loss = 0
                        for i, sample in tqdm(
                            list(enumerate(samples_batched)),
                            position=0,
                        ):
                            advantages = torch.clamp(
                                sample["advantages"],
                                -5,
                                5,
                            ).float()

                            for j in tqdm(
                                range(num_train_timesteps),
                                desc="Timestep",
                                position=1,
                                leave=False,
                            ):
                                _, log_prob = diffusion.netG.module.p_sample_with_logprob(
                                    sample["latents"][:, j], sample["timesteps"][0, j],
                                    condition_x=diffusion.data['SR'],
                                    prev_x=sample["next_latents"][:, j]
                                )
                                ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                                unclipped_loss = -advantages * ratio
                                clipped_loss = -advantages * torch.clamp(
                                    ratio,
                                    1.0 - 1e-4,
                                    1.0 + 1e-4,
                                )
                                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                                total_loss += loss.item()

                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()

                    # log
                    if current_step % opt['train']['print_freq'] == 0:
                        logs = diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            tb_logger.add_scalar(k, v, current_step)
                        logger.info(message)

                        logger.info(f"")

                        if wandb_logger:
                            wandb_logger.log_metrics(logs)
                            wandb_logger.log_metrics({
                                "rewards_mean": rewards_mean,
                                "steps": current_step,
                            })
                            wandb_logger.log_metrics({
                                "rewards_std": rewards_std,
                                "steps": current_step,
                            })
                            wandb_logger.log_metrics({
                                "loss": total_loss / (len(samples_batched) * num_train_timesteps),
                                "steps": current_step
                            })

                    # validation
                    if current_step % opt['train']['val_freq'] == 0:
                        avg_psnr = 0.0
                        idx = 0
                        result_path = '{}/{}'.format(opt['path']
                                                     ['results'], current_epoch)
                        os.makedirs(result_path, exist_ok=True)

                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['val'], schedule_phase='val')
                        for _, val_data in enumerate(val_loader):
                            idx += 1
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                            # generation
                            Metrics.save_img(
                                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                            avg_psnr += Metrics.calculate_psnr(
                                sr_img, hr_img)

                            if wandb_logger:
                                wandb_logger.log_image(
                                    f'validation_{idx}',
                                    np.concatenate((fake_img, sr_img, hr_img), axis=1)
                                )

            break
