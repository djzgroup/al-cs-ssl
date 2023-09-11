import os

import torch
import tqdm
import time
import glob
import random

import numpy as np


from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.datasets.kitti.kitti_dataset_AL import create_kitti_infos
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.eval_utils.eval_utils import eval_one_epoch_al, eval_one_epoch_al_coreset



def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)
            
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item() , batch_size)
            
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )
                    
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
            
            # save intermediate ckpt every {ckpt_save_time_interval} seconds         
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1
                
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, 
                
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

def train_model_al(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                   cfg, args, output_dir, dist,
                   train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                   merge_all_iters_to_one_epoch=False, use_logger_to_record=False,
                   logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False, use_amp=False):
    for it in range(args.iteration):
        logger.info('******************Start Active Learning (%s)********************' % (it + 1))
        # When al_iter > 0
        if it > 0:
            train_set, train_loader, train_sampler = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=False, workers=args.workers,
                logger=logger,
                training=True,
                merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
                total_epochs=args.epochs,
                seed=666 if args.fix_random_seed else None,
                if_labeled='train_labeled',
                iter_al=it
            )
            # Build Model && optimizer && scheduler
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batch_norm(model)
            model.cuda()

            optimizer = build_optimizer(model, cfg.OPTIMIZATION)

            model.train()

            if dist:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
            logger.info(model)

            lr_scheduler, lr_warmup_scheduler = build_scheduler(
                optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
                last_epoch=-1, optim_cfg=cfg.OPTIMIZATION
            )

        # Find the bound of object.
        max_vol = 0
        progress_bar = tqdm.tqdm(total=len(train_loader), leave=True, desc='Traverse', dynamic_ncols=True)
        for i ,batch_dict in enumerate(train_loader):
            batch_max = (batch_dict['gt_boxes'][:, :, 3] * batch_dict['gt_boxes'][:, :, 4] * batch_dict['gt_boxes'][:, :, 5]).max()
            max_vol = max_vol if max_vol > batch_max else batch_max
        progress_bar.close()
        cfg.MODEL.ROI_HEAD.MAX_VOL = max_vol

        logger.info('******************Active Learning (%s)********************' % (it + 1))
        with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
            total_it_each_epoch = len(train_loader)
            dataloader_iter = iter(train_loader)
            accumulated_iter = start_iter
            for cur_epoch in tbar:
                if train_sampler is not None:
                    train_sampler.set_epoch(cur_epoch)
                # Train one epoch.
                if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                    cur_scheduler = lr_warmup_scheduler
                else:
                    cur_scheduler = lr_scheduler
                accumulated_iter = train_one_epoch(
                    model, optimizer, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,
                    cur_epoch=cur_epoch, total_epochs=total_epochs,
                    use_logger_to_record=use_logger_to_record,
                    logger=logger, logger_iter_interval=logger_iter_interval,
                    ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                    show_gpu_stat=show_gpu_stat
                )
                trained_epoch = cur_epoch + 1

                if trained_epoch == total_epochs and it != args.iteration-1:
                    logger.info('******************Start Active Learning********************')
                    test_set, test_loader, sampler = build_dataloader(
                        dataset_cfg=cfg.DATA_CONFIG,
                        class_names=cfg.CLASS_NAMES,
                        batch_size=args.batch_size,
                        dist=dist, workers=args.workers, logger=logger, training=False,
                        if_labeled='train_unlabeled', iter_al=it
                    )
                    eval_output_dir = output_dir / 'eval' / 'eval_with_train_AL'
                    eval_output_dir.mkdir(parents=True, exist_ok=True)
                    _, uncertainty_list = eval_one_epoch_al(cfg, args, model, test_loader, cur_epoch, logger, it,
                                   dist_test=False, result_dir=eval_output_dir)
                    # _, uncertainty_list = eval_one_epoch_al_coreset(cfg, args, model, test_loader, cur_epoch, logger, it,
                    #                                         train_loader, dist_test=False, result_dir=eval_output_dir)
                    # Active Selection
                    if it != args.iteration - 1:
                        active_select(args, cfg, it, uncertainty_list)
                        # Update pkl && gtbase
                        import yaml
                        from pathlib import Path
                        from easydict import EasyDict
                        dataset_cfg = EasyDict(yaml.load(
                            open('/workspace/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset_AL.yaml')))
                        PROJ_DIR = (Path(__file__).resolve().parent / '../../').resolve()
                        create_kitti_infos(
                            dataset_cfg=dataset_cfg,
                            class_names=['Car', 'Pedestrian', 'Cyclist'],
                            data_path=PROJ_DIR / 'data' / 'kitti',
                            save_path=PROJ_DIR / 'data' / 'kitti',
                            iter_al=(it + 1)
                        )

                if trained_epoch % total_epochs == 0 and rank == 0:
                    ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)
                    # if ckpt_list.__len__() >= max_ckpt_save_num:
                    #     for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    #         os.remove(ckpt_list[cur_file_idx])
                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d_%d' % (trained_epoch, it + 1))
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
                # ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                # ckpt_list.sort(key=os.path.getmtime)
                # ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d_%d' % (trained_epoch, it + 1))
                # save_checkpoint(
                #     checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                # )

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)

def active_select(args, cfg, it, modify_list=None):
    """
    Through evaluation Update train_labeled.txt & train_unlabeled.txt.
    :param args: args.
    :param cfg: cfgs.
    :param it: number of iteration.
    :param modify_list: return of evaluation, list of uncertain frame
    :return: NULL
    """
    kitti_imagesets_dir = cfg.ROOT_DIR / 'data' / 'kitti' / 'ImageSets'
    if it == 0: # The first iteration
        kitti_labeled_txt_dir = kitti_imagesets_dir / 'train_labeled.txt'
        kitti_unlabeled_txt_dir = kitti_imagesets_dir / 'train_unlabeled.txt'
    else:
        kitti_labeled_txt_dir = kitti_imagesets_dir / ('train_labeled_' + str(it) + '.txt')
        kitti_unlabeled_txt_dir = kitti_imagesets_dir / ('train_unlabeled_' + str(it) + '.txt')
    labeled_list = np.loadtxt(kitti_labeled_txt_dir, dtype=str)
    unlabeled_list = np.loadtxt(kitti_unlabeled_txt_dir, dtype=str)

    if args.strategy == 'Random':
        # Random sampling
        active_idx = random.sample(range(0, unlabeled_list.shape[0]), int(args.ratio * 3712))
        active_list = np.take(unlabeled_list, active_idx)
        # labeled data
        labeled_list = sorted(np.concatenate((labeled_list, active_list), axis=0))
        np.savetxt(kitti_imagesets_dir / ('train_labeled_' + str(it + 1) + '.txt'), np.array(labeled_list), fmt='%s')
        # unlabeled data
        unlabeled_list = np.delete(unlabeled_list, active_idx)
        np.savetxt(kitti_imagesets_dir / ('train_unlabeled_' + str(it + 1) + '.txt'), np.array(unlabeled_list), fmt='%s')
    elif args.strategy == 'Entropy':
        active_list = np.array(modify_list)
        # labeled data
        labeled_list = sorted(np.concatenate((labeled_list, active_list), axis=0))
        np.savetxt(kitti_imagesets_dir / ('train_labeled_' + str(it + 1) + '.txt'), np.array(labeled_list), fmt='%s')
        # unlabeled data
        unlabeled_list = np.setdiff1d(unlabeled_list, active_list)
        np.savetxt(kitti_imagesets_dir / ('train_unlabeled_' + str(it + 1) + '.txt'), np.array(unlabeled_list), fmt='%s')
    elif args.strategy == 'Least_conf':
        active_list = np.array(modify_list)
        # labeled data
        labeled_list = sorted(np.concatenate((labeled_list, active_list), axis=0))
        np.savetxt(kitti_imagesets_dir / ('train_labeled_' + str(it + 1) + '.txt'), np.array(labeled_list), fmt='%s')
        # unlabeled data
        unlabeled_list = np.setdiff1d(unlabeled_list, active_list)
        np.savetxt(kitti_imagesets_dir / ('train_unlabeled_' + str(it + 1) + '.txt'), np.array(unlabeled_list), fmt='%s')
    elif args.strategy == 'CoreSet':
        active_list = np.array(modify_list)
        # labeled data
        labeled_list = sorted(np.concatenate((labeled_list, active_list), axis=0))
        np.savetxt(kitti_imagesets_dir / ('train_labeled_' + str(it + 1) + '.txt'), np.array(labeled_list), fmt='%s')
        # unlabeled data
        unlabeled_list = np.setdiff1d(unlabeled_list, active_list)
        np.savetxt(kitti_imagesets_dir / ('train_unlabeled_' + str(it + 1) + '.txt'), np.array(unlabeled_list), fmt='%s')
