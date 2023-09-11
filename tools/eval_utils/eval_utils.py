import pickle
import random
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_one_epoch_al(cfg, args, model, dataloader, epoch_id, logger, it, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    # al temp
    frame_uncertainty_list = []
    frame_uncertainty_dict = {}

    # cal weight if true
    if args.weight:
        weight_car, weight_ped, weight_cyc = cal_weight(cfg, it, logger)

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            # embeddings = pred_dicts[0]['embeddings'].view(-1, 128, cfg.MODEL.ROI_HEAD.SHARED_FC[-1])

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        # Insert AL selection


        # For each batch, Cal Entropy for each frame, Save them to fud.
        for x in range(len(annos)):
            frame_uncertainty = 0
            for y in range(annos[x]['name'].shape[0]):
                cur_prob = annos[x]['score'][y]
                if args.strategy == 'Entropy':
                    if args.aggregation == 'Avg':
                        frame_uncertainty += -cur_prob * np.log2(cur_prob) / annos[x]['name'].shape[0]
                    elif args.aggregation == 'Sum':
                        if args.weight:
                            if annos[x]['name'][y] == 'Car':
                                frame_uncertainty += -cur_prob * np.log2(cur_prob) * weight_car
                            elif annos[x]['name'][y] == 'Pedestrian':
                                frame_uncertainty += -cur_prob * np.log2(cur_prob) * weight_ped
                            elif annos[x]['name'][y] == 'Cyclist':
                                frame_uncertainty += -cur_prob * np.log2(cur_prob) * weight_cyc
                    elif args.aggregation == 'Max':
                        frame_uncertainty = max(frame_uncertainty, -cur_prob * np.log2(cur_prob))
                elif args.strategy == 'Least_conf' and args.strategy == "Random":
                    if args.aggregation == 'Avg':
                        frame_uncertainty += cur_prob / annos[x]['name'].shape[0]
                    elif args.aggregation == 'Sum':
                        if args.weight:
                            if annos[x]['name'][y] == 'Car':
                                frame_uncertainty += cur_prob * weight_car
                            elif annos[x]['name'][y] == 'Pedestrian':
                                frame_uncertainty += cur_prob * weight_ped
                            elif annos[x]['name'][y] == 'Cyclist':
                                frame_uncertainty += cur_prob * weight_cyc
                        else:
                            frame_uncertainty += cur_prob
                    elif args.aggregation == 'Max':
                        if frame_uncertainty == 0:
                            frame_uncertainty = cur_prob
                        else:
                            frame_uncertainty = min(frame_uncertainty, cur_prob)
                # elif args.strategy == 'CoreSet':
                #     embeddings = annos[x]['']


            frame_uncertainty_dict[annos[x]['frame_id']] = frame_uncertainty
    # true >, false <

    if args.strategy == "Entropy":
        frame_uncertainty_dict_sorted = sorted(frame_uncertainty_dict.items(), key=lambda d: d[1], reverse=True)
    elif args.strategy == "Least_conf":
        frame_uncertainty_dict_sorted = sorted(frame_uncertainty_dict.items(), key=lambda d: d[1], reverse=False)
    elif args.strategy == "Random":
        frame_uncertainty_dict_sorted = sorted(frame_uncertainty_dict.items(), key=lambda d: d[1], reverse=False)
        random.shuffle(frame_uncertainty_dict_sorted)

    # Add frame id to list.
    for i in range(int(args.ratio * 3712)):
        frame_uncertainty_list.append(frame_uncertainty_dict_sorted[i][0])

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict, frame_uncertainty_list

def eval_one_epoch_al_coreset(cfg, args, model, dataloader, epoch_id, logger, it,
                              labeled_dataloader,
                              dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    # al temp
    frame_uncertainty_list = []
    frame_uncertainty_dict = {}
    labeled_embeddings = []
    unlabeled_embeddings = []
    unlabeled_frame_ids = []

    # cal weight if true
    if args.weight:
        weight_car, weight_ped, weight_cyc = cal_weight(cfg, it, logger)

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval_labeled', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            for batch_idx in range(len(pred_dicts)):
                unlabeled_frame_ids.append(pred_dicts[batch_idx])
            embeddings = pred_dicts[batch_idx]['embeddings'].view(1, -1)
            unlabeled_embeddings.append(embeddings)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        # Insert AL selection


        # For each batch, Cal Entropy for each frame, Save them to fud.

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # Feed Labeled data
    print("Start Eval Labeled data.")
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(labeled_dataloader), leave=True, desc='eval_labeled', dynamic_ncols=True)
    for i, batch_dict in enumerate(labeled_dataloader):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            for batch_idx in range(len(pred_dicts)):
                unlabeled_frame_ids.append(pred_dicts[batch_idx])
                embeddings = pred_dicts[0]['embeddings'].view(1, -1)
            labeled_embeddings.append(embeddings)

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    print("********Start CoreSet Searching...********")
    selected_idx = furthest_first(torch.cat(unlabeled_embeddings, 0), torch.cat(labeled_embeddings, 0), int(3712 * args.ratio))

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    selected_list = []
    for idx in selected_idx:
        selected_list.append(str(idx.item()).zfill(6))
    return ret_dict, selected_list

def cal_weight(cfg, it, logger):
    db_infos = {}
    len_car = 0
    len_ped = 0
    len_cyc = 0
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    class_names = cfg.CLASS_NAMES
    for class_name in class_names:
        db_infos[class_name] = []

    # Read pkl file
    db_info_dir = cfg.ROOT_DIR / 'data' / 'kitti'

    with open(str(db_info_dir / 'kitti_dbinfos_train_labeled.pkl'), 'rb') as f:
        infos = pickle.load(f)
        [db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]


    # filter
    new_db_infos = {}
    removed_difficulty = [-1]
    min_gt_points_list = ['Car:5', 'Pedestrian:5', 'Cyclist:5']
    for key, dinfos in db_infos.items():
        pre_len = len(dinfos)
        new_db_infos[key] = [
            info for info in dinfos
            if info['difficulty'] not in removed_difficulty
        ]
        logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))


    for name_num in min_gt_points_list:
        name, min_num = name_num.split(':')
        min_num = int(min_num)
        if min_num > 0 and name in new_db_infos.keys():
            filtered_infos = []
            for info in new_db_infos[name]:
                if info['num_points_in_gt'] >= min_num:
                    filtered_infos.append(info)


            logger.info('Database filter by min points %s: %d => %d' % (name, len(new_db_infos[name]), len(filtered_infos)))
            new_db_infos[name] = filtered_infos
            if name == 'Car':
                len_car = len(filtered_infos)
            elif name == 'Pedestrian' :
                len_ped = len(filtered_infos)
            elif name == 'Cyclist':
                len_cyc = len(filtered_infos)

    ratio_car = (len_car + len_ped + len_cyc) / len_car
    ratio_ped = (len_car + len_ped + len_cyc) / len_ped
    ratio_cyc = (len_car + len_ped + len_cyc) / len_cyc
    return ratio_car, ratio_ped, ratio_cyc

def furthest_first(x1, x2, n):
    selected = []
    x1 = x1.view(x1.shape[0], -1)
    x2 = x2.view(x2.shape[0], -1)
    min_distance = squared_distance(x1, x2).mean(1)
    for i in range(n):
        idx = torch.argmax(min_distance)
        selected.append(idx)
        if i < (n - 1):
            distance = squared_distance(x1, x1[idx, :].unsqueeze(0))
            for j in range(x1.shape[0]):
                min_distance[j] = min(min_distance[j], distance[j, 0])
    return selected

def squared_distance(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x_norm = (x ** 2).sum(1).unsqueeze(1)
    y_norm = (y ** 2).sum(1).unsqueeze(0)
    xy = -2.0 * torch.matmul(x, y.t())
    dist = x_norm + y_norm + xy
    return torch.clamp(dist, min=0.0)


if __name__ == '__main__':
    pass
