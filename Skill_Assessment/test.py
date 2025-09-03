import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch import nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

proj_root = os.path.dirname(os.path.abspath(__file__))
ds_root = './data/jigsaws'

from dataset.skill_video_dataset import SkillVideoDataset
from utils.ImageShow import *
from utils import ShapingLoss

pt_save_root = os.path.join(proj_root, 'model_param')
visual_save_root = os.path.join(proj_root, 'group_vis_res')


def part_center_variance(part_centers):
    # part_centers: K x 256
    k, ch = part_centers.shape

    for i in range(k):
        for j in range(i + 1, k):
            variance = (part_centers[i] - part_centers[j]).pow(2).sum()
            print(f'Vairance btw {i}-th and {j}-th center: {variance.item():.3f}.')


def heatmap_regu(heatmaps, position_masks):
    bs, heatmap_t, _, heatmap_h, heatmap_w = heatmaps.shape
    bs, mask_t, mask_h, mask_w = position_masks.shape

    t_times = round(mask_t / heatmap_t)
    sampled_position_masks = position_masks.detach() if t_times == 1 else position_masks[:, ::t_times, :, :].detach()

    heatmap_regu_loss = -sampled_position_masks * (heatmaps.squeeze(2) + 1e-7).log().contiguous()
    heatmap_regu_loss = heatmap_regu_loss.mean()
    return heatmap_regu_loss


def train(args, split_index, device, save_label):
    print(f'--Task: {args.task}, Split Type: {args.val_split}, Split Index: {split_index} ...')
    if args.extractor in ['r2d', 'r2d_50', 'r2d_101']:
        frames_per_sample = 1
    else:
        frames_per_sample = 8 if '4layer' in args.extractor else 4
    num_samples = args.num_samples * frames_per_sample
    frames_per_timestep = 1

    if args.extractor in ['r2d', 'r2d_50', 'r2d_101']:
        from model_def.PartGroup_SkillNet2D import PartGroup_SkillNet2D
        model = PartGroup_SkillNet2D(args.num_parts, args.extractor, args.context, args.aggregate,
                                     args.avgpool_parts, args.scene_node, args.attention, args.multi_lstms,
                                     args.prepro, args.no_pastpro, args.simple_pastpro,
                                     final_score_bias=15, final_score_weight=25).to(device)
    else:
        from model_def.PartGroup_SkillNet3D import PartGroup_SkillNet3D
        model = PartGroup_SkillNet3D(args.num_parts, args.extractor, args.context, args.aggregate,
                                     args.avgpool_parts, args.scene_node, args.attention, args.multi_lstms,
                                     args.prepro, args.no_pastpro, args.simple_pastpro,
                                     final_score_bias=15, final_score_weight=25).to(device)

    return_position_masks = True if args.heatmap_regu_weight > 0 else False
    video_datasets = {x: SkillVideoDataset(ds_root, x == 'train', task=args.task, debug=args.debug,
                                           split_type=args.val_split, split_index=split_index,
                                           frames_per_timestep=frames_per_timestep,
                                           sampled_timestep_num=num_samples,
                                           balanced_train_sample=args.balanced_train_sample,
                                           noised_train_label=args.noised_train_label,
                                           train_sample_augment=args.train_sample_augment,
                                           test_sample_augment=args.test_sample_augment,
                                           return_position_masks=return_position_masks,
                                           score_norm_bias=0, score_norm_weight=1,
                                           ) for x in ['val']}
    print({x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['val']})
    # dataloaders = {x: DataLoader(video_datasets[x], batch_size=args.batch_size, shuffle=(x=='train'),
    #                   num_workers=128) for x in ['train', 'val']}
    batch_size_dict = {'val': 1}
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=batch_size_dict[x], shuffle=(x == 'train'),
                                 num_workers=36) for x in ['val']}

    if multi_gpu:
        print('Use', num_devices, 'GPUs!')
        model = nn.DataParallel(model, device_ids=list(range(num_devices)))
    model.load_state_dict(torch.load(args.pt_path))
    model.eval()

    criterion = nn.MSELoss()

    val_rho_history = []
    # for epoch in range(args.num_epochs):
    print('-' * 10)
    if args.num_parts > 1:
        part_center_variance(model.part_grouping.part_centers)
    # Each epoch has a training and validation phase
    for phase in ['val']:
        pred_scores = []
        gt_scores = []
        all_video_names = []
        running_loss = 0.0

        for samples in tqdm(dataloaders[phase]):
            inputs = samples[0].to(device)  # BxLx3x112x112
            labels = samples[1].to(device, dtype=torch.float)  # B
            batch_video_names = samples[2]

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                outputs, part_assigns, part_att = model(inputs)  # Bx1, BxLx3x7x7, BxLx3/BxLx1x14x14
                outputs = outputs.squeeze(1)  # B

                main_loss = criterion(outputs, labels)
                # print(f'main: {main_loss.item():.4f}')
                loss = main_loss

                if args.shaping_weight > 0:
                    part_assigns_ = torch.cat(part_assigns.unbind(dim=1), dim=0).contiguous()  # L*B x CxHxW
                    shaping_loss = ShapingLoss.ShapingLoss(
                        part_assigns_, radius=2,
                        std=0.4, num_parts=args.num_parts,
                        alpha=1, beta=0.001
                    ) * args.shaping_weight
                    # print(f'shaping: {shaping_loss.item():.4f}')
                    loss += shaping_loss  # + tconsist_loss #+ sconsist_loss

                if args.heatmap_regu_weight > 0:
                    position_masks = samples[4].to(device, dtype=torch.float)
                    heatmap_regu_loss = heatmap_regu(part_att, position_masks) * args.heatmap_regu_weight
                    # print(f'heatmap regu: {heatmap_regu_loss.item():.4f}')
                    loss += heatmap_regu_loss

            running_loss += loss.item() * inputs.size(0)
            pred_scores.append(torch.round(outputs))
            gt_scores.append(labels)
            all_video_names += batch_video_names

        pred_scores = torch.cat(pred_scores, dim=0).detach().to('cpu').numpy()
        gt_scores = torch.cat(gt_scores, dim=0).to('cpu').numpy()

        mae = mean_absolute_error(gt_scores, pred_scores)
        mse = mean_squared_error(gt_scores, pred_scores)

        epoch_rho, epoch_pvalue = stats.spearmanr(gt_scores, pred_scores)

        print(f'{phase} MAE: {mae:.4f} MSE: {mse:.4f} P-value: {epoch_pvalue:.4f}.')
        if phase == 'val':
            print('Predicted scores:', end=" ")
            [print(f'{pred_score:.2f}', end=" ") for pred_score in pred_scores]
            print()
            print('GT scores:       ', end=" ")
            [print(f'{gt_score:.2f}', end=" ") for gt_score in gt_scores]
            print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=str, default='r2p1d',
                        choices=['r2p1d', 'r2p1d_4layer', 'r3d', 'r3d_4layer', 'r2d', 'r2d_50', 'r2d_101'],
                        help='The type of CNN feature extractor. We defaulty use R(2+1)D.')
    parser.add_argument("--context", type=str, default='bilstm',
                        choices=['lstm', 'bilstm', 'gcn', 'transformer', 'none'],
                        help='The type temporal context modeling network. Default is bidirectional LSTMs.')
    parser.add_argument("--aggregate", type=str, default='avgpool',
                        choices=['mean', 'avgpool', 'final', 'lstm'],
                        help='Spatiotemporal aggregation mode. Default is avgpool.')

    parser.add_argument("--val_split", type=str, default='SuperTrialOut',
                        choices=['SuperTrialOut', 'UserOut', 'FourFolds'])
    parser.add_argument("--task", type=str, default='Suturing',
                        choices=['Suturing', 'Knot_Tying', 'Needle_Passing', 'Across'])
    parser.add_argument("--split_index", type=int, default=2)
    parser.add_argument("--pt_path", type=str,
                        default='./model_param/SkillSuturing_SuperTrialOut_r2p1d_bilstm_avgpool_3parts_np_sp_rt_sn_ap_shape10_lr3e-05_2.pt')
    parser.add_argument("--num_samples", type=int, default=32, help='Equals to T in the paper.')
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--multi_gpu", action='store_true')

    parser.add_argument("--randseed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--schedule_step", type=int, default=20)
    parser.add_argument("--scene_node", action='store_true',
                        help='Set to be True to be consistent with paper.', default=True)
    parser.add_argument("--num_parts", type=int, default=3)
    parser.add_argument("--no_pastpro", action='store_true', default=True)
    parser.add_argument("--shaping_weight", type=float, default=10, help='Default is 10.')
    parser.add_argument("--heatmap_regu_weight", type=float, default=0,
                        help='If >0, use positional regularization. We use 20 in the paper.')

    # Unfrequently used arguments
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--debug", action='store_true', default=False)  # not in use
    parser.add_argument("--attention", action='store_true', default=False)  # not in use
    parser.add_argument("--avgpool_parts", action='store_true', default=True)  # not in use
    parser.add_argument("--multi_lstms", action='store_true', default=False)
    parser.add_argument("--prepro", action='store_true', default=True)  # not in use
    parser.add_argument("--simple_pastpro", action='store_true', default=True)  # not in use
    parser.add_argument("--rolling_train", action='store_true', default=True)  # not in use
    parser.add_argument("--freeze_extractor", action='store_true', default=False)
    parser.add_argument("--freeze_half_extractor", action='store_true', default=False)  # not in use
    parser.add_argument("--freeze_central", action='store_true', default=False)  # not in use
    parser.add_argument("--init_extractor", action='store_true', default=False)  # not in use
    parser.add_argument("--tconsist_start_from", type=int, default=0)  # not in use
    parser.add_argument("--train_sample_augment", type=int, default=1)  # not in use
    parser.add_argument("--test_sample_augment", type=int, default=1)  # not in use
    parser.add_argument("--balanced_train_sample", action='store_true', default=True)  # not in use
    parser.add_argument("--noised_train_label", action='store_true', default=True)  # not in use

    parser.add_argument("--visualize", action='store_true',
                        help='If true, the assignment maps will be saved in ./group_vis_res file', default=True)
    parser.add_argument("--save_separately", action='store_true', default=True)
    parser.add_argument("--save_checkpoint", action='store_true', default=True)
    parser.add_argument("--extra_label", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.randseed)
    np.random.seed(args.randseed)
    torch.manual_seed(args.randseed)
    torch.cuda.manual_seed_all(args.randseed)

    model_type = f'{args.extractor}_{args.context}_{args.aggregate}'
    save_label = f"Skill{args.task}_{args.val_split}_{model_type}_{args.num_parts}parts"
    if args.no_pastpro:
        save_label += "_np"
    if args.simple_pastpro:
        save_label += "_sp"
    if args.multi_lstms:
        save_label += "_ml"
    if args.attention:
        save_label += "_att"
    if args.rolling_train:
        save_label += "_rt"
    if args.init_extractor:
        save_label += "_ie"
    if args.freeze_extractor:
        save_label += "_fe"
    if args.freeze_half_extractor:
        save_label += "_fhe"
    if args.freeze_central:
        save_label += "_fc"
    if args.scene_node:
        save_label += "_sn"
    if args.avgpool_parts:
        save_label += "_ap"

    if args.shaping_weight > 0:
        save_label += f"_shape{args.shaping_weight}"
    if args.heatmap_regu_weight > 0:
        save_label += f"_htmpregu{args.heatmap_regu_weight}"

    if args.tconsist_start_from > 0:
        save_label += f"_ts{args.tconsist_start_from}"
    save_label += f"_lr{args.learning_rate}"
    if args.extra_label != None:
        save_label += f"_{args.extra_label}"
    print(save_label)

    multi_gpu = args.multi_gpu
    num_devices = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_nums = {'SuperTrialOut': 5, 'UserOut': 8, 'FourFolds': 4}
    split_num = split_nums[args.val_split]

    if args.split_index == 0:
        split_ids = list(range(1, split_num + 1))
        if args.task == 'Needle_Passing':
            split_ids.remove(6)
        print(f'Run all {split_num} split.')
    elif args.split_index in list(range(1, split_num + 1)):
        split_ids = [args.split_index]
        print(f'Only run the {args.split_index}-th split.')
        if args.task == 'Needle_Passing' and args.split_index == 6:
            print(f'Needle_Passing task has no split 6. Skip.')
    else:
        raise Exception(f'Given split index is wrong: {args.split_index}')

    rhos, l1s = [0] * len(split_ids), [0] * len(split_ids)
    for i, split_id in enumerate(split_ids):
        train(args, split_id, device, save_label)



