import argparse
from dataset_utils.img_seq import load_data, add_datasets_args
from dataset_utils.img_seq_universal_datamodule import UniversalDataModule 
import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.strategies.ddp import DDPStrategy
from ckpt_utils.img_seq_rec_universal_checkpoints import UniversalCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model_utils.img_seq_optimizer_utils import add_optimizer_args, configure_optimizers, get_total_steps
from models.img_seq_rec_base import Img_Seq_Rep_Rec, Linear_pred, Img_title_fuse_Rec, Img_title_fuse_Rec_VanillTrans, Img_title_fuse_Rec_VanillCrossTrans
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import pickle
import torch.nn as nn
import numpy as np
import os
import logging
import time
import random
import torch.backends.cudnn as cudnn
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string)


def cal_recall(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def recalls_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_recall(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['Recall@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def calculate_metrics(scores, labels, metric_ks):
    metrics = recalls_and_ndcgs_k(scores, labels, metric_ks)
    return metrics


class Collator():
    def __init__(self, args):
        self.img_load_path = '../data/' + args.datasets_name + '/img/'
        self.len_seq = args.seq_len
        
        with open('../data/' + args.datasets_name + '/asin2id.pickle', 'rb') as f:
            self.id_asin_dict = pickle.load(f)

        self.prob_like = len(self.id_asin_dict)+1
        
    def __call__(self, inputs):
        examples = []
        for input_id_temp in inputs:
            input_temp = list(input_id_temp.values())[0]
            id_temp = list(input_id_temp.keys())[0]
            
            example = {}
            seqs_temp = input_temp[-(self.len_seq+1):]
            seq_temp = seqs_temp[:-1]
            
            label = seqs_temp[-1]
            seq_temp_id = [self.id_asin_dict[i]+1 for i in seq_temp]
            label_id = self.id_asin_dict[label] + 1
            seq_temp_id_pad = [0] * (self.len_seq - len(seq_temp_id)) + seq_temp_id
            mask_temp = [0] * (self.len_seq - len(seq_temp_id)) + len(seq_temp_id) * [1]
            
            example['seq_id'] =  seq_temp_id_pad
            example['mask_seq'] = mask_temp 
            example['label_id'] = label_id
            example['sample_id'] = id_temp
            example['preprob'] = torch.zeros([self.prob_like])
            examples.append(example)
        return default_collate(examples)


class Seq_img_Rec(LightningModule):
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Seq img Rec')
        parser.add_argument('--seq_len', type=int, default=10)
        parser.add_argument('--hidden_size', type=int, default=768)
        parser.add_argument('--attn_head', type=int, default=16)
        parser.add_argument('--n_blocks', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.8)
        parser.add_argument('--fuse_gl_flag', type=bool, default=False)
        parser.add_argument('--info', type=str, default=None)
        parser.add_argument('--emb_title_model', type=str, default=None)
        parser.add_argument('--emb_img_model', type=str, default=None)
        parser.add_argument('--emb_title_train', type=bool, default=True)
        parser.add_argument('--emb_img_train', type=bool, default=True)
        return parent_parser
    
    def __init__(self, args, nums_sample, logger):
        super().__init__()
        
        with open('../data/' + args.datasets_name + '/pairs_freq.pickle', 'rb') as f:
            self.pairs_freq = pickle.load(f)

        emb = torch.load('../data/' + args.datasets_name + '/' + args.emb_img_model)
        pad_emb = torch.mean(emb, dim=0)
        embs = torch.cat([pad_emb.unsqueeze(0), emb], dim=0)
        self.item_emb_img = torch.nn.Embedding.from_pretrained(embs)
        self.item_emb_img.weight.requires_grad = args.emb_img_train
        
   
        emb_title = torch.load('../data/' + args.datasets_name + '/' + args.emb_title_model)
        pat_emb_title = torch.mean(emb_title, dim=0)
        embs_title = torch.cat([pat_emb_title.unsqueeze(0), emb_title], dim=0)
        self.item_emb_title = torch.nn.Embedding.from_pretrained(embs_title)
        self.item_emb_title.weight.requires_grad = args.emb_title_train
        
        
        self.loss_weight = torch.zeros(nums_sample)
        self.seq_rep_model = Img_title_fuse_Rec(args) 
        # self.seq_rep_model = Img_title_fuse_Rec_VanillTrans(args)  ## Vanilla Transformer
        # self.seq_rep_model = Img_title_fuse_Rec_VanillCrossTrans(args)  ## Vanilla Transformer cross attention
        
        self.pred_layer = Linear_pred(args, self.item_emb_img.weight.shape[0])
        self.penalty_loss = True
        self.kl_loss_aug = nn.KLDivLoss(log_target=True)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_ind = nn.CrossEntropyLoss(reduction='none')
        self.alpha_1 = 0.05 
        self.alpha_2 = 0.001
        self.alpha_3 = 0.001
        self.logger_save = logger
        self.save_hyperparameters(args)
        # self.apply(self._init_weights)
        
    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))
    
    def configure_optimizers(self):
        return configure_optimizers(self)
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def loss_weight_ce(self, id_samples, prod, gd):
        _, ind_pre = torch.topk(prod, k=5)
        flag = (gd.unsqueeze(1) == ind_pre).float().sum(dim=-1).to(prod.device)
        self.loss_weight = self.loss_weight.to(prod.device)
        self.loss_weight[id_samples] = self.loss_weight[id_samples] + flag
        tau = 0.07
        loss_weigh_penty = 1 - torch.softmax(self.loss_weight/tau, dim=0)[id_samples]
        loss_ce = self.loss_ce_ind(prod, gd)
        loss_ind = loss_weigh_penty * loss_ce
        return torch.mean(loss_ind)
    
    def loss_norm_2(self):
        return torch.norm(self.item_emb_img.weight, p=2)
    
    def pos_neg_gen(self,):
        pos_items = []
        for i in range(self.item_emb_img.weight.shape[0]-1):  
            if i in self.pairs_freq:
                pos_list = self.pairs_freq[i]
                pos_list_item = [j[0] for j in pos_list]
                pos_list_prob = [j[1] for j in pos_list]
                pos_item_idx = random.choices(list(range(len(pos_list_prob))), weights=pos_list_prob, k=1)[0]
                pos_items.append(pos_list_item[pos_item_idx]+1)
            else:
                pos_items.append(i+1)
        return pos_items
    
    def loss_uniformity_alignment(self):
        pos_items = self.pos_neg_gen()
        items_emb = self.item_emb_img.weight[1:, :]
        pos_items_emb = self.item_emb_img(torch.tensor(pos_items).long().to(items_emb.device))
        align_loss = torch.norm(items_emb - pos_items_emb, dim=-1).mean()**2
        
        return align_loss
        
    def training_step(self, batch, batch_idx):
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_img_pt = torch.stack(batch['seq_id'], dim=1)
        
        seq_rep_img = self.item_emb_img(seq_img_pt)
        seq_rep_title = self.item_emb_title(seq_img_pt)
       
        rep_seq = self.seq_rep_model(seq_rep_img, seq_rep_title, masks)
        prod = self.pred_layer(rep_seq)
        
        if self.penalty_loss:    
            loss = self.loss_weight_ce(batch['sample_id'], prod, batch['label_id'].long())
        else:
            loss = self.loss_ce(prod, batch['label_id'].long())
        
        seq_rep_img_aug = seq_rep_img + torch.randn_like(seq_rep_img)
        rep_seq_aug = self.seq_rep_model(seq_rep_img_aug, seq_rep_title, masks)
        prod_aug = self.pred_layer(rep_seq_aug)
        prod_aug = F.log_softmax(prod_aug, dim=-1)
        prod = F.log_softmax(prod, dim=-1)
        loss_kl = self.kl_loss_aug(prod, prod_aug).mean(-1)
        loss_align = self.loss_uniformity_alignment()
        # loss = loss + self.alpha_1*torch.exp(loss_kl)
        ## 
        loss = loss + self.alpha_1*torch.exp(loss_kl) + self.alpha_2*torch.log(self.loss_norm_2())+self.alpha_3*torch.log(loss_align)
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        if self.trainer.global_rank == 0 and self.global_step == 100:
            report_memory('Seq rec')
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        self.seq_rep_model.eval()
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_img_pt = torch.stack(batch['seq_id'], dim=1)
        seq_rep_img = self.item_emb_img(seq_img_pt)
        seq_rep_title = self.item_emb_title(seq_img_pt)
        rep_seq = self.seq_rep_model(seq_rep_img, seq_rep_title, masks)
        prod = self.pred_layer(rep_seq)
        
        metrics = calculate_metrics(prod, batch['label_id'].unsqueeze(1), metric_ks=[5, 10, 20, 50])
        self.log("Metrics", metrics)
        return {"metrics":  metrics}
    
    def validation_epoch_end(self, validation_step_outputs) -> None:
        print('validation_epoch_end')
        metrics_all = self.all_gather(validation_step_outputs)
        val_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'Recall@50': [], 'NDCG@50': []}
        val_metrics_dict_mean = {}
        for temp in metrics_all:
            for key_temp, val_temp in temp['metrics'].items():
                val_metrics_dict[key_temp].append(torch.mean(val_temp).cpu().item())

        for key_temp, values_temp in val_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            val_metrics_dict_mean[key_temp] = values_mean
        print(val_metrics_dict_mean)
        self.log("Val_Metrics", val_metrics_dict_mean)
        self.log("Recall@10", val_metrics_dict_mean['Recall@10'])
        self.logger_save.info("Val Metrics: {}".format(val_metrics_dict_mean))
        self.logger_save.info("Recall@10: {}".format(val_metrics_dict_mean))
                
    def test_step(self,  batch, batch_idx):
        self.seq_rep_model.eval()
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_img_pt = torch.stack(batch['seq_id'], dim=1)
        seq_rep_img = self.item_emb_img(seq_img_pt)
        seq_rep_title = self.item_emb_title(seq_img_pt)
      
        rep_seq = self.seq_rep_model(seq_rep_img, seq_rep_title, masks)
        prod = self.pred_layer(rep_seq)
        metrics = calculate_metrics(prod, batch['label_id'].unsqueeze(1), metric_ks=[5, 10, 20, 50])
        
        return {"metrics":  metrics}

    def test_epoch_end(self, test_step_outputs) -> None:
        print('test_epoch_end')
        test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'Recall@50': [], 'NDCG@50': []}
        test_metrics_dict_mean = {}
        metrics_all = self.all_gather(test_step_outputs)
        for temp in metrics_all:
            for key_temp, val_temp in temp['metrics'].items():
                test_metrics_dict[key_temp].append(torch.mean(val_temp).cpu().item())

        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print(test_metrics_dict_mean)
        self.logger_save.info("Test Metrics: {}".format(test_metrics_dict_mean))
        self.logger_save.info("Recall@10: {}".format(test_metrics_dict_mean['Recall@10']))
        self.log("Test_Metrics", test_metrics_dict_mean)
        self.log("Recall@10", test_metrics_dict_mean['Recall@10'])
        

def main():
    args_parser = argparse.ArgumentParser()
    args_parser = add_optimizer_args(args_parser)
    args_parser = add_datasets_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = Seq_img_Rec.add_module_specific_args(args_parser)
    custom_parser = [
        '--datasets_path_train', '../data/Sports/imgs_seq_5_train.pickle',
        '--datasets_path_test', '../data/Sports/imgs_seq_5_test.pickle',
        '--datasets_path_val', '../data/Sports/imgs_seq_5_val.pickle',
        '--datasets_name', 'Sports',
        '--train_batchsize', '128',
        '--val_batchsize', '128',
        '--test_batchsize', '128',
        '--seq_len', '10',
        '--info', 'Liner pred, emb nofinetune, fuse global local seq rep',
        '--learning_rate', '5e-4',
        '--min_learning_rate', '5e-5',
        '--fuse_gl_flag', 'False', 
        '--random_seed', '512',
        '--dropout', '0.8',
        '--emb_title_model', 'title_emb_clip_vit_large_patch14_len10.pt',
        '--emb_img_model', 'img_emb_clip_vit_large_patch14.pt',
        '--emb_title_train', 'True',
        '--emb_img_train', 'True',
        ]  
    args = args_parser.parse_args(args=custom_parser)
    fix_random_seed_as(args.random_seed)
    
    if not os.path.exists('../log/'):
        os.makedirs('../log/')
    if not os.path.exists('../log/' + args.datasets_name):
        os.makedirs('../log/' + args.datasets_name)
    logging.basicConfig(level=logging.INFO, filename='../log/' + args.datasets_name + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
    logger = logging.getLogger(__name__)
    print(args.info)
    logger.info(args.info)
    print(args)
    logger.info(args)
    
    datasets = load_data(args)
    
    
    collate_fn = Collator(args)
    datamodule = UniversalDataModule(collate_fn=collate_fn, args=args, datasets=datasets)
    
    checkpoint_callback = UniversalCheckpoint(args)
    early_stop_callback_step = EarlyStopping(monitor='Recall@10', min_delta=0.00, patience=3, verbose=False, mode='max')
    trainer = Trainer(devices=1, accelerator="gpu", strategy=DDPStrategy(find_unused_parameters=True), callbacks=[checkpoint_callback, early_stop_callback_step], max_epochs=50,  check_val_every_n_epoch=1)
    
    model = Seq_img_Rec(args, datasets['train'].__len__(), logger)  
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule) 


if __name__ == "__main__":
    main()
