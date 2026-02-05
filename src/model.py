import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

from torchmetrics.functional import retrieval_average_precision, retrieval_precision

from src.coprompt import MultiModalPromptLearner, Adapter, TextEncoder
from src.utils import load_clip_to_cpu #, retrieval_average_precision
from src.losses import loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_model(m):
    m.requires_grad_(False)
    
def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
            
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, classnames, clip_model, clip_model_distill
    ):
        super().__init__()
        clip_model.apply(freeze_all_but_bn)
        clip_model_distill.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype
        self.prompt_learner_photo = MultiModalPromptLearner(cfg, classnames, clip_model, type='photo')
        self.prompt_learner_sketch = MultiModalPromptLearner(cfg, classnames, clip_model, type='sketch')
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        self.adapter_photo = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        
        self.model_distill = clip_model_distill
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1
    
    def get_logits(self, img_tensor, type='photo'):
        if type=='photo':
            prompt_learner = self.prompt_learner_photo
        else:
            prompt_learner = self.prompt_learner_sketch
        tokenized_prompts = prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = prompt_learner()
        
        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        ) # (n_classes, 512)
        
        image_features = self.image_encoder(
                img_tensor.type(self.dtype), shared_ctx, deep_compound_prompts_vision
            ) # (batch_size, 768)
            
        x_a = self.adapter_photo(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # image_features = F.normalize(image_features, dim=-1)
        # text_features = F.normalize(text_features, dim=-1)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        return logits, image_features_normalize, image_features
        
    def forward(self, x):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        pos_logits, photo_features_norm, photo_feature = self.get_logits(photo_tensor)
        sk_logits, sk_feature_norm, sk_feature = self.get_logits(sk_tensor, type='sketch')
        _, neg_feature, _ = self.get_logits(neg_tensor)
        
        return photo_features_norm, sk_feature_norm, photo_aug_tensor, sk_aug_tensor, \
            neg_feature, label, pos_logits, sk_logits, photo_feature, sk_feature
            
    def extract_feature(self, image, type='photo'):
        _, feature, _ = self.get_logits(image, type=type)
        return feature
            
class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        clip_model = load_clip_to_cpu(args)
        
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(args, design_details=design_details)
        
        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_metric = 1e-3
        
        self.model = CustomCLIP(cfg=args, classnames=classname, clip_model=clip_model, clip_model_distill=clip_model_distill)
    
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1
        )
        
        return [optimizer] , [scheduler]
    
    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        features = self.forward(batch)
        
        loss, loss_dict = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        for k, v in loss_dict.items():
            self.log(k, v, on_step=False, on_epoch=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, type='sketch')
            self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, type='photo')
            self.val_step_outputs_ph.append((feat, label))
    
    def on_validation_epoch_end(self):
        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)
        
        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])
        
        all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))
        
        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        precision = torch.zeros(len(query_feat_all))
        
        # Track per-class Average Precision
        class_ap = {}
        class_count = {}
        
        if self.args.dataset == "sketchy_2":
            map_k = 200
            p_k = 200
        else:
            map_k = 0
            if self.args.dataset == "quickdraw":
                p_k = 200
            else:
                p_k = 100
                
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            distance = self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
            target[np.where(all_photo_category == category)] = True
            
            if map_k != 0:
                top_k_actual = min(map_k, len(gallery)) 
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=top_k_actual)
            else: 
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
            
            # Track per-class AP
            if category not in class_count:
                class_count[category] = 0
                class_ap[category] = 0.0
            class_count[category] += 1
            class_ap[category] += ap[idx].item()
                
            precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=p_k)
            # tot_pos = int(target.sum().item())
            # top_k_actual = min(p_k, tot_pos)
            # idx_pos = distance.topk(top_k_actual, largest=True, sorted=True, dim=-1).indices
            # target_filtered = target[idx_pos]
            # precision[idx] = target_filtered.float().mean()
            
        mAP = torch.mean(ap)
        precision = torch.mean(precision)
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        
        print(f"\nEpoch: {self.current_epoch + 1}/{self.trainer.max_epochs}")
        
        if map_k != 0:
            print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
        else:
            print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
        train_loss = self.trainer.callback_metrics.get("train_loss", None)

        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")
            
        keys = ["loss_triplet", "loss_photo_skt", "loss_distill", "loss_ce", "loss_mcc"]
        for k in keys:
            loss_val = self.trainer.callback_metrics.get(k, None)
            if loss_val is not None:
                print(f"{k}: {loss_val.item():.6f}")
        
        # Analyze per-class performance (Average Precision)
        class_mean_ap = {cat: (class_ap[cat] / class_count[cat]) if class_count[cat] > 0 else 0 
                         for cat in class_count.keys()}
        
        # Sort by AP (ascending) to find worst performing classes
        sorted_classes = sorted(class_mean_ap.items(), key=lambda x: x[1])
        
        print("\nTop 10 classes with lowest AP:")
        for i, (cat_idx, mean_ap) in enumerate(sorted_classes[:10]):
            class_name = self.classname[cat_idx] if cat_idx < len(self.classname) else f"Class_{cat_idx}"
            print(f"  {i+1}. {class_name}: AP={mean_ap:.4f} ({class_count[cat_idx]} samples)")
        
        # Force flush stdout
        import sys
        sys.stdout.flush()

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()