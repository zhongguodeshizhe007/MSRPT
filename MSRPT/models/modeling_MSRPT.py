# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import torch
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn as nn
import numpy as np
from scipy import ndimage
import models.configs as configs
from models.embed import Embeddings 
from models.embed import Encoder
from models.modules import Block

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, sd=None, tee=None, sex=None, age=None, TypeOfPlace=None, Mode=None, Frequency=None,Distance=None,NumOfAccidents=None,NumOfAnger=None,DSLHW=None,DSLRR=None,DrivingClose=None,BeatDriver=None,SoundingHorn=None,UseEarpiece=None):
        embedding_output, sd, tee, sex, age, TypeOfPlace, Mode, Frequency,Distance,NumOfAccidents,NumOfAnger,DSLHW,DSLRR,DrivingClose,BeatDriver,SoundingHorn,UseEarpiece = self.embeddings(input_ids, sd, tee, sex, age, TypeOfPlace, Mode, Frequency,Distance,NumOfAccidents,NumOfAnger,DSLHW,DSLRR,DrivingClose,BeatDriver,SoundingHorn,UseEarpiece)
        text = torch.cat((sd, tee, sex, age,TypeOfPlace, Mode, Frequency,Distance,NumOfAccidents,NumOfAnger,DSLHW,DSLRR,DrivingClose,BeatDriver,SoundingHorn,UseEarpiece), 1)
        encoded, attn_weights = self.encoder(embedding_output, text)
        return encoded, attn_weights


class MSRPT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=5, zero_head=False, vis=False):
        super(MSRPT, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, sd=None, tee=None, sex=None, age=None, TypeOfPlace=None, Mode=None, Frequency=None,Distance=None,NumOfAccidents=None,NumOfAnger=None,DSLHW=None,DSLRR=None,DrivingClose=None,BeatDriver=None,SoundingHorn=None,UseEarpiece=None,labels=None):
        x, attn_weights = self.transformer(x, sd, tee, sex, age,TypeOfPlace, Mode, Frequency,Distance,NumOfAccidents,NumOfAnger,DSLHW,DSLRR,DrivingClose,BeatDriver,SoundingHorn,UseEarpiece)
        logits = self.head(torch.mean(x, dim=1))

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.float())
            return loss
        else:
            return logits, attn_weights, torch.mean(x, dim=1)

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            # print(posemb.size(), posemb_new.size())
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'MSRPT': configs.get_MSRPT_config(),
}
