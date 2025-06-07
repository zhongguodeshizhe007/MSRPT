# coding=utf-8
import torch
import torch.nn as nn
# coding=utf-8
import copy
from models.modules import Block
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import models.configs as configs

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.sd_len
        num_tee = config.tee_len

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.sd_embeddings = Linear(768, config.hidden_size)  
        self.tee_embeddings = Linear(1, config.hidden_size)  
        self.sex_embeddings = Linear(1, config.hidden_size)  
        self.age_embeddings = Linear(1, config.hidden_size)
        self.TypeOfPlace_embeddings = Linear(1, config.hidden_size)
        self.Mode_embeddings = Linear(1, config.hidden_size)
        self.Frequency_embeddings = Linear(1, config.hidden_size)
        self.Distance_embeddings = Linear(1, config.hidden_size)
        self.NumOfAccidents_embeddings = Linear(1, config.hidden_size)
        self.NumOfAnger_embeddings = Linear(1, config.hidden_size)
        self.DSLHW_embeddings = Linear(1, config.hidden_size)
        self.DSLRR_embeddings = Linear(1, config.hidden_size)
        self.DrivingClose_embeddings = Linear(1, config.hidden_size)
        self.BeatDriver_embeddings = Linear(1, config.hidden_size)
        self.SoundingHorn_embeddings = Linear(1, config.hidden_size)
        self.UseEarpiece_embeddings = Linear(1, config.hidden_size)  
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+n_patches, config.hidden_size))
        self.pe_sd = nn.Parameter(torch.zeros(1, tk_lim, config.hidden_size))
        self.pe_tee = nn.Parameter(torch.zeros(1, num_tee, config.hidden_size))
        self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_TypeOfPlace = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_Mode = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_Frequency = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_Distance = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_NumOfAccidents = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_NumOfAnger = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_DSLHW = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_DSLRR = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_DrivingClose = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_BeatDriver = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_SoundingHorn = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_UseEarpiece = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_sd = Dropout(config.transformer["dropout_rate"])
        self.dropout_tee = Dropout(config.transformer["dropout_rate"])
        self.dropout_sex = Dropout(config.transformer["dropout_rate"])
        self.dropout_age = Dropout(config.transformer["dropout_rate"])
        self.dropout_TypeOfPlace = Dropout(config.transformer["dropout_rate"])
        self.dropout_Mode = Dropout(config.transformer["dropout_rate"])
        self.dropout_Frequency = Dropout(config.transformer["dropout_rate"])
        self.dropout_Distance = Dropout(config.transformer["dropout_rate"])
        self.dropout_NumOfAccidents = Dropout(config.transformer["dropout_rate"])
        self.dropout_NumOfAnger = Dropout(config.transformer["dropout_rate"])
        self.dropout_DSLHW = Dropout(config.transformer["dropout_rate"])
        self.dropout_DSLRR = Dropout(config.transformer["dropout_rate"])
        self.dropout_DrivingClose = Dropout(config.transformer["dropout_rate"])
        self.dropout_BeatDriver = Dropout(config.transformer["dropout_rate"])
        self.dropout_SoundingHorn = Dropout(config.transformer["dropout_rate"])
        self.dropout_UseEarpiece = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, sd, tee, sex, age, TypeOfPlace, Mode, Frequency,Distance,NumOfAccidents,NumOfAnger,DSLHW,DSLRR,DrivingClose,BeatDriver,SoundingHorn,UseEarpiece):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x) # 16*16 --> CNN --> 1*1
        sd = self.sd_embeddings(sd)
        tee = self.tee_embeddings(tee)
        sex = self.sex_embeddings(sex)
        age = self.age_embeddings(age)
        TypeOfPlace = self.TypeOfPlace_embeddings(TypeOfPlace)
        Mode = self.Mode_embeddings(Mode)
        Frequency = self.Frequency_embeddings(Frequency)
        Distance = self.Distance_embeddings(Distance)
        NumOfAccidents = self.NumOfAccidents_embeddings(NumOfAccidents)
        NumOfAnger = self.NumOfAnger_embeddings(NumOfAnger)
        DSLHW = self.DSLHW_embeddings(DSLHW)
        DSLRR = self.DSLRR_embeddings(DSLRR)
        DrivingClose = self.DrivingClose_embeddings(DrivingClose)
        BeatDriver = self.BeatDriver_embeddings(BeatDriver)
        SoundingHorn = self.SoundingHorn_embeddings(SoundingHorn)
        UseEarpiece = self.UseEarpiece_embeddings(UseEarpiece)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        sd_embeddings = sd + self.pe_sd
        tee_embeddings = tee + self.pe_tee
        sex_embeddings = sex + self.pe_sex
        age_embeddings = age + self.pe_age
        TypeOfPlace_embeddings = TypeOfPlace + self.pe_TypeOfPlace
        Mode_embeddings = Mode + self.pe_Mode
        Frequency_embeddings = Frequency + self.pe_Frequency
        Distance_embeddings = Distance + self.pe_Distance
        NumOfAccidents_embeddings = NumOfAccidents + self.pe_NumOfAccidents
        NumOfAnger_embeddings = NumOfAnger + self.pe_NumOfAnger
        DSLHW_embeddings = DSLHW + self.pe_DSLHW
        DSLRR_embeddings = DSLRR + self.pe_DSLRR
        DrivingClose_embeddings = DrivingClose + self.pe_DrivingClose
        BeatDriver_embeddings = BeatDriver + self.pe_BeatDriver
        SoundingHorn_embeddings = SoundingHorn + self.pe_SoundingHorn
        UseEarpiece_embeddings = UseEarpiece + self.pe_UseEarpiece

        embeddings = self.dropout(embeddings)
        sd_embeddings = self.dropout_sd(sd_embeddings)
        tee_embeddings = self.dropout_tee(tee_embeddings)
        sex_embeddings = self.dropout_sex(sex_embeddings)
        age_embeddings = self.dropout_age(age_embeddings)
        TypeOfPlace_embeddings = self.dropout_TypeOfPlace(TypeOfPlace_embeddings)
        Mode_embeddings = self.dropout_Mode(Mode_embeddings)
        Frequency_embeddings = self.dropout_Frequency(Frequency_embeddings)
        Distance_embeddings = self.dropout_Distance(Distance_embeddings)
        NumOfAccidents_embeddings = self.dropout_NumOfAccidents(NumOfAccidents_embeddings)
        NumOfAnger_embeddings = self.dropout_NumOfAnger(NumOfAnger_embeddings)
        DSLHW_embeddings = self.dropout_DSLHW(DSLHW_embeddings)
        DSLRR_embeddings = self.dropout_DSLRR(DSLRR_embeddings)
        DrivingClose_embeddings = self.dropout_DrivingClose(DrivingClose_embeddings)
        BeatDriver_embeddings = self.dropout_BeatDriver(BeatDriver_embeddings)
        SoundingHorn_embeddings = self.dropout_SoundingHorn(SoundingHorn_embeddings)
        UseEarpiece_embeddings = self.dropout_UseEarpiece(UseEarpiece_embeddings)
        return embeddings, sd_embeddings, tee_embeddings, sex_embeddings, age_embeddings,TypeOfPlace_embeddings,Mode_embeddings,Frequency_embeddings,Distance_embeddings,NumOfAccidents_embeddings,NumOfAnger_embeddings,DSLHW_embeddings,DSLRR_embeddings,DrivingClose_embeddings,BeatDriver_embeddings,SoundingHorn_embeddings,UseEarpiece_embeddings


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i < 2:
                layer = Block(config, vis, mm=True)
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, text=None):
        attn_weights = []
        
        for (i, layer_block) in enumerate(self.layer):
            if i == 2:
                hidden_states = torch.cat((hidden_states, text), 1)  
                hidden_states, weights = layer_block(hidden_states)
            elif i < 2:
                hidden_states, text, weights = layer_block(hidden_states, text)
            else:
                hidden_states, weights = layer_block(hidden_states)

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights