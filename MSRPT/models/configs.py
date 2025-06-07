import ml_collections

def get_MSRPT_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072 # 根据实际情况设置
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.2
    config.transformer.dropout_rate = 0.3 
    config.classifier = 'token'
    config.representation_size = None
    config.sd_len = 20
    config.tee_len = 9
    return config
