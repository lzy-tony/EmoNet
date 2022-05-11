class Config:
    name = "Text-CNN"
    batch_size = 64
    lr = 1e-3

    emb_dim = 50
    kernels = [2, 3, 4, 5]
    kernel_num = 100
    dropout_rate = 0.4
    weight_decay = 5e-3
    class_num = 2
