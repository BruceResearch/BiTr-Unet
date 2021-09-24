
config = dict()

config["base_path"] = "../"
config["initial_learning_rate"] = 1e-4
# config["image_shape"] = (128, 192, 160)
config["normalizaiton"] = "group_normalization"
config["mode"] = "trilinear"
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["loss_k1_weight"] = 0.1
config["loss_k2_weight"] = 0.1
config["focal_alpha"] = 0.99
config["focal_gamma"] = 2
# config["data_path"] = config["base_path"] + "data/MICCAI_BraTS_2018_Data_Training"
config["data_path"] = config["base_path"] + "data/MICCAI_BraTS2020_TrainingData"
config["training_patients"] = []
config["validation_patients"] = []
# augmentation
config["intensity_shift"] = True
config["scale"] = True
config["flip"] = True
config["L2_norm"] = 1e-5
config["patience"] = 5
config["lr_decay"] = 0.7
config["checkpoint"] = True  # Boolean. If True, will save the best model as checkpoint.
config["label_containing"] = True  # Boolean. If True, will generate label with overlapping.
config["VAE_enable"] = True  # Boolean. If True, will enable the VAE module.
config["focal_enable"] = False  # Boolean. If True, will enable the focal loss.
if config["focal_enable"]:
    config["initial_learning_rate"] *= 2