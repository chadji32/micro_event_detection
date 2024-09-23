class Config:
    def __init__(self, path="Dataset", image_size=128, step_size=10, clip_duration=30, num_frames=5, num_epochs=10, lr=0.0001, batch_size=16, augmentation=None, model="model/model_no_aug.pt", log_file="model/logs_no_aug.txt", prediction_file="model/predictions_no_aug.csv", save_roc_path="models/roc_no_aug.png"):
        self.path = path
        self.image_size = image_size
        self.step_size = step_size
        self.clip_duration = clip_duration
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.augmentation = augmentation if augmentation is not None else []
        self.model = model
        self.log_file = log_file
        self.prediction_file = prediction_file
        self.save_roc_path = save_roc_path
        

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
