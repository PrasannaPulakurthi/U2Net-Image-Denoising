from abc import abstractmethod, ABC


class Config(ABC):
    def __init__(self):
        args = self.parse_args()
        
        self.img_size = args.img_size
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.noise_std = args.noise_std
        self.data_dir = args.data_dir

        self.model_name = args.model_name
        self.loss_type = args.loss_type
        self.load_epoch = args.load_epoch

        self.exp_name = args.exp_name
        self.output_dir = args.output_dir
        self.model_dir = args.model_dir
        self.train_img_dir = args.train_img_dir
        self.test_img_dir = args.test_img_dir
        
        self.gpu = args.gpu
        

    @abstractmethod
    def parse_args(self):
        raise NotImplementedError

