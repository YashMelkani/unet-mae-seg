import os
import glob
import random
import yaml

from concurrent.futures import ThreadPoolExecutor

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer, seed_everything

from torch.utils.data import ConcatDataset, DataLoader

from att_unet import AttUNet
from dataset import MAEHeartDataset, SEGHeartDataset

def create_mae_dataset(vid_path, n_frames):
    return MAEHeartDataset(vid_path, n_frames=n_frames, balance=True, augment=True) # seeded by lightning seed_everything method

def create_seg_dataset(vid_path, n_frames):
    return SEGHeartDataset(vid_path, n_frames=n_frames, balance=True, augment=True) # seeded by lightning seed_everything method

def init_mae_loaders(config, seed=0):
    
    base_dir = config['base_dir']
    n_frames = config['n_frames']
    train_split = config['train_split']
    batch_size = config['bs']
        
    dirs = ['YASH/YASH/Misc/AHA Obesity/',
            'YASH/Videos/',
            'GIRISH/ALF & TRF 5 WEEK SCIENCE 2015/',
            'GIRISH/Dilated Cardiomyopathy Model/',
            'SHARED_VIDEOS/machine_learning/'
           ]
    
    vid_paths = []
    
    datasets = []
    for d in dirs:
        for vid_path in glob.glob(os.path.join(base_dir, d, '**/*.avi'), recursive = True):
            
            if os.path.getsize(vid_path) < 1e6: # ignore corrupted videos
                continue
            if 'GM_Uploaded' in vid_path and 'Laminopathy Project' not in vid_path: # do not include the videos used for segmentation model evaluation (laminopathy study vids are fine)
                continue
            vid_paths.append(vid_path)
    
    n_vids = len(vid_paths)
    # random.seed(seed) # seeded by lightning seed_everything method
    random.shuffle(vid_paths)
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(create_mae_dataset, vid_paths, [n_frames] * n_vids)
        
    datasets = list(results)
    
    n_train_vids = int(n_vids * train_split)
    print('N train vids:', n_train_vids)
    print('N val vids:', n_vids - n_train_vids)
    
    train_dataset = ConcatDataset(datasets[:n_train_vids])
    val_dataset = ConcatDataset(datasets[n_train_vids:])
                
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, vid_paths[:n_train_vids], vid_paths[n_train_vids:]
 
def init_seg_loaders(config, seed=0):
    base_dir = config['base_dir']
    n_frames = config['n_frames']
    train_split = config['train_split']
    batch_size = config['bs']
    
    vid_paths = []
    
    content = os.listdir(base_dir)
    for f in content:
        f_path = os.path.join(base_dir, f)
        if os.path.isdir(f_path):
            vid_paths.append(f_path)

    vid_paths = sorted(vid_paths)
    n_vids = len(vid_paths)
    # random.seed(seed) # seeded by lightning seed_everything method
    random.shuffle(vid_paths)
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(create_seg_dataset, vid_paths, [n_frames] * n_vids)
    
    datasets = list(results)
    
    n_train_vids = int(n_vids * train_split)
    print('N train vids:', n_train_vids)
    print('N val vids:', n_vids - n_train_vids)
        
    train_dataset = ConcatDataset(datasets[:n_train_vids])
    val_dataset = ConcatDataset(datasets[n_train_vids:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, vid_paths[:n_train_vids], vid_paths[n_train_vids:]
    
@rank_zero_only
def save_meta_files(config, train_vids, val_vids, task):
    
    save_dir = config['train']['save_dir']
    name = config['train']['name']
    
    save_dir = os.path.join(save_dir, name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    #save config
    path = os.path.join(save_dir, f'{task}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    #save training/validation vids
    path = os.path.join(save_dir, f'{task}_vids.txt')
    f = open(path, 'w')
    
    f.write(f'TRAINING (n = {len(train_vids)})\n')
    for vid in train_vids:
        f.write(f'{vid}\n')
    f.write('\n')
    
    f.write(f'VALIDATION (n = {len(val_vids)})\n')
    for vid in val_vids:
        f.write(f'{vid}\n')
    
    f.close()
    
def init_model(config, task='mae', last_ckpt = None):
    
    if task == 'mae':
        
        if last_ckpt is None: # init from scratch
            model = AttUNet(img_ch = config['n_input_channels'],
                            output_ch = config['n_output_channels'])

        else: # resume from last ckpt
            model = AttUNet.load_from_checkpoint(last_ckpt)
    
        model.set_training_task('mae')
        
    elif task == 'seg':
        
        pretrain_ckpt = config['pretrain_ckpt']
        
        if last_ckpt is None and pretrain_ckpt is None: # init from scratch
            model = AttUNet(img_ch = config['n_input_channels'],
                            output_ch = config['n_output_channels'])

        elif last_ckpt is None: # init model from pretraining checkpoint
            model = AttUNet.load_from_checkpoint(pretrain_ckpt)
            
        else: # resume from last ckpt
            model = AttUNet.load_from_checkpoint(last_ckpt)
    
        model.set_training_task('seg')
    
    return model


def train(config, model, train_loader, val_loader, last_ckpt=None):
    
    # world_size = int(os.environ['WORLD_SIZE'])
    world_size = 2
    n_node = max(1, world_size // 4) # returns 1 if <= 4 gpus else, for 8, 12, 16 ... will return correct # of nodes
    
    name = config['name']
    save_dir = config['save_dir']
    epochs = config['epochs']
    val_freq = config['val_freq']
    
    # callbacks
    best_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:05d}",
    )
    curr_ckpt_callback = ModelCheckpoint(
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    logger = CSVLogger(save_dir, name=name)
    
    trainer = Trainer(devices = min(world_size, 4), # up-to 4 devices per node 
                      num_nodes = n_node, 
                      strategy = 'ddp',
                      check_val_every_n_epoch = val_freq, 
                      max_epochs = epochs,
                      accelerator = "gpu", 
                      callbacks = [best_ckpt_callback, curr_ckpt_callback, lr_monitor],
                      logger = logger,
                      enable_progress_bar = False)
    
    print(f"Starting training with {world_size} GPUs over {n_node} nodes")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=last_ckpt)
    
    print("Training completed")
    
if __name__ == '__main__':
   
    config_path = './configs/seg.yaml'
    
    last_ckpt = None
    # last_ckpt = './results/'
    
    seed_everything(0, workers=True)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    task = config['task']
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    
    if task == 'mae':
        train_loader, val_loader, train_vids, val_vids = init_mae_loaders(data_config, seed=0)
    elif task == 'seg':
        train_loader, val_loader, train_vids, val_vids = init_seg_loaders(data_config, seed=0)
    
    save_meta_files(config, train_vids, val_vids, task)
    
    model = init_model(model_config, task=task, last_ckpt=last_ckpt)
    
    train(train_config, model, train_loader, val_loader)
    
    
