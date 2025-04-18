# AttUNet for Drosophila Cardiac Analysis

Trained an Attention UNet to segment Drosophila hearts in cardiac recordings from from optical microscopy. I pretrain the AttUNet loosely following the methodology outlined in the paper *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2022). 


## Future Work

I am currently working on replicating this work with Vision Transformers following *Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles* (Ryali et al., 2023) I also need to do the following

- scale lr with n_gpus (lr should be param in config)
- integrate new data balancing approach in MAEHeartDataset
- train with elastic deform augmentation for longer
- create animations...

