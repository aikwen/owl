from owl.core import dataset
from owl.utils import img_aug
from owl.utils import types
from pathlib import Path

if __name__ == '__main__':
    transform_configs = [
        types.ResizeConfig(width=512, height=512, p = 1),
        types.RotateConfig(p=0),
        types.VFlipConfig(p=0),
        types.HFlipConfig(p=0),
        types.JpegConfig(quality_low=70, quality_high=100, p=0),
        types.GblurConfig(kernel_low=3, kernel_high=15, p=0),
    ]
    transform = img_aug.aug_compose(transform_configs)

    loader = dataset.create_dataloader([Path('example')],
                                       transform,
                                       1,
                                       num_workers=2,
                                       shuffle=False)
    import matplotlib.pyplot as plt
    import numpy as np
    for i, batch in enumerate(loader):
        batch: dataset.DataSetBatch
        tp, gt, tp_name, gt_name = (batch["tp_tensor"],
                                    batch["gt_tensor"],
                                    batch["tp_name"],
                                    batch["gt_name"])
        # tp, gt, tp_name, gt_name = batch
        print(tp_name, "shape:", tp.shape)
        print(gt_name, "shape:", gt.shape)
        tp_img = tp[0].permute(1, 2, 0).numpy().astype(np.uint8)
        mask_display = gt[0][0].numpy()
        # 展示图像
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(tp_img)
        ax[1].imshow(mask_display, cmap='gray')
        plt.show()
