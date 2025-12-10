from owl.core import dataset
from pathlib import Path

if __name__ == '__main__':
    loader = dataset.create_dataloader([Path('example')],
                                       None,
                                       1,
                                       num_workers=0,
                                       shuffle=False)
    import matplotlib.pyplot as plt
    import numpy as np
    for i, batch in enumerate(loader):
        batch: dataset.DataSetBatch
        print(batch)
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
