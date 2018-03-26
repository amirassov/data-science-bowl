# dstorch

Данный репозиторий содержит мое решение конкурса [Data-Science-Bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018).

При решении задачи были использованы идеи и куски кода из следующих проектов:
- https://github.com/ternaus/TernausNet
- https://github.com/ternaus/robot-surgery-segmentation
- https://github.com/asanakoy/kaggle_carvana_segmentation
- https://github.com/lopuhin/mapillary-vistas-2017

Статьи:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- [Deep Watershed Transform for Instance Segmentation](https://arxiv.org/abs/1611.08303)
- [MULTICLASS WEIGHTED LOSS FOR INSTANCE SEGMENTATION OF CLUTTERED CELLS](https://arxiv.org/abs/1802.07465)


Что было сделано:
- Unet + augmentation + TTA = 0.346
- Unet + augmentation + TTA + watershed = 0.411
- TernausNet34 + augmentation + TTA + watershed = 0.466
- TernausNet34 + augmentation + TTA + watershed + scaling = ?

Заметки:
- Параметры: 100 эпох обучаем с `lr = 0.0005`, дальше 50 эпох запускаем `cyclic_lr`.
- Нужно добавить масштабирования картинок по площади.
- image для `watershed` считать как сумму центров.
- Добавить дополнительные данные
