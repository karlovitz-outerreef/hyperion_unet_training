# Training

Example training command:

```
python hyperion_unet_training/training/train.py --train-dir tmp/train/ --val-dir tmp/val/ --model-dir tmp/experiments/models/ --config-file tmp/experiments/configs/default_config.json --logdir tmp/experiments/logs/ --batch-size 1 --num-workers 0 --epochs 3
```