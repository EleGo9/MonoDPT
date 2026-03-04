export WANDB_API_KEY='your_api_key'
python tools/train_val.py --config configs/monodpt.yaml
CUDA_VISIBLE_DEVICES=0 python tools/demo.py --config configs/monodpt.yaml --ckpt /path/to/ckpt/pth -onnx
