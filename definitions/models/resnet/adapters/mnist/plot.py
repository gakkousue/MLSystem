# definitions/models/resnet/adapters/mnist/plot.py
import os
import matplotlib.pyplot as plt
import torch
import torchvision

def plot_transformed_sample(data_sample, output_dir):
    """
    Adapterによる変換後のデータ(Tensor)を受け取り、可視化して保存する。
    data_sample: (input_tensor, label) のタプルを想定
    """
    img_tensor, label = data_sample
    
    # img_tensor は (C, H, W) の形状で、正規化されている可能性がある
    # 視認のために正規化を解除（簡易的に0-1へクリップ）して表示する
    
    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    # グリッド画像の作成
    grid = torchvision.utils.make_grid(img_tensor, normalize=True)
    
    plt.figure(figsize=(4, 4))
    # (C, H, W) -> (H, W, C)
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.title(f"Transformed Sample (Label: {label})")
    plt.axis("off")
    
    save_path = os.path.join(output_dir, "adapter_transformed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved adapter plot: {save_path}")