#!/usr/bin/env python
from pathlib import Path

import numpy as np
import random
import pickle
import argparse
# import cupy # cupyは使用しない
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def load_dataset(fnames, sanity_check=False):
    dat = []
    lab = []
    num = []
    cnt = -1
    for c, fname in enumerate(fnames):
        k = 0
        with open(fname) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                if line.rstrip() == '-------------------':
                    k += 1
                    if sanity_check and k == 101:
                        break
                    cnt += 1
                    line = fin.readline()
                    axis = np.array([float(v) for v in line.rstrip().split()[2:]], dtype=np.float32)
                    dat.append({'axis': axis, 'part': []})
                    lab.append(c)
                    num.append(0)
                    continue
                part = np.array([float(v) for v in line.strip().split()], dtype=np.float32)
                dat[cnt]['part'].append(part)

    max_part = 0
    for d in dat:
        if max_part < len(d['part']):
            max_part = len(d['part'])

    for i in range(len(dat)):
        a = np.zeros((max_part, 7), dtype=np.float32)
        l = len(dat[i]['part'])
        if l == 0:
            l = 1
        else:
            a[:l,:] = np.array(dat[i]['part'])
        dat[i]['part'] = a
        num[i] = l
    
    return dat, num, lab


class RCNPDataset(Dataset):
    def __init__(self, dat, num, lab, split='train'):
        if split == 'train':
            self.dat = [dat[i] for i in ind1]
            self.num = np.array([num[i] for i in ind1], dtype=np.float32)
            self.lab = [lab[i] for i in ind1]
        elif split == 'validation':
            self.dat = [dat[i] for i in ind2]
            self.num = np.array([num[i] for i in ind2], dtype=np.float32)
            self.lab = [lab[i] for i in ind2]
        elif split == 'test':
            self.dat = [dat[i] for i in ind3]
            self.num = np.array([num[i] for i in ind3], dtype=np.float32)
            self.lab = [lab[i] for i in ind3]
        else:
            raise ValueError('split must be either train, validation, or test')
        
        lab = np.array(self.lab, dtype=np.int32)
        print('%s: %d (%d %d %d)' % (split, len(lab), np.sum(lab==0), np.sum(lab==1), np.sum(lab==2)))
        
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, i):
        # PyTorchのDataLoaderはデフォルトでnumpy配列をTensorに変換
        axis = self.dat[i]['axis']
        part = self.dat[i]['part']
        num = self.num[i]
        lab = self.lab[i]
        return axis, part, num, lab


class BaselineModel(nn.Module):
    def __init__(self, n_units=100, n_out=3):
        super(BaselineModel, self).__init__()
        self.lp1 = nn.Linear(7, n_units) # 入力次元を明示的に指定
        self.lp2 = nn.Linear(n_units, n_units)
        self.lc1 = nn.Linear(n_units + 3, n_units) # axisの次元は3
        self.lc2 = nn.Linear(n_units, n_units)
        self.lc3 = nn.Linear(n_units, n_units)
        self.lc4 = nn.Linear(n_units, n_units)
        self.lc5 = nn.Linear(n_units, n_out)
            
    # __call__
    def forward(self, a, p, n):
        n_bat, n_par, n_dim = p.shape 
        nonl = torch.relu
        
        h = p.view(-1, n_dim) # F.reshape
        h = nonl(self.lp1(h))
        h = nonl(self.lp2(h))
        h = h.view(n_bat, n_par, -1) # F.reshape
        
        # F.sum(h, axis=1) / n.reshape((-1, 1))
        h = torch.sum(h, dim=1) / n.view(-1, 1)

        h = torch.cat((h, a), dim=1) # F.concat
        h = nonl(self.lc1(h))
        h = nonl(self.lc2(h))
        h = nonl(self.lc3(h))
        h = nonl(self.lc4(h))
        return self.lc5(h)


def evaluate(fname, ds):
    import sklearn.metrics
    
    device = torch.device("cpu")
    model = BaselineModel(n_units=30, n_out=3)
    model.load_state_dict(torch.load(fname, map_location=device))
    model.to(device)
    model.eval() # 評価モードに設定
    
    bsize = 2048

    it = DataLoader(ds, batch_size=bsize, shuffle=False)
    
    y_list = []
    t_list = []
    
    with torch.no_grad(): # 勾配計算を無効化
        for batch in it:
            a, p, n, t_batch = batch
            a, p, n, t_batch = a.to(device), p.to(device), n.to(device), t_batch.to(device)
            outputs = model(a, p, n)
            res = F.softmax(outputs, dim=1)
            y_list.append(F.softmax(res, dim=1).cpu().numpy())
            t_list.append(t_batch.cpu().numpy())

    y = np.concatenate(y_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    
    z = np.argmax(y, axis=1)
    print(z[:100])
    print(t[:100])
    print('ACC:', sklearn.metrics.accuracy_score(t, z))
    print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(t, z))
    with open('out.txt', 'w') as fout:
        for i in range(len(y)):
            fout.write('%d %f %f %f\n' % (t[i], y[i,0], y[i,1], y[i,2]))


data_directory = Path('../data/ILC.2019.09-low-level-data/')
data_directory.mkdir(parents=True, exist_ok=True) # データディレクトリがなければ作成

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rebuild_dataset', action='store_true')
    parser.add_argument('-d', '--device'  , type=int, default=-1)
    parser.add_argument('-e', '--evaluate', type=str, default=None)
    args = parser.parse_args()
    
    if args.rebuild_dataset:
        fnames = [
            str(data_directory / 'bb_data1.txt'),
            str(data_directory / 'cc_data1.txt'),
            str(data_directory / 'uds_data1.txt')
        ]
        dat, num, lab = load_dataset(fnames, sanity_check=False)

        l = len(dat)
        ind = list(range(l))
        random.shuffle(ind)
        s1 = int(l * 0.6)
        s2 = int(l * 0.8)
        ind1 = ind[:s1]
        ind2 = ind[s1:s2]
        ind3 = ind[s2:]
        

        ds_train = RCNPDataset(dat, num, lab, split='train')
        ds_valid = RCNPDataset(dat, num, lab, split='validation')
        ds_test  = RCNPDataset(dat, num, lab, split='test')
    
        with open(data_directory / 'rcnpdataset_train.pickle', 'wb') as fout:
            pickle.dump(ds_train, fout)
        with open(data_directory / 'rcnpdataset_valid.pickle', 'wb') as fout:
            pickle.dump(ds_valid, fout)
        with open(data_directory / 'rcnpdataset_test.pickle', 'wb') as fout:
            pickle.dump(ds_test, fout)
    else:
        with open(data_directory / 'rcnpdataset_train.pickle', 'rb') as fin:
            ds_train = pickle.load(fin)
        with open(data_directory / 'rcnpdataset_valid.pickle', 'rb') as fin:
            ds_valid = pickle.load(fin)
        with open(data_directory / 'rcnpdataset_test.pickle', 'rb') as fin:
            ds_test = pickle.load(fin)


    if args.evaluate is not None:
        evaluate(args.evaluate, ds_test)
        exit(0)
            
    # PyTorchの学習ループ
    device = torch.device("cpu")
  
    batchsize = 1024
    it_train = DataLoader(ds_train, batchsize, shuffle=True)
    it_valid = DataLoader(ds_valid, batchsize, shuffle=False)
    it_test  = DataLoader(ds_test , batchsize, shuffle=False)

    model = BaselineModel(n_units=30, n_out=3).to(device)
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # trainer 手動でループを実装
    num_epochs = 1000
    output_dir = Path('result')
    output_dir.mkdir(exist_ok=True)
    
    # extensions.ExponentialShift
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    # extensions.LogReportやPlotReportのため
    history = {
        'epoch': [], 'main/loss': [], 'main/accuracy': [], 
        'validation/main/loss': [], 'validation/main/accuracy': [], 'elapsed_time': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 訓練パート
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for i, (a, p, n, t) in enumerate(it_train):
            a, p, n, t = a.to(device), p.to(device), n.to(device), t.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(a, p, n)
            loss = criterion(outputs, t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * a.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == t.data)

            # extensions.PrintReport
            total_iterations = (epoch - 1) * len(it_train) + i
            if total_iterations > 0 and total_iterations % 200 == 0:
                 print(f"Epoch {epoch}, Iteration {i}")


        epoch_train_loss = train_loss / len(ds_train)
        epoch_train_acc = train_corrects.double() / len(ds_train)

        # 検証パート (extensions.Evaluator) 
        model.eval()
        valid_loss = 0.0
        valid_corrects = 0
        with torch.no_grad():
            for a, p, n, t in it_valid:
                a, p, n, t = a.to(device), p.to(device), n.to(device), t.to(device, dtype=torch.long)
                outputs = model(a, p, n)
                loss = criterion(outputs, t)
                
                valid_loss += loss.item() * a.size(0)
                _, preds = torch.max(outputs, 1)
                valid_corrects += torch.sum(preds == t.data)
        
        epoch_valid_loss = valid_loss / len(ds_valid)
        epoch_valid_acc = valid_corrects.double() / len(ds_valid)

        # スケジューラの更新
        scheduler.step()

        # ログ
        elapsed_time = time.time() - start_time
        history['epoch'].append(epoch)
        history['main/loss'].append(epoch_train_loss)
        history['main/accuracy'].append(epoch_train_acc.item())
        history['validation/main/loss'].append(epoch_valid_loss)
        history['validation/main/accuracy'].append(epoch_valid_acc.item())
        history['elapsed_time'].append(elapsed_time)

        # extensions.PrintReport (エポックごと)
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Valid Loss: {epoch_valid_loss:.4f}, Acc: {epoch_valid_acc:.4f} | "
              f"Time: {elapsed_time:.2f}s")

        # extensions.snapshot / snapshot_object (20エポックごと)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), output_dir / f'model_epoch-{epoch}.pth')
            torch.save(model, output_dir / f'snapshot_epoch-{epoch}.pth')

    # extensions.PlotReport
    # Loss
    plt.figure()
    plt.plot(history['epoch'], history['main/loss'], label='main/loss')
    plt.plot(history['epoch'], history['validation/main/loss'], label='validation/main/loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / 'loss.png')
    
    # Accuracy
    plt.figure()
    plt.plot(history['epoch'], history['main/accuracy'], label='main/accuracy')
    plt.plot(history['epoch'], history['validation/main/accuracy'], label='validation/main/accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / 'accuracy.png')

    print("Training finished.")


    model.eval()
    y_list = []
    t_list = []

    with torch.no_grad():
        for a, p, n, t in it_test:
            a, p, n = a.to(device), p.to(device), n.to(device)
            outputs = model(a, p, n)
            prob = F.softmax(outputs, dim=1)

            y_list.append(prob.cpu().numpy())
            t_list.append(t.numpy())

    y = np.concatenate(y_list, axis=0)  # shape (N,3)
    t = np.concatenate(t_list, axis=0)  # shape (N,)

    # Save confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(t, np.argmax(y, axis=1))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")  # 枠や目盛りを消す

    row_labels = ["bb event", "cc event", "uds event"]
    col_labels = ["b", "c", "uds"]

    table = ax.table(
        cellText=cm,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    table.scale(1.2, 1.6)  
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()



    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    class_names = ["bb", "cc", "uds"]
    colors = ["red", "blue", "green"]

    for true_cls in range(3):
        mask = (t == true_cls)
        probs_true = y[mask]

        for pred_cls in range(3):
            ax = axes[true_cls, pred_cls]
            ax.hist(probs_true[:, pred_cls], bins=50, color=colors[true_cls], alpha=0.7, density=True)

            ax.set_xlim(0, 1)
            if pred_cls == 0:
                ax.set_ylabel(f"{class_names[true_cls]} events")
            if true_cls == 2:
                ax.set_xlabel(f"{class_names[pred_cls]} likeness")

            ax.set_title(f"{class_names[true_cls]}-{class_names[pred_cls]}")

    plt.tight_layout()
    plt.savefig(output_dir / "likeness_histograms.png")
    plt.close()


    # ROCカーブの計算と描画
    print("Generating ROC curves...")

    def calculate_roc_points(t, y_prob, signal_label, background_label):
        signal_probs = y_prob[t == signal_label, signal_label]
        background_probs = y_prob[t == background_label, signal_label]
        
        n_signal = len(signal_probs)
        n_background = len(background_probs)

        thresholds = np.linspace(0, 1, 1001)
        
        efficiencies = []
        misids = []
        
        for th in thresholds:
            eff = np.sum(signal_probs >= th) / n_signal if n_signal > 0 else 0.
            misid = np.sum(background_probs >= th) / n_background if n_background > 0 else 0.
            
            efficiencies.append(eff)
            misids.append(misid)
            
        return np.array(efficiencies), np.array(misids)

    class_names = ["bb", "cc", "uds"]
    
    for signal_idx in range(3):
        plt.figure(figsize=(7, 6))

        for background_idx in range(3):
            if signal_idx == background_idx:
                continue

            effs, misids = calculate_roc_points(t, y, signal_idx, background_idx)
            mask = misids > 0
            
            plt.plot(effs[mask], misids[mask], 
                     label=f'BG: {class_names[background_idx]}')

        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--')
        plt.xlabel(f'{class_names[signal_idx]} tagging efficiency')
        plt.ylabel('Mis-identification Fraction')
        plt.title(f'ROC Curve for {class_names[signal_idx]}-tagging')
        plt.legend()
        plt.savefig(output_dir / f'roc_{class_names[signal_idx]}_tagging.png')
        plt.close()

    print("ROC curves generated.")