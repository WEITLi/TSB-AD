from typing import Dict
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import os
import pandas as pd
from time import time
from pprint import pprint

from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset


def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]  # cut
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])  # pad
        windows.append(w if 'DTAAD' in args.model or 'Attention' in args.model or 'TranAD' in args.model else w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    # loader = [i[:, debug:debug+1] for i in loader]
    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s = []
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    if 'Attention' in model.name: 
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        res = []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                ae = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s, y_pred = [], []
            for d in data:
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            model.to(torch.device(args.Device))
            for i, d in enumerate(data):
                d = d.to(torch.device(args.Device))
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item());
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item());
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                d = d.to(torch.device(args.Device))
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            xs = []
            for d in data:
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])  # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1;
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                d = d.to(torch.device(args.Device))
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
            # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], [] 
        if training:
            for d, _ in dataloader:  
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0] 
                window = d.permute(1, 0, 2)  
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)  
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'DTAAD' in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * l(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * l(z[1].permute(1, 0, 2),elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            for d, _ in dataloader:
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, bs, feats)
                z = model(window)
                z = z[1].permute(1, 0, 2)
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    else:
        model.to(torch.device(args.Device))
        data = data.to(torch.device(args.Device))
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
                      'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    ### Plot curves
    if not args.test:
        if 'TranAD' or 'DTAAD' in model.name: testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    ### Plot attention
    if not args.test:
        if 'DTAAD' in model.name:
            plot_attention(model, 1, f'{args.model}_{args.dataset}')

    ### Scores
    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = df.append(result, ignore_index=True)
    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)
    # pprint(getresults2(df, result))


class DTAAD_Model(nn.Module):
    def __init__(self, feats, n_window=10, device='cpu'):
        super(DTAAD_Model, self).__init__()
        self.name = 'DTAAD'
        self.n_window = n_window
        self.n_feats = feats
        self.device = device
        self.n_hidden = 64
        self.batch = 128
        
        # 定义模型组件
        self.encoder = nn.GRU(feats, self.n_hidden, batch_first=True)
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(self.n_hidden, 4, batch_first=True)
        
        # 解码器
        self.decoder = nn.GRU(self.n_hidden, self.n_hidden, batch_first=True)
        self.output_layer1 = nn.Linear(self.n_hidden, feats)
        self.output_layer2 = nn.Linear(self.n_hidden, feats)
        
        # 可选：激活函数
        self.relu = nn.ReLU()
        self.lr = 1e-3

    def forward(self, x):
        # x 形状: [batch_size, feats, seq_len]
        
        # 重新整形为 GRU 的输入形状: [batch_size, seq_len, feats]
        x = x.permute(0, 2, 1)
        
        # 编码
        enc_output, hidden = self.encoder(x)
        
        # 自注意力机制
        attn_output, attn_weights = self.attention(enc_output, enc_output, enc_output)
        
        # 解码
        dec_output, _ = self.decoder(attn_output)
        
        # 产生预测
        y1 = self.output_layer1(dec_output)
        y2 = self.output_layer2(dec_output)
        
        return y1, y2


class DTAAD():
    def __init__(self,
                 win_size=10,
                 feats=1,
                 lr=1e-3,
                 batch_size=128,
                 epochs=5,
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
        
        self.win_size = win_size
        self.feats = feats
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        
        # 模型初始化
        self.model = DTAAD_Model(feats=feats, n_window=win_size, device=self.device).double().to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.loss = nn.MSELoss(reduction='none')
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=5)
        
        # 用于异常分数计算
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]
        
        # 数据准备
        train_data = torch.DoubleTensor(tsTrain)
        train_dataset = TensorDataset(train_data, train_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        valid_data = torch.DoubleTensor(tsValid)
        valid_dataset = TensorDataset(valid_data, valid_data)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 训练过程
        _lambda = 0.8
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (d, _) in loop:
                d = d.to(self.device)
                local_bs = d.shape[0]
                
                # 转换为滑动窗口形式
                windows = []
                for i in range(local_bs):
                    if i + self.win_size <= local_bs:
                        w = d[i:i + self.win_size]
                    else:
                        padding = self.win_size - (local_bs - i)
                        w = torch.cat([d[i:], d[0].repeat(padding, 1)])
                    windows.append(w)
                window = torch.stack(windows).permute(0, 2, 1)
                
                # 获取最后一个时间步作为目标
                elem = window[:, :, -1].view(1, local_bs, self.feats)
                
                self.optimizer.zero_grad()
                
                # 模型前向传播
                z = self.model(window)
                
                # 计算损失
                l1 = _lambda * self.loss(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * self.loss(z[1].permute(1, 0, 2), elem)
                loss = torch.mean(l1)
                
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            # 验证
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with torch.no_grad():
                for idx, (d, _) in loop:
                    d = d.to(self.device)
                    local_bs = d.shape[0]
                    
                    # 转换为滑动窗口形式
                    windows = []
                    for i in range(local_bs):
                        if i + self.win_size <= local_bs:
                            w = d[i:i + self.win_size]
                        else:
                            padding = self.win_size - (local_bs - i)
                            w = torch.cat([d[i:], d[0].repeat(padding, 1)])
                        windows.append(w)
                    window = torch.stack(windows).permute(0, 2, 1)
                    
                    # 获取最后一个时间步作为目标
                    elem = window[:, :, -1].view(1, local_bs, self.feats)
                    
                    # 模型前向传播
                    z = self.model(window)
                    
                    # 计算损失
                    l1 = _lambda * self.loss(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * self.loss(z[1].permute(1, 0, 2), elem)
                    loss = torch.mean(l1)
                    
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    # 计算逐样本的重构误差，用于异常检测
                    mse = torch.mean(l1, dim=(0, 2))  # 形状: [local_bs]
                    scores.append(mse.cpu())
            
            valid_loss = avg_loss / max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs:
                # 拟合高斯分布用于异常检测
                if len(scores) > 0:
                    scores_tensor = torch.cat(scores, dim=0)
                    if scores_tensor.numel() > 0:
                        self.mu = torch.mean(scores_tensor)
                        self.sigma = torch.std(scores_tensor)
                        print(f"分布参数 Mu: {self.mu.item()}, Sigma: {self.sigma.item()}")
                    else:
                        print("警告：验证分数为空，无法拟合高斯分布。")
                        self.mu = torch.tensor(0.0)
                        self.sigma = torch.tensor(1.0)
                else:
                    print("警告：验证分数列表为空，无法拟合高斯分布。")
                    self.mu = torch.tensor(0.0)
                    self.sigma = torch.tensor(1.0)
                
                if self.early_stopping.early_stop:
                    print("早停激活，训练提前结束。")
                    break
    
    def decision_function(self, data):
        """计算异常分数，分数越高表示越可能是异常"""
        self.model.eval()
        
        # 准备数据
        data_tensor = torch.DoubleTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        _lambda = 0.8
        scores = []
        all_outputs = []
        
        with torch.no_grad():
            for d, _ in dataloader:
                d = d.to(self.device)
                local_bs = d.shape[0]
                
                # 转换为滑动窗口形式
                windows = []
                for i in range(local_bs):
                    if i + self.win_size <= local_bs:
                        w = d[i:i + self.win_size]
                    else:
                        padding = self.win_size - (local_bs - i)
                        w = torch.cat([d[i:], d[0].repeat(padding, 1)])
                    windows.append(w)
                window = torch.stack(windows).permute(0, 2, 1)
                
                # 获取最后一个时间步作为目标
                elem = window[:, :, -1].view(1, local_bs, self.feats)
                
                # 模型前向传播
                z = self.model(window)
                
                # 计算损失/重构误差
                l1 = _lambda * self.loss(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * self.loss(z[1].permute(1, 0, 2), elem)
                
                # 记录输出用于可视化
                all_outputs.append(z[1].permute(1, 0, 2).cpu().numpy())
                
                # 计算每个样本的异常分数
                sample_scores = torch.mean(l1, dim=(0, 2))  # 形状: [local_bs]
                
                # 使用高斯分布归一化异常分数
                if self.mu is not None and self.sigma is not None:
                    normalized_scores = (sample_scores - self.mu) / (self.sigma + self.eps)
                    scores.append(normalized_scores.cpu().numpy())
                else:
                    scores.append(sample_scores.cpu().numpy())
        
        # 合并所有批次的分数
        self.__anomaly_score = np.concatenate(scores)
        
        # 可以选择保存预测结果以进行可视化
        self.y_hats = all_outputs
        
        return self.__anomaly_score
    
    def anomaly_score(self) -> np.ndarray:
        """返回之前计算的异常分数"""
        if self.__anomaly_score is None:
            print("请先调用 decision_function 来计算异常分数。")
            return np.array([])
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        """获取模型的预测值（用于可视化）"""
        if self.y_hats is None:
            print("请先调用 decision_function 来计算预测值。")
            return None
        return np.concatenate(self.y_hats, axis=1)

# 辅助函数：将数据转换为滑动窗口形式
def convert_to_windows(data, win_size):
    windows = []
    for i, g in enumerate(data):
        if i >= win_size:
            w = data[i - win_size:i]  # 切分
        else:
            w = torch.cat([data[0].repeat(win_size - i, 1), data[0:i]])  # 填充
        windows.append(w)
    return torch.stack(windows)
