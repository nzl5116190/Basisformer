import argparse
import torch
import sys
sys.path.append('.')
from data_provider.data_factory import data_provider
from torch import optim
from model import Basisformer
from torch import nn
import time
import numpy as np
from evaluate_tool import metric
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pyplot import plot_seq_feature
from adabelief_pytorch import AdaBelief
import logging
import random


def vali(vali_data, vali_loader, criterion, epoch, writer, flag='vali'):
    total_loss = []
    model.eval()
    count_error = 0
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            f_dim = -1 if args.features == 'MS' else 0
            origin = batch_y[:, :args.seq_len, f_dim:].to(device)
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            real_batch_x = batch_x
            
            outputs,m,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index.float().to(device),batch_y,train=False,y_mark=batch_y_mark)
            
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss_raw = criterion(pred, true)
            loss = loss_raw.mean()

            total_loss.append(loss)

            if i == 0:
                fig = plot_seq_feature(outputs, batch_y, real_batch_x, flag)
                writer.add_figure("figure_{}".format(flag), fig, global_step=epoch)
                    
    total_loss = np.average(total_loss)
        
    model.train()
    return total_loss

def train():
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_and_print('[Info] Number of parameters: {}'.format(num_params))
    train_set, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args,flag='val')
    test_data, test_loader = data_provider(args,flag='test')
    
    para1 = [param for name,param in model.named_parameters() if 'map_MLP' in name]
    para2 = [param for name,param in model.named_parameters() if 'map_MLP' not in name]
    # optimizer = AdaBelief(model.parameters(), lr=args.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
    optimizer = AdaBelief([{'params':para1,'lr':5e-3},{'params':para2,'lr':args.learning_rate}], eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
    # optimizer = AdaBelief(model.parameters(), lr=args.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
    criterion = nn.MSELoss()
    criterion_view = nn.MSELoss(reduction='none')

    train_steps = len(train_loader)

    writer = SummaryWriter(os.path.join(record_dir,'event'))

    best_loss = 0
    count_error = 0
    count = 0
    

    for epoch in range(args.train_epochs):
        train_loss = []
        loss_pred = []
        loss_of_ce = []
        l_s = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(train_loader):
            optimizer.zero_grad()

            # to cuda
            batch_x = batch_x.float().to(device) # (B,L,C)
            batch_y = batch_y.float().to(device) # (B,L,C)
            batch_y_mark = batch_y_mark.float().to(device)
            
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            outputs,loss_infonce,loss_smooth,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index.float().to(device),batch_y,y_mark=batch_y_mark)
            
            loss_p = criterion(outputs, batch_y)
            lam1 = args.loss_weight_prediction
            lam2 = args.loss_weight_infonce
            lam3 = args.loss_weight_smooth
        
            # if loss_p > 5:
            #     count_error = count_error +1
            #     writer.add_scalar('error_loss', loss_p, global_step=count_error)
            #     fig = plot_seq_feature(outputs, batch_y,batch_x,error=True,input=batch_x)
            #     writer.add_figure("figure_error", fig, global_step=count_error)
            #     log_and_print(loss_p)
                
            loss = lam1 * loss_p + lam2 * loss_infonce  + lam3 * loss_smooth
            train_loss.append(loss.item())
            loss_pred.append(loss_p.item())
            loss_of_ce.append(loss_infonce.item())
            l_s.append(loss_smooth.item())
            loss.backward()
            optimizer.step()


            if (i+1) % (train_steps//5) == 0:
                log_and_print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))


        log_and_print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        loss1 = np.average(loss_pred)
        log_and_print('loss_pred:{0}'.format(loss1))
        loss2 = np.average(loss_of_ce)
        log_and_print('loss entropy:{0}'.format(loss2))
        loss3 = np.average(l_s)
        log_and_print('loss smooth:{0}'.format(loss3))
        vali_loss = vali(vali_data, vali_loader, criterion_view, epoch, writer, 'vali')
        test_loss = vali(test_data, test_loader, criterion_view, epoch, writer, 'test')

        log_and_print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss))

        fig = plot_seq_feature(outputs, batch_y, batch_x)
        writer.add_figure("figure_train", fig, global_step=epoch)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('vali_loss', vali_loss, global_step=epoch)
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        
        ckpt_path = os.path.join(record_dir,args.check_point)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
                
        if best_loss == 0:
            best_loss = vali_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
        else:
            if vali_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
                best_loss = vali_loss
                count = 0
            else:
                count = count + 1

        torch.save(model.state_dict(), os.path.join(ckpt_path, 'final_checkpoint.pth'))
        
        if count >= args.patience:
            break
    return


def test(setting='setting',test=True):
    test_data, test_loader = data_provider(args,flag='test')
    if test:
        log_and_print('loading model')
        model.load_state_dict(torch.load(os.path.join(record_dir,args.check_point, 'valid_best_checkpoint.pth')))
    
    preds = []
    trues = []

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(test_loader):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            outputs,m,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index.float().to(device),batch_y,train=False,y_mark=batch_y_mark)
                
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  
            true = batch_y  

            preds.append(pred)
            trues.append(true)
            
    t2 = time.time()
    log_and_print('total_time:{0}'.format(t2-t1))
    log_and_print('avg_time:{0}'.format((t2-t1)/len(test_data)))

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


    mae, mse, rmse, mape, mspe = metric(preds, trues)
    log_and_print('mse:{}, mae:{}'.format(mse, mae))
    return 

def log_and_print(text):
    logging.info(text)
    print(text)
    return    

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]
    
#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
    parser.add_argument('--is_training', type=bool, default=True, help='train or test')
    parser.add_argument('--device', type=int, default=0, help='gpu dvice')

    # data loader
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='all_six_datasets/traffic', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                            'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:mondfthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str default='tanh'

    # model define
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--heads', type=int, default=16, help='head in attention')
    parser.add_argument('--d_model', type=int, default=100, help='dimension of model')
    parser.add_argument('--N', type=int, default=10, help='number of learnable basis')
    parser.add_argument('--block_nums', type=int, default=2, help='number of blocks')
    parser.add_argument('--bottleneck', type=int, default=2, help='reduction of bottleneck')
    parser.add_argument('--map_bottleneck', type=int, default=20, help='reduction of mapping bottleneck')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--tau', type=float, default=0.07, help='temperature of infonce loss')
    parser.add_argument('--loss_weight_prediction', type=float, default=1.0, help='weight of prediction loss')
    parser.add_argument('--loss_weight_infonce', type=float, default=1.0, help='weight of infonce loss')
    parser.add_argument('--loss_weight_smooth', type=float, default=1.0, help='weight of smooth loss')


    #checkpoint_path
    parser.add_argument('--check_point',type=str,default='checkpoint',help='check point path, relative path')

    args = parser.parse_args()
    
    record_dir = os.path.join('records',args.data_path.split('.')[0],'features_'+args.features,\
                              'seq_len'+str(args.seq_len)+','+'pred_len'+str(args.pred_len))
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    
    if args.is_training:
        logger_file = os.path.join(record_dir,'train.log')
    else:
        logger_file = os.path.join(record_dir,'test.log')
        
    if os.path.exists(logger_file):
        with open(logger_file, "w") as file:
            file.truncate(0)
    logging.basicConfig(filename=logger_file, level=logging.INFO)
    
    log_and_print('Args in experiment:')
    log_and_print(args)

    device = init_dl_program(args.device, seed=0,max_threads=8) if torch.cuda.is_available() else "cpu"
    # device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"
    model = Basisformer(args.seq_len,args.pred_len,args.d_model,args.heads,args.N,args.block_nums,args.bottleneck,args.map_bottleneck,device,args.tau)

    log_and_print(model)
    model.to(device)  ##
    if args.is_training:
        train()
    test()


