import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_imagenet
import numpy as np
from torch.autograd import Variable
from utils import *
from modules import *
from datetime import datetime 
import dataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/resnet18_ste_2_v1_seed_1_wd_3_1e-4_3e-5_dali')   # Best_Prec1 62.9808 	Best_Prec5 84.2731

# zh
# writer = SummaryWriter('./runs_wus/resnet18_ieblock_seed_1_wd_3_1e-4_3e-5_dali_2')

# writer = SummaryWriter('./runs_wus/resnet18_ieblock_seed_1_wd_3_1e-4_3e-5_dali_dzh1_1.0')
# writer = SummaryWriter('./runs_wus/resnet18_ieblock_seed_1_wd_3_1e-4_0_dali_dzh1_1.0')

# 两阶段量化
# writer = SummaryWriter('./runs_wus/resnet18_ieblock_seed_1_wd_3_1e-4_3e-5_dali_fp-w')
# retrain  base fp-w: 65.7
# writer = SummaryWriter('./runs_wus/resnet18_ieblock_seed_1_wd_3_1e-4_3e-5_dali_retrain_dzh_1.0')

# resnet-34
# writer = SummaryWriter('./runs_wus/resnet34_ieblock_seed_1_wd_3_1e-4_3e-5_bt_128_dali')   # san1-2
# writer = SummaryWriter('./runs_wus/resnet34_ieblock_seed_1_wd_3_1e-4_3e-5_bt_128_dali_dzh_1.0')   # san1-2

# 两阶段量化
# writer = SummaryWriter('./runs_wus/resnet34_ieblock_seed_1_wd_3_1e-4_3e-5_bt_128_dali_fp-w')
# retrain  base fp-w: 68.7
# writer = SummaryWriter('./runs_wus/resnet34_ieblock_seed_1_wd_3_1e-4_3e-5_bt_128_dali_retrain_dzh_1.0')

# nb  w: 1 a: san1-2
# writer = SummaryWriter('./runs_wus/resnet18_nb_seed_1_wd_3_1e-4_0_dali') #
# writer = SummaryWriter('./runs_wus/resnet18_nb_seed_1_wd_3_1e-4_0_dali_dzh_1.0')

# writer = SummaryWriter('./runs_wus/resnet18_nb_seed_1_wd_3_1e-4_0_dali_fp-w')

# writer = SummaryWriter('./runs_wus/resnet18_nb_seed_1_wd_3_5e-5_0_dali_retrain_dzh_1.0')
# writer = SummaryWriter('./runs_wus/resnet18_nb_seed_1_wd_3_5e-4_0_dali_retrain_dzh_1.0')


# resnet-34
# writer = SummaryWriter('./runs_ieee/resnet34_ieblock_san_1e-4_1e-5_bt_128_dali_dzh1')

# resnet-18
# writer = SummaryWriter('./runs_ieee/resnet18_ieblock_san_1e-4_3e-5_dali')   # 62.13
# writer = SummaryWriter('./runs_ieee/resnet18_ieblock_san_1e-4_1e-5_dali_dzh1')   # 62.78
# writer = SummaryWriter('./runs_ieee/resnet18_ieblock_san_1e-4_3e-5_dali_dzh1')   # 62.79
# writer = SummaryWriter('./runs_ieee/resnet18_ieblock_san_5e-4_5e-5_dali_dzh1')   # 62.38
# writer = SummaryWriter('./runs_ieee/resnet18_ieblock_san_5e-4_1e-5_dali_dzh1')   # 62.53


class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        global w_need_stop_wd, w_need_de_wd
        global g_zh_po, g_zh_ne, g_isZh_po, g_isZh_ne, g_isZh, g_zh_th, de_wd

        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                count_Conv2d = count_Conv2d + 1

        self.num_of_params = count_Conv2d
        self.saved_params = []
        self.target_modules = []
        self.last_saved_params = []
        # 统计权值翻转情况
        self.w_flip_record = []
        self.w_flip_record_period = []
        self.reset_w_need_stop_wd = []
        self.init_zh = []
        self.w_border_po = []
        self.w_border_ne = []
        self.w_flip_record_period2 = []

        # 抑制权值更新
        self.w_begin_status_record = []
        self.tth = []

        self.clip_th = 0
        self.total_nums = 0

        self.grad_normal_ema = []
        self.flip_count = 0
        self.current_ep = 0
        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)          # 上一个epoch的weight
                self.last_saved_params.append(tmp)
                self.w_flip_record.append(torch.zeros_like(tmp))
                self.w_flip_record_period.append(torch.zeros_like(tmp))
                self.w_flip_record_period2.append(torch.zeros_like(tmp))
                self.w_begin_status_record.append(torch.zeros_like(tmp))
                self.tth.append(torch.zeros_like(tmp))
                self.reset_w_need_stop_wd.append(torch.ones_like(tmp))
                # w_need_de_wd.append(torch.zeros_like(tmp))
                # de_wd.append(torch.zeros_like(tmp))
                self.init_zh.append(torch.zeros_like(tmp))
                self.target_modules.append(m.weight)   # 当前epoch更新后的 weight
                self.grad_normal_ema.append(0)
                # g_isZh.append(torch.zeros_like(tmp))
                # g_zh_th.append(torch.zeros_like(tmp))
                self.w_border_po.append(torch.zeros_like(tmp))
                self.w_border_ne.append(torch.zeros_like(tmp))

        self.count_time = 197
        self.count_time_epoch = 5
        self.maintain_sign_count = 0
        # self.w_scale = 1 + 1e-1
        self.w_scale_up = 1 + 1e-1
        self.w_scale_low = 1 - 1e-1

        self.unflip_rate = []

        self.update_ep = 0

        self.update_init = True

        for i in range(self.num_of_params):
            self.unflip_rate.append(1.0)

    def save_params(self, train_iter):
        # 每个iter都执行
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def calculate_flip(self, ep, train_iter, cur_lr):

        if self.current_ep != ep:
            writer.add_scalar('flip_count', self.flip_count / 195, ep - 1)
            self.flip_count = 0
            for index in range(self.num_of_params):
                if ep - self.update_ep == 2:
                    self.w_flip_record_period2[index] = torch.zeros_like(self.w_flip_record_period2[index])
                self.w_flip_record_period[index] = torch.zeros_like(self.w_flip_record_period[index])
            if ep - self.update_ep == 2:
                self.update_ep = ep

        self.total_nums = 0
        layer3_index = [17, 16, 15, 14, 13, 12]
        layer2_index = [11, 10, 9, 8, 7, 6]
        layer1_index = [5, 4, 3, 2, 1, 0]

        for index in range(self.num_of_params):
            w_sign_change = self.saved_params[index].sign() != self.target_modules[index].data.sign()
            self.w_flip_record[index][w_sign_change] += 1
            self.w_flip_record_period[index][w_sign_change] += 1
            self.w_flip_record_period2[index][w_sign_change] += 1

            tmp_0 = torch.zeros_like(self.saved_params[index])
            self.total_nums += (self.w_flip_record[index] == tmp_0).sum()
            self.flip_count += (self.saved_params[index].sign() != self.target_modules[index].data.sign()).sum()

        if self.current_ep != ep:
            self.current_ep = ep

        writer.add_scalar('w_unflip_nums', self.total_nums, train_iter)


iter_count_train = 0
iter_count_val = 0

def main():
    global args, best_prec1, best_prec5, conv_modules
    best_prec1 = 0
    best_prec5 = 0

    args.print_freq=int(256/args.batch_size*500)

    if args.evaluate:
        args.results_dir = '/tmp'
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.resume and not args.evaluate:
        with open(os.path.join(save_path,'config.txt'), 'w') as args_file:
            args_file.write(str(datetime.now())+'\n\n')
            for args_n,args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v,int) else args_v
                args_file.write(str(args_n)+':  '+str(args_v)+'\n')

        setup_logging(os.path.join(save_path, 'logger.log'))
        logging.info("saving to %s", save_path)
        logging.info("run arguments: %s", args)
    else: 
        setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        if args.seed > 0:
            set_seed(args.seed) 
        else: 
            cudnn.benchmark = True
    else:
        args.gpus = None

    if args.dataset=='tinyimagenet':
        num_classes=200
        model_zoo = 'models_imagenet.'
    elif args.dataset=='imagenet':
        num_classes=1000
        model_zoo = 'models_imagenet.'
    elif args.dataset=='cifar10': 
        num_classes=10
        model_zoo = 'models_cifar.'
    elif args.dataset=='cifar100': 
        num_classes=100
        model_zoo = 'models_cifar.'

    #* create model
    if len(args.gpus)==1:
        model = eval(model_zoo+args.model)(num_classes=num_classes).cuda()
    else: 
        model = nn.DataParallel(eval(model_zoo+args.model)(num_classes=num_classes))
    if not args.resume:
        logging.info("creating model %s", args.model)
        logging.info("model structure: ")
        for name,module in model._modules.items():
            logging.info('\t'+str(name)+': '+str(module))
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

    bin_op = BinOp(model)

    #* evaluate
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
            return 
        else: 
            checkpoint = torch.load(args.evaluate, map_location=torch.device('cpu'))
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                        args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = os.path.join(save_path,'checkpoint.pth.tar')
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)


    # retrain
    # base_path = './runs_backup'
    # model_path = '/resnet18_sc+bn_s4-2_seed_1_wd_3_1e-4_3e-5_dali_63.6/result'  # 预训练模型
    # checkpoint_file = os.path.join(base_path + model_path, 'model_best.pth.tar')
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint['state_dict'])
    # print('预训练模型加载完成')
    #
    # args.lr = 0.0001
    # args.epochs = 10

    # base_path = './save_models'
    # # model_path = '/resnet18_ieblock_seed_1_wd_3_1e-4_3e-5_dali_fp-w-65.7'  # 预训练模型
    # # model_path = '/resnet34_ieblock_seed_1_wd_3_1e-4_3e-5_bt_128_dali_fp-w-68.7'  # 预训练模型
    # model_path = '/resnet18_nb_seed_1_wd_3_1e-4_0_dali_fp-w-64.1'  # 预训练模型
    # checkpoint_file = os.path.join(base_path + model_path, 'model_best.pth.tar')
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint['state_dict'])
    # logging.info('预训练模型加载完成')


    criterion = nn.CrossEntropyLoss().cuda()
    criterion = criterion.type(args.type)
    model = model.type(args.type)

    if args.evaluate:
        if args.use_dali:
            val_loader = dataset.get_imagenet(
                        type='val',
                        image_dir=args.data_path,
                        batch_size=args.batch_size_test,
                        num_threads=args.workers,
                        crop=224,
                        device_id='cuda:0',
                        num_gpus=1)
        else:
            val_loader = dataset.get_imagenet_torch(
                        type='val',
                        image_dir=args.data_path,
                        batch_size=args.batch_size_test,
                        num_threads=args.workers,
                        device_id='cuda:0'
                        )
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    #* load dataset
    if args.dataset=='imagenet':
        if args.use_dali:
            train_loader = dataset.get_imagenet(
                            type='train',
                            image_dir=args.data_path,
                            batch_size=args.batch_size,
                            num_threads=args.workers,
                            crop=224,
                            device_id='cuda:0',
                            num_gpus=1)
            val_loader = dataset.get_imagenet(
                            type='val',
                            image_dir=args.data_path,
                            batch_size=args.batch_size_test,
                            num_threads=args.workers,
                            crop=224,
                            device_id='cuda:0',
                            num_gpus=1)
        else:
            train_loader = dataset.get_imagenet_torch(
                            type='train',
                            image_dir=args.data_path,
                            batch_size=args.batch_size,
                            num_threads=args.workers,
                            device_id='cuda:0',
                            )
            val_loader = dataset.get_imagenet_torch(
                            type='val',
                            image_dir=args.data_path,
                            batch_size=args.batch_size_test,
                            num_threads=args.workers,
                            device_id='cuda:0'
                            )             
    else: 
        train_loader, val_loader = dataset.load_data(
                                    dataset=args.dataset,
                                    data_path=args.data_path,
                                    batch_size=args.batch_size,
                                    batch_size_test=args.batch_size_test,
                                    num_workers=args.workers)

    # 选择 量化卷积层
    conv_param = (param for name, param in model.named_parameters() if ('layer' in name and 'conv' in name))
    param = (param for name, param in model.named_parameters() if not ('layer' in name and 'conv' in name))

    #* optimizer settings
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            [
                {'params':conv_param,'initial_lr':args.lr,'weight_decay': 5e-4},
                # {'params':param,'initial_lr':args.lr,'weight_decay': 3e-5}
                {'params':param,'initial_lr':args.lr,'weight_decay': 1e-4}
            ],
            lr=args.lr,
            momentum=args.momentum,
                                    )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params':conv_param,'initial_lr':args.lr},
                                      {'params':param,'initial_lr':args.lr,'weight_decay':0.}],
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay) 
    else: 
        logging.error("Optimizer '%s' not defined.", args.optimizer)

    if args.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warm_up*4, eta_min = 0, last_epoch=args.start_epoch)
    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    elif args.lr_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (1.0-(epoch-args.warm_up*4)/(args.epochs-args.warm_up*4)), last_epoch=-1)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        logging.info("criterion: %s", criterion)
        logging.info('scheduler: %s', lr_scheduler)

    #* record names of conv_modules
    conv_modules=[]
    for name, module in model.named_modules():
        if isinstance(module,BinarizeConv2d):
            conv_modules.append(module)


    for epoch in range(args.start_epoch+1, args.epochs):
        time_start = datetime.now()
        #* warm up
        if args.warm_up and epoch < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch+1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])
            break

        for module in conv_modules:
            module.epoch = torch.tensor([epoch]).float()
            module.lr = optimizer.param_groups[0]['lr']
            # print(module.lr)

        #* training
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer, bin_op)

        #* adjust Lr
        if epoch >= 4 * args.warm_up:
            lr_scheduler.step()

        #* evaluating
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, criterion, epoch, bin_op)

        #* remember best prec
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = max(val_prec1, best_prec1)
            best_prec5 = max(val_prec5, best_prec5)
            best_epoch = epoch
            best_loss = val_loss

        #* save model
        if epoch % 1 == 0:
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_optimizer = optimizer.state_dict()
            model_scheduler = lr_scheduler.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'optimizer': model_optimizer,
                'lr_scheduler': model_scheduler,
            }, is_best, path=save_path)

        if args.time_estimate > 0 and epoch % args.time_estimate==0:
            time_end = datetime.now()
            cost_time,finish_time = get_time(time_end-time_start,epoch,args.epochs)
            logging.info('Time cost: '+cost_time+'\t'
                        'Time of Finish: '+finish_time)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        writer.add_scalar('train_prec1', train_prec1, epoch + 1)
        writer.add_scalar('val_prec1', val_prec1, epoch + 1)


    logging.info('*'*50+'DONE'+'*'*50)
    logging.info('\n Best_Epoch: {0}\t'
                     'Best_Prec1 {prec1:.4f} \t'
                     'Best_Prec5 {prec5:.4f} \t'
                     'Best_Loss {loss:.3f} \t'
                     .format(best_epoch+1, prec1=best_prec1, prec5=best_prec5, loss=best_loss))

total_iter = 0

def forward(data_loader, model, criterion, epoch=0, bin_op=None, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    global iter_count_train, iter_count_val
    global total_iter

    if args.use_dali:
        for i, batch_data in enumerate(data_loader):
            #* measure data loading time
            data_time.update(time.time() - end)
            inputs = batch_data[0]['data']
            target = batch_data[0]['label'].squeeze().long()
            batchsize = args.batch_size if training else args.batch_size_test
            len_dataloader = int(np.ceil(data_loader._size/batchsize))
            if args.gpus is not None:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            input_var = Variable(inputs.type(args.type))
            target_var = Variable(target)

            #* compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            if training:
                iter_count_train = iter_count_train + 1
            else:
                iter_count_val = iter_count_val + 1

            if type(output) is list:
                output = output[0]

            if training:
                lr = optimizer.param_groups[0]['lr']
                bin_op.save_params(iter_count_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bin_op.calculate_flip(epoch, iter_count_train, lr)
                total_iter += 1

            #* measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            #* measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i * batchsize, data_loader._size,
                                phase='TRAINING' if training else 'EVALUATING',
                                batch_time=batch_time,
                                data_time=data_time, loss=losses, 
                                top1=top1, top5=top5))
    else:
        for i, (inputs, target) in enumerate(data_loader):
            #* measure data loading time
            data_time.update(time.time() - end)
            if args.gpus is not None:
                inputs = inputs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            input_var = Variable(inputs.type(args.type))
            target_var = Variable(target)
            batchsize = args.batch_size if training else args.batch_size_test

            #* compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            if type(output) is list:
                output = output[0]

            if training:
                #* back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #* measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            #* measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i * batchsize, len(data_loader) * batchsize,
                                phase='TRAINING' if training else 'EVALUATING',
                                batch_time=batch_time,
                                data_time=data_time, loss=losses,
                                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, bin_op):
    model.train()
    return forward(data_loader, model, criterion, epoch, bin_op,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch, bin_op):
    model.eval()
    return forward(data_loader, model, criterion, epoch, bin_op,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
