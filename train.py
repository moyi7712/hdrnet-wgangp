import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from Config import Config
from DataPipe import PipeLine
from Network import Train_Strategy
config = Config('Config.yaml')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = Train_Strategy(config=config.HyperPara(), strategy=strategy)
dataset = PipeLine(config=config.Dataset(), batch_size=net.global_batch_size).DatasetTrain()
dataset = strategy.experimental_distribute_dataset(dataset)
is_first = True
with strategy.scope():
    pre_epoch = net.epoch.numpy()
    pre_step = net.this_step.numpy()
    step = pre_step
    for epoch in range(pre_epoch, pre_epoch+200):

        for inputs, label in dataset:
            is_D = True if step % 2 == 0 else False
            net.epoch.assign(epoch)
            if not step%2==0 and is_first:
                step+=1
            loss_dict = net.distributed_step(inputs, label, epoch, is_D, is_first)
            is_first = False
            with net.summary_writer.as_default():
                for key in loss_dict.keys():
                    tf.summary.scalar(key, loss_dict[key], step=step)
            if step % 20 == 0:
                print('step:{}, epoch:{}'.format(step,epoch))
                net.ckpt_manager.save()
            net.this_step.assign(step)
            step += 1
