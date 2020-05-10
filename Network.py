import tensorflow as tf
from Model import Generator, Discriminator
import os, random
import Utils


class Train(object):
    def __init__(self, config):
        self.base_lr = config.learning_rate
        self.config = config
        self.batch_size = config.batch_size
        self.G = Generator(config=config.Model.Generator)
        self.D = Discriminator(config=config.Model.Discriminator)
        self.optmizer_G = tf.keras.optimizers.Adam(learning_rate=self.base_lr, amsgrad=False)
        self.optmizer_D = tf.keras.optimizers.Adam(learning_rate=self.base_lr, amsgrad=False)
        self.epoch = tf.Variable(name='epoch', initial_value=0)
        self.this_step = tf.Variable(name='this_step', initial_value=0)

        checkpoint = tf.train.Checkpoint(optimizer_G=self.optmizer_G,
                                         optimizer_D=self.optmizer_D,
                                         G=self.G,
                                         D=self.D,
                                         epoch=self.epoch,
                                         this_step=self.this_step)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.prefix, 'summary'))
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.prefix, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint)

    def regularizer_loss(self, model):
        return tf.nn.scale_regularization_loss(tf.add_n(getattr(model, 'losses')))

    def lr_schedul(self, epoch):
        if epoch > 10:
            return self.base_lr / 2
        else:
            return self.base_lr




class Train_Strategy(Train):
    def __init__(self, config, strategy):
        super(Train_Strategy, self).__init__(config=config)
        self.strategy = strategy
        if strategy:
            self.global_batch_size = strategy.num_replicas_in_sync * self.batch_size

    def strategy2Tensor(self, loss):
        return tf.reduce_mean(self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None))

    def average_loss(self, loss):
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)

    def train_G(self, inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            fake = self.G(inputs)
            score_f = self.D(fake)
            # score_t = self.D(labels)
            MSE = tf.reduce_mean(tf.square(labels-fake), axis=[1, 2, 3])
            loss_total = self.average_loss(-score_f + 100*MSE)
            # loss_total = self.average_loss(-score_f + score_t + tf.reduce_mean(tf.square(inputs - fake), axis=[1, 2, 3]))
        grad = tape.gradient(loss_total, self.G.trainable_variables)
        self.optmizer_G.apply_gradients(zip(grad, self.G.trainable_variables))
        loss_out = [loss_total, score_f, MSE]
        return loss_out

    def train_D(self, inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            fake = self.G(inputs)
            score_fake = self.D(fake)
            score_true = self.D(labels)
            gp, gradients = self.gp(self.D, labels, fake)
            loss_total = self.average_loss(-score_true + score_fake + 10*gp)
        grad = tape.gradient(loss_total, self.D.trainable_variables)
        self.optmizer_D.apply_gradients(zip(grad, self.D.trainable_variables))
        loss_out = [loss_total, score_fake, score_true, gp, gradients]
        return loss_out

    def gp(self, discriminator, real, fake):
        sigma = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1., seed=234)
        interpolates = sigma * real + (1 - sigma) * fake
        gradients = tf.gradients(discriminator(interpolates, training=True), [interpolates])[0]
        gp = (tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])) - 1)**2
        return gp, tf.reduce_mean(gradients, axis=[1, 2, 3])



    @tf.function(experimental_relax_shapes=True)
    def distributed_step(self, inputs, labels, epoch, is_D,is_first):
        self.optmizer_G.learning_rate = self.lr_schedul(epoch)
        self.optmizer_D.learning_rate = self.lr_schedul(epoch)
        # is_D = not is_D
        loss_dict = {}
        if is_D:


            loss = self.strategy.experimental_run_v2(self.train_G, args=(inputs, labels))
            loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]
            loss_dict['loss_total_G'] = loss[0]
            loss_dict['score_fake_G'] = loss[1]
            loss_dict['MSE_G'] = loss[2]
        if (not is_D) or is_first:
            loss = self.strategy.experimental_run_v2(self.train_D, args=(inputs, labels))
            loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]
            loss_dict['loss_total_D'] = loss[0]
            loss_dict['score_fake_D'] = loss[1]
            loss_dict['score_true_D'] = loss[2]
            loss_dict['gp'] = loss[3]
            loss_dict['gradients'] = loss[4]

        return loss_dict