import os
import logging
import time
from scipy import stats
from datetime import datetime
import tensorflow as tf
from sklearn.cluster import KMeans
from fn_loss import build_embedding_loss, build_dist_loss, HardmaxLoss
from fn_head import build_embedding_head, build_dist_head
from fn_backbone import build_doubleHead
from preprocess import extract_fn

import sys
import numpy as np
import cv2
tf.keras.backend.set_floatx('float32')
MAX_IMAGE_SUMMARY = 1

class LocalDisNet(object):

    def __init__(self, sess, flags):
        self.sess = sess
        self.backbone_fn = build_doubleHead
        alphas = np.array([2.0, 2.0, 4.0, 6.0, 8.0])
        self.alphas = alphas.repeat(int(flags.training_epoches / alphas.shape[0])).tolist()

        self.flags = flags
        self.dtype = tf.float32

        self.checkpoint_dir = os.path.join(self.flags.model_dir, "checkpoint")
        self.summary_dir = os.path.join(self.flags.model_dir, "summary")

        self.image_w = 512
        self.image_h = 512

    def build_test(self):
        self.input = tf.placeholder(tf.float32, (None, self.image_h, self.image_w, self.flags.image_channels))
        img_normalized = tf.image.per_image_standardization(tf.squeeze(self.input, axis=-1))
        img_normalized = tf.expand_dims(img_normalized, axis=-1)
        features1, features2 = self.backbone_fn(inputs=img_normalized, features=16)
        self.embedding = build_embedding_head(features1, self.alphas[-1], self.flags.embedding_dim)  # self.alphas[-1]
        print("embedding branch built.")
        if self.flags.dist_branch:
            self.dist = build_dist_head(features2)
            print("distance regression branch built.")
        self.saver = tf.train.Saver(max_to_keep=2, name='checkpoint')

    def train(self, batch_size, training_epoches, train_dir, val_dir=None):
        
        ######################
        #### prepare data ####
        ######################

        preprocess_f = lambda sample: extract_fn(sample, 
                                                 image_channels=self.flags.image_channels,
                                                 image_depth=self.flags.image_depth,
                                                 dist_map=self.flags.dist_branch)
        # config training dataset
        train_tf = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        train_ds = tf.data.TFRecordDataset(train_tf)
        train_ds = train_ds.map(preprocess_f)
        train_ds = train_ds.shuffle(buffer_size=100)
        # train_ds = train_ds.repeat(training_epoches)
        train_ds = train_ds.batch(batch_size)
        train_iterator = train_ds.make_initializable_iterator()
        train_handle = self.sess.run(train_iterator.string_handle())
        # config validation dataset
        if val_dir is not None:
            val_tf = [os.path.join(val_dir, f) for f in os.listdir(val_dir)]
            val_ds = tf.data.TFRecordDataset(val_tf)
            val_ds = val_ds.map(preprocess_f)
            val_ds = val_ds.batch(batch_size)
            val_iterator = val_ds.make_initializable_iterator()
            val_handle = self.sess.run(val_iterator.string_handle())
        # make iterator        
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_ds.output_types, train_ds.output_shapes)
        sample = iterator.get_next()

        ########################################
        #### build the network and training ####
        ########################################

        # prepare aux and summary training data
        self._make_aux()
        img_normalized = tf.image.per_image_standardization(tf.squeeze(sample['image/image'], axis=-1))
        img_normalized = tf.expand_dims(img_normalized, -1)
        tf.summary.image('input_image', img_normalized, max_outputs=MAX_IMAGE_SUMMARY)
        tf.summary.image('ground_truth', tf.cast(sample['image/label'] * 10, dtype=tf.uint8), max_outputs=MAX_IMAGE_SUMMARY)
        if self.flags.dist_branch:
            tf.summary.image('distance_map', sample['image/dist_map']*255, max_outputs=MAX_IMAGE_SUMMARY)
        features1, features2 = self.backbone_fn(inputs=img_normalized, features=16)
        # build embedding branch
        alpha = tf.placeholder(tf.float32, shape=[])
        embedding = build_embedding_head(features1, alpha, self.flags.embedding_dim)
        embedding_loss = build_embedding_loss(embedding, sample['image/label'], sample['image/neighbor'], include_bg=self.flags.include_bg)
        tf.summary.scalar('loss_embedding', embedding_loss)
        tf.summary.image('emb_dim1-3', embedding[:, :, :, 0:3], max_outputs=MAX_IMAGE_SUMMARY)
        # build distance regression branch
        if self.flags.dist_branch:
            dist = build_dist_head(features2)
            dist_loss = build_dist_loss(dist, sample['image/dist_map'])
            train_loss = embedding_loss + dist_loss
            tf.summary.scalar('loss_dist', dist_loss)
            tf.summary.image('output_dist', dist, max_outputs=MAX_IMAGE_SUMMARY)
        else:
            train_loss = embedding_loss
        tf.summary.scalar('loss', train_loss)

        # build optimizer
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.flags.lr, global_step, 5000, 0.9, staircase=True)
        tf.summary.scalar('lr', lr)
        opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss, global_step=global_step, name="opt")

        # summary and checkpoint
        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, 'train'), graph=self.sess.graph)
        if val_dir is not None:
            val_writer = tf.summary.FileWriter(
                os.path.join(self.summary_dir, 'val'), graph=self.sess.graph)
        summary_proto = tf.Summary()

        # ========================= get number of trainable variables
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print("The total number trainable variables {}".format(total_parameters))

        ########################
        #### start training ####
        ########################

        self.saver = tf.train.Saver(max_to_keep=5, name='checkpoint')
        # t_step = self.restore_weights()
        # if t_step <= 0:
        t_step = 0
        self.sess.run(tf.global_variables_initializer())
        logging.info("{}: Init new training".format(datetime.now()))

        for epoch in range(training_epoches):
            step_alpha = self.alphas[epoch] if epoch < len(self.alphas) else self.alphas[-1]
            print(step_alpha)
            t_time = time.time()
            self.sess.run(train_iterator.initializer)
            while True:
                try:
                    t_step = t_step + 1
                    if t_step % self.flags.summary_steps == 0 or t_step == 1:
                        loss, _, c_summary = self.sess.run([train_loss, opt, summary], feed_dict={handle: train_handle,
                                                                                                  alpha: step_alpha})
                        train_writer.add_summary(c_summary, t_step)
                        time_periter = (time.time() - t_time) / self.flags.summary_steps
                        logging.info("Epoch {:>8d}; Iteration {} ({:.4f}s/iter)".format(epoch, t_step, time_periter))
                        t_time = time.time()
                    else:
                        loss, _ = self.sess.run([train_loss, opt], feed_dict={handle: train_handle, alpha: step_alpha})
                        logging.info("Epoch {:>8d}; Iteration {} loss: {}".format(epoch, t_step, loss))


                    if val_dir is not None and t_step % self.flags.validation_steps == 0:
                        v_step = 0
                        self.sess.run(val_iterator.initializer)
                        losses = []
                        while True:
                            v_step = v_step + 1
                            try:
                                l = self.sess.run([train_loss], feed_dict={handle: val_handle, alpha: step_alpha})
                                losses.append(l)
                                logging.info("Validation step {} loss: {}".format(v_step, l))
                            except Exception as e:
                                val_summary = tf.Summary(value=[
                                    tf.Summary.Value(tag="loss_val", simple_value=np.mean(losses))])
                                val_writer.add_summary(val_summary, t_step)
                                break
                except Exception as e:
                    break
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model'),
                        global_step=t_step)
        logging.info("{}: Iteration_{} Saved checkpoint".format(datetime.now(), t_step))

    def restore_model(self, ckp_dir=None):
        self.build_test()
        return self.restore_weights(os.path.join(ckp_dir, "checkpoint"))

    def restore_weights(self, ckp_dir=None):
        if ckp_dir is None:
            ckp_dir = self.checkpoint_dir
        latest_checkpoint = tf.train.latest_checkpoint(ckp_dir)
        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Restore model from step {}. Loaded checkpoint {}"
                         .format(datetime.now(), step_num, latest_checkpoint))
            return step_num
        else:

            return 0

    def _make_aux(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        log_file = self.flags.model_dir + "/log.log"
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                            filename=log_file,
                            level=logging.DEBUG,
                            filemode='w')
        logging.getLogger().addHandler(logging.StreamHandler())


    def segment_from_seed(self, imgs, seed_thres=0.5, similarity_thres=0.7, resize=True, with_dist=True):

        '''
        imgs: list of images (numpy array)
        min_sz: minimal size of object
        resize: resize segments to the same size of imgs
        '''
        import postprocessing as pp
        from skimage.filters import gaussian

        imgs_input = []
        for i in range(len(imgs)): 
            img = np.squeeze(imgs[i])
            if img.shape[0:2] != (self.image_h, self.image_w):
                imgs_input.append(cv2.resize(img, (self.image_h, self.image_w)))
            else:
                imgs_input.append(img)
        imgs_input = np.array(imgs_input)

        if len(imgs_input.shape) == 3:
            imgs_input = np.expand_dims(imgs_input, axis=-1)

        if with_dist:
            embs, dist = self.sess.run([self.embedding, self.dist], feed_dict={self.input: imgs_input})
        else:
            embs = self.sess.run([self.embedding], feed_dict={self.input: imgs_input})

        segs = []
        for i in range(len(embs)):
            # =========================================
            # K means with unfixed seeds number
            # =========================================
            # # get seeds
            # dist = np.squeeze(gaussian(dist[i], sigma=3))
            # seeds = pp.get_seeds(dist, thres=seed_thres)
            # # seed to instance mask
            # emb = pp.smooth_emb(embs[i], radius=3)
            # # emb = embs[i]
            # seg = pp.mask_from_seeds(emb, seeds, similarity_thres=similarity_thres)
            # # remove noise
            # seg = pp.remove_noise(seg, dist, min_size=10, min_intensity=0.1)
            # segs.append(seg)

            # ========================================
            # K means with fix number of clusters
            # ========================================
            # pixel_embedding = np.reshape(np.squeeze(embs[i]), (-1, 4))
            # pixel_label = KMeans(n_clusters=4).fit_predict(pixel_embedding)
            # predict_mask = np.reshape(pixel_label, np.squeeze(embs[i][..., -1]).shape)
            predict_mask = np.argmax(embs[i], axis=-1).squeeze()
            background_label = stats.mode(predict_mask, axis=None)[0][0]
            seg = predict_mask.copy()
            if background_label != 0:
                seg[predict_mask == 0] = background_label
                seg[predict_mask == background_label] = 0
            segs.append(seg)

        if resize:
            for i in range(len(segs)):
                segs[i] = cv2.resize(segs[i], (imgs[i].shape[1], imgs[i].shape[0]), interpolation=cv2.INTER_NEAREST)

        return segs

    
    