import json
import os
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from src.models.text.embeddings import BoW2VecFilterStop, AveWord2VecFilterStop
from src.models.w2vv import W2VV_MS
from src.util.bigfile import BigFile

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.utils import generic_utils
import tensorboard_logger as tb_logger



INFO = __file__
trainCollection = "flickr8kenctrain"
valCollection = "flickr8kencval"
testCollection = "flickr8kenctest"

checkpoint_dir = ""  # TODO
VOCABULARY_FILE = os.environ["DATA_PATH"] + "/interim/articles/vocabulary.pickle"
W2V_FILE = os.environ["DATA_PATH"] + ""  # TODO
FEATURE_FILE = os.environ["DATA_PATH"] + ""  # TODO


def cal_val_perf(all_errors, opt=None):
    # validation metric: MAP
    i2t_map_score = 0
    i2t_map_score = i2t_map(all_errors, n_caption=opt.n_caption)
    currscore = i2t_map_score
    return currscore


def process():
    # output visualization script
    runfile_vis = 'do_visual.sh'
    open(runfile_vis, 'w').write('port=$1\ntensorboard --logdir %s --port $port' % checkpoint_dir)
    os.system('chmod +x %s' % runfile_vis)

    val_per_hist_file = os.path.join(checkpoint_dir, 'val_per_hist.txt')
    model_file_name = os.path.join(checkpoint_dir, 'model.json')
    model_img_name = os.path.join(checkpoint_dir, 'model.png')

    tb_logger.configure(checkpoint_dir, flush_secs=5)

    # text embedding (text representation)

    bow2vec = BoW2VecFilterStop(VOCABULARY_FILE)
    w2v2vec = AveWord2VecFilterStop(W2V_FILE)

    n_text_layers = [bow2vec.ndims + w2v2vec.ndims, 2048, 2048]

    img_feats = BigFile(FEATURE_FILE)
    # val_img_feats = BigFile(FEATURE_FILE)

    # define word2visualvec model
    model = W2VV_MS(n_text_layers)

    model.save_json_model(model_file_name)
    model.plot(model_img_name)
    opt = {}

    opt.clipnorm = 5.0
    opt.optimizer = 'rmsprop'
    opt.learning_rate = 0.0001

    model.compile_model('mse', opt=opt)
    # model.init_model(opt.init_model_from)

"""
TODO continue going through the code and then run it ON THE FLICKR DATASET
"""


    # training set
    caption_file = os.path.join(rootpath, trainCollection, 'TextData', '%s.caption.txt' % trainCollection)
    trainData = PairDataSet_MS(caption_file, opt.batch_size, text2vec, bow2vec, w2v2vec, img_feats, flag_maxlen=True,
                               maxlen=opt.sent_maxlen)

    val_sent_file = os.path.join(rootpath, valCollection, 'TextData', '%s.caption.txt' % valCollection)
    val_img_list, val_sents_id, val_sents = readImgSents(val_sent_file)

    losser = get_losser(opt.simi_fun)()

    best_validation_perf = 0
    n_step = 0
    count = 0
    lr_count = 0
    best_epoch = -1
    val_per_hist = []
    for epoch in range(opt.max_epochs):
        print('\nEpoch', epoch)
        print("Training...")
        print("learning rate: ", model.get_lr())
        tb_logger.log_value('lr', model.get_lr(), step=n_step)

        train_progbar = generic_utils.Progbar(trainData.datasize)
        trainBatchIter = trainData.getBatchData()
        for minibatch_index in xrange(trainData.max_batch_size):
            # for minibatch_index in xrange(10):
            n_step += 1
            img_X_batch, text_X_batch = trainBatchIter.next()
            loss_batch = model.model.train_on_batch(text_X_batch, img_X_batch)
            train_progbar.add(img_X_batch.shape[0], values=[("loss", loss_batch)])

            tb_logger.log_value('loss', loss_batch, step=n_step)
            tb_logger.log_value('n_step', n_step, step=n_step)

        print("\nValidating...")
        all_errors = pred_mutual_error_ms(val_img_list, val_sents, model, text2vec, bow2vec, w2v2vec, val_img_feats,
                                          losser, opt=opt)

        this_validation_perf = cal_val_perf(all_errors, opt=opt)
        tb_logger.log_value('val_accuracy', this_validation_perf, step=n_step)

        val_per_hist.append(this_validation_perf)

        print('previous_best_performance: %.3f' % best_validation_perf)
        print('current_performance: %.3f' % this_validation_perf)

        fout_file = os.path.join(checkpoint_dir, 'epoch_%d.h5' % (epoch))

        lr_count += 1
        if this_validation_perf > best_validation_perf:
            best_validation_perf = this_validation_perf
            count = 0

            # save model
            model.model.save_weights(fout_file)
            if best_epoch != -1:
                os.system('rm ' + os.path.join(checkpoint_dir, 'epoch_%d.h5' % (best_epoch)))
            best_epoch = epoch

        else:
            # when the validation performance has decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_count > 2:
                model.decay_lr(0.5)
                lr_count = 0
            count += 1
            if count > 10:
                print("Early stopping happened")
                break

    sorted_epoch_perf = sorted(zip(range(len(val_per_hist)), val_per_hist), key=lambda x: x[1], reverse=True)
    with open(val_per_hist_file, 'w') as fout:
        for i, perf in sorted_epoch_perf:
            fout.write("epoch_" + str(i) + " " + str(perf) + "\n")

    print(os.path.join(checkpoint_dir, 'epoch_%d.h5' % sorted_epoch_perf[0][0]))
    # os.system('./'+runfile)
    os.system('cp %s/epoch_%d.h5 %s/best_model.h5' % (checkpoint_dir, sorted_epoch_perf[0][0], checkpoint_dir))


if __name__ == "__main__":
    sys.exit(main())
