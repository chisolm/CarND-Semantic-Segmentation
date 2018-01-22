import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import argparse
import time
import datetime
import imageio


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path);
    graph = tf.get_default_graph();
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name);

    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name);
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name);
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name);
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name);

    print("tensor keep_prob: ", keep_prob.get_shape(), type(keep_prob))
    print("tensor shapes: input, layer3_out, layer4_out, layer7_out",
            input_image.get_shape(),
            layer3_out.get_shape(),
            layer4_out.get_shape(),
            layer7_out.get_shape())
    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    std_dev = 1e-3
    # 1x1 convolution
    with tf.name_scope("1x1"):
        conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Deconvolution layer
    with tf.name_scope("deconv_1"):
        deconv_1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5), name="deconv_1")

    # 1x1 convolution of skip layer 4
    with tf.name_scope("skip_l4"):
        skip_l4_1x1 = tf.layers.conv2d_transpose(vgg_layer4_out, num_classes, 1, padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        add_layer4_out = tf.add(deconv_1, skip_l4_1x1)

    # Deconvolution layer
    with tf.name_scope("deconv_2"):
        deconv_2 = tf.layers.conv2d_transpose(add_layer4_out, num_classes, 4, 2, padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5), name="deconv_2")

    # 1x1 convolution of skip layer 4
    with tf.name_scope("skip_l3"):
        skip_l3_1x1 = tf.layers.conv2d_transpose(vgg_layer3_out, num_classes, 1, padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        add_layer3_out = tf.add(deconv_2, skip_l3_1x1)

    # Deconvolution layer
    with tf.name_scope("deconv_3"):
        deconv_3 = tf.layers.conv2d_transpose(add_layer3_out, num_classes, 16, 8, padding= 'same',
                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5), name="deconv_3")

    return deconv_3
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # loss function
    with tf.name_scope('cross_entropy'):
        # lesson 10.9 classification and loss:
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    tf.summary.scalar('global_step', global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    with tf.name_scope('train_op'):
        train_op = optimizer.minimize(cross_entropy_loss)


    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def optimize_IoU(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # This is an attempt at the Intsersection over Union measurement of loss.
    # However when used as a loss function it led to extremely inconsistent 
    # behavior epoch to epoch.  I was unable to resolve why it failed.

    # Reference for implementation:
    # reference: http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html


    with tf.name_scope("iou_data"):
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        correct_label = tf.reshape(correct_label, (-1, num_classes))
        # intersection
        inter=tf.reduce_sum(tf.multiply(logits,correct_label))
        # union
        add = tf.add(logits,correct_label)
        mul = tf.multiply(logits,correct_label)
        union=tf.reduce_sum(tf.subtract(add, mul))
        tf.summary.scalar('inter', inter)
        tf.summary.scalar('add', add)
        tf.summary.scalar('mul', mul)
        tf.summary.scalar('union', union)

        # loss function
        IoU_loss = tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))

        tf.summary.scalar('IoU_loss', IoU_loss)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        tf.summary.scalar('global_step', global_step)

        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        train_op = optimizer.minimize(IoU_loss, global_step=global_step)

    return logits, train_op, IoU_loss

#tests.test_optimize(optimize_IoU)


# def create_sample_output(logits):
#     """ define prediction probabilities and classes """

#     # I think I'm running into trouble with not having fixed dimensions
#     num_classes = 2
#     with tf.name_scope("sample_out"):
#         local_logits = tf.identity(logits, name='local_logits')
#         prediction_softmax = tf.nn.softmax(local_logits, name="prediction_softmax")
#         prediction_class = tf.cast(tf.greater(prediction_softmax, 0.5), dtype=tf.float32, name='prediction_class')
#         prediction_class_idx = tf.cast(tf.argmax(prediction_class, axis=3), dtype=tf.uint8, name='prediction_class_idx')
#         tf.summary.image('prediction_class_idx', tf.expand_dims(tf.div(tf.cast(prediction_class_idx, dtype=tf.float32), float(num_classes)), -1), max_outputs=2)


#         image_shape = (160, 576)
#         #im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})
#         #im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])


#         segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
#         mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
#         mask = scipy.misc.toimage(mask, mode="RGBA")
#         street_im = scipy.misc.toimage(image)
#         street_im.paste(mask, box=None, mask=mask)


def train_nn(args, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, train_writer):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # most function in get batches fn
    counter = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        for image, label in get_batches_fn(batch_size):
            # training
            counter += 1

            #print("trainable_variables: ")
            #for i in tf.trainable_variables():
            #    print(i)

            # merge is probably empty:
            # TypeError: Fetch argument None has invalid type <class 'NoneType'>

            merge = tf.summary.merge_all()
            summary, _, loss = sess.run([merge, train_op, cross_entropy_loss],
                            feed_dict={input_image: image, correct_label: label,
                            keep_prob: args.keep_prob, learning_rate: args.learning_rate})
            # Done parameterize learning and keep_prob
            print("Loss: = {:.3f}".format(loss))
            train_writer.add_summary(summary, counter)
            pass
    pass
#tests.test_train_nn(train_nn)

def load_model(sess, model_dir):
        # load saved model
        tf.saved_model.loader.load(sess, ["FCN8"], model_dir)
        # we need to re-assign the following ops to instance variables for prediction
        # we cannot continue training from this state as other instance variables are undefined
        graph = tf.get_default_graph()
        #print("graph: ", 
        #    [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='Variable']
        #    )
        #print("graph: ", 
        #    [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='VariableV2']
        #    )
        #print("graph: ", 
        #    [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='ResourceVariable']
        #    )
        input_image = graph.get_tensor_by_name("image_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        #prediction_class = graph.get_tensor_by_name("predictions/prediction_class:0")
        #prediction_class = graph.get_tensor_by_name("label:0")
        logits = graph.get_tensor_by_name("logits:0")
        return input_image, keep_prob, logits

def save_model(sess, model_dir, tag):
    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
    #builder.add_meta_graph_and_variables(sess, [tag])
    builder.add_meta_graph_and_variables(sess, tag)
    builder.save()


def predict_files(args, image_shape):

    runs_dir = './runs'
    with tf.Session() as sess:
        input_image, keep_prob, logits = load_model(sess, args.model_dir)
        helper.save_inference_samples(runs_dir, args.inference_dir, sess, image_shape, logits, keep_prob, input_image)


def run(args, image_shape):
    num_classes = 2
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:


        epochs = args.epochs
        batch_size = args.batch
        # learning_rate = 0.0001
        num_classes = 2
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes), name="correct_label")
        learning_rate = tf.placeholder(dtype = tf.float32, name="learning_rate")

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(args.training_data, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Done: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)


        with tf.name_scope("data"):
            # tf.summary.image('input_images', input_image, max_outputs=2)
            tf.summary.image('input_images', input_image)

        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        #create_sample_output(logits)

        train_writer = tf.summary.FileWriter('./logs/1/train' + logging_time, sess.graph)

        # Done: Train NN using the train_nn function
        tf.set_random_seed(42)
        sess.run(tf.global_variables_initializer())

        # Want to save just a couple images
        with tf.name_scope("output_image"):
            # tf.summary.image('input_images', input_image, max_outputs=2)
            tf.summary.image('input_images', input_image)

        train_nn(args, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
            keep_prob, learning_rate, train_writer)

        # Done: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, args.inference_dir, sess, image_shape, logits, keep_prob, input_image)

        if (args.save == True):
            save_model(sess, args.model_dir, ["FCN8"])
        train_writer.close()

        # OPTIONAL: Apply the trained model to a video

import matplotlib
matplotlib.use('GTKAgg')   # generate ? output by default
from moviepy.editor import VideoFileClip
#from IPython.display import HTML

class movie_config:
    def __init__(self, sess, logits, input_image_tensor, keep_prob_tensor, image_shape):
        self.sess = sess
        self.logits = logits
        self.input_image_tensor = input_image_tensor
        self.keep_prob_tensor = keep_prob_tensor
        self.image_shape = image_shape


movie_data = []

def process_image(image):
    finalimg = helper.eval_single_image(movie_data.sess, image, movie_data.logits,
        movie_data.keep_prob_tensor, movie_data.input_image_tensor, movie_data.image_shape)
    return finalimg

def eval_movie(args, image_shape):

    global movie_data
    imageio.plugins.ffmpeg.download()

    with tf.Session() as sess:
        input_image, keep_prob, logits = load_model(sess, args.model_dir)
        movie_data = movie_config(sess, logits, input_image, keep_prob, image_shape)
        #helper.save_inference_samples(runs_dir, args.inference_dir, sess, image_shape, logits, keep_prob, input_image)

        video_debug_output1 = args.video_output
        video_input1 = VideoFileClip(args.video_input) # .subclip(22,30)  # 8 second test at first
        processed_video = video_input1.fl_image(process_image)
        #%time processed_video.write_videofile(video_debug_output1, audio=False)
        processed_video.write_videofile(video_debug_output1, audio=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Process flags.')
    parser.add_argument('action', help='train or predict on saved model: [train | predict]',
                        type=str, choices=['train', 'predict', 'predict_movie'])
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to run')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=False)
    parser.add_argument('--batch', type=int, default=10, help='Batch size. default: 128')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='Keep probability. default: 0.5')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate. default: 0.00005')
    parser.add_argument('--model_dir', type=str, default=("logs/model" + logging_time),
                        help='Dir to save model. default: logs/model')
    parser.add_argument('--training_data', help="training images dir, default: data/data_road/training",
                    type=str, default="data/data_road/training")
    parser.add_argument('--inference_dir', help="testing images dir, default: data/data_road/testing",
                    type=str, default="data/data_road/testing")
    parser.add_argument('--input_image_pat', help="input images for prediction, as pattern default: data/data_road/testing/image_2/*.png",
                    type=str, default="data/data_road/testing/image_2/*.png")
    parser.add_argument('--video_input', type=str, default="project_video.mp4",
                        help='File for video input. default: project_video.mp4')
    parser.add_argument('--video_output', type=str, default="project_video_output.mp4",
                        help='File name for video output. default: project_video_output.mp4')

    #parser.add_argument('--augmentbase', action='store_true', default=False, help='Augment the data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    logging_time = datetime.datetime.now().strftime("%y.%m.%d.%H:%M")
    args = parse_args()

    # shape for output images (and processing of images)
    # Using roughly half size of supplies images, should really derive from images
    # and network strides
    image_shape = (160, 576)


    # set tensorflow logging
    tf.logging.set_verbosity(tf.logging.INFO)

    if args.action=='train':
        run(args, image_shape)
    elif args.action=='predict':
        print('input_image_pat={}'.format(args.input_image_pat))
        predict_files(args, image_shape)
    elif args.action=='predict_movie':
        eval_movie(args, image_shape)

