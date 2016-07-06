import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import numpy as np



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    # depth, height, width, channels
    # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME')


def count_chords(ys):
    counts = np.zeros(25)
    for el in ys:
        counts += el
        # for n in range(0,len(c)):
        #     counts[n] += c[n]
    d = {}
    for i in range(0,len(counts)):
        if counts[i] > 0:
            d[str(i)] = counts[i]
    print d

def count_chords_cathegorical(ys):
    counts = np.zeros(25)
    for s in ys:
        for el in s:
            counts[el] = counts[el] + 1
    print counts
    l = len(ys[0])
    print np.max(counts)/float(l)







def train_cnn(dg):
    session_run_id = "1"

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None,1000])
        x_re = tf.reshape(x, [-1,1,1000,1])
    with tf.name_scope('ground_truth'):
        y_ = tf.placeholder(tf.float32, shape=[None, 25])


    #convolutions
    # height, width, input channels, output channels
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([1, 128, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv1d(x_re, W_conv1) + b_conv1)
    with tf.name_scope('pool12x2'):
        h_pool2 = max_pool_2x2(h_conv1)


    with tf.name_scope('fc1'):
        h_conv1_flat = tf.reshape(h_pool2, [-1, 1000 * 32])

        W_fc1 = weight_variable([1000 * 32, 25])
        b_fc1 = bias_variable([25])

        h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

    # with tf.name_scope('fc2'):
    #     W_fc2 = weight_variable([1024, 25])
    #     b_fc2 = bias_variable([25])
    #     h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1,W_fc2) + b_fc2)
    with tf.name_scope('output'):
        y = tf.nn.softmax(h_fc1)
    # W = tf.Variable(tf.zeros([1000, 25]))
    # b = tf.Variable(tf.zeros([25]))
    #
    # y = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(h_conv1_flat, W) + b))

    # sess.run(tf.initialize_all_variables())

    with tf.name_scope('cross_entropy'):
        # diff = y_ * tf.log(y)
        # with tf.name_scope('total'):
        #     cross_entropy = -tf.reduce_mean(diff)
        # tf.scalar_summary('cross entropy', cross_entropy)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    with tf.name_scope('accuracy_computation'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy_training',accuracy)

        # testing / validation
    with tf.name_scope('gt_histogram'):
        tf.histogram_summary('gt_histogram',y_)


    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./summaries/' + session_run_id,sess.graph)
    # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')


    sess.run(tf.initialize_all_variables())
    for i in range(100000):


        # summary, acc, ts, ce = sess.run([merged, accuracy, train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1]})


        # if (i % 10 == 0):
        # summary, acc = sess.run([, accuracy], feed_dict={x: batch[0], y_: batch[1]})
        if i % 10 == 0:
            # batch = dg.get_test()
            # # summary = sess.run(merged,feed_dict={x: batch[0][0:10], y_: batch[1][0:10]})
            # acc_total = 0
            #
            # for chunk_start in range(0, len(batch[0]) - 100, 100):
            #     acc = sess.run(accuracy_validation, feed_dict={x: batch[0][i:i + 100], y_: batch[1][i:i + 100]})
            #     acc_total = acc_total * float(i) / float(i + 1) + acc * 1./ float(i + 1)
            #
            # summary = tf.Summary(value=[tf.Summary.Value(tag='total_accuracy', simple_value=acc_total)])
            # train_writer.add_summary(summary,i)
            batch = dg.next_batch(1000)
            summary, acc, ts, ce = sess.run([merged, accuracy, train_step, cross_entropy],
                                            feed_dict={x: batch[0], y_: batch[1]})
            train_writer.add_summary(summary, i)
            current_song_ctr, total_songs = dg.get_song_counter()
            epoch = dg.get_epochs()
            print('Accuracy at step %s: %s, Cross entropy: %s, song %i of %i, epochs %i' % (i, acc, ce, current_song_ctr, total_songs, epoch))
        else:
            batch = dg.next_batch(1000)
            summary, acc, ts, ce = sess.run([merged, accuracy, train_step, cross_entropy],
                                            feed_dict={x: batch[0], y_: batch[1]})



def train_softmax(dg):
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 1000])
    y_ = tf.placeholder(tf.float32, shape=[None, 25])

    W = tf.Variable(tf.zeros([1000, 25]))
    b = tf.Variable(tf.zeros([25]))

    y = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x, W) + b))

    sess.run(tf.initialize_all_variables())

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    #testing / validation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(100000):
        #print "iteration",i
        # batch = mnist.train.next_batch(50)
        batch,stats = dg.next_batch(100,statistics=True)
        ts,ce = sess.run([train_step,cross_entropy],feed_dict={x: batch[0], y_: batch[1]})
        # _, loss_val, W_val = train_step.run([train_step, cross_entropy, W],feed_dict={x: batch[0], y_: batch[1]})

        # print "ts:",ts
        if (i %1000 == 0):
            print "it:",i,"ce:",ce,"correct:",sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1]})
            print stats
        #print "y:",y
        # if i%1000 == 0:
        # print sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1]})
        # batch = dg.next_batch(10000)
        # print(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]}))
        # print batch[1]
        # print(sess.run(correct_prediction, feed_dict={x: batch[0], y_: batch[1]}))