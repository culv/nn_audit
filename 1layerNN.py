from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import re

# create a weight tensor
def weight_variable(shape, name):
    initial = tf.random_normal(shape)
    print(initial)
    return tf.Variable(initial, name=name)

# create a bias tensor
def bias_variable(shape, name):
    initial = tf.random_normal(shape)
    print(initial)
    return tf.Variable(initial, name=name)

# get test accuracy of network
def get_accuracy(inp, output, labels_placeholder, images, labels):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(labels_placeholder,1))
    # tf.argmax(y,1) returns the index of the highest entry in a tensor along the specified axis
    # tf.equal checks to see if two arguments are equal

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.cast converts correct_prediction True and False values into 1's and 0's
    return sess.run(accuracy, feed_dict={inp: images, labels_placeholder: labels})

### save ###
# this method saves checkpoints and the final model to the checkpoint_dir      
def save(sess, saver, checkpoint_dir, model_dir, step):
    model_name = "1layerNN.model" # filename to save model parameters in
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir) # join checkpoint_dir and current directory

    if not os.path.exists(checkpoint_dir): # if checkpoint_dir doesn't exist, make it
        os.makedirs(checkpoint_dir)
    # use TensorFlow Saver class to save the model
    # it will save under the filename DCGAN.model-(step)
    saver.save(sess,
        os.path.join(checkpoint_dir, model_name),
            global_step=step)

### load ###
# this method loads a saved model checkpoint
def load(saver, sess, checkpoint_dir, model_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Successfully read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

# sigmoid function
def sigmoid(np_array_in):
    return 1/(1+np.exp(-np_array_in))

# plot accuracy throughout training
def plot_acc(iter, fig, ax, test_acc, train_acc=None):
    plt.cla()
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('iterations')
    ax.set_title('neural net accuracy on test set (MNIST, 100 hidden neurons, batch size=100)')
    ax.set_xlim(0,5501)
    ax.set_ylim(0,101)
    ax.set_xticks(np.arange(0,5501,500))
    ax.set_yticks(np.arange(0,101,10))
    ax.set_yticks(np.arange(0,100,5), minor=True)
    ax.grid()
    ax.grid(which='minor')
    dom = np.arange(0, iter+1, 50)
    ax.plot(dom, 100.*np.array(test_acc))
    if train_acc is not None:
        ax.plot(dom, 100.*np.array(train_acc), c='r')
    plt.pause(0.1)

if __name__ == '__main__':
    
    # dataset
    dataset = 'MNIST'

    # get the MNIST data set
    # 55,000 training data points in mnist.train
    # 10,000 test data points in mnist.test
    # 5,000 validation data points in mnist.validation
    mnist = input_data.read_data_sets('MNIST/', one_hot=True)
    # labels are in one-hot vectors where only one component is 1, rest are 0
    # label vectors have length of 10, ith component=1 is label of i (i can be 0-9)
    # each MNIST data point has a handwritten digit (28x28 pixels) and a label (0-9)
    
    # mnist.train.images is a tensor that is 55000x784
    # 55000 = number of images, 784 = the 28x28 image flattened into a single vector
    # each entry is a float in [0,1]
    
    # mnist.train.labels is a tensor that is 55000x10
    # 55000 = number of images, 10 = number of labels
    
    # hidden layer size, batch size
    hidden_neur = 100
    batch_size = 100
    
    # epochs, iterations
    epochs = 10
    iters = np.ceil(epochs*55000./batch_size).astype('int')
    
    # learning rate and beta parameter for Adam gradient descent
    learning_rate = 0.005
    beta1 = 0.5
    
    # names of directories where checkpoints and this neural net will be saved
    checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoint')
    model_dir = '{}_hidden{}_batch{}'.format(dataset, hidden_neur, batch_size)
    
    # Create the model

    # you give a placeholder an input when you ask TensorFlow
    # to run a computation
    # the [None, 784] means it will be a tensor with a size of Nonex784
    # where None means that dimension can be any length (so we can put in
    # any number of 1x784 training image vectors that we want)
    x = tf.placeholder(tf.float32, [None, 784])
    
    # Variable is a modifiable tensor that lives in the TensorFlow graph
    # it can be used and modified by different computations
    # model parameters are usually Variables
    W_fc1 = weight_variable([784,hidden_neur],'weight1')
    b_fc1 = bias_variable([hidden_neur],'bias1')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([hidden_neur,10],'weight2')
    b_fc2 = bias_variable([10],'bias2')
    y = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # NOTE: do xW (instead of Wx) because you don't know how many samples you want
    # to feed in, but you know you want to get 10 probabilities
    # corresponding to 0-9 labels out
    
    # xW gives you a length 10 vector for every input image
    # you add the bias b to each of these vectors
    # so even though xW is (input#)x10 and b is 1x10 it's OK
    # because of broadcasting (only size of trailing axes must match or
    # one of them must be 1)
 
    # one-hot labels
    y_ = tf.placeholder(tf.int64, [None, 10])    
    
    
    # start TensorFlow session and initialize any variables        
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # for saving/loading model
    saver = tf.train.Saver()

    # train or no
    train = False
    if train:    
        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
                )
       # use Adam gradient descent to minimize cross entropy loss
        train_step = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(cross_entropy)
        # this adds an operation to the TensorFlow graph that
        # implements backpropagation and Adam gradient descent 

        # list to hold accuracy
        test_acc_vec = []
        train_acc_vec = []
        
        # start interactive pyplot and set up plot
        plt.ion()
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.pause(0.1) # interactive plot will freeze if you don't pause
    
        # start training!
        for iter in range(iters):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # gets 100 random images and their labels
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                                        # feed dict fills placeholders
            
            # calculate neural net accuracy on test set every 50 iterations of training
            if (np.mod(iter, 50) == 0):
                test_acc = get_accuracy(x, y, y_, mnist.test.images, mnist.test.labels)
#                train_acc = get_accuracy(x, y, y_, mnist.train.images, mnist.train.labels)
                test_acc_vec.append(test_acc)
#                train_acc_vec.append(train_acc)
            # save network every 500 iterations
            if (np.mod(iter, 500) == 0) and (iter != 0):
                save(sess, saver, checkpoint_dir, model_dir, iter)
                print('saved model: iter={}'.format(iter))

            # update interactive plot every 100 iterations and print test set accuracy to terminal
            if (np.mod(iter, 100) == 0):
                plot_acc(iter, fig, ax, test_acc_vec)#, train_acc=train_acc_vec)
                print('iter {} test set accuracy: {:.2f}%'.format(iter+1, test_acc*100))
    
        # save final plot
        plot_acc(iter, fig, ax, test_acc_vec)#, train_acc=train_acc_vec)
        fname = '{}_training.png'.format(model_dir)
        plt.savefig(fname)
    
        # save final trained model
        save(sess, saver, checkpoint_dir, model_dir, iter)
        print('saved model: iter={}'.format(iter))


    
    
    if not train:
        # load in trained model
        load(saver, sess, checkpoint_dir, model_dir)
    
        # grab MNIST training data
        mnist_train = mnist.test.images    

        # method 1: grab trained network weights and calculate neuron activations
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
    
        # calculate layer 1 neuron activations = ReLU(inputs*weights+bias)
        lay1_neuron_act = np.maximum(np.matmul(mnist_train, values[0]) + values[1], 0)
        lay2_neuron_act = sigmoid(np.matmul(lay1_neuron_act, values[2]) + values[3])
        neuron_data_from_weights = np.concatenate((lay1_neuron_act,lay2_neuron_act), axis=1)
    
        # method 2: grab neuron activations directly from tensorflow
        tf_lay1, tf_lay2 = sess.run([h_fc1, y], feed_dict={x: mnist_train})
        # concatenate both layers into a single vector
        neuron_data = np.concatenate((tf_lay1, tf_lay2), axis=1)
        
        what_set = 'test'
        neuron_data_fn = 'mnist_{}neurons_{}_set_neuron_data.gz'.format(hidden_neur, what_set)
        print(" [*] Saving neuron data...")
        np.savetxt(neuron_data_fn, neuron_data)
        print(" [*] Saved neuron data to {}".format(neuron_data_fn))
    
        print(" [*] Loading neuron data...")
        test = np.loadtxt(neuron_data_fn)
        print(" [*] Loaded neuron data from {}".format(neuron_data_fn))
    
        check_file = test==neuron_data
        print(check_file.all())
        check_methods = np.abs(neuron_data-neuron_data_from_weights)<1e-4
        print(check_methods.all())
