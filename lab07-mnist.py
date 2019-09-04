import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1)) # 1 -> row
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#argmax는 최댓값이 위치한 자리를 리턴
is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15 # 전체 데이터셋을 15번 학습시키겠다
batch_size = 100 # 한번에 몇개씩?
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch

        print("Epoch: ", '%04d' % (epoch+1), 'cost= ', '{:9f}'.format(avg_cost))
    print("Accuracy: ", accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y:mnist.test.labels}
    ))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Labels: ", sess.run(tf.math.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.math.argmax(hypothesis, 1),
                                feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()