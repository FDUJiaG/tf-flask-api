import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from mnist import model

x = tf.compat.v1.placeholder("float", [None, 784])
sess = tf.compat.v1.Session()

with tf.compat.v1.variable_scope("regression"):
    y1, varibles = model.regression(x)
saver = tf.compat.v1.train.Saver(varibles)
saver.restore(sess, "mnist/data/regression.ckpt")

with tf.compat.v1.variable_scope("convolutional"):
    keep_prob = tf.compat.v1.placeholder("float")
    y2, varibles = model.convolutional(x, keep_prob)
saver = tf.train.Saver(varibles)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


app = Flask(__name__)


@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=8000)
