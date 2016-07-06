import tensorflow as tf

# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.InteractiveSession()

# Evaluate the tensor `c`.
print(sess.run(c))