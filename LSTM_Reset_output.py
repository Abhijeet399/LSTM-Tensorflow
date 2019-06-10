import tensorflow as tf
import numpy as np
import math
import time
import os
tf.__version__

data_dir = "C:/Users/bhatt/Desktop/Jupyter Notebook/"
neurons=512
def load_data(kind):
    """
    Akind -  ANN / LSTM
    """
    data = np.load(os.path.join(data_dir, "%s.npz"%kind))
    train_x, train_y = data["train_x"], data["train_y"]
    val_x, val_y = data["val_x"], data["val_y"]
    return train_x, train_y, val_x, val_y

trainData,trainLabel,valData,valLabel=load_data('LSTM')

batch_size=98

print(trainData.shape)
print(trainLabel.shape)
print(valData.shape)
print(valLabel.shape)

def create_batches(start,batchSize):
    return trainData[start:start+batchSize],trainLabel[start:start+batchSize]

x=tf.placeholder(tf.float32,shape=[None,50, 2048])
y_ = tf.placeholder(tf.float32, shape=[None,5])

# class Input(object):
#     def __init__(self, batch_size, num_steps, data):
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#         self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

# cell=tf.nn.rnn_cell.LSTMCell(256)

def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    state_c, state_h=cell.zero_state(batch_size, tf.float32)
    # state_variables=tf.concat(state_c, state_h)
    # for state_c, state_h in cell.zero_state(batch_size, tf.float32):
    #     state_variables.append(tf.contrib.rnn.LSTMStateTuple(
    #         tf.Variable(state_c, trainable=False),
    #         tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return state_c, state_h

def get_state_update_op(state_c, state_h, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    # for state_c, state_h , new_state in zip(state_c, state_h, new_states):
        # Assign the new state to the state variables on this layer

    # update_ops.extend([tf.assign(new_states[0],((state_c).eval())),
    #                tf.assign(new_states[1],((state_h).eval()))])
    # update_ops.extend([tf.assign(new_states[0],state_c),
    #                    tf.assign(new_states[1],state_h)])
    update_ops.extend([state_c.assign(new_state[0]),
                          state_h.assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)

def reset():
	# Define an op to reset the hidden state to zeros
	update_ops = []
	# for state_variable in rnn_tuple_state:
	# Assign the new state to the state variables on this layer
	update_ops.extend([state_c.assign(tf.zeros_like(state_c)),
	                   state_h.assign(tf.zeros_like(state_h))])
	# Return a tuple in order to combine all update_ops into a single operation.
	# The tuple's actual value should not be used.
	return tf.tuple(update_ops)

# initial_state = cell.zero_state(batch_size, dtype=tf.float32)

# cell=tf.nn.rnn_cell.LSTMCell(256,initial_state=rnn_tuple_state)

# For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
# states_c,states_h = get_state_variables(batch_size, cell)
cell=tf.nn.rnn_cell.LSTMCell(512)

initial_state = cell.zero_state(batch_size, tf.float32)

outputs, state = tf.nn.dynamic_rnn(cell, x ,dtype=tf.float32)

# Add an operation to update the train states with the last state tensors.
# update_op = get_state_update_op(states_c,states_h, state)

output=tf.reshape(outputs,[-1,50*neurons])

def reset_c_h(cell):
	state_c, state_h=cell.zero_state(batch_size, tf.float32)
	state_c=tf.zeros_like(state_c)
	state_h=tf.zeros_like(state_h)
	initialstates=tf.nn.rnn_cell.LSTMStateTuple(state_c,state_h)
	cell=tf.nn.rnn_cell.LSTMCell(512,initialstates)
	return cell

def initializeLayer(inputs,outputs):
    return tf.Variable(tf.truncated_normal([inputs,outputs],stddev=0.1)), tf.Variable(tf.zeros([outputs]))

def lossCompute():
	print(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)))
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

def trainstep(alpha, crossentropy):
    return tf.train.GradientDescentOptimizer(alpha).minimize(crossentropy)

def findaccuracy(y,y_):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_prediction,accuracy

with tf.Session() as sess:
	# # Compute the zero state array of the right shape once
	# zero_state = sess.run(initial_state)

	# # Start with a zero vector and update it 
	# cur_state = zero_state

	W1,B1=initializeLayer(50*neurons,neurons)
	W2,B2=initializeLayer(neurons,5)
	sess.run(tf.global_variables_initializer())
    #x = tf.placeholder(tf.float32, shape=[None,102400])
    #y_ = tf.placeholder(tf.float32, shape=[None, 5])
	y1 = tf.matmul(output,W1) + B1
    #fy1 = tf.nn.leaky_relu(y1)
	y = tf.matmul(y1,W2) + B2
	cross_entropy=lossCompute()
	train_step = trainstep(1e-3,cross_entropy)
	correct_prediction,accuracy=findaccuracy(y,y_)
	for i in range(25):
		for j in range(6):
			reset_c_h(cell)
			# cur_state, _ = sess.run([state, ...], feed_dict={initial_state=cur_state, ...})
			batchData,batchLabel=create_batches(j*98,98)
			train_step.run(feed_dict={x: batchData, y_: batchLabel})
		print("accuracy", accuracy.eval(feed_dict={x: valData, y_: valLabel}))
