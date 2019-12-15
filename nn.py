#!/usr/bin/python

"""
Extended for batch training but converging very slowly.
An avarage is used after running one batch in this case
epoch on delta_out of all neurons.

Maybe convergance would be faster if the size of the batch
would not be as big as a whole data.
"""

import random
import time
import math
import sys
import os
import json
import copy
import pickle
import numpy as np

from scipy import array, optimize
from optparse import OptionParser
from pylab import subplot, plot, grid, title, ion, clf
from pylab import show, setp, rcParams, imshow, draw
# rand, zeros, bar, cla, cm, mpl
from multiprocessing import cpu_count
# from multiprocessing import Pool


def get_uniqe_id():
    return "%s_%s" % (int(time.time()), int(random.random()*1000))


class Sigmoid:
    """ Sigmoid activation function.  """
    # from my tests: np.finfo(np.float64).max = 1.7976931348623157e+308
    # and exp(a) < max, thus a < ln(max) => clamp
    # on a should be < 709 to avoid overflow
    clamp = 500

    def __call__(self, x):
        # clamp values to avoid numerical overflow/underflow
        if x >= self.clamp:
            x = self.clamp
        elif x <= -self.clamp:
            x = -self.clamp
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(self, x):
        return x * (1.0 - x)


class ArcTan:
    """
    Hyperbolic Tangent activation function
    (commonly referred to as atan or tan^-1)
    """

    def __call__(self, x):
        return math.tanh(x)

    def derivative(self, x):
        return 1.0 - (x ** 2)


class CustomPrint():

    def __init__(self, uid):
        self.old_stdout = sys.stdout
        self.uid = uid

    def write(self, text):
        text = text.rstrip()
        if len(text) == 0:
            return
        self.old_stdout.write(self.uid + " " + text + '\n')

    def flush(self):
        self.old_stdout.flush()


class LayerOutputSnapshot(object):

    def __init__(self, size_limit, layer_id, neurons_ids):
        self.size_limit = size_limit
        self.outputs = [[]] * size_limit
        self.layer_id = layer_id
        self.neurons_ids = neurons_ids


class TrainingSnapshot(object):

    def __init__(
            self, neurons_group=None,
            norm_inputs=None, norm_targets=None,
            active_layer_output_snapshot=None):
        self.neurons_group = copy.deepcopy(neurons_group)
        self.norm_inputs = copy.deepcopy(norm_inputs)
        self.norm_targets = copy.deepcopy(norm_targets)
        self.active_layer_output_snapshot = copy.deepcopy(
            active_layer_output_snapshot)


class Context(object):

    def __init__(self):
        self.network_command_fpath = ""
        self.last_training_snapshot = TrainingSnapshot()
        self.training_snapshots = []
        self.snapshot_layer = False

    def save(
            self, file_path_last_training_snapshot=None,
            file_path_training_snapshots=None):
        if file_path_last_training_snapshot:
            with open(file_path_last_training_snapshot, "w") as fd:
                pickle.dump(self.training_snapshots[-1], fd)
        if file_path_training_snapshots:
            with open(file_path_training_snapshots, "w") as fd:
                pickle.dump(self.training_snapshots, fd)

    def load(
            self, file_path_last_training_snapshot,
            file_path_training_snapshots=None):
        if file_path_last_training_snapshot:
            with open(file_path_last_training_snapshot, "r") as fd:
                self.last_training_snapshot = pickle.load(fd)
        if file_path_training_snapshots:
            with open(file_path_training_snapshots, "r") as fd:
                self.training_snapshots = pickle.load(fd)


def get_chunks_indexes(interval_size, count):
    """
    To generate count amount of chunks
    in interval <0, interval_size>

    e.g.:
    >>> get_chunks_indexes(10, 3)
    [(0, 3), (3, 6), (6, 10)
    """
    n = interval_size / count
    i = (count - interval_size % count) * n
    idx = range(n, i, n) + range(i, interval_size, n+1)
    idx1 = [0] + idx
    idx2 = idx + [interval_size]
    return zip(idx1, idx2)


def print_result(result):
    print
    for i in result:
        print i


def print_neurons(neurons, neuron_ids):
    for i, nid in enumerate(neuron_ids):
        print i, neurons[nid]


def random_weight(smallwt):
    # return 2.0 * (random.random() - 0.5) * smallwt
    # return 20 * (random.random() - 0.5)
    return random.random() - 0.5


class NormalizedArray:
    """ Array normalized by using linear transformation
        y = ax + b
    """

    def __init__(self, array, lower_limit=0., upper_limit=1.):
        self.array = copy.copy(array)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.limits, self.encode, self.decode = self._norms()
        self.norm_array = self._norm_array()

    def _linear(self, a, b, c, d):
        """
        Returns coefficients of linear eqasion a, b. y = ax + b
        from range (a,b) to (c,d)
        """
        """
        if b == a: raise ValueError(
            "Mapping not possible due to equal limits
            ")
        """
        if b == a:
            c1 = 0.0
            c2 = (c + d) / 2.
        else:
            c1 = (d - c) / (b - a)
            c2 = c - a * c1
        return c1, c2

    def _norms(self):
        """ Gets normalization information from an array

        (self.lower_limit, self.upper_limit) is a range of normalization.
        in_array is 2-dimensional, normalization parameters are computed
        for each column...
        """
        limits = []
        encode_linear = []
        decode_linear = []
        t_array = array(self.array).transpose()
        for col in t_array:
            max_arr = max(col)
            min_arr = min(col)
            limits += [(min_arr, max_arr)]
            encode_linear += [self._linear(
                min_arr, max_arr, self.lower_limit, self.upper_limit)]
            decode_linear += [
                self._linear(
                    self.lower_limit,
                    self.upper_limit, min_arr, max_arr)]
        return array(limits), array(encode_linear), array(decode_linear)

    def _norm_array(self):
        """ Normalize 2-dimensional array linearly column by column.  """
        self.array = array(self.array).transpose()
        if not self.array.dtype == np.dtype("float"):
            self.array = self.array.astype("float")
        i = self.array.shape[0]
        for ii in xrange(i):
            self.array[ii] = self.array[ii] * self.encode[ii, 0] \
                + self.encode[ii, 1]
        return self.array.transpose()


def de_norm_array(array, decode_coeff):
    new_array = []
    for i in range(0, len(array)):
        new_array.append(array[i] * decode_coeff[i, 0] + decode_coeff[i, 1])
    return new_array


# class NeuronsGroup(object):
class NeuronGroup(object):

    def __init__(
            self, neuron_count=None, layers=None,
            inactivate_neurons_biases=False, training_mode=1,
            activation_function=Sigmoid):
        self.start_neurons_ids = []
        self.activation_function = activation_function()
        self.neuron_count = neuron_count
        self.training_mode = training_mode
        self.input_ids = []
        self.output_ids = []
        self.layers = layers
        self.inactivate_neurons_biases = inactivate_neurons_biases
        self.neurons = []
        self.epoch = 0
        self.epoch_error = 0
        self.init_time = time.time()
        self.epoch_time = None
        self.smallwt = 0.5
        # speed of learning
        self.eta = 0.9
        # momentum
        self.alpha = 0.3
        self.output_value_zones = (
            (0.85, 1.0),
            (0.15, 0.85),
            (-0.1, 0.15)
        )
        if neuron_count:
            for i in range(0, self.neuron_count):
                # delta_weight = 0.0
                # input_ptr = None
                output = 0
                if self.inactivate_neurons_biases:
                    bias = 0.0
                else:
                    # bias = 0.0
                    bias = random_weight(self.smallwt)
                delta_bias = random_weight(self.smallwt)
                gradient_bias = 0.0
                """
                [[ 0 weight, 1 delta_weight,
                2 input_ptr, 3 gradient_weight]...]
                """
                w_dw_ip_gw = []
                output_ptr = []
                delta_out = 0.0
                delta_out_batch_sum = 0.0
                delta_bias_batch_sum = 0.0
                training_lock = False
                output_values_counters = [0, 0, 0]
                """
                0 i            1 bias    2 delta_bias
                3 w_dw_ip_gw   4 output  5 output_ptr
                6 delta_out    7 eta     8 alpha
                9 delta_out_batch_sum
                10 delta_bias_batch_sum  11 gradient_bias
                12 training_lock
                13 output_values_counters
                """
                neuron = [
                    i, bias, delta_bias, w_dw_ip_gw, output,
                    output_ptr, delta_out, self.eta, self.alpha,
                    delta_out_batch_sum, delta_bias_batch_sum,
                    gradient_bias, training_lock, output_values_counters
                ]
                self.neurons.append(neuron)

    def set_start_neurons_ids(self, neurons_ids):
        self.start_neurons_ids = copy.copy(neurons_ids)

    def reset_init_time(self):
        self.init_time = time.time()

    def wire(self, neuron_connection_map):
        for i, j in neuron_connection_map:
            # print "i %s, j %s" % (i,j)
            self.neurons[i][5].append(j)
            self.neurons[j][3].append([
                random_weight(self.smallwt), 0.0, i, 0.0])

    def set_input_and_output_ids(self, inp, out):
        self.input_ids = inp
        self.output_ids = out
        self.set_start_neurons_ids(inp)

    def get_results(self):
        results = []
        for id in self.output_ids:
            results.append(self.neurons[id][4])
        return results

    def get_neurons_up_to_layer_position(self, position):
        neurons = []
        if position > 1:
            length = sum(
                self.layers[0:position]
            )
            neurons = self.neurons[0:length]
        return neurons

    def set_neurons(self, neurons):
        for neuron in neurons:
            nid = neuron[0]
            saved_output_ptr = self.neurons[nid][5]
            self.neurons[nid] = neuron
            self.neurons[nid][5] = saved_output_ptr

    def copy_old_neuron(self, neuron_id, old_neuron):
        # Copy a whole neuron without its INT_PTR & OUT_PTR
        saved_output_ptr = self.neurons[neuron_id][5]
        saved_input_ptr = self.neurons[neuron_id][3]
        self.neurons[neuron_id] = copy.copy(old_neuron)
        self.neurons[neuron_id][5] = saved_output_ptr
        self.neurons[neuron_id][3] = saved_input_ptr
        # Copy old neuron INT_PTR at beginning of new INT_PTR
        for index in range(0, len(self.neurons[neuron_id][3])):
            if index < len(old_neuron[3]):
                self.neurons[neuron_id][3][index][0] = old_neuron[3][index][0]
                self.neurons[neuron_id][3][index][1] = old_neuron[3][index][1]
                self.neurons[neuron_id][3][index][3] = old_neuron[3][index][3]

    def copy_in_old_neurons(self, old_neurons_group):
        for layer_index, layer_size in enumerate(self.layers[1:]):
            # did layer exit in old network ?
            if layer_index < len(old_neurons_group.layers[1:]):
                for neuron_index_in_layer in range(0, layer_size):
                    expected_old_neuron_id = sum(old_neurons_group.layers[0:layer_index+1]) + neuron_index_in_layer
                    new_neuron_id = sum(self.layers[0:layer_index+1]) + neuron_index_in_layer
                    # did neuron exist in old layer ?
                    if expected_old_neuron_id < sum(old_neurons_group.layers[0:layer_index+2]):
                        #print "layer_index %s expected_old_neuron_id %s new_neuron_id %s" % (layer_index, expected_old_neuron_id, new_neuron_id)
                        self.copy_old_neuron(
                            new_neuron_id,
                            old_neurons_group.neurons[expected_old_neuron_id])

    def get_biases_and_weights(self, get_biases=False):
        out = []
        next_neurons_ids = get_next_neurons(self.neurons, self.input_ids)
        while len(next_neurons_ids) != 0:
            for index in next_neurons_ids:
                if get_biases:
                    out.append(self.neurons[index][1])
                for w_wd_ip in self.neurons[index][3]:
                    out.append(w_wd_ip[0])
            next_neurons_ids = get_next_neurons(self.neurons, next_neurons_ids)
        return array(out).astype("float")

    def set_biases_and_weights(self, biases_and_weights, set_biases=False):
        counter = 0
        next_neurons_ids = get_next_neurons(self.neurons, self.input_ids)
        while len(next_neurons_ids) != 0:
            # print "next_neurons_ids", next_neurons_ids
            for i in next_neurons_ids:
                # print "next_neurons_ids i, counter", i, counter
                # print "biases_and_weights", biases_and_weights
                if set_biases:
                    self.neurons[i][1] = biases_and_weights[counter]
                    counter += 1
                for j in range(0, len(self.neurons[i][3])):
                    self.neurons[i][3][j][0] = biases_and_weights[counter]
                    counter += 1
            next_neurons_ids = get_next_neurons(self.neurons, next_neurons_ids)

    def get_gradient_of_biases_and_weights(self, get_gradient_of_biases=False):
        out = []
        next_neurons_ids = get_next_neurons(self.neurons, self.input_ids)
        while len(next_neurons_ids) != 0:
            for index in next_neurons_ids:
                if get_gradient_of_biases:
                    out.append(self.neurons[index][11])
                for w_wd_ip_gw in self.neurons[index][3]:
                    out.append(w_wd_ip_gw[3])
            next_neurons_ids = get_next_neurons(self.neurons, next_neurons_ids)
        return array(out).astype("float")

    def reset_gradient_of_biases_and_weights(self):
        for i, neuron in enumerate(self.neurons):
            self.neurons[i][11] = 0.0
            for j, w_wd_ip_gw in enumerate(self.neurons[i][3]):
                self.neurons[i][3][j][3] = 0.0

    def back_propagate_error(
            self, neurons_ids, target_vector,
            batch_learning=False, gradient=False):
        """ To update error (delta_out) neurons layer

            neurons     - all neurons
            neurons_ids - neurons ids in n layer
        """
        if sorted(neurons_ids) == sorted(self.output_ids):
            for i, nid in enumerate(neurons_ids):
                if not self.neurons[nid][12]:
                    self.neurons[nid][6] = (target_vector[i] - self.neurons[nid][4]) * self.activation_function.derivative(self.neurons[nid][4])
                    self.neurons[nid][2] = self.neurons[nid][7] * self.neurons[nid][6] + self.neurons[nid][8] * self.neurons[nid][2]
                    if gradient:
                        update_neuron_biases_and_weights_gradients(self.neurons, nid)
        else:
            for nid in neurons_ids:
                # n_sum = self.neurons[nid][1]
                n_sum = 0.0
                for n_nid in self.neurons[nid][5]:
                    for w_wd_ip_gw in self.neurons[n_nid][3]:
                        if w_wd_ip_gw[2] == self.neurons[nid][0]:
                            n_sum += w_wd_ip_gw[0] * self.neurons[n_nid][6]
                            break  # only one is connected to previous one
                if not self.neurons[nid][12]:
                    self.neurons[nid][6] = n_sum * self.activation_function.derivative(self.neurons[nid][4])
                    self.neurons[nid][2] = self.neurons[nid][7] * self.neurons[nid][6] + self.neurons[nid][8] * self.neurons[nid][2]
                    if gradient:
                        update_neuron_biases_and_weights_gradients(
                            self.neurons, nid)
        return get_previous_neurons(self.neurons, neurons_ids)

    def update_weights(self, neurons_ids, *args, **kwargs):
        """ To update delta_weight, weight, delta_bias & bias of neurons

            neurons     - all neurons
            neurons_ids - neurons ids for update
        """
        for nid in neurons_ids:
            """
            neurons[nid][2] = neurons[nid][7] * neurons[nid][6]
                + neurons[nid][8] * neurons[nid][2]
            """
            if not self.neurons[nid][12]:
                self.neurons[nid][1] += self.neurons[nid][2]
                for i, w_wd_ip_gw in enumerate(self.neurons[nid][3]):
                    self.neurons[nid][3][i][1] = self.neurons[nid][7] * self.neurons[w_wd_ip_gw[2]][4] * self.neurons[nid][6] + self.neurons[nid][8] * w_wd_ip_gw[1]
                    self.neurons[nid][3][i][0] += self.neurons[nid][3][i][1]
        return get_previous_neurons(self.neurons, neurons_ids)

    def spike_on_input(self, input_vector):
        next_neurons_ids = set()
        for i, nid in enumerate(self.input_ids):
            self.neurons[nid][4] = input_vector[i]
            next_neurons_ids.update(self.neurons[nid][5])
        return next_neurons_ids

    def get_layer_outputs(self, layer_neurons_ids):
        outputs = []
        for i, nid in enumerate(layer_neurons_ids):
            outputs.append(self.neurons[nid][4])
        return outputs

    def set_layer_outputs(self, layer_neurons_ids, input_vector):
        next_neurons_ids = set()
        for i, nid in enumerate(layer_neurons_ids):
            self.neurons[nid][4] = input_vector[i]
            next_neurons_ids.update(self.neurons[nid][5])
        return next_neurons_ids

    def spike_on_output(
            self, target_vector, output_value_zones,
            batch_learning=False, gradient=False):
        error = 0.0
        for i, nid in enumerate(sorted(self.output_ids)):
            n_sum = self.neurons[nid][1]
            see = []
            for w_wd_ip_gw in self.neurons[nid][3]:
                n_sum += self.neurons[w_wd_ip_gw[2]][4] * w_wd_ip_gw[0]
                see.append("%s * %s" % (
                    self.neurons[w_wd_ip_gw[2]][4], w_wd_ip_gw[0]))
            try:
                self.neurons[nid][4] = self.activation_function(n_sum)
            except Exception, e:
                sys.stderr.write("spike_on_output %s %s %s %s\n" % (
                    e, self.neurons[nid][4], -n_sum, see))
            update_neuron_output_value_zones_counters(
                output_value_zones,
                self.neurons,
                nid
            )
            error += (target_vector[i] - self.neurons[nid][4]) ** 2 * 0.5
            self.neurons[nid][6] = (target_vector[i] - self.neurons[nid][4]) * self.neurons[nid][4] * (1.0 - self.neurons[nid][4])
            self.neurons[nid][2] = self.neurons[nid][7] * self.neurons[nid][6] + self.neurons[nid][8] * self.neurons[nid][2]
            if batch_learning:
                self.neurons[nid][9] += self.neurons[nid][6]
                self.neurons[nid][10] += self.neurons[nid][2]
        return error

    def spike_on_group_output(self, output_value_zones, neurons_ids):
        next_neurons_ids = set()
        for nid in neurons_ids:
            n_sum = self.neurons[nid][1]
            for w_wd_ip in self.neurons[nid][3]:
                n_sum += self.neurons[w_wd_ip[2]][4] * w_wd_ip[0]
            try:
                self.neurons[nid][4] = self.activation_function(n_sum)
            except Exception, e:
                sys.stderr.write(
                    "spike_on_group_output %s %s %s\n" % (
                        e, self.neurons[nid][4], -n_sum))
                # self.neurons[nid][4] = 0.0
            update_neuron_output_value_zones_counters(
                output_value_zones,
                self.neurons,
                nid
            )
            next_neurons_ids.update(self.neurons[nid][5])
        return next_neurons_ids

    def sse_error(self, target_vector):
        error = 0.0
        for i, nid in enumerate(self.output_ids):
            error += (target_vector[i] - self.neurons[nid][4]) ** 2 * 0.5
        return error

    def network_propagate_inputs_forward(
            self, input_vec,
            target_vec, batch_learning=False, gradient=False):
        """ To propagate inputs through the network from
            the starting layer to the last layer.
        """
        # print "network_propagate_inputs_forward", self.start_neurons_ids
        next_n_ids = self.set_layer_outputs(self.start_neurons_ids, input_vec)
        # print "next_n_ids", next_n_ids
        while len(next_n_ids) != 0:
            next_n_ids = self.spike_on_group_output(
                self.output_value_zones, next_n_ids)
        return self.sse_error(target_vec)

    def execute_function_on_network_walking_backwards(
            self, function, *args, **kwargs):
        """ To execute function function with its arguments on network
            walking from last towards the first layer. The first input
            layer is not processed.
        """
        previous_neurons_ids = function(self.output_ids, *args, **kwargs)
        #while sorted(previous_neurons_ids) != sorted(self.input_ids):
        while sorted(previous_neurons_ids) != sorted(self.start_neurons_ids):
            previous_neurons_ids = function(
                previous_neurons_ids, *args, **kwargs)

    def network_update_weights_backwards(self):
        """ To update network neurons weights backwards.
            Starting from the last layer and continuing towards
            to the first layer.
        """
        self.execute_function_on_network_walking_backwards(self.update_weights)

    def network_propagate_error_backwards(
            self, target_vec, batch_learning=False, gradient=False):
        """ To update network neurons error (delta_out) backwards.
            Starting from the last layer and continuing towards
            to the first layer.
        """
        self.execute_function_on_network_walking_backwards(
            self.back_propagate_error,
            target_vec,
            batch_learning,
            gradient
        )

# global variables
context = Context()

norm_inputs = None
norm_targets = None
neurons_group = None
errors = []
epochs = []

custom_print = CustomPrint(get_uniqe_id())
sys.stdout = custom_print
save_net_fpath = None
report_every_epoch = 10

next_layer_output_snapshot = None


def get_neuron_layer_index(neurons_group, neuron_id):
    neuron_layer_index = -1
    neuron_index_in_layer = -1
    layer_added_up = copy.copy(neurons_group.layers)
    for index, layer in enumerate(neurons_group.layers):
        if index > 0:
           layer_added_up[index] += layer_added_up[index-1]
    for index, neuron in enumerate(neurons_group.layers):
        if neuron_id < layer_added_up[index]:
            neuron_layer_index = index
            break
    if neuron_layer_index != -1:
        neuron_index_in_layer = neurons_group.layers[neuron_layer_index] - \
            (layer_added_up[neuron_layer_index] - neuron_id)
    return neuron_layer_index, neuron_index_in_layer


def reset_output_values_couters(neurons):
    for nid, neuron in enumerate(neurons):
        for idx, value in enumerate(neurons[nid][13]):
            neurons[nid][13][idx] = 0


def print_output_values_couters(neurons):
    zones = []
    for neuron in neurons:
        zones.append(neuron[13])
    print zones


def update_neuron_output_value_zones_counters(
        output_value_zones, neurons, neuron_id):
    for index, zone in enumerate(output_value_zones):
        down, up = zone
        if neurons[neuron_id][4] <= up and neurons[neuron_id][4] > down:
            neurons[neuron_id][13][index] += 1
            break


def set_neurons_training_lock(neurons, neurons_ids, lock=True):
    for nid in neurons_ids:
        neurons[nid][12] = lock


def set_individual_neurons_training_locks(neurons, neurons_ids, neurons_locks):
    for index, nid in enumerate(neurons_ids):
        neurons[nid][12] = neurons_locks[index]


def get_all_neurons_layers(neurons, input_neurons_ids):
    neurons_layers = [set(input_neurons_ids)]
    neurons_layers.append(
        get_next_neurons(neurons, neurons_layers[-1])
    )
    while len(neurons_layers[-1]) != 0:
        neurons_layers.append(
            get_next_neurons(neurons, neurons_layers[-1])
        )
    return neurons_layers[0:-1]


def get_next_neurons(neurons, neurons_ids):
    next_ids = set()
    for nid in neurons_ids:
        for next_id in neurons[nid][5]:
            next_ids.add(next_id)
    return next_ids


def get_previous_neurons(neurons, neurons_ids):
    previous_n_ids = set()
    for nid in neurons_ids:
        for w_wd_ip in neurons[nid][3]:
            previous_n_ids.add(w_wd_ip[2])
    return previous_n_ids


def apply_layers_locks(neurons_group, layers_lock_map):
    next_n_ids = set(neurons_group.input_ids)
    layer_counter = 0
    while next_n_ids:
        if type(layers_lock_map[layer_counter]) is type(True):
            set_neurons_training_lock(
                neurons_group.neurons,
                next_n_ids,
                layers_lock_map[layer_counter]
            )
        else:
            set_individual_neurons_training_locks(
                neurons_group.neurons,
                next_n_ids,
                layers_lock_map[layer_counter]
            )
        next_n_ids = get_next_neurons(neurons_group.neurons, next_n_ids)
        layer_counter += 1


def multiprocess_function(pool, function_ptr, args, **kw_args):
    ids = set()
    results = []
    neurons_ids_copy = []
    if "neurons_ids" in kw_args:
        neurons_ids_copy[:] = kw_args["neurons_ids"]
    sequences = get_chunks_indexes(len(neurons_ids_copy), pool._processes)
    for sequence in sequences:
        if "neurons_ids" in kw_args:
            kw_args["neurons_ids"] = neurons_ids_copy[sequence[0]:sequence[1]]
        print "neurons_ids", kw_args["neurons_ids"]
        results += [pool.apply_async(function_ptr, (args, kw_args))]
    for r in results:
        ids.add(r.get())
    print "ids", ids
    return ids


"""
# save for multiprocessing
def multiprocess_spike_on_input(pool, input_vector, neurons, neurons_ids):
    ids = set()
    results = []
    sequences = get_chunks_indexes(len(neurons_ids), pool._processes)
    for sequence in sequences:
        results += [
            pool.apply_async(
                spike_on_input,
                (input_vector, neurons, neurons_ids[sequence[0]:sequence[1]])
            )
        ]
    for r in results:
        ids.update(r.get())
    return list(ids)


# save for multiprocessing
def multiprocess_spike_on_group_output(pool, neurons, neurons_ids):
    if type(set([])) == type(neurons_ids):
        neurons_ids = list(neurons_ids)
    ids = set()
    results = []
    sequences = get_chunks_indexes(len(neurons_ids), pool._processes)
    for sequence in sequences:
        results += [
            pool.apply_async(
                spike_on_group_output,
                (neurons, neurons_ids[sequence[0]:sequence[1]])
            )
        ]
    for r in results:
        ids.update(r.get())
    return list(ids)

def multiprocess_spike_on_output(
        pool, target_vector, neurons,
        neurons_ids, batch_learning=False,
        gradient=False):
    error = 0.0
    results = []
    sequences = get_chunks_indexes(len(neurons_ids), pool._processes)
    for sequence in sequences:
        results += [
            pool.apply_async(
                spike_on_output,
                (target_vector, neurons,
                    neurons_ids[sequence[0]:sequence[1]],
                    batch_learning, gradient))
        ]
    for r in results:
        error += r.get()
    return error
"""


def avarage_neurons_daltas_per_batch(neurons, size):
    for n in neurons:
        # print "delta_out", n[6], n[9], size, n[9] / size
        # print "delta_bias, n[2], n[10], size, n[10] / size
        n[6] = n[9] / size
        n[2] = n[10] / size
        n[9] = 0.0
        n[10] = 0.0


def apply_accumulated_neurons_daltas_per_batch(neurons):
    for n in neurons:
        n[6] = n[9]
        n[2] = n[10]
        n[9] = 0.0
        n[10] = 0.0


def reset_neurons_daltas_batch_sum(neurons):
    for n in neurons:
        n[9] = 0.0
        n[10] = 0.0


def update_neuron_biases_and_weights_gradients(neurons, neuron_id):
    """
     bias' gradient
     very strange, it seems like using neurons[neuron_id][6]
     works better than neurons[neuron_id][2] for bias gradient
     when using it for optimisation
    """
    # neurons[neuron_id][11] += neurons[neuron_id][6]
    neurons[neuron_id][11] += neurons[neuron_id][2]
    for i, w_wd_ip_gw in enumerate(neurons[neuron_id][3]):
        # weights' gradients
        neurons[neuron_id][3][i][3] += neurons[neuron_id][6] * \
            neurons[w_wd_ip_gw[2]][4]

"""
def multiprocess_back_propagate_error(
        pool, neurons,
        neurons_ids, batch_learning=False,
        gradient=False):
    ids = set()
    results = []
    sequences = get_chunks_indexes(len(neurons_ids), pool._processes)
    for sequence in sequences:
        results += [
            pool.apply_async(
                back_propagate_error,
                (neurons, neurons_ids[sequence[0]:sequence[1]],
                    batch_learning, gradient)
            )
        ]
    for r in results:
        ids.update(r.get())
    return list(ids)
"""

"""
def multiprocess_update_weights(pool, neurons, neurons_ids):
    ids = set()
    results = []
    sequences = get_chunks_indexes(len(neurons_ids), pool._processes)
    for sequence in sequences:
        results += [
            pool.apply_async(
                update_weights,
                (neurons, neurons_ids[sequence[0]:sequence[1]])
            )
        ]
    for r in results:
        ids.update(r.get())
    return list(ids)
"""


def get_neurons_delta_out(neurons, neurons_ids):
    return [neurons[nid][6] for nid in neurons_ids]


def get_neurons_inputs(neurons, neurons_ids):
    return [neurons[neurons[nid][3][2]][4] for nid in neurons_ids]


def get_detla_of_biases_and_weights(neurons, input_neurons_ids):
    """ To get gradient of neurons' biases and weights """
    out = []
    next_neurons_ids = get_next_neurons(neurons, input_neurons_ids)
    while len(next_neurons_ids) != 0:
        for index in next_neurons_ids:
            # is following line correct ?
            # addition form delta bias should be = neurons[index][2] * 1
            out.append(neurons[index][2])
            for w_wd_ip in neurons[index][3]:
                out.append(neurons[index][6] * neurons[w_wd_ip[2]][4])
        next_neurons_ids = get_next_neurons(neurons, next_neurons_ids)
    return array(out).astype("float")


def get_layout_tepmlate(neurons_group, segments_per_neuron=1):
    rows = max(neurons_group.layers) * (segments_per_neuron + 1) + 1
    columns = 2 * len(neurons_group.layers) - 1
    return np.tile(
        np.nan,
        (rows, columns)
    )


def get_biases_and_weights_layout(neurons_group):
    neurons = neurons_group.neurons
    template = get_layout_tepmlate(neurons_group)
    next_neurons_ids = get_next_neurons(
        neurons,
        neurons_group.input_ids
    )
    t_column = 1
    while len(next_neurons_ids) != 0:
        t_row = 1
        for index, nid in enumerate(next_neurons_ids):
            template[t_row, t_column] = neurons[nid][1]
            for w_wd_ip in neurons[nid][3]:
                template[t_row, t_column] += w_wd_ip[0]
            template[t_row, t_column] /= (len(neurons[nid][3]) + 1)
            t_row += 2
        t_column += 2
        next_neurons_ids = get_next_neurons(
            neurons,
            next_neurons_ids
        )
    return template


def get_response(neurons, neurons_ids, input):
    global neurons_group
    response = []
    for nid in neurons_ids:
        n_sum = neurons[nid][1]
        for w_wd_ip in neurons[nid][3]:
            n_sum += input * w_wd_ip[0]
        try:
            response.append(neurons_group.activation_function(n_sum))
        except Exception, e:
            response.append(0.0)
            sys.stderr.write("%s %s\n" % (n_sum, e))
    return response


def get_response_layout(neurons_group, input):
    template = get_layout_tepmlate(neurons_group)
    next_neurons_ids = get_next_neurons(
        neurons_group.neurons,
        neurons_group.input_ids
    )
    t_column = 1
    while len(next_neurons_ids) != 0:
        t_row = 1
        for r in get_response(neurons_group.neurons, next_neurons_ids, input):
            template[t_row, t_column] = r
            t_row += 2
        t_column += 2
        next_neurons_ids = get_next_neurons(
            neurons_group.neurons,
            next_neurons_ids
        )
    return template


def get_neuron_zone_percentage(neurons, neuron_id, zone_index):
    one_perc = float(sum(neurons[neuron_id][13])) / 100
    return float(neurons[neuron_id][13][zone_index]) / one_perc


def get_neuron_zones_percentages(neurons, neuron_id):
    z_percentages = []
    for index, zone in enumerate(neurons[neuron_id][13]):
        z_percentages.append(
            get_neuron_zone_percentage(neurons, neuron_id, index)
        )
    return z_percentages


def get_all_output_value_zones_layout(neurons_group):
    template = get_layout_tepmlate(
        neurons_group, len(neurons_group.output_value_zones))
    next_neurons_ids = get_next_neurons(
        neurons_group.neurons,
        neurons_group.input_ids
    )
    t_column = 1
    while len(next_neurons_ids) != 0:
        t_row = 1
        for nid in next_neurons_ids:
            for r in get_neuron_zones_percentages(neurons_group.neurons, nid):
                template[t_row, t_column] = r
                t_row += 1
            t_row += 1
        t_column += 2
        next_neurons_ids = get_next_neurons(
            neurons_group.neurons,
            next_neurons_ids
        )
    return template


def get_output_value_zone_layout(neurons_group, zone_index):
    template = get_layout_tepmlate(neurons_group)
    next_neurons_ids = get_next_neurons(
        neurons_group.neurons,
        neurons_group.input_ids
    )
    t_column = 1
    while len(next_neurons_ids) != 0:
        t_row = 1
        for nid in next_neurons_ids:
            template[t_row, t_column] = get_neuron_zone_percentage(
                neurons_group.neurons,
                nid,
                zone_index
            )
            t_row += 2
        t_column += 2
        next_neurons_ids = get_next_neurons(
            neurons_group.neurons,
            next_neurons_ids
        )
    return template


def get_biases(neurons, input_neurons_ids):
    out = []
    next_neurons_ids = get_next_neurons(neurons, input_neurons_ids)
    while len(next_neurons_ids) != 0:
        for index in next_neurons_ids:
            out.append(neurons[index][1])
        next_neurons_ids = get_next_neurons(neurons, next_neurons_ids)
    return array(out).astype("float")


def get_weights(neurons, input_neurons_ids):
    out = []
    next_neurons_ids = get_next_neurons(neurons, input_neurons_ids)
    while len(next_neurons_ids) != 0:
        for index in next_neurons_ids:
            for w_wd_ip in neurons[index][3]:
                out.append(w_wd_ip[0])
        next_neurons_ids = get_next_neurons(neurons, next_neurons_ids)
    return array(out).astype("float")


def get_biases_average(neurons, input_neurons_ids):
    return get_biases(neurons, input_neurons_ids).mean()


def get_biases_standart_deviation(neurons, input_neurons_ids):
    return get_biases(neurons, input_neurons_ids).std()


def get_weights_average(neurons, input_neurons_ids):
    return get_weights(neurons, input_neurons_ids).mean()


def get_weights_standart_deviation(neurons, input_neurons_ids):
    return get_weights(neurons, input_neurons_ids).std()


def error_function(
        biases_and_weights, neurons_group,
        norm_inputs, norm_targets, data_sequence):
    start = data_sequence[0]
    end = data_sequence[-1]
    length = end - start
    batch_learning = False
    error = 0.0
    # neurons_group.reset_gradient_of_biases_and_weights()
    neurons_group.set_biases_and_weights(
        copy.copy(biases_and_weights),
        not neurons_group.inactivate_neurons_biases
    )
    # compute error over all inputs
    for r in random.sample(range(start, end), length):
        set_error = neurons_group.network_propagate_inputs_forward(
            norm_inputs.norm_array[r],
            norm_targets.norm_array[r],
            batch_learning
        )
        error += set_error
    return error


def error_gradient_function(
        biases_and_weights, neurons_group,
        norm_inputs, norm_targets, data_sequence):
    start = data_sequence[0]
    end = data_sequence[-1]
    length = end - start
    neurons_group.reset_gradient_of_biases_and_weights()
    # g = neurons_group.get_gradient_of_biases_and_weights(
    neurons_group.get_gradient_of_biases_and_weights(
        not neurons_group.inactivate_neurons_biases
    )
    neurons_group.set_biases_and_weights(
        copy.copy(biases_and_weights),
        not neurons_group.inactivate_neurons_biases)
    # compute gradient over all inputs
    for r in random.sample(range(start, end), length):
        # for r in range(start, end):
        neurons_group.network_propagate_inputs_forward(
            norm_inputs.norm_array[r],
            norm_targets.norm_array[r],
            batch_learning=False,
            gradient=True
        )
        neurons_group.network_propagate_error_backwards(
            norm_targets.norm_array[r], batch_learning=False, gradient=True
        )
    return neurons_group.get_gradient_of_biases_and_weights(
        not neurons_group.inactivate_neurons_biases)


def multiprocess_error_function(
        biases_and_weights, pool,
        neuron_group, norm_inputs, norm_targets, data_sequences,
        number_of_paralel_process):
    res = []
    for data_sequence in data_sequences:
        # print "data_sequence:", data_sequence
        res += [pool.apply_async(
            error_function,
            (biases_and_weights, neuron_group, norm_inputs,
                norm_targets, data_sequence))]
    c = [r.get() for r in res]
    r = sum(c)
    # print "multiprocess_error_function error:", r
    # return r / number_of_paralel_process
    return r


def mutliprocess_error_gradient_function(
        biases_and_weights, pool,
        neuron_group, norm_inputs, norm_targets, data_sequences,
        number_of_paralel_process):
    res = []
    for data_sequence in data_sequences:
        res += [pool.apply_async(
            error_gradient_function,
            (biases_and_weights, neuron_group,
                norm_inputs, norm_targets, data_sequence))]
    c = [r.get() for r in res]
    r = sum(c)
    # print "mutliprocess_error_gradient_function gradient:", r
    # return r / number_of_paralel_process
    return r


class NetworkGraph(object):
    """ Network graph of neural network where
        every layer is fully connected to following
        layer.
    """

    def __init__(self, layers):
        self.layers = layers
        self.connections = self._generate_connections()
        self.set_input_and_output_ids()
        self.count = sum(self.layers)

    def set_input_and_output_ids(self):
        self.input_ids = range(0, self.layers[0])
        last_id = sum(self.layers)
        self.output_ids = range(last_id - self.layers[-1], last_id)

    def _generate_connections(self):
        """
        from (2, 3, 2, 1)

        0 2 5
        1 3 6 7
          4

        generates

        [
            [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4],
            [2, 5], [2, 6], [3, 5], [3, 6], [4, 5], [4, 6],
            [5, 7], [6, 7]
        ]
        """
        connections = []
        counter_i = 0
        for l in range(0, len(self.layers) - 1):
            connections_ij = []
            counter_j = counter_i + self.layers[l]
            for i in range(counter_i, counter_i + self.layers[l]):
                for j in range(counter_j, counter_j + self.layers[l+1]):
                    connections_ij.append([i, j])
            counter_i += self.layers[l]
            connections.extend(connections_ij)
        return connections


def plot_distribution(data, subplot_id):
    sp = subplot(subplot_id)
    grid(True)
    number_of_bars = 30
    n, bins, patches = sp.hist(
        data, number_of_bars, normed=1, histtype="bar"
    )
    setp(patches, "facecolor", "g", "alpha", 0.75)
    return subplot_id + 1


def plot_data(x, y, subplot_id):
    subplot(subplot_id)
    plot(x, y, "b--")
    grid(True)
    # title("Progress of network error.")
    return subplot_id + 1


def plot_targets_versus_learned(targets_and_learned, subplot_id):
    # print "targets_and_learned", targets_and_learned
    for i, (target, learned) in enumerate(targets_and_learned):
        subplot(subplot_id)
        plot(target, "b--", learned, "k-")
        grid(True)
        # title("Targets versus learned - output " + str(i))
        subplot_id += 1
    return subplot_id


def build_targets_versus_learned(
        neuron_group, norm_inputs, norm_targets):
    outputs_count = len(norm_targets.norm_array[0])
    records_count = len(norm_targets.norm_array)
    targets_versus_learned = np.empty((records_count, 2, outputs_count))
    de_norm_outpus = []
    for target in norm_targets.norm_array:
        de_norm_outpus.append(
            de_norm_array(target, norm_targets.decode)
        )
    for i in range(0, len(norm_inputs.norm_array)):
        # set_error = neurons_group.network_propagate_inputs_forward(
        neurons_group.set_start_neurons_ids(neurons_group.input_ids)
        neurons_group.network_propagate_inputs_forward(
            norm_inputs.norm_array[i],
            norm_targets.norm_array[i]
        )
        results = de_norm_array(
            neuron_group.get_results(), norm_targets.decode)
        for j in range(0, outputs_count):
            # targets_versus_learned[i][0][j] = inputs_and_targets["targets"][i][j]
            targets_versus_learned[i][0][j] = de_norm_outpus[i][j]
            targets_versus_learned[i][1][j] = results[j]
    # print "targets_versus_learned", targets_versus_learned
    # print "targets_versus_learned transposed",
    #    targets_versus_learned.transpose()
    return targets_versus_learned.transpose()


def draw_error(data=None):
    global context, norm_inputs, norm_targets, neurons_group
    # train_data = data["training_data"]
    neurons_group = data["training_snapshot"].neurons_group
    training_snapshot = data["training_snapshot"]
    epochs_errors = []
    for ts in context.training_snapshots:
        epochs_errors.append([ts.neurons_group.epoch, ts.neurons_group.epoch_error])
    transposed_epochs_errors = array(epochs_errors).transpose()

    rcParams['figure.figsize'] = 15, 10
    subplot_num = 4
    subplot_id = (len(training_snapshot.norm_inputs.norm_array[0]) + subplot_num) * 100 + 11
    bw = sorted(neurons_group.get_biases_and_weights(
        not neurons_group.inactivate_neurons_biases))
    plot_data(
        transposed_epochs_errors[0][0:data["step"]],
        transposed_epochs_errors[1][0:data["step"]],
        subplot_id)
    plot_data(range(0, len(bw)), bw, subplot_id + 1)
    plot_distribution(
        get_biases(
            neurons_group.neurons,
            neurons_group.input_ids), subplot_id + 2)
    plot_distribution(
        get_weights(
            neurons_group.neurons,
            neurons_group.input_ids),
        subplot_id + 3)
    plot_targets_versus_learned(
        build_targets_versus_learned(
            neurons_group,
            training_snapshot.norm_inputs,
            training_snapshot.norm_targets
        ),
        subplot_id + 4
    )


def draw_outputs_values_zones(data=None):
    global neurons_group
    zones = str(neurons_group.output_value_zones)
    # print_output_values_couters(neuron_group.neurons)
    layout = get_all_output_value_zones_layout(neurons_group)
    print "Neurons' outputs values zones " + zones
    print layout
    imshow(layout, interpolation='nearest')
    title("Neurons' outputs values zones " + zones)


def draw_biases_and_weights_average(data=None):
    global neurons_group
    # layout = neurons_group.get_biases_and_weights_layout(
    #    not neurons_group.inactivate_neurons_biases)
    layout = get_biases_and_weights_layout(neurons_group)
    print layout
    imshow(layout, interpolation='nearest')
    title("Average of neurons' biases and weights.")


def replay_network_training(
        data={"speed": 1, "command": "show_biases_and_weights_average"}):
    global context, neurons_group
    commands_map = {
        "show_biases_and_weights_average": draw_biases_and_weights_average,
        "show_error": draw_error,
        "show_outputs_values_zones": draw_outputs_values_zones,
    }
    ion()
    show()
    try:
        for index, step in enumerate(range(
                0, len(context.training_snapshots), data["speed"])):
            print_training_stats(
                context.training_snapshots[step].neurons_group)
            #inputs = de_norm_array(
            #    context.training_snapshots[step].norm_inputs.norm_array,
            #    context.training_snapshots[step].norm_inputs.decode)
            #print "BBB"
            data.update({
                "step": step,
                "training_snapshot": context.training_snapshots[step],
            })
            commands_map[data["command"]](data)
            draw()
            time.sleep(0.05)
            clf()
    except Exception, e:
        print e
        pass
    time.sleep(40)


def run_network(data):
    global norm_inputs, norm_targets, neurons_group
    result = []
    error = 0.0
    neurons_group.set_start_neurons_ids(neurons_group.input_ids)
    length = len(norm_inputs.norm_array)
    for r in range(0, length):
        set_error = neurons_group.network_propagate_inputs_forward(
            norm_inputs.norm_array[r],
            norm_targets.norm_array[r]
        )
        error += set_error
        result.append([
            data["inputs"][r],
            data["targets"][r],
            de_norm_array(neurons_group.get_results(), norm_targets.decode)
        ])
    print "\nError = %s" % error
    print_result(result)
    return
    rcParams['figure.figsize'] = 15, 10
    subplot_num = 1
    subplot_id = (len(data["targets"][0]) + subplot_num) * 100 + 11
    plot_targets_versus_learned(
        build_targets_versus_learned(
            neurons_group, norm_inputs, norm_targets),
        subplot_id
    )
    title("Response to input: " + str(input))
    show()
    # time.sleep(20)


def optimize_callback(biases_and_weights):
    global context, report_every_epoch, norm_inputs, \
        norm_targets, neurons_group
    neurons_group.epoch += 1
    error = 0
    if neurons_group.epoch % report_every_epoch == 0:
        neurons_group.set_biases_and_weights(
            copy.copy(biases_and_weights),
            not neurons_group.inactivate_neurons_biases)
        for r in range(0, len(norm_inputs.norm_array)):
            set_error = neurons_group.network_propagate_inputs_forward(
                norm_inputs.norm_array[r],
                norm_targets.norm_array[r]
            )
            error += set_error
        neurons_group.epoch_error = error
        neurons_group.epoch_time = time.time()
        print_training_stats(neurons_group)
        # training_snapshot.snapshot_epoch(neurons_group)
        process_commands(context.network_command_fpath)
        test_conditional_commands()


def normalize_inputs(train_data):
    global norm_inputs, norm_targets
    norm_inputs = NormalizedArray(train_data["inputs"], 0.15, 0.85)
    norm_targets = NormalizedArray(train_data["targets"], 0.15, 0.85)


def load_network(last_epoch_network_file_path, all_epoch_network_file_path):
    global context, neurons_group
    context.load(last_epoch_network_file_path, all_epoch_network_file_path)
    neurons_group = context.last_training_snapshot.neurons_group
    # neurons_group.training_mode = 1
    # neurons_group.inactivate_neurons_biases = False
    # neurons_group.activation_function = Sigmoid()


def adjust_loaded_network():
    """ Adjust loaded network if there was output snapshot taken
        and amount if inputs or inputs themself changed.
    """
    global norm_inputs, context
    flag = False
    # print neurons_group.input_ids, context.last_training_snapshot.neurons_group.input_ids
    if context.last_training_snapshot.active_layer_output_snapshot != None:
        # if len(norm_inputs.norm_array) == len(context.last_training_snapshot.norm_inputs.norm_array):
        #     if not (norm_inputs.norm_array == context.last_training_snapshot.norm_inputs.norm_array).all():
        #         flag = True
        if len(neurons_group.input_ids) != len(context.last_training_snapshot.neurons_group.start_neurons_ids):
            flag = True
        if flag:
            neurons_group.set_start_neurons_ids(neurons_group.input_ids)
            train_from_snapshot({
                "position": context.last_training_snapshot.active_layer_output_snapshot.layer_id
            })


def build_network(layers, inactivate_neurons_biases, training_mode):
    global context, neurons_group
    network_graph = NetworkGraph(layers)
    neurons_group = NeuronGroup(
        network_graph.count, network_graph.layers,
        inactivate_neurons_biases, training_mode)
    context.neurons_group = neurons_group
    neurons_group.wire(network_graph.connections)
    neurons_group.set_input_and_output_ids(
        network_graph.input_ids, network_graph.output_ids)


def stop_training(data=None):
    global context, save_net_fpath
    context.save(save_net_fpath, save_net_fpath + ".epochs")
    print "Network saved to", save_net_fpath, save_net_fpath + ".epochs"
    sys.exit(0)

def modify_layers(data=None):
    global neurons_group
    """ data format

        {
            "position" : 2,
            "layers_neurons": [8, 4]
        }
    """
    old_neurons_group = copy.copy(neurons_group)

    saved_epoch = neurons_group.epoch
    saved_epoch_error = neurons_group.epoch_error
    saved_init_time = neurons_group.init_time
    saved_epoch_time = neurons_group.epoch_time
    saved_inactivate_neurons_biases = neurons_group.inactivate_neurons_biases
    saved_training_mode = neurons_group.training_mode

    new_layers = neurons_group.layers[0:data["position"]]
    new_layers.extend(data["layers_neurons"])
    print "New network layers", new_layers
    build_network(
        new_layers,
        saved_inactivate_neurons_biases,
        saved_training_mode)

    neurons_group.copy_in_old_neurons(old_neurons_group)
    neurons_group.epoch = saved_epoch
    neurons_group.epoch_error = saved_epoch_error
    neurons_group.init_time = saved_init_time
    neurons_group.epoch_time = saved_epoch_time

    locks_map = [True] * data["position"]
    locks_map.extend([False] * len(data["layers_neurons"]))
    print "New layers locks", locks_map

    apply_layers_locks(neurons_group, locks_map)

def add_layers(data=None):
    global neurons_group
    """ data format

        {
            "position" : 2,
            "layers_neurons": [8, 4]
        }
    """
    saved_neurons = neurons_group.get_neurons_up_to_layer_position(
        data["position"]
    )

    saved_epoch = neurons_group.epoch
    saved_epoch_error = neurons_group.epoch_error
    saved_init_time = neurons_group.init_time
    saved_epoch_time = neurons_group.epoch_time
    saved_inactivate_neurons_biases = neurons_group.inactivate_neurons_biases
    saved_training_mode = neurons_group.training_mode

    new_layers = neurons_group.layers[0:data["position"]]
    new_layers.extend(data["layers_neurons"])
    print "New network layers", new_layers
    build_network(
        new_layers,
        saved_inactivate_neurons_biases,
        saved_training_mode)

    neurons_group.set_neurons(saved_neurons)
    neurons_group.epoch = saved_epoch
    neurons_group.epoch_error = saved_epoch_error
    neurons_group.init_time = saved_init_time
    neurons_group.epoch_time = saved_epoch_time

    locks_map = [True] * data["position"]
    locks_map.extend([False] * len(data["layers_neurons"]))
    print "New layers locks", locks_map

    apply_layers_locks(neurons_group, locks_map)


def status(data=None):
    global neurons_group
    print "Layers", neurons_group.layers


def set_layers_locks(data=None):
    global neurons_group
    apply_layers_locks(neurons_group, data["locks"])
    print "New layers locks", data["locks"]


def train_from_snapshot(data=None):
    global context, neurons_group, next_layer_output_snapshot, norm_inputs
    neurons_ids = get_all_neurons_layers(
        neurons_group.neurons, neurons_group.input_ids)[data["position"]]
    next_layer_output_snapshot = LayerOutputSnapshot(
        len(norm_inputs.norm_array), data["position"], neurons_ids)
    print "Set snapshotting from %s layer with neuron_ids %s" % (
        data["position"], neurons_ids)
    context.snapshot_layer = True


def show_layers_locks(data=None):
    global neurons_group
    locks = []
    for neurons_ids in get_all_neurons_layers(neurons_group.neurons, neurons_group.input_ids):
        layer_locks = []
        for nid in neurons_ids:
            layer_locks.append(neurons_group.neurons[nid][12])
        locks.append(layer_locks)
    print "Layers locks %s" % locks


def show_response(neurons_group, input):
    layout = get_response_layout(neurons_group, input)
    print layout
    imshow(layout, interpolation='nearest')
    title("Response to input: " + str(input))
    show()


def max_input_response(data=None):
    global neurons_group
    show_response(neurons_group, 0.85)


def min_input_response(data=None):
    global neurons_group
    show_response(neurons_group, 0.15)


def show_biases_and_weights_average(data=None):
    draw_biases_and_weights_average(data)
    show()


def show_outputs_values_zones(data=None):
    draw_outputs_values_zones(data)
    show()


def show_outputs_values_zone(data=None):
    global neurons_group
    zone = str(neurons_group.output_value_zones[data["zone_index"]])
    layout = get_output_value_zone_layout(neurons_group, data["zone_index"])
    print "Neurons' outputs values zone " + zone
    print layout
    imshow(layout, interpolation='nearest')
    title("Neurons' outputs values zone " + zone)
    show()


def execute_command(command):
    global neurons_group
    commands_map = {
        "stop_training": stop_training,
        "add_layers": add_layers,
        "status": status,
        "set_layers_locks": set_layers_locks,
        "show_biases_and_weights_average": show_biases_and_weights_average,
        "max_input_response": max_input_response,
        "min_input_response": min_input_response,
        "show_outputs_values_zone": show_outputs_values_zone,
        "show_outputs_values_zones": show_outputs_values_zones,
        "train_from_snapshot": train_from_snapshot,
        "modify_layers": modify_layers,
        "show_layers_locks" : show_layers_locks,
    }

    print "command", command
    try:
        commands_map[command["command"]](command)
    except Exception, e:
        sys.stderr.write("Execute command error:" + str(e) + "\n")
        pass


conditional_commands = []


def test_conditional_commands():
    global conditional_commands, neurons_group
    left_commands = []
    for command in conditional_commands:
        if "condition_epoch>" in command:
            if neurons_group.epoch > command["condition_epoch>"]:
                try:
                    execute_command(command)
                except Exception, e:
                    sys.stderr.write(
                        "test_conditional_commands error:", str(e) + "\n")
            else:
                left_commands.append(command)
    conditional_commands = left_commands


def process_commands(network_command_fpath="/tmp/network_command"):
    global conditional_commands

    if os.path.exists(network_command_fpath):
        with open(network_command_fpath, "r") as fd:
            print "Found", network_command_fpath
            try:
                commands = json.load(fd)
            except Exception as e:
                sys.stderr.write(
                    "Error when processing command " + str(e) + "\n")
                return
            try:
                os.remove(network_command_fpath)
                print "Removed", network_command_fpath
            except:
                pass

            if type(commands) == type([]):
                for command in commands:
                    if "command" in command:
                        if "conditional" in command:
                            conditional_commands.append(command)
                        else:
                            execute_command(command)
            else:
                print "Invalid command(s) ..."


def print_training_stats(neurons_group):
    print "Epoch %s : Error %s : Time %s : Weights_average %s : \
        Weights_standard_deviation %s : Biases_average %s : \
        Biases_standard_deviation %s" % (
        neurons_group.epoch,
        neurons_group.epoch_error,
        str(neurons_group.epoch_time - neurons_group.init_time),
        get_weights_average(neurons_group.neurons, neurons_group.input_ids),
        get_weights_standart_deviation(
            neurons_group.neurons, neurons_group.input_ids),
        get_biases_average(neurons_group.neurons, neurons_group.input_ids),
        get_biases_standart_deviation(
            neurons_group.neurons, neurons_group.input_ids))


def get_inputs():
    global neurons_group, context, norm_inputs
    if set(neurons_group.start_neurons_ids) == set(neurons_group.input_ids):
        return norm_inputs.norm_array
    else:
        return context.last_training_snapshot.active_layer_output_snapshot.outputs


def get_targets():
    global neurons_group, context, norm_targets
    if set(neurons_group.start_neurons_ids) == set(neurons_group.input_ids):
        return norm_targets.norm_array
    else:
        return context.last_training_snapshot.norm_targets.norm_array


def back_propagation_network_training(
        epochs, max_error=0.0001, batch_learning=False,
        command_net_fpath="/tmp/network_command",
        batch_samples_percentage=100,
        number_of_paralel_process=cpu_count()):
    global norm_inputs, norm_targets, training_snapshots, \
        neurons_group, report_every_epoch, save_net_fpath, \
        next_layer_output_snapshot, context

    neurons_group.eta = 0.7
    neurons_group.alpha = 0.3

    # pool = Pool(number_of_paralel_process)
    length = len(get_inputs())

    limit = neurons_group.epoch + epochs
    while neurons_group.epoch < limit:
        neurons_group.epoch += 1

        error = 0.0
        # layer_counter = 0

        counter = 0
        max_weight_update_counter = int(
            length * batch_samples_percentage / 100)
        reset_output_values_couters(neurons_group.neurons)
        for r in random.sample(range(0, length), length):
            # layer_counter, set_error = network_propagate_2(
            set_error = neurons_group.network_propagate_inputs_forward(
                # pool,
                get_inputs()[r],
                get_targets()[r],
                batch_learning
            )
            error += set_error
            # network_propagate_error_backwards_3(
            #   pool, neurons_group, layer_counter, batch_learning)
            neurons_group.network_propagate_error_backwards(
                get_targets()[r], batch_learning, False)

            # do snapshot of layer if necessary
            if context.snapshot_layer:
                next_layer_output_snapshot.outputs[r] = \
                    neurons_group.get_layer_outputs(
                        next_layer_output_snapshot.neurons_ids)
                if all(next_layer_output_snapshot.outputs):
                    context.snapshot_layer = False
                    context.last_training_snapshot = TrainingSnapshot(
                        neurons_group,
                        norm_inputs,
                        norm_targets,
                        next_layer_output_snapshot)
                    neurons_group.set_start_neurons_ids(
                        next_layer_output_snapshot.neurons_ids)
                    print "Training from snapshot ready to go. Starting %s layer with neurons ids %s overall %s records" % (
                        next_layer_output_snapshot.layer_id,
                        next_layer_output_snapshot.neurons_ids,
                        len(next_layer_output_snapshot.outputs))

            # network_update_weights_2(pool, neurons_group, layer_counter)
            neurons_group.network_update_weights_backwards()

        if neurons_group.epoch % report_every_epoch == 0:
            neurons_group.epoch_error = error
            neurons_group.epoch_time = time.time()
            print_training_stats(neurons_group)
            context.training_snapshots.append(TrainingSnapshot(
                neurons_group,
                norm_inputs,
                norm_targets,
                context.last_training_snapshot.active_layer_output_snapshot))
            process_commands(command_net_fpath)
            test_conditional_commands()

        if error < max_error:
            print "\n\nNETWORK DATA - EPOCH " + str(neurons_group.epoch)
            break

    biases = get_biases(neurons_group.neurons, neurons_group.input_ids)
    weights = get_weights(neurons_group.neurons, neurons_group.input_ids)
    print "biases : " + str(sorted(biases))
    print "weights : " + str(sorted(weights))


def min_l_bfgs_b_network_training(
        epochs, max_error=0.0001, batch_learning=False):
    global context, norm_inputs, norm_targets, neurons_group

    neurons_group.eta = 0
    neurons_group.alpha = 0

    length = len(norm_inputs.norm_array)
    biases_and_weights = neurons_group.get_biases_and_weights(
        not neurons_group.inactivate_neurons_biases)
    data_sequence = get_chunks_indexes(length, 1)[0]
    extra_args = (neurons_group, norm_inputs, norm_targets, data_sequence)
    res = optimize.fmin_l_bfgs_b(
        error_function,
        biases_and_weights,
        fprime=error_gradient_function,
        maxfun=epochs,
        # approx_grad=True,
        # callback=optimize_callback,
        # bounds=((-25.0, 25.0),)*len(biases_and_weights),
        # epsilon=0.0001,
        # m=40,
        # factr=10,
        disp=1,
        args=extra_args
    )
    neurons_group.set_biases_and_weights(
        res[0], not neurons_group.inactivate_neurons_biases)


def fmin_bfgs_network_training(
        epochs, max_error=0.0001, batch_learning=False):
    global context, norm_inputs, norm_targets, neurons_group

    # if neurons_group.inactivate_neurons_biases:
    #    neurons_group.eta = 0
    #    neurons_group.alpha = 0

    length = len(norm_inputs.norm_array)
    biases_and_weights = neurons_group.get_biases_and_weights(
        not neurons_group.inactivate_neurons_biases)
    data_sequence = get_chunks_indexes(length, 1)[0]
    extra_args = (neurons_group, norm_inputs, norm_targets, data_sequence)
    res = optimize.fmin_bfgs(
        error_function,
        biases_and_weights,
        fprime=error_gradient_function,
        maxiter=epochs,
        callback=optimize_callback,
        gtol=1e-7,
        # ftol=0.001,
        # stepmx=10,
        # eta=0.005,
        args=extra_args
    )
    neurons_group.set_biases_and_weights(
        res, not neurons_group.inactivate_neurons_biases)


def fmin_tnc_network_training(epochs, max_error=0.0001, batch_learning=False):
    global context, norm_inputs, norm_targets, neurons_group

    # neurons_group.eta = 1
    # neurons_group.alpha = 1

    length = len(norm_inputs.norm_array)
    biases_and_weights = neurons_group.get_biases_and_weights(
        not neurons_group.inactivate_neurons_biases)
    data_sequence = get_chunks_indexes(length, 1)[0]
    extra_args = (neurons_group, norm_inputs, norm_targets, data_sequence)
    res = optimize.fmin_tnc(
        error_function,
        biases_and_weights,
        fprime=error_gradient_function,
        args=extra_args,
        bounds=((-25., 25.),)*len(biases_and_weights),
        disp=5,
        maxCGit=200,
        maxfun=epochs,
        eta=0.00025,
        # stepmx=1.0,
        # this one has some affect
        # rescale=3.0
        # pgtol=1
        # xtol=5
        # ftol=0.4
        # fmin=0.4
        # this one has some affect
        # accuracy=0.1
        # offset=[[0.0,0.0],]*len(biases_and_weights)
    )
    neurons_group.set_biases_and_weights(
        res[0], not neurons_group.inactivate_neurons_biases)


def train_network(
        epochs, max_error, batch_learning,
        number_of_paralel_process, command_net_fpath):
    global context, norm_inputs, norm_targets, save_net_fpath
    neurons_group.reset_init_time()
    print "Layers " + str(neurons_group.layers) + \
        " records " + str(len(norm_inputs.norm_array))
    if neurons_group.training_mode == 1:
        back_propagation_network_training(
            epochs, max_error, batch_learning,
            command_net_fpath, 100, number_of_paralel_process)
    elif neurons_group.training_mode == 2:
        min_l_bfgs_b_network_training(epochs, max_error, batch_learning)
    elif neurons_group.training_mode == 3:
        fmin_bfgs_network_training(epochs, max_error, batch_learning)
    elif neurons_group.training_mode == 4:
        fmin_tnc_network_training(epochs, max_error, batch_learning)
    context.save(save_net_fpath, save_net_fpath + ".epochs")


def parse_options(argv):
    parser = OptionParser()
    parser.add_option(
        "-d", "--data", dest="data",
        help="file path to training / test data file")
    parser.add_option(
        "-i", "--inactivate_neurons_biases",
        default=False, dest="inactivate_neurons_biases",
        help="inactivate neuron biases by setting them to value = 1",
        action="store_true")
    parser.add_option(
        "-l", "--load_len", dest="load_len",
        help="file path to load last epoch network file")
    parser.add_option(
        "-q", "--load_aen", default=False,
        dest="load_aen", help="load all epochs network file",
        action="store_true")
    parser.add_option(
        "-g", "--command", default="/tmp/network_command",
        dest="command", help="file path to command file")
    parser.add_option(
        "-k", "--layers", default="4,7,4",
        dest="layers", help="initial layers configuration, default 4,7,4")
    parser.add_option(
        "-s", "--save", default="/tmp/saved_network",
        dest="save", help="file path to save last epoch network")
    parser.add_option(
        "-t", "--test_mode", default=False,
        dest="test_mode", help="run network in test mode with input data",
        action="store_true")
    parser.add_option(
        "-y", "--replay_mode", default=False,
        dest="replay_mode", help="run network in replay mode",
        action="store_true")
    parser.add_option(
        "-m", "--replay_mode_number", default=1,
        dest="replay_mode_number",
        help="1 - to show output values zones, 2 - to show weights / biases and error, 3 - to show weights")
    parser.add_option(
        "-w", "--training_mode", default=1,
        dest="training_mode", help="Network training mode. 1 - back propagation, 2 - min_l_bfgs_b, 3 - fmin_bfgs, 4 - fmin_tnc")
    parser.add_option(
        "-f", "--replay_mode_speed", default=1,
        dest="replay_mode_speed", help="replay mode speed")
    parser.add_option(
        "-b", "--batch_learning", default=False,
        dest="batch_learning", help="run network in batch training mode",
        action="store_true")
    parser.add_option(
        "-n", "--number_of_paralel_process",
        default=cpu_count(), dest="number_of_paralel_process",
        help="number of paralel process to run when optimizing")
    parser.add_option(
        "-p", "--batch_samples_percentage",
        default="100", dest="batch_samples_percentage",
        help="percentage after which weights are updated")
    parser.add_option(
        "-e", "--epochs", default=100,
        dest="epochs", help="number of training epochs")
    parser.add_option(
        "-r", "--report_every_epoch",
        default=10, dest="report_every_epoch", help="report every epoch")
    return parser.parse_args(argv)


def main():
    global context, norm_inputs, norm_targets, neurons_group, \
        report_every_epoch, save_net_fpath, training_snapshot, start_time

    options, args = parse_options(sys.argv[1:])

    if not options.data:
        sys.stderr.write("Not provided training data file.")
        return 1
    else:
        with open(options.data, "r") as fd:
            train_data = json.load(fd)
            normalize_inputs(train_data)

    # layers = (3, 5, 3, 1)
    # layers = (4, 30, 15, 7, 4)
    # 4, 4, 7, 4 Error = 0.0348790062138
    # 4, 4, 8, 4 Error = 0.000653684537344
    # layers = [4, 4, 7, 4]
    # layers = [4, 4, 4]
    expected_error = 0.0000001
    report_every_epoch = int(options.report_every_epoch)
    save_net_fpath = options.save

    context.network_command_fpath = options.command

    layers = []
    if options.layers:
        layers = [int(i) for i in options.layers.split(",")]

    if options.load_len:
        if options.load_aen or options.replay_mode:
            load_network(options.load_len, options.load_len + ".epochs")
        else:
            load_network(options.load_len, None)
    else:
        build_network(
            layers, options.inactivate_neurons_biases,
            int(options.training_mode))

    speed = int(options.replay_mode_speed)
    if options.replay_mode:
        if options.replay_mode_number == "1":
            replay_network_training(
                data={
                    "speed": speed,
                    "command": "show_outputs_values_zones"})
        elif options.replay_mode_number == "2":
            replay_network_training(
                data={
                    "speed": speed,
                    "command": "show_error", "training_data": train_data})
        elif options.replay_mode_number == "3":
            replay_network_training(
                data={
                    "speed": speed,
                    "command": "show_biases_and_weights_average"})
        else:
            pass
        return 0

    if options.test_mode:
        if options.load_len:
            run_network(train_data)
        else:
            sys.stderr.write("Not provided file path to load network.")
            return 1
    else:
        adjust_loaded_network()
        train_network(
            int(options.epochs),
            expected_error,
            options.batch_learning,
            int(options.number_of_paralel_process),
            options.command
        )


if __name__ == "__main__":
    sys.exit(main())
