#!/usr/bin/python
import sys
import pprint
import copy

import numpy as np
from scipy import array
from optparse import OptionParser

class NormalizedArrayAll:
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
        Returns coefficients of linear eqasion a, b.  where y = ax + b
        from range (a,b) to (c,d)
        """
        #if b == a: raise ValueError("Mapping not possible due to equal limits")
        if b == a:
            c1 = 0.0
            c2 = ( c + d ) / 2.
        else:
            c1 = ( d - c ) / ( b - a )
            c2 = c - a * c1
        return c1, c2


    def _norms(self):
        """ Gets normalization information from an array

        (self.lower_limit, self.upper_limit) is a range of normalization.
        in_array is 2-dimensional, normalization parameters are computed
        for all array elements ...
        """
        limits = []
        encode_linear = []
        decode_linear = []
        f_array = array(self.array).flatten()
        max_arr = max(f_array)
        min_arr = min(f_array)
        limits = (min_arr, max_arr)
        encode_linear = (self._linear(min_arr, max_arr, self.lower_limit, self.upper_limit))
        decode_linear = (self._linear(self.lower_limit, self.upper_limit, min_arr, max_arr))
        return array(limits), array(encode_linear), array(decode_linear)


    def _norm_array(self):
        """ Normalize 2-dimensional array linearly.  """
        if not self.array.dtype == np.dtype("float"):
            self.array = self.array.astype("float")
        i = self.array.shape[0]
        for ii in xrange(i):
            self.array[ii] = self.array[ii] * self.encode[0] + self.encode[1]
        return self.array

def init_results():
    global epochs_to_report
    results = {}
    for epoch in epochs_to_report:
        results.update({
            "Epoch %s " % epoch: {
                "Error": 0, "Time": 0,
                "Weight_average": 0, "Weight_standard_deviation": 0,
                "Biases_average": 0, "Biases_standard_deviation": 0},
        })
    return results


def init_tests(runs=36):
    global tests_cases
    tests = []
    tests.append({
        "count": runs,
        "start": ["START", "1. Starting from", "network without change, a lower bound"],
        "stop": ["END", "1. Starting from", "network without change, a lower bound"],
    })
    tests.append({
        "count": runs,
        "start": ["START", "10. Starting from", "network without change, an upper bound"],
        "stop": ["END", "10. Starting from", "network without change, an upper bound"],
    })
    tests.append({
        "count": runs,
        "start": ["START", "11. Starting from", "network without change, an upper bound"],
        "stop": ["END", "11. Starting from", "network without change, an upper bound"],
    })
    #return tests
    tests.append({
            "count": runs,
            "start": ["START", "2. Starting from", "network to bigger", "with old neurons locked", ", %s" % tests_cases[0]],
            "stop": ["END", "2. Starting from", "network to bigger", "with old neurons locked", ", %s" % tests_cases[0]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "3. Starting from", "network to bigger", "with old neurons locked", ", %s" % tests_cases[1]],
            "stop": ["END", "3. Starting from", "network to bigger", "with old neurons locked", ", %s" % tests_cases[1]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "4. Starting from", "network to bigger", "with layers", ", %s" % tests_cases[0]],
            "stop": ["END", "4. Starting from", "network to bigger", "with layers", ", %s" % tests_cases[0]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "5. Starting from", "network to bigger", "with layers", ", %s" % tests_cases[1]],
            "stop": ["END", "5. Starting from", "network to bigger", "with layers", ", %s" % tests_cases[1]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "6. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[0]],
            "stop": ["END", "6. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[0]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "7. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[1]],
            "stop": ["END", "7. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[1]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "8. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[0]],
            "stop": ["END", "8. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[0]],
    })
    tests.append({
            "count": runs,
            "start": ["START", "9. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[1]],
            "stop": ["END", "9. Starting from", "network to bigger", "with all neurons unlocked", ", %s" % tests_cases[1]],
    })
    #pprint.pprint(tests)
    return tests


def process_run(line, results):
    for key in results.keys():
        if line.find(key) > -1:
            le = line.split()
            results[key]["Error"] += float(le[5])
            results[key]["Time"] += float(le[8])
            results[key]["Weight_average"] += float(le[11])
            results[key]["Weight_standard_deviation"] += float(le[14])
            results[key]["Biases_average"] += float(le[17])
            results[key]["Biases_standard_deviation"] += float(le[20])
    return results


def average_run(results, count):
    for key in results.keys():
        results[key]["Error"] /= count
        results[key]["Time"] /= count
        results[key]["Weight_average"] /= count
        results[key]["Weight_standard_deviation"] /= count
        results[key]["Biases_average"] /= count
        results[key]["Biases_standard_deviation"] /= count


def convert_to_array(results):
    converted_results = []
    for result in results:
        line = []
        for key in sorted(result.keys()):
            if key.startswith("Epoch"):
                line.append([
                    int(key.rstrip().split()[1]),
                    result[key]["Error"],
                    result[key]["Time"],
                    result[key]["Weight_average"],
                    result[key]["Weight_standard_deviation"],
                    result[key]["Biases_average"],
                    result[key]["Biases_standard_deviation"]
                ])
        converted_results.append(line)
    return array(converted_results)

def print_sorted_lines(lines):
    lines.sort()
    for line in lines:
        print line[1]


def print_csv_index(results, converted_results, index):
    transposed_results = converted_results.transpose()
    #print "transposed_results", transposed_results
    header = "Epoch,"
    for result in results:
        header += "%s," % result["test"].replace(',','')
    print header.rstrip(",")
    lines = []
    for index, rec in enumerate(transposed_results[index]):
        lines.append([
            transposed_results[0][index][0],
            str(transposed_results[0][index][0]) + "," + ",".join([str(i) for i in rec])
        ])
    print_sorted_lines(lines)


def print_combination(results, combination, transposed_results):
    header = "Epoch,"
    for result in results:
        header += "%s," % result["test"].replace(',','')
    print header.rstrip(",")
    lines = []
    for index, rec in enumerate(combination):
        lines.append([
            transposed_results[0][index][0],
            str(transposed_results[0][index][0]) + "," + ",".join([str(i) for i in rec])
        ])
    print_sorted_lines(lines)


def print_errors(results, converted_results):
    print "Error"
    print_csv_index(results, converted_results, 1)


def print_training_time(results, converted_results):
    print "Training time"
    print_csv_index(results, converted_results, 2)


def print_weights_average(results, converted_results):
    print "Weights average"
    print_csv_index(results, converted_results, 3)


def print_weights_standart_deviation(results, converted_results):
    print "Weights standard_deviation"
    print_csv_index(results, converted_results, 4)

def print_biases_average(results, converted_results):
    print "Biases average"
    print_csv_index(results, converted_results, 5)

def print_biases_standart_deviation(results, converted_results):
    print "Biases standard_deviation"
    print_csv_index(results, converted_results, 6)

def print_weights_average_and_weights_standart_deviation(results, converted_results):
    print "Norm Weights average + Norm Weights standard deviation"
    transposed_results = converted_results.transpose()
    norm_wa = NormalizedArrayAll(transposed_results[3], 0.0, 1.0)
    norm_wsd = NormalizedArrayAll(transposed_results[4], 0.0, 1.0)
    combination = norm_wa.norm_array + norm_wsd.norm_array
    print_combination(results, combination, transposed_results)

def print_abs_weights_average_and_weights_standart_deviation(results, converted_results):
    print "Norm Absolute Weights average + Norm Weights standard deviation"
    transposed_results = converted_results.transpose()
    norm_wa = NormalizedArrayAll(abs(transposed_results[3]), 0.0, 1.0)
    norm_wsd = NormalizedArrayAll(transposed_results[4], 0.0, 1.0)
    combination = norm_wa.norm_array + norm_wsd.norm_array
    print_combination(results, combination, transposed_results)

def print_biases_average_and_biases_standart_deviation(results, converted_results):
    print "Norm Biases average + Norm Biases standard deviation"
    transposed_results = converted_results.transpose()
    norm_ba = NormalizedArrayAll(transposed_results[5], 0.0, 1.0)
    norm_bsd = NormalizedArrayAll(transposed_results[6], 0.0, 1.0)
    combination = norm_ba.norm_array + norm_bsd.norm_array
    print_combination(results, combination, transposed_results)

def print_abs_biases_average_and_biases_standart_deviation(results, converted_results):
    print "Norm Absolute Biases average + Norm Biases standard deviation"
    transposed_results = converted_results.transpose()
    norm_ba = NormalizedArrayAll(abs(transposed_results[5]), 0.0, 1.0)
    norm_bsd = NormalizedArrayAll(transposed_results[6], 0.0, 1.0)
    combination = norm_ba.norm_array + norm_bsd.norm_array
    print_combination(results, combination, transposed_results)

def print_errors_and_training_time(results, converted_results):
    print "Norm Error + Norm Training time"
    transposed_results = converted_results.transpose()
    norm_errors = NormalizedArrayAll(transposed_results[1], 0.0, 1.0)
    norm_traning_time = NormalizedArrayAll(transposed_results[2], 0.0, 1.0)
    combination = norm_errors.norm_array + norm_traning_time.norm_array
    print_combination(results, combination, transposed_results)

def print_errors_and_training_time_and_abs_weights_average(results, converted_results):
    print "Norm Error + Norm Training time + Norm Absolute Weight average"
    average = []
    error_and_training_time = []
    transposed_results = converted_results.transpose()
    norm_errors = NormalizedArrayAll(transposed_results[1], 0.0, 1.0)
    norm_traning_time = NormalizedArrayAll(transposed_results[2], 0.0, 1.0)
    norm_biases_and_weights_average = NormalizedArrayAll(abs(transposed_results[3]), 0.0, 1.0)
    combination = norm_errors.norm_array + norm_traning_time.norm_array + norm_biases_and_weights_average.norm_array
    print_combination(results, combination, transposed_results)

def print_errors_and_training_time_and_weights_standatd_deviation(results, converted_results):
    print "Norm Error + Norm Training time + Norm Weights standard deviation"
    average = []
    error_and_training_time = []
    transposed_results = converted_results.transpose()
    norm_errors = NormalizedArrayAll(transposed_results[1], 0.0, 1.0)
    norm_traning_time = NormalizedArrayAll(transposed_results[2], 0.0, 1.0)
    norm_biases_and_weights_standard_deviation = NormalizedArrayAll(transposed_results[4], 0.0, 1.0)
    combination = norm_errors.norm_array + norm_traning_time.norm_array + norm_biases_and_weights_standard_deviation.norm_array
    print_combination(results, combination, transposed_results)

def process_log(lines):
    tests = init_tests()
    in_test = False
    counter = 0
    run_results = init_results()
    all_results = []
    end_tag = []
    test_line = None

    for test in tests:
        for line in lines:
            if all([line.find(i) > -1 for i in test["start"]]):
                #print "Start--", line
                in_test = True
                counter += 1
                end_tag = test["stop"]
                test_line = line
                if not run_results.has_key("test"):
                    full_test_name = test_line.split(" ", 10)[-1].rsplit(",", 1)[-2]
            elif in_test and all([line.find(i) > -1 for i in end_tag]):
                #print "End--", line
                #if counter == test["count"]:
                #    break
                in_test = False
                end_tag = []
            else:
                if in_test:
                    #print test["start"]
                    #if line.find("Epoch 6800") > -1:
                    #    print line
                    run_results = process_run(line, run_results)
        average_run(run_results, counter)
        run_results.update({"test": full_test_name, "counter": counter})
        all_results.append(run_results)
        if not counter == test["count"]:
            print "There were %s runs expected, but only %s were found of <%s> :(" % (test["count"], counter, run_results["test"])
        print test
        pprint.pprint(run_results)
        print
        counter = 0
        run_results = init_results()
        #sys.exit()
    coverted = convert_to_array(all_results)
    print_errors(all_results, coverted)
    print
    print_training_time(all_results, coverted)
    print
    print_weights_average(all_results, coverted)
    print
    print_weights_standart_deviation(all_results, coverted)
    print
    print_biases_average(all_results, coverted)
    print
    print_biases_standart_deviation(all_results, coverted)
    #print
    #print_weights_average_and_weights_standart_deviation(all_results, coverted)
    #print
    #print_abs_weights_average_and_weights_standart_deviation(all_results, coverted)
    #print
    #print_biases_average_and_biases_standart_deviation(all_results, coverted)
    #print
    #print_abs_biases_average_and_biases_standart_deviation(all_results, coverted)
    print
    print_errors_and_training_time(all_results, coverted)
    #print
    #print_errors_and_training_time_and_abs_weights_average(all_results, coverted)
    #print
    #print_errors_and_training_time_and_weights_standatd_deviation(all_results, coverted)


def parse_options(argv):
    parser = OptionParser()
    parser.add_option("-l", "--load", dest="load", help="file path to the log for analysis")
    parser.add_option("-e", "--epoch", dest="epoch", help="initial epoch")
    return parser.parse_args(argv)


def main():
    global log, epochs_to_report, tests_cases
    options, args = parse_options(sys.argv[1: ])

    if not options.load:
        sys.stderr.write("Not provided the log file.")
        return 1
    if not options.epoch:
        sys.stderr.write("Initial epoch was not provided.")
        return 1

    #tests_cases = [7500, 15000]
    tests_cases = [32500, 40000]

    first_report_epoch = int(options.epoch)
    end_report_epoch = 50000
    report_epoch_step = 50
    epochs_to_report= range(first_report_epoch, end_report_epoch, report_epoch_step)

    with open(options.load, "r") as fd:
        process_log(fd.readlines())


if __name__ == "__main__":
    sys.exit(main())
