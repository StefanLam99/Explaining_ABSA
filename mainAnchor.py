import Anchor
import time
import numpy as np
from classifier import *
import statistics

def main ():
    start_whole = time.time()
    tau = 0.75
    num_instances = 200
    pert_left = True
    pert_right = True
    batch_size = 3
    epsilon = 0.25
    delta = 0.05
    initial_value = 20
    list_anchors = [''] * num_instances
    mean_vector = [0] * num_instances
    instance_counter = 0
    b = classifier('Olaf')
    for instance_index in range(0, num_instances):
        previous_anchor = []
        coverage_astar = 0.2
        start = time.time()
        perturbed_instances_left, instance_sentiment, instance_left, b, instance_info = Anchor.get_perturbations(True, False, b, instance_index)
        perturbed_instances_right, instance_sentiment, instance_right, b, instance_info = Anchor.get_perturbations(False, True,b, instance_index)
        instance = instance_left + instance_right
        print('instance', instance)
        perturbed_instances = [''] * len(perturbed_instances_right)
        for i in range(len(perturbed_instances_left)):
            perturbed_instances[i] = perturbed_instances_left[i] + perturbed_instances_right[i]
        print('pert instances', perturbed_instances)
        possible_anchor_list = Anchor.possible_anchor(previous_anchor, instance, coverage_astar, perturbed_instances)

        while possible_anchor_list != []:
            bbest = Anchor.bbest_anchors(batch_size, possible_anchor_list, epsilon, delta, perturbed_instances_left,
                                         perturbed_instances_right,
                                         instance_sentiment, initial_value, pert_left, pert_right, b,
                                         instance_info[0], instance_info[1], instance_info[2], instance_info[3],
                                         instance_info[4], instance_info[5], instance_info[6])
            print('bbest', bbest, type(bbest))
            if max(Anchor.get_lb_vector(bbest)) > tau:
                for y in range(len(bbest)):
                    if bbest[y][1] > tau:
                        anchor_cov = Anchor.get_coverage(bbest[y][3], perturbed_instances)
                        if anchor_cov > coverage_astar:
                            coverage_astar = anchor_cov
                            bbest_anchor = bbest[y]
            previous_anchor = [bbest[j][3] for j in range(len(bbest))]
            possible_anchor_list = Anchor.possible_anchor(previous_anchor, instance, coverage_astar,
                                                          perturbed_instances)
        end = time.time()
        print(end - start)
        anchor_mean = bbest_anchor[0]
        final_anchor = bbest_anchor[3]
        print('anchor', final_anchor, anchor_mean)
        mean_vector[instance_index] = anchor_mean
        list_anchors[instance_index] = final_anchor
        instance_counter += 1
        print('mean vector', mean_vector)
        print('instance counter', instance_counter)

    fid_mean = statistics.mean(mean_vector)
    fid_stdev = statistics.stdev(mean_vector)
    end_whole = time.time()
    print('mean vector', mean_vector)
    print('anchor list', list_anchors)
    print('fidelity', fid_mean, fid_stdev)

    print(end_whole - start_whole)
    return bbest


output = main()