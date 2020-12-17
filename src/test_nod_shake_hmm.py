'''
test_hmm.py
Author: Anantharaman Narayana Iyer
Date: 7 Sep 2014
'''
import json
import os
import sys

from myhmm import MyHmm

models_dir =  ''#

seq0 = ('up', 'up', 'up', 'stationary', 'down', 'left')
seq1 = ('stationary', 'up', 'stationary', 'stationary', 'left', 'left', 'right')
seq2 = ('up', 'up', 'up', 'stationary', 'down', 'down')
seq3 = ('stationary', 'left', 'left', 'right')
seq4 = ('up', 'stationary', 'down', 'down', 'stationary', 'up', 'up', 'stationary', 'down', 'stationary', 'stationary', 'up', 'stationary', 'stationary', 'down')

observation_list = [seq0, seq1, seq2, seq3, seq4]

if __name__ == '__main__':
    #test the forward algorithm and backward algorithm for same observations and verify they produce same output
    #we are computing P(O|model) using these 2 algorithms.
    nod_file = "nod.json" # this is the model file name - you can create one yourself and set it in this variable
    shake_file = "shake.json"
    hmm_nod = MyHmm(os.path.join(models_dir, nod_file))
    hmm_shake = MyHmm(os.path.join(models_dir, shake_file))
    
    total1 = total2 = 0 # to keep track of total probability of distribution which should sum to 1
    for obs in observation_list:
        for name, hmm in zip(["Nod", "Shake"], [hmm_nod, hmm_shake]):
            p1 = hmm.forward(obs)
            p2 = hmm.backward(obs)
            total1 += p1
            total2 += p2
            print(name, " Observations = ", obs, " Fwd Prob = ", p1, " Bwd Prob = ", p2, " total_1 = ", total1, " total_2 = ", total2)
    print("The new model parameters after 0 iteration are: ")
    print("A = ", hmm.A)
    print("B = ", hmm.B)
    print("pi = ", hmm.pi)

    observations = seq2
    print("Learning the model through Forward-Backward Algorithm for the observations")
    model_file = "nod.json"
    hmm = MyHmm(os.path.join(models_dir, model_file))
    hmm.forward_backward(observations)

    print("The new model parameters after 1 iteration are: ")
    print("A = ", hmm.A)
    print("B = ", hmm.B)
    print("pi = ", hmm.pi)
    
