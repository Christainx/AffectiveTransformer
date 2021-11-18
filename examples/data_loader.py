# -*- coding=utf-8 -*-

'''
Created on May 18, 2020

@author: zhj
'''
import pickle
import csv


def load_epa():
    csvfile = open( './EnglishWords_EPAs_List.csv', encoding = 'UTF-8-sig' )
    csvreader = csv.reader( csvfile )

    wordlist = []
    dict_e = {}
    dict_p = {}
    dict_a = {}
    for info in csvreader:
        if len( info ) < 4:
            continue
        word = info[0]
        wordlist.append( word )
        dict_e[word] = info[1]
        dict_p[word] = info[2]
        dict_a[word] = info[3]

    fpout = open( 'epa.pkl', 'wb' )
    pickle.dump( [wordlist, dict_e, dict_p, dict_a], fpout )


def load_vad():
    fp = open( './NRC-VAD-Lexicon.txt', encoding = 'UTF-8-sig' )
    content = fp.readlines()

    wordlist = []
    dict_e = {}
    dict_p = {}
    dict_a = {}
    skip1st = True
    for info in content:
        if skip1st:
            skip1st = False
            continue
        info = info.strip().split( '\t' )
        if len( info ) < 4:
            continue
        word = info[0]
        wordlist.append( word )
        dict_e[word] = info[1]
        dict_p[word] = info[2]
        dict_a[word] = info[3]

    fpout = open( 'vad.pkl', 'wb' )
    pickle.dump( [wordlist, dict_e, dict_p, dict_a], fpout )


load_epa()
load_vad()
