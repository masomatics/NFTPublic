import argparse
import itertools
import yaml
import functools
import copy
import pdb
import os
import numpy as np
import re

'''
Helper for Automated shell generator. 
The files in this code Recurse over yaml dicts and search for 
the nested dict key with multiple inputs. 
'''
specialkeys = ['device']
def recursive_enumeration(mydict, loc):
    separator='.'
    locs = {}
    for item in mydict.items():
        if len(loc) > 0:
            deeper_loc = separator.join([loc, str(item[0])])
        else:
            deeper_loc = str(item[0])
        #base case
        if type(item[1]) != dict:
            #if len(str(item[1]).split(',')) > 1 or str(item[0]) in specialkeys:
            items_list = str(item[1]).split(' and ')
            locs.update({deeper_loc : items_list})

        #recursion case
        else:
            #recursive_locs, locs_for_naming = recursive_enumeration(item[1], deeper_loc)
            recursive_locs= recursive_enumeration(item[1], deeper_loc)
            locs.update(recursive_locs)

    return locs


def format_list(input_list):
    for k in range(len(input_list)):
        input = input_list[k].replace('(','[')
        input = input.replace(')',']')
        input = input.replace(', ',',')
        input_list[k] = input
    return input_list




def generate_attr(variation_config, base_config,
         log_dir = None, mode='pfkube', maxgpu=None):


    #identify the list_defined_keys
    #dict_of_lists, nameloc = recursive_enumeration(variation_config, '')
    dict_of_lists = recursive_enumeration(variation_config, '')

    list_of_lists = []
    dict_locs = []
    for key in dict_of_lists:
        dict_locs = dict_locs + [key]
        dictlist = format_list(dict_of_lists[key])
        list_of_lists = list_of_lists + [dictlist]


    #create all combinations
    combos = list(itertools.product(*list_of_lists))
    combos = [list(combo) for combo in combos]


    #keys to be used for comparison in this experiment:
    compare_keys = []
    for j in range(len(dict_locs)):
        if len(dict_of_lists[dict_locs[j]]) > 1:
            compare_keys = compare_keys + [dict_locs[j]]

    compare_keys = np.array(compare_keys)
    output_names_list = []
    output_dir_paths = []

    #make --attr commandlists
    separator = '_'
    attr_list = ['--config_path=%s --attr '%base_config] * len(combos)
    for k in range(len(combos)):
        output_dir_str=[]
        for i in range(len(dict_locs)):

            if mode == 'screen' and dict_locs[i] == 'device':
                attr_list[k] = attr_list[k] + dict_locs[i] \
                               + '=%s ' % (np.mod(k, maxgpu))
            else:
                attr_list[k] = attr_list[k] + dict_locs[i] \
                             + '=%s '%(combos[k][i])

            dict_locs_end = dict_locs[i].split('.')[-1]
            if len(dict_of_lists[dict_locs[i]]) > 1:
                output_dir_str = output_dir_str + [str(dict_locs_end) + \
                                                   str(combos[k][i])]
        output_dir_name = separator.join(output_dir_str)
        #Format the dir_name
        output_dir_name = output_dir_name.replace('.', '')
        output_dir_name = output_dir_name.replace('npz', '')
        output_dir_name = output_dir_name.replace('/', '_')


        if output_dir_name in output_names_list:
            output_dir_name = output_dir_name + 'id%s'%k
        output_names_list = output_names_list + [output_dir_name]

        output_dir_path = os.path.join(log_dir, output_dir_name)
        output_dir_paths = output_dir_paths + [output_dir_name]
        attr_list[k] = attr_list[k] + 'log_dir=' + output_dir_path

    return attr_list, compare_keys, output_dir_paths

def dirname_to_jobname(log_dir):
    jobname = log_dir.split('/')[-1]
    if jobname[-1] == '0':
        jobname = jobname + 'zero'
    if jobname[-1] == '1':
        jobname = jobname + 'one'
    jobname = re.sub(r'\d+', '', jobname)

    jobname = jobname.replace('_', '-').lower()
    jobname = jobname.split('-')
    clean_name = []
    for k in range(len(jobname)):
        if len(jobname[k]) > 0:
            clean_name = clean_name + [jobname[k]]
    jobname = ('-').join(clean_name)
    return jobname


def dirname_to_jobname_ex(log_dir):
    jobname = log_dir.split('/')[-1]
    if jobname[-1] == '0':
        jobname = jobname + 'zero'
    if jobname[-1] == '1':
        jobname = jobname + 'one'
    #jobname = re.sub(r'\d+', '', jobname)
    jobname = jobname.replace('.', '').lower()

    jobname = jobname.replace('_', '-').lower()
    jobname = jobname.split('-')
    clean_name = []
    for k in range(len(jobname)):
        if len(jobname[k]) > 0:
            clean_name = clean_name + [jobname[k]]
    jobname = ('-').join(clean_name)
    return jobname

