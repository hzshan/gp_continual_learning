import argparse, datetime, os, pickle
import numpy as np

# Some utility classes/functions for running codes on the cluster.
# Put together on Apr.26, 2022.


def initialize():
    data_path = None
    output_home_path = None

    if os.getcwd()[:3] == '/n/':
        ON_CLUSTER = True
        output_home_path = '/n/home11/haozheshan/ContinualLearning2022/outputs/'
        data_path = '/n/home11/haozheshan/ContinualLearning2022/'
        os.chdir(data_path)
    elif os.getcwd()[:3] == '/Us':
        ON_CLUSTER = False
        data_path = None
        output_home_path = None
    else:
        raise ValueError('Current working path not understood. Path:' + os.getcwd())
    return ON_CLUSTER, data_path, output_home_path


class ClusterResultOrganizer:
    def __init__(self, local_path, batch_name, max_ind=300):
        self.max_ind = max_ind
        self.all_data_obj = []
        self.all_args = []
        self.local_path = local_path
        self.batch_name = batch_name
        self.args = None

    def load_all(self):
        for file_ind in range(self.max_ind):
            file_path = f'{self.local_path}/{self.batch_name}/{self.batch_name}_{file_ind}.results'
            if os.path.isfile(file_path):
                _obj = pickle.load(open(file_path, 'rb'))
                self.all_data_obj.append(_obj)
                self.all_args.append(_obj['args'])
                self.args = _obj['args']

        print('=================== Cluster organizer ===================')
        if len(self.all_data_obj) > 0:
            print(f'{len(self.all_data_obj)} data objects loaded from folder "{self.batch_name}".')
        else:
            print(f'!!!!! No data file was found !!!!!')
        print('=================== Cluster organizer ===================')

    def organize_results(self, value_key, args_key):
        organized_results = {}

        for obj_ind in range(len(self.all_data_obj)):
            if args_key not in vars(self.all_args[obj_ind]).keys():
                raise ValueError(f'args key {args_key} not found in results')

            param_name = str(vars(self.all_args[obj_ind])[args_key])

            if value_key not in self.all_data_obj[obj_ind].keys():
                print(f'Organizer: data with key <<{value_key}>> were not found')
                return None
            numpied = np.array(self.all_data_obj[obj_ind][value_key])
            if param_name in organized_results.keys():
                organized_results[param_name].append(numpied)
            else:
                organized_results[param_name] = [numpied]

        for param_name in organized_results.keys():
            organized_results[param_name] = np.array(organized_results[param_name])
        return organized_results


class Args:
    """
    Simple wrapper for argumentparser. Makes code cleaner.
    Updated Apr 26, 2022
    """
    def __init__(self, description=None):
        self.p = argparse.ArgumentParser(description=description)
        self.p.add_argument('-f')  # necessary because sometimes Python passes a "-f" argument
        # print('Warning: argument type specified by the default value.')
        self.add('cluster', default=0)
        self.add('BATCH_NAME', default='BATCH_NAME')
        self.add('TRIAL_IND', default=0)
        self.boolean_vars = []

    def add(self, name, default, ptype=None, help=None, optional=True):
        raw_name = name
        if optional:
            name = '--' + name
        if ptype is None:
            ptype = type(default)

        if ptype == bool:
            self.boolean_vars.append(raw_name)
            ptype = str
        self.p.add_argument(name, type=ptype, default=default, help=help)

    def parse_args(self):
        args, unknown = self.p.parse_known_args()

        # parse boolean variables. This allows using "True" and "False" in the command prompts.
        for var_name in self.boolean_vars:
            str_var = str(vars(args)[var_name])
            if str_var == 'True':
                vars(args)[var_name] = True
            elif str_var == 'False':
                vars(args)[var_name] = False
            else:
                raise ValueError(f'Value for -{var_name} need to be "True" or "False", but got {vars(args)[var_name]}')
        args.f = None
        return args


class Logger:
    def __init__(self, output_path : str, run_name : str, only_print=False):
        self.prefix = output_path + run_name
        self.output_path = output_path
        self.run_name = run_name
        self.only_print = only_print
        self.log(f'Name of this run: {run_name}')

    def log(self, text: str):
        # this checks whether the folder exists; if not, always recreate it

        now = datetime.datetime.now()
        dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
        date_text = dt_string + ' | '
        text = date_text + text
        print(text)
        if not self.only_print:
            if os.path.isdir(self.output_path) is False:
                os.mkdir(self.output_path)
            file = open(self.prefix + '.log', 'a')
            file.write('\n' + text)
            file.close()

    def finish(self, results_obj=None):
        os.rename(self.prefix + '.log', self.output_path + 'FINISHED_' + self.run_name + '.log')
        if results_obj is not None:

            # this checks whether the folder exists; if not, always recreate it
            if os.path.isdir(self.output_path) is False:
                os.mkdir(self.output_path)
            pickle.dump(results_obj, open(self.output_path + f'{self.run_name}.results', 'wb'))



class ArgsReader:
    def __init__(self):
        self.dict = {}

    def add_args(self, args_obj):
        for _name in vars(args_obj).keys():
            if type(vars(args_obj)[_name]) in [float, int]:
                if _name not in self.dict.keys():
                    self.dict[_name] = [vars(args_obj)[_name]]
                else:
                    self.dict[_name].append(vars(args_obj)[_name])

    def get_ind_var(self):
        ind_label = None
        ind_axis = None
        for _name in self.dict.keys():
            if len(np.unique(self.dict[_name])) > 1 and _name != 'TRIAL_IND':
                ind_label = _name
                ind_axis = np.array(self.dict[_name])
        return ind_label, ind_axis