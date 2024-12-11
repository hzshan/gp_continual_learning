import argparse, datetime, os, pickle
import numpy as np

# Some utility classes/functions for running codes on the cluster.
# Put together on Apr.26, 2022.

def list_folders(path, keyword1, keyword2='', keyword3=''):
    pathlist = os.listdir(path)
    print(f'Generating a list at {str(datetime.datetime.now())[:16]}')
    counter = 0
    filtered_list = []
    for item in pathlist:
        if keyword1 in item and keyword2 in item and keyword3 in item:
            print(f'[{counter}] {item}')
            counter += 1
            filtered_list.append(item)
    return filtered_list



def initialize():
    """
    Hacky way to initialize the path variables by detecting whether it's 
    currently running on the cluster or locally.
    """
    data_path = None
    output_home_path = None

    if os.getcwd()[:3] == '/n/':
        ON_CLUSTER = True
        output_home_path = '/n/home11/haozheshan/ContinualLearning2022/outputs/'
        data_path = '/n/home11/haozheshan/ContinualLearning2022/datasets'
        os.chdir(data_path)
    elif os.getcwd()[:3] == '/Us':
        ON_CLUSTER = False
        data_path = None
        output_home_path = None
    else:
        raise ValueError('Current working path not understood. Path:' + os.getcwd())
    return ON_CLUSTER, data_path, output_home_path


class ClusterResultOrganizer:
    def __init__(self, local_path, batch_name, sort_by_key=None, order_by_seed_number=False, verbose=True):
        # self.sorted_data_obj = {}
        # self.sorted_seed_numbers = {}
        self.all_args = []
        self.all_data_obj = []
        self.all_seed_numbers = []
        self.local_path = local_path
        self.batch_name = batch_name
        self.args = None
        self.sort_by_key = sort_by_key
        self.multiseed = False
        self.order_by_seed_number = order_by_seed_number  # whether to order the rows of each data matrix by the seed number used
        self.file_name_list = []
        self.file_path = None
        self.verbose = verbose

        self.packed_results = False
        if self.batch_name.endswith('.packed_results'):
            print('Loading a packed result file.')
            self.packed_results = True
    
        self.load_all()
        if self.verbose:
            print('Available data keys are ' + str(self.all_data_obj[0].keys()))
        
    def print_args(self):
        assert self.args is not None, 'No args found.'
        arg_dict = vars(self.args)
        for key, val in arg_dict.items():
            print(f'{key}: {val}')

    def load_all(self):
        
        if self.packed_results:
            packed_results = pickle.load(open(f'{self.local_path}/{self.batch_name}', 'rb'))
            self.all_data_obj = packed_results['data_obj_list']
        
            for _result_obj in self.all_data_obj:
                self.all_args.append(_result_obj['args'])
                if self.order_by_seed_number:
                    self.all_seed_numbers.append(_result_obj['args'].seed)
                self.args = _result_obj['args']
        else:
            filepath = f'{self.local_path}/{self.batch_name}/'
            self.file_path = filepath
            file_list = os.listdir(filepath)
            # print(filepath)
            num_result_files = 0
            for file_name in file_list:
                if file_name.endswith('.results'):
                    num_result_files += 1
                    try:
                        _obj = pickle.load(open(filepath + file_name, 'rb'))
                    except:
                        print(f'Loading {filepath + file_name} failed.')
                        continue
                    self.file_name_list.append(file_name)
                    self.all_data_obj.append(_obj)
                    self.all_args.append(_obj['args'])
                    if self.order_by_seed_number:
                        self.all_seed_numbers.append(_obj['args'].seed)
                    self.args = _obj['args']
            
            if num_result_files == 0:
                raise RuntimeError(f'No result files found under {filepath}.')

        # attempt to identify the variable that is different between jobs. 
        # This is done by using the first key found to have multiple values
        # between jobs.
        if self.sort_by_key is None:
            list_of_all_keys = list(vars(self.all_args[0]).keys())
            for key in list_of_all_keys:
                all_unique_values = []
                for _args in self.all_args:
                    if vars(_args)[key] not in all_unique_values:
                        all_unique_values.append(vars(self.args)[key])
                if len(all_unique_values) > 1 and key not in [
                    'TRIAL_IND',
                    'seed'
                    ]:
                    break
            self.sort_by_key = key
            print('No key was specified.'
                  'Automatically using key <<' + self.sort_by_key + '>>'
                  ' to sort the results.')

        values_for_sort_by_key = []

        for _args in self.all_args:
            if vars(_args)[self.sort_by_key] not in values_for_sort_by_key:
                values_for_sort_by_key.append(vars(_args)[self.sort_by_key])
        sort_by_key_message = (f'For key <<{self.sort_by_key}>>,'
        f'the values are {values_for_sort_by_key}')
        
        if 'NSEEDS' in vars(self.args).keys():
            self.multiseed = True
            print('"NSEEDS" found in the arguments.'
                  'Assuming that each file contains multiple random seeds.')

        # print('=================== Cluster organizer ===================')
        if len(self.all_data_obj) > 0:
            print(f'{len(self.all_data_obj)} data objects loaded from folder "{self.batch_name}".')
        else:
            print(f'!!!!! No data file was found !!!!!')
        if self.verbose:
            print(sort_by_key_message)
        # print('=================== Cluster organizer ===================')

    def organize_results(self, value_key, as_arrays=False):
        organized_results = {}
        organized_results_seed_numbers = {}

        for obj_ind in range(len(self.all_data_obj)):
            if self.sort_by_key not in vars(self.all_args[obj_ind]).keys():
                raise ValueError(f'args key {self.sort_by_key} not found in results')

            param_name = str(vars(self.all_args[obj_ind])[self.sort_by_key])

            if value_key not in self.all_data_obj[obj_ind].keys():
                print(f'Organizer: data with key <<{value_key}>> were not found')
                return None, None
            numpied = np.array(self.all_data_obj[obj_ind][value_key])
            if param_name in organized_results.keys():
                organized_results[param_name].append(numpied)
                if self.order_by_seed_number:
                    organized_results_seed_numbers[param_name].append(
                        self.all_seed_numbers[obj_ind])
            else:
                organized_results[param_name] = [numpied]
                if self.order_by_seed_number:
                    organized_results_seed_numbers[param_name] =\
                          [self.all_seed_numbers[obj_ind]]

        # sort results by the random seed number
        for param_name in organized_results.keys():
            organized_results[param_name] = np.array(organized_results[param_name])
            if self.order_by_seed_number:
                organized_results[param_name] =\
                      organized_results[param_name][np.argsort(organized_results_seed_numbers[param_name])]
            
            if self.multiseed:
                organized_results[param_name] = organized_results[param_name][0]
        
        if as_arrays:
            return dict_to_arrays(organized_results)
        else:
            return organized_results

def dict_to_arrays(data_dictionary):
    if data_dictionary is None:
        return None, None
    keys = list(data_dictionary.keys())
    numerical_keys = None
    if '.' in keys[0]:
        numerical_keys = [float(key) for key in keys]
    else:
        numerical_keys = [int(key) for key in keys]
    
    numerical_keys = np.sort(np.array(numerical_keys))

    data = []
    for key in numerical_keys:
        data.append(data_dictionary[str(key)])
    
    data = np.array(data)
    return numerical_keys, data



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
        self.log('Run finished.')
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