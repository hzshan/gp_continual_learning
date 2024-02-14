"""
A short utility script to combine individual .log and .results files
into a single .packed_results file. This is useful for quick FTP transfer.

Expected to be launched with the following bash script:
    module load python
    source activate continual
    cd /n/home11/haozheshan/ContinualLearning2022 || exit

    python pack_data.py --batch_to_pack $1

where $1 is the name of the folder containing the .log and .results files.

Updated Feb. 13, 2024
"""
import cluster_utils
import pickle, os

output_home_path = '/n/home11/haozheshan/ContinualLearning2022/outputs'

argparser = cluster_utils.Args()
argparser.add(name='batch_to_pack', default='folder_name',
              help='this is the name of the result folder')
args = argparser.parse_args()

filepath = f'{output_home_path}/{args.batch_to_pack}/'

file_list = os.listdir(filepath)

packed_data = {
    'data_obj_list': [],
    'text_file_list': [],
}

for file_name in file_list:
    if file_name.endswith('.results'):
        try:
            _obj = pickle.load(open(filepath + file_name, 'rb'))
        except:
            print(f'Loading {filepath + file_name} failed.')
            continue
        packed_data['data_obj_list'].append(_obj)
    elif file_name.endswith('.txt'):
        packed_data['text_file_list'].append(open(filepath + file_name, 'r').read())

pickle.dump(packed_data,
            open(f'{output_home_path}/{args.batch_to_pack}.packed_results', 'wb'))