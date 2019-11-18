import os
from datetime import datetime


def create_csv(results, results_dir='../data/results'):

    csv_file_name = 'results_'
    csv_file_name += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_file_name), 'w') as f:

        f.write('user_id,item_list')

        for key, value in results.items():
            f.write('\n' + str(key) + ',')
            i = 0
            for val in value:
                f.write(str(val))
                if i != 9:
                    f.write(' ')
                    i += 1
