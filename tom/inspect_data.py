
from os import lseek
import numpy as np
import csv
import random

# Pick a single time series file
fileName = './data/00000e74ad.npy'
data = np.load(fileName)

print(type(data))
print(data.shape)
print(data.dtype)
print(data.strides)
print(data.data)
print(data)


# Write out a sample submission (random)
out_file = open('sample1.csv', 'w')

with open('./data/sample_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            out_file.write(row[0] + ',' + row[1] + '\n')
            line_count += 1
        else:
            #print(f'\t{row[0]} is file, prob is {row[1]}.')random.uniform(0, 1)
            foo = random.uniform(0, 1)
            if foo >=0.5:
                out_file.write(row[0] + ',0\n')
            else:
                out_file.write(row[0] + ',1\n')


            line_count += 1
    print(f'Processed {line_count} lines.')

out_file.close()
