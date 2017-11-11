import fileinput


"""
Simple script to repair delimiters in data set

author: @dawidkski

"""

# variable definition
DIRECTORY_PATH = "/home/dave/notes/machine-learning/regression/MNK/data/"
FILENAME = "data_testing"

with open(DIRECTORY_PATH + FILENAME, "r+") as data_file:
  data = data_file.readlines()
  data_file.seek(0)
  for line in data:
    line = line.strip().split()
    data_file.write("{0},{1}\n".format(line[0], line[1]))
  data_file.truncate()


