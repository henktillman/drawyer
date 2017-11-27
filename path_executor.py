import pickle, pdb

with open('path.pickle', 'rb') as handle:
  path = pickle.load(handle)


pdb.set_trace()

print('hello')