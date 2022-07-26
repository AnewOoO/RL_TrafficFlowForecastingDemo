from numpy import load

data = load('pems04.npz')
lst = data.files
print(lst)
print(data['data'])
print(type(data['data']))
print(data['data'].shape)