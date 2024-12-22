import numpy as np

F = np.zeros(100)
F[[1, 3, 4, 2]] = 1

print(F[[1, 3, 4, 2]])

# F = np.delete(F, [1, 4, 3, 2])

print(len(F[1:-1]))

print(np.array([1, 4, 2, 3, 5, 7]) - np.array([1, 3, 2]))
