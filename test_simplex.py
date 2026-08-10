import itertools, numpy as np
def enumerate_simplex(n, step=0.02):
    grid = np.arange(0.0, 1.0 + step/2, step)
    if n == 1:
        yield (1.0,); return
    for w in itertools.product(grid, repeat=n):
        if abs(sum(w) - 1.0) < step/2:
            yield w

n3 = list(enumerate_simplex(3, 0.02))
print(f'3-way step 0.02: {len(n3)} configs')
print('sample:', n3[:5], '...', n3[-3:])
n4 = list(enumerate_simplex(4, 0.05))
print(f'4-way step 0.05: {len(n4)} configs')
print('sample:', n4[:5], '...', n4[-3:])
print('all sum==1?', all(abs(sum(w) - 1.0) < 1e-9 for w in n3 + n4))
