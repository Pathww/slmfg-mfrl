def ids_2dto1d(i, j, M=10, N=10):
    assert 0 <= i < M and 0 <= j < N
    index = int(i * N + j)
    return index


def ids_1dto2d(ids, M=10, N=10):
    i = ids // N
    j = ids - N * i
    return i, j



