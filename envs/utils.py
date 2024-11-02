def ids_2dto1d(i, j, M, N):
    assert 0 <= i < M and 0 <= j < N
    index = int(i * N + j)
    return index


def ids_1dto2d(ids, M, N):
    i = ids // N
    j = ids - N * i
    return i, j



