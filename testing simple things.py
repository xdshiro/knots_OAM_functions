def permute2(seq):
    if not seq:
        yield seq
    for i in range(len(seq)):
        rest = seq[:i] + seq[i + 1:]
        print(f'rest: {rest}')
        for x in permute2(rest):
            yield seq[i:i + 1] + x


for x in permute2('123'):
    print(x)
