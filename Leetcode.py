def twoSum(nums, target):
    for element in enumerate(nums):
        if target - element[1] in nums[element[0] + 1:]:
            return [element[0], element[0] + 1 + nums[element[0] + 1:].index(target - element[1])]
    return (0)


print(f'Twinkle, twinkle, little star,\n\t How I wonder '
      f'what you are!\n\t\t Up above the world so high,\n\t\t Like '
      f'a diamond in the sky.\n Twinkle, twinkle, '
      f'little star,\n\t How I wonder what you are')
print(print(f'Twinkle, twinkle, little star,\n\t How I wonder '
            f'what you are!\n\t\t Up above the world so high,\n\t\t Like '
            f'a diamond in the sky.\n Twinkle, twinkle, '
            f'little star,\n\t How I wonder what you are'))


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

a = 'Zina'
b = a[len(a) - 1: -1]
print(b)