def twoSum(nums, target):
    for element in enumerate(nums):
        if target - element[1] in nums[element[0] + 1:]:
            return [element[0], element[0] + 1 + nums[element[0] + 1:].index(target - element[1])]
    return 0


print(twoSum(
    [3, 2, 4],
    6)
)
