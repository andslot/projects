def read_file(file: str):


def part1():
    result = 0
    file = open("test.txt", "r").read().split("\n")
    digits = []
    symbols = {}
    for i, line in enumerate(file):
         digit = 0
         d_len = 0
         for j, elem in enumerate(line):
            if elem.isnumeric():
                 digit *= 10
                 digit += int(elem)
                 d_len += 1
            elif not elem.isnumeric() and digit == 0:
                digits.append(digit, set([(c, k) for k in range(j - d_len - 1, j + 2) for c in range(i - 1, i + 2)]))
                digit = 0
                d_len = 0
                if elem not in ".\n":
                    symbols[(i, j)] = elem

    

                 
                 
    return result

if __name__ == "__main__":
    print(part1())