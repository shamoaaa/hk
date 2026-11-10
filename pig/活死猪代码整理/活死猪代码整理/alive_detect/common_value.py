from multiprocessing import Manager


# 使用 Manager 创建共享变量
manager = Manager()
total_num = manager.Value('i', 0)
finish_num = manager.Value('i', 0)
confusion_matrix = manager.list([manager.list([0] * 3) for _ in range(3)])

def get_total_num():
    return total_num.value

def set_total_num(value):
    total_num.value = value


def get_finish_num():
    return finish_num.value

def add_finish_num():
    finish_num.value = finish_num.value + 1


def set_confusion_matrix(row, col):
    if col >= 0:
        confusion_matrix[row][col] += 1

def get_confusion_matrix():
    for row in confusion_matrix:
        print(list(row))
