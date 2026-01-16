'''
use example:
(main.py)
from write_log import enable_dual_output

enable_dual_output("my_output.txt") # --> 이런 식으로 놓기만 하면 이후의 print()들을 모두 my_output.txt 경로에 저장

print("이건 콘솔에도 보이고 파일에도 저장돼요!")
print("print 그대로 사용 가능!")

'''

import sys

class PrintAndSave:
    '''
    save the contents from print() into text file(.txt)
    '''
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # 콘솔 출력
        self.log.write(message)       # 파일 출력
        self.log.flush()

    def flush(self):
        # flush 메서드도 구현해야 sys.stdout처럼 동작
        self.terminal.flush()
        self.log.flush()

def enable_dual_output(filename="output.txt"):
    sys.stdout = PrintAndSave(filename)

def write_log(model_name: str, **kwargs):
    pass

