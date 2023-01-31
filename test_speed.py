import tinytorch
import time

if __name__ == '__main__':
    
    a = tinytorch.rand(600000)
    print(a)
    b = tinytorch.rand(6)
    print(b)
    now = time.time()
    c = a + b
    print(c)
    print(f"time usage: {time.time()- now}")