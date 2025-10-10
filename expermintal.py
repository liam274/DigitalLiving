import subprocess
import time
import sys

l: list[float]=[]
ls: list[str]=[]
for _ in range(10):
    start: float = time.time()
    subprocess.run(["python", "main.py"],stdout=sys.stdout)
    l.append(time.time()-start)
    with open("output.txt","r",encoding="utf-8") as file:
        ls.append(file.readlines()[-1])
print(l,ls,sep="\n\n\n")