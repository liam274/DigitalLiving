import subprocess
import sys
result=subprocess.run(["python","main.py"],stdout=sys.stdout,stdin=sys.stdin,stderr=sys.stderr)
if result.returncode!=0:
    with open("state.stat","w") as file:
        file.write("")
input("Press enter to continue...")