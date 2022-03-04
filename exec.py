import sys
import subprocess

args = sys.argv[1:]
if len(args) == 0:
    raise ValueError("Must provide a path to an executable file.")
print(f"Running {args[0]} with arguments {args[1:]}")

output = subprocess.run(args, capture_output=True)
print(output.stdout)

