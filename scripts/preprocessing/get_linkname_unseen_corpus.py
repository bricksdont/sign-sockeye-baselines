import sys

original_basename = sys.argv[1]

parts = original_basename.split(".")

if parts[0] == "dev" or parts[0] == "test":
    parts[0] += "_unseen"

linkname = ".".join(parts)

print(linkname)
