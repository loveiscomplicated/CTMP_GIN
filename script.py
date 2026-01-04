import argparse

# Create the parser
parser = argparse.ArgumentParser(description="This is a simple argument parser.")

# Add arguments
parser.add_argument("--first", dest="var1", help="First variable")
parser.add_argument("--second", dest="var2", help="Second variable")

# Parse the arguments
args = parser.parse_args()

print(args.var1)
print(args.var2)