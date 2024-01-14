# Work flow #

# Step 1: Build dataset #
with open ('BTB_make_data.py') as f:
    exec(f.read())

# Step 2: Make predictions #
with open('prediction.py') as f:
    exec(f.read())

# Step 3: Do comparisons #
with open('compare.py') as f:
    exec(f.read())