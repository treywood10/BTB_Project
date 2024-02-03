# Work flow #

# Step 1: Build dataset #
with open ('BTB_make_data.py') as f:
    exec(f.read())

# Step 2: Do comparisons #
with open('compare.py') as f:
    exec(f.read())

# Step 3: Make predictions #
with open('prediction.py') as f:
    exec(f.read())

# Step 4: Accuracy plot #
with open('acc_plot.py') as f:
    exec(f.read())
