import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

name = f"robust_10percent_pix2pix_2_bs32_ep1_lambda1000_vanilla"
output_dir = f"./output/{name}"

# loss function
fname = f"{output_dir}/loss_log.txt"

df = pd.read_csv(fname, sep=" ", header=0)
header = df.columns

plt.figure(figsize=(6,5))
plt.xlabel("iteration")
plt.ylabel("loss")
plt.yscale('log')

epoch = df["#iter"]
key_list = header[4:]
for i, key in enumerate(key_list):
    plt.plot(epoch, df[key], label=key)
plt.legend(loc="upper right")

# Get the current date and time
current_datetime = dt.datetime.now()

# Format the date and time as a string to use in a file name
# For example, "YYYY-MM-DD_HH-MM-SS"
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

plt.savefig(f"{name}_log_loss.png")