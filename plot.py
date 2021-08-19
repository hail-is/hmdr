import matplotlib.pyplot as plt

def plot(data, title):
    plt.style.use('ggplot')
    f = plt.figure()
    f, axes = plt.subplots(nrows = 3, ncols = 1, sharex=True, figsize=(10, 10))

    ticks = list(range(len(data)))

    axes[0].scatter(y=[record[1] for record in data],
                    x=ticks, label='simulated')
    axes[0].scatter(y=[record[2] for record in data],
                    x=ticks, label='estimated')
    axes[0].set_title("trace heritability")
    axes[0].legend()

    axes[1].scatter(y=[record[3] for record in data],
                    x=ticks)
    axes[1].set_title("variance explained")

    axes[2].scatter(y=[record[4] for record in data],
                    x=ticks, label='simulated')
    axes[2].scatter(y=[record[5] for record in data],
                    x=ticks, label='estimated')
    axes[2].legend()
    axes[2].set_title("variance of trace heritability")
    f.suptitle(title)
    plt.show()

for epochs in [1, 5, 25]:
    for recon_weight in [0.0, 1.0]:
        fname = f'log-linear_tranform_variance3-{epochs}-{recon_weight}'
        try:
            with open(fname) as fd:
                data = [[float(j) for j in x.split(',')] for i, x in enumerate(fd.readlines())]
                plot(data, f'epochs={epochs} recon_weight={recon_weight}')
        except FileNotFoundError:
            pass
