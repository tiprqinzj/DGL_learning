import matplotlib.pyplot as plt

if __name__ == '__main__':



    files = ['canonical4_2023-12-22/rank{}_train_loss.dat'.format(i) for i in range(100)]


    for i, file in enumerate(files):

        with open(file) as f:
            lines = f.read().splitlines()
            lines = [float(l) for l in lines]

        if i == 0:
            losses = [0] * len(lines)
        
            
        for i in range(len(losses)):
            losses[i] += lines[i]
    

    losses = [l / 100 for l in losses]

    with open('canonical4_2023-12-22/training_loss_summary.dat', 'w') as f:

        for i, l in enumerate(losses):
            f.write('{}\t{:.2f}\n'.format(i+1, l))

    # print every 500 steps
    avg_loss = 0
    for step, loss in enumerate(losses):
        avg_loss += loss
        if (step + 1) % 10000 == 0:
            print('steps {}, avg loss = {:.2f}'.format(step+1, avg_loss / 10000))
            avg_loss = 0
    
    print('steps {}, avg loss = {:.2f}'.format(step+1, avg_loss / ((step + 1) % 10000)))


    fig, ax = plt.subplots()

    ax.plot(losses)
    ax.set_ylim([-5, 105])

    plt.savefig('canonical4_2023-12-22/training_loss.png', dpi=300, bbox_inches='tight')

