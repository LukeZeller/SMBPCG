import argparse

from gan import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--output', default=None, help='Where to store samples and models')
    parser.add_argument('--save_frequency', type=int, default=0, help='Number of iters between model file and sample output.')
    parser.add_argument('--saveD', action='store_false', help='Save discriminator models along with generator. Necessary for multiple training sessions.')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--level', default='', help='Name of level file (within data/full_levels/ascii directory).')
    parser.add_argument('--seed', type=int, default=-1, help='Manually set seed for training.')
    opt = parser.parse_args()

    train.train(opt)
