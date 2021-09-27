import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils.transforms import normalize
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def track_data():
    folder = ''
    fnames = ['2d_adapt_search.pkl',
              '2d_adapt_search_2_sigma_0_01.pkl']

    fig, axes = plt.subplots(4 + 2, 2, sharex='all', sharey='all')
    fig.set_size_inches(8, 5)
    plt.subplots_adjust(top=0.949,
                        bottom=0.097,
                        left=0.067,
                        right=0.967,
                        hspace=0.13,
                        wspace=0.056)

    for jj, name in enumerate(fnames):
        data = pd.read_pickle(folder + name)
        data = data[:250]

        # select valid data
        good_data = data.loc[data['EMIT'].notnull()]
        bad_data = data.loc[data['EMIT'].isnull()]

        if 'DQ4' in good_data.keys():
            names = ['FocusingSolenoid', 'MatchingSolenoid', 'DQ4', 'DQ5']
        else:
            names = ['FocusingSolenoid', 'MatchingSolenoid']

        # get data bounds
        bounds = [data.min(), data.max()]
        bounds = torch.tensor(np.vstack([ele[names].to_numpy() for ele in
                                         bounds]).astype(np.float))

        good_x = torch.from_numpy(good_data[names].to_numpy())
        good_x_idx = torch.from_numpy(good_data['state_idx'].to_numpy())
        bad_x = torch.from_numpy(bad_data[names].to_numpy())
        bad_x_idx = torch.from_numpy(bad_data['state_idx'].to_numpy())

        good_x = normalize(good_x, bounds)
        bad_x = normalize(bad_x, bounds)

        # get lengthscale
        ls = np.load(name.split('.')[0] + '_lengthscale_trace.npy')[:50]

        n_params = len(names)

        ax = axes.T[jj]
        for i in range(n_params):
            ax[i].plot(good_x_idx, good_x[:, i], '+', label='Valid measurement')
            ax[i].plot(bad_x_idx, bad_x[:, i], '+', label='Invalid measurement')

            if jj == 0:
                ax[i].set_ylabel(names[i].replace('Solenoid', ''))
            # ax2.plot(ls[:, -1], ls[:, i])
        ax[-2].plot(ls[:, -1], ls[:, -2] / max(ls[:, -2]))
        # ax[-1].plot(ls[:, -1], ls[:, -3])

        # add fraction of valid points
        vfrac = []
        for kk in range(1, 50+1):
            vfrac += [data.iloc[:kk*5]['EMIT'].notnull().sum()/(5*kk)]

        ax[-1].plot(vfrac)

        if jj == 0:
            ax[-2].set_ylabel('$<\sigma>_{norm}$')
            ax[-1].set_ylabel('$N_{val}/N_{tot}$')

            ax[0].legend()
        ax[-1].set_xlabel('Sample index')
        ax[-1].set_xlim(0, 50)

    lbl = ['a', 'b']
    for a, label in zip(axes[0], lbl):
        a.text(-0.0, 1.05, f'({label})', ha='right',
               va='bottom', transform=a.transAxes,
               fontdict={'size': 12})

    fig.savefig('neurips_comparison.svg')


track_data()

plt.show()
