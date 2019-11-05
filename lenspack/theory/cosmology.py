# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
from subprocess import call
from glob import glob


class nicaea(object):
    """Python wrapper for NICAEA."""
    def __init__(self, path, Om0=0.25, Ode0=0.75, h=0.7, Ob0=0.044,
                 sigma8=0.8, ns=0.95, w0=-1.0, w1=0.0, Onu0=0.0, Neff=0.0,
                 zs=1.0, linear=False):
        """Compute theoretical weak-lensing quantities in a given cosmology.

        See NICAEA documentation for further details at
        https://nicaea.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        path : str
            Full path to base NICAEA directory.
        Om0, ..., Neff : float, optional
            Cosmological parameters.
        zs : float, optional
            Redshift of source galaxies. Only a single redshift is currently
            supported. Default is 1.0.
        linear : bool, optional
            Whether to compute using only the linear matter power spectrum.
            Default is False, in which case the nonlinear halofit prescription
            of Takahashi et al. (2012) is used.

        Notes
        -----
        This wrapper works by copying the necessary parameter files from
        NICAEA's default directory into a temporary directory. Computations
        are then carried out in this temporary directory, which can later
        be easily removed.

        References
        ----------
        * Kilbinger, Benabed, Guy, et al., A&A 497, 667 (2009)
        * https://github.com/CosmoStat/nicaea

        Examples
        --------
        # Plot two-point correlation functions
        >>> nic = nicaea('.../nicaea_2.7', Om0=0.3, Ode0=0.7)
        >>> nic.compute()
        >>> theta, xi_p = nic.xi_p.T
        >>> xi_m = nic.xi_m[:, 1]
        >>> nic.destroy()

        >>> fig, ax = plt.subplots(1, 1)
        >>> ax.loglog(theta, xi_p, label='xi_p')
        >>> ax.loglog(theta, xi_m, label='xi_m')
        >>> ax.set_xlabel('theta [arcmin]')
        >>> ax.set_ylabel('Two-point correlation function')
        >>> ax.legend(loc=0)
        >>> plt.show()

        """
        # Path to NICAEA
        self.path = path

        # Cosmological parameters
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.h = h
        self.Ob0 = Ob0
        self.sigma8 = sigma8
        self.w0 = w0
        self.w1 = w1
        self.Onu0 = Onu0
        self.Neff = Neff
        self.ns = ns
        self.zs = zs
        self.linear = linear

        self.update_parfiles()

    def update_parfiles(self):
        """Modify default parameter files to reflect user inputs."""
        # Path to default NICAEA parameter files
        parpath = os.path.join(self.path, 'par_files')
        assert os.path.exists(parpath), "Path does not exist."

        # Create temporary directory for modified parameter files and output
        if not hasattr(self, 'tmpdir'):
            self.tmpdir = tempfile.mkdtemp()
            self._saved_umask = os.umask(0o077)

        # Copy NICAEA parameter files to temporary directory
        call(['cp', os.path.join(parpath, 'cosmo.par'), self.tmpdir])
        call(['cp', os.path.join(parpath, 'cosmo_lens.par'), self.tmpdir])
        call(['cp', os.path.join(parpath, 'nofz.par'), self.tmpdir])
        call(['cp', os.path.join(parpath, 'nofz_single'), self.tmpdir])
        call(['cp', os.path.join(parpath, 'nofz_1'), self.tmpdir])
        call(['cp', os.path.join(parpath, 'nofz_2'), self.tmpdir])

        # A clumsy function to modify parameter files
        def edit_cosmo(parfile, key, value):
            with open(os.path.join(self.tmpdir, parfile), 'r') as f:
                data = f.readlines()
            for i, line in enumerate(data):
                if line.startswith(key):
                    split = line.split('\t')
                    if key == 'snonlinear':
                        print(split)
                        split[1] = value
                    else:
                        for j, val in enumerate(split[1:-2]):
                            if not val == '':
                                split[j+1] = str(value)
                    data[i] = '\t'.join(split)
                    break
            with open(os.path.join(self.tmpdir, parfile), 'w') as f:
                f.writelines(data)

        # Update cosmology
        edit_cosmo('cosmo.par', 'Omega_m', self.Om0)
        edit_cosmo('cosmo.par', 'Omega_de', self.Ode0)
        edit_cosmo('cosmo.par', 'w0_de', self.w0)
        edit_cosmo('cosmo.par', 'w1_de', self.w1)
        edit_cosmo('cosmo.par', 'h_100', self.h)
        edit_cosmo('cosmo.par', 'Omega_b', self.Ob0)
        edit_cosmo('cosmo.par', 'Omega_nu_mass', self.Onu0)
        edit_cosmo('cosmo.par', 'Neff_nu_mass', self.Neff)
        edit_cosmo('cosmo.par', 'normalization', self.sigma8)
        edit_cosmo('cosmo.par', 'n_spec', self.ns)
        if self.linear:
            edit_cosmo('cosmo.par', 'snonlinear', 'linear')

        # Update n(z)
        with open(os.path.join(self.tmpdir, 'nofz.par'), 'r') as f:
            data = f.readlines()
        for i, line in enumerate(data):
            split = line.split()
            if line.startswith('Nzbin'):
                split[1] = '1\n'
                data[i] = '\t\t'.join(split)
            elif line.startswith('nzfile'):
                split[1] = 'nofz_single\n'
                data[i] = '\t\t'.join(split[:2])
        with open(os.path.join(self.tmpdir, 'nofz.par'), 'w') as f:
            f.writelines(data)

        with open(os.path.join(self.tmpdir, 'nofz_single'), 'r') as f:
            data = f.readlines()
        data[1] = str(self.zs) + "\n"
        data[2] = str(self.zs) + "\n"
        with open(os.path.join(self.tmpdir, 'nofz_single'), 'w') as f:
            f.writelines(data)

    def compute(self, tmin=0.5, tmax=500, nt=20, lmin=10, lmax=1e5, nl=128):
        """Run NICAEA's weak-lensing calculator.

        Parameters
        ----------
        tmin, tmax : float, optional
            Lower and upper bounds on angular scale theta. Defaults are
            (0.5, 500).
        nt : int, optional
            Number of theta bins. Default is 10.
        lmin, lmax : float, optional
            Lower and upper bounds on multipole ell. Defaults are (10, 1e5).
        nl : int, optional
            Number of ell bins. Default is 128.

        Four new attributes are generated as follows.

        name         description
        ----         -----------
        xi_p         Two-point correlation function xi+
        xi_m         Two-point correlation function xi-
        P_kappa      Convergence power spectrum (continuous ell)
        P_kappa_d    Convergence power spectrum (discrete ell)

        More outputs could be harvested in the future, such as gammasqr etc.

        """
        assert hasattr(self, 'tmpdir'), "Need to run update_parfiles()."

        # Remember where we came from
        homedir = os.getcwd()

        # Go into tmp dir
        os.chdir(self.tmpdir)
        # Call lensingdemo binary
        call(["lensingdemo", "-L", "{} {} {}".format(lmin, lmax, nl),
              "-T", "{} {} {}".format(tmin, tmax, nt)])

        # Load xi_pm and power spectra
        self.xi_p = np.loadtxt('xi_p')
        self.xi_m = np.loadtxt('xi_m')
        self.P_kappa = np.loadtxt('P_kappa')
        self.P_kappa_d = np.loadtxt('P_kappa_d')

        # Return to original directory
        os.chdir(homedir)

    def destroy(self, verbose=True):
        """Remove any temporary directory and its contents."""
        # Do nothing if there is no current tmp dir
        assert hasattr(self, 'tmpdir'), "Nothing to destroy."

        # Find and delete all files in the tmp dir
        files = glob(os.path.join(self.tmpdir, "*"))
        for f in files:
            os.remove(f)
        os.umask(self._saved_umask)
        os.rmdir(self.tmpdir)
        if verbose:
            print("Removed {}".format(self.tmpdir))

        # Remove attributes related to the tmp dir
        del(self.tmpdir)
        del(self._saved_umask)
