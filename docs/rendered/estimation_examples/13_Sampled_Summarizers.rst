Test Sampled Summarizers
========================

**Author:** Sam Schmidt

**Last successfully run:** April 26, 2023

June 28 update: I modified the summarizers to output not just N sample
N(z) distributions (saved to the file specified via the ``output``
keyword), but also the single fiducial N(z) estimate (saved to the file
specified via the ``single_NZ`` keyword). I also updated NZDir and
included it in this example notebook

.. code:: ipython3

    import os
    import rail
    import numpy as np
    import pandas as pd
    import tables_io
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code:: ipython3

    from rail.estimation.algos.k_nearneigh import KNearNeighInformer, KNearNeighEstimator

.. code:: ipython3

    from rail.estimation.algos.var_inf import VarInfStackSummarizer
    from rail.estimation.algos.naive_stack import NaiveStackSummarizer
    from rail.estimation.algos.point_est_hist import PointEstHistSummarizer
    from rail.estimation.algos.nz_dir import NZDirInformer, NZDirSummarizer
    from rail.core.data import TableHandle, QPHandle
    from rail.core.stage import RailStage

.. code:: ipython3

    import qp

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

To create some N(z) distributions, we’ll want some PDFs to work with
first, for a quick demo let’s just run some photo-z’s using the
KNearNeighEstimator estimator using the 10,000 training galaxies to
generate ~20,000 PDFs using data from healpix 9816 of cosmoDC2_v1.1.4
that are included in the RAIL repo:

.. code:: ipython3

    knn_dict = dict(zmin=0.0, zmax=3.0, nzbins=301, trainfrac=0.75,
                    sigma_grid_min=0.01, sigma_grid_max=0.07, ngrid_sigma=10,
                    nneigh_min=3, nneigh_max=7, hdf5_groupname='photometry')

.. code:: ipython3

    pz_train = KNearNeighInformer.make_stage(name='inform_KNN', model='demo_knn.pkl', **knn_dict)

.. code:: ipython3

    # Load up the example healpix 9816 data and stick in the DataStore
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

.. code:: ipython3

    # train knnpz
    pz_train.inform(training_data)


.. parsed-literal::

    split into 7669 training and 2556 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.023333333333333334 and numneigh=7
    
    
    
    Inserting handle into data store.  model_inform_KNN: inprogress_demo_knn.pkl, inform_KNN




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1424801540>



.. code:: ipython3

    pz = KNearNeighEstimator.make_stage(name='KNN', hdf5_groupname='photometry',
                                  model=pz_train.get_handle('model'))
    qp_data = pz.estimate(test_data)


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_KNN: inprogress_output_KNN.hdf5, KNN
    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating PZ PDF for rows 20,000 - 20,449


So, ``qp_data`` now contains the 20,000 PDFs from KNearNeighEstimator,
we can feed this in to three summarizers to generate an overall N(z)
distribution. We won’t bother with any tomographic selections for this
demo, just the overall N(z). It is stored as ``qp_data``, but has also
been saved to disk as ``output_KNN.fits`` as an astropy table. If you
want to read in this data to grab the qp Ensemble at a later stage, you
can do this via qp with a ``ens = qp.read("output_KNN.fits")``

I coded up **quick and dirty** bootstrap versions of the
``NaiveStackSummarizer``, ``PointEstHistSummarizer``, and
``VarInference`` sumarizers. These are not optimized, not parallel
(issue created for future update), but they do produce N different
bootstrap realizations of the overall N(z) which are returned as a qp
Ensemble (Note: the previous versions of these degraders returned only
the single overall N(z) rather than samples).

Naive Stack
-----------

Naive stack just “stacks” i.e. sums up, the PDFs and returns a qp.interp
distribution with bins defined by np.linspace(zmin, zmax, nzbins), we
will create a stack with 41 bins and generate 20 bootstrap realizations

.. code:: ipython3

    stacker = NaiveStackSummarizer.make_stage(zmin=0.0, zmax=3.0, nzbins=41, nsamples=20, output="Naive_samples.hdf5", single_NZ="NaiveStack_NZ.hdf5")

.. code:: ipython3

    naive_results = stacker.summarize(qp_data)


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10000


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Inserting handle into data store.  output: inprogress_Naive_samples.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_NaiveStack_NZ.hdf5, NaiveStackSummarizer


The results are now in naive_results, but because of the DataStore, the
actual *ensemble* is stored in ``.data``, let’s grab the ensemble and
plot a few of the bootstrap sample N(z) estimates:

.. code:: ipython3

    newens = naive_results.data

.. code:: ipython3

    fig, axs = plt.subplots(figsize=(8,6))
    for i in range(0, 20, 2):
        newens[i].plot_native(axes=axs, label=f"sample {i}")
    axs.plot([0,3],[0,0],'k--')
    axs.set_xlim(0,3)
    axs.legend(loc='upper right')




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f13c9769a20>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_19_1.png


The summarizer also outputs a **second** file containing the fiducial
N(z). We saved the fiducial N(z) in the file “NaiveStack_NZ.hdf5”, let’s
grab the N(z) estimate with qp and plot it with the native plotter:

.. code:: ipython3

    naive_nz = qp.read("NaiveStack_NZ.hdf5")
    naive_nz.plot_native(xlim=(0,3))




.. parsed-literal::

    <Axes: xlabel='redshift', ylabel='p(z)'>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_21_1.png


Point Estimate Hist
-------------------

PointEstHistSummarizer takes the point estimate mode of each PDF and
then histograms these, we’ll again generate 41 bootstrap samples of this
and plot a few of the resultant histograms. Note: For some reason the
plotting on the histogram distribution in qp is a little wonky, it
appears alpha is broken, so this plot is not the best:

.. code:: ipython3

    pointy = PointEstHistSummarizer.make_stage(zmin=0.0, zmax=3.0, nzbins=41, nsamples=20, single_NZ="point_NZ.hdf5", output="point_samples.hdf5")

.. code:: ipython3

    %%time
    pointy_results = pointy.summarize(qp_data)


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10000


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Inserting handle into data store.  output: inprogress_point_samples.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_point_NZ.hdf5, PointEstHistSummarizer
    CPU times: user 16 s, sys: 5.1 s, total: 21.1 s
    Wall time: 21.1 s


.. code:: ipython3

    pens = pointy_results.data

.. code:: ipython3

    fig, axs = plt.subplots(figsize=(8,6))
    pens[0].plot_native(axes=axs, fc = [0, 0, 1, 0.01])
    pens[1].plot_native(axes=axs, fc = [0, 1, 0, 0.01])
    pens[4].plot_native(axes=axs, fc = [1, 0, 0, 0.01])
    axs.set_xlim(0,3)
    axs.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f13c6183d60>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_26_1.png


Again, we have saved the fiducial N(z) in a separate file,
“point_NZ.hdf5”, we could read that data in if we desired.

VarInfStackSummarizer
---------------------

VarInfStackSummarizer implements Markus’ variational inference scheme
and returns qp.interp gridded distribution. VarInfStackSummarizer tends
to get a little wonky if you use too many bins, so we’ll only use 25
bins. Again let’s generate 20 samples and plot a few:

.. code:: ipython3

    runner=VarInfStackSummarizer.make_stage(name='test_varinf', zmin=0.0,zmax=3.0,nzbins=25, niter=10, nsamples=20,
                                        output="sampletest.hdf5", single_NZ="varinf_NZ.hdf5")

.. code:: ipython3

    %%time
    varinf_results = runner.summarize(qp_data)


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 20449


.. parsed-literal::

    Inserting handle into data store.  output_test_varinf: inprogress_sampletest.hdf5, test_varinf
    Inserting handle into data store.  single_NZ_test_varinf: inprogress_varinf_NZ.hdf5, test_varinf
    CPU times: user 1.02 s, sys: 49.2 ms, total: 1.07 s
    Wall time: 1.07 s


.. code:: ipython3

    vens = varinf_results.data
    vens




.. parsed-literal::

    <qp.ensemble.Ensemble at 0x7f13c4fd1930>



Let’s plot the fiducial N(z) for this distribution:

.. code:: ipython3

    varinf_nz = qp.read("varinf_NZ.hdf5")
    varinf_nz.plot_native(xlim=(0,3))




.. parsed-literal::

    <Axes: xlabel='redshift', ylabel='p(z)'>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_33_1.png


NZDir
-----

NZDirSummarizer is a different type of summarizer, taking a weighted set
of neighbors to a set of training spectroscopic objects to reconstruct
the redshift distribution of the photometric sample. I implemented a
bootstrap of the **spectroscopic data** rather than the photometric
data, both because it was much easier computationally, and I think that
the spectroscopic variance is more important to take account of than
simple bootstrap of the large photometric sample. We must first run the
``inform_NZDir`` stage to train up the K nearest neigh tree used by
NZDirSummarizer, then we will run ``NZDirSummarizer`` to actually
construct the N(z) estimate.

Like PointEstHistSummarizer NZDirSummarizer returns a qp.hist ensemble
of samples

.. code:: ipython3

    inf_nz = NZDirInformer.make_stage(n_neigh=8, hdf5_groupname="photometry", model="nzdir_model.pkl")

.. code:: ipython3

    inf_nz.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  model: inprogress_nzdir_model.pkl, NZDirInformer




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f13c7388cd0>



.. code:: ipython3

    nzd = NZDirSummarizer.make_stage(leafsize=20, zmin=0.0, zmax=3.0, nzbins=31, model="nzdir_model.pkl", hdf5_groupname='photometry',
                           output='NZDir_samples.hdf5', single_NZ='NZDir_NZ.hdf5')

.. code:: ipython3

    nzd_res = nzd.estimate(test_data)


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10000
    Inserting handle into data store.  single_NZ: inprogress_NZDir_NZ.hdf5, NZDirSummarizer
    Inserting handle into data store.  output: inprogress_NZDir_samples.hdf5, NZDirSummarizer
    Process 0 running estimator on chunk 10000 - 20000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449


.. code:: ipython3

    nzd_ens = nzd_res.data

.. code:: ipython3

    nzdir_nz = qp.read("NZDir_NZ.hdf5")

.. code:: ipython3

    fig, axs = plt.subplots(figsize=(10,8))
    nzd_ens[0].plot_native(axes=axs, fc = [0, 0, 1, 0.01])
    nzd_ens[1].plot_native(axes=axs, fc = [0, 1, 0, 0.01])
    nzd_ens[4].plot_native(axes=axs, fc = [1, 0, 0, 0.01])
    axs.set_xlim(0,3)
    axs.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f13c4b2fdc0>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_41_1.png


As we also wrote out the single estimate of N(z) we can read that data
from the second file written (specified by the ``single_NZ`` argument
given in NZDirSummarizer.make_stage above, in this case “NZDir_NZ.hdf5”)

.. code:: ipython3

    nzdir_nz = qp.read("NZDir_NZ.hdf5")

.. code:: ipython3

    nzdir_nz.plot_native(xlim=(0,3))




.. parsed-literal::

    <Axes: xlabel='redshift', ylabel='p(z)'>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_44_1.png


Results
-------

All three results files are qp distributions, NaiveStackSummarizer and
VarInfStackSummarizer return qp.interp distributions while
PointEstHistSummarizer returns a qp.histogram distribution. Even with
the different distributions you can use qp functionality to do things
like determine the means, modes, etc… of the distributions. You could
then use the std dev of any of these to estimate a 1 sigma “shift”, etc…

.. code:: ipython3

    zgrid = np.linspace(0,3,41)
    names = ['naive', 'point', 'varinf', 'nzdir']
    enslist = [newens, pens, vens, nzd_ens]
    results_dict = {}
    for nm, en in zip(names, enslist):
        results_dict[f'{nm}_modes'] = en.mode(grid=zgrid).flatten()
        results_dict[f'{nm}_means'] = en.mean().flatten()
        results_dict[f'{nm}_std'] = en.std().flatten()

.. code:: ipython3

    results_dict




.. parsed-literal::

    {'naive_modes': array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
     'naive_means': array([0.90186015, 0.90594959, 0.90998413, 0.90789723, 0.91061713,
            0.91218896, 0.90767783, 0.90533378, 0.90791096, 0.90745548,
            0.91191782, 0.90432094, 0.90620407, 0.9087479 , 0.90410093,
            0.91160362, 0.90904143, 0.90900628, 0.89714565, 0.9139707 ]),
     'naive_std': array([0.45914047, 0.45888175, 0.45996851, 0.45790462, 0.45714307,
            0.46346092, 0.45902071, 0.45676502, 0.45571704, 0.45996679,
            0.45816316, 0.4557761 , 0.45894757, 0.45615512, 0.45516466,
            0.45840581, 0.4548986 , 0.4602565 , 0.45546553, 0.45910612]),
     'point_modes': array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
     'point_means': array([0.87747268, 0.87966422, 0.88390687, 0.88292592, 0.88652242,
            0.88416824, 0.88270433, 0.88085392, 0.88398525, 0.88127683,
            0.88581149, 0.88120195, 0.88141923, 0.88272697, 0.87882182,
            0.88615408, 0.88490508, 0.88213304, 0.87551255, 0.88610463]),
     'point_std': array([0.41965477, 0.41628602, 0.41871585, 0.41564559, 0.41745585,
            0.4206483 , 0.41603666, 0.41589367, 0.41726652, 0.41629308,
            0.41392928, 0.41791324, 0.41622493, 0.41328677, 0.41322766,
            0.4182433 , 0.41442477, 0.41762939, 0.41780264, 0.41593474]),
     'varinf_modes': array([0.9  , 0.9  , 0.9  , 0.9  , 0.975, 0.9  , 0.9  , 0.9  , 0.9  ,
            0.9  , 0.9  , 0.9  , 0.9  , 0.9  , 0.9  , 0.9  , 0.9  , 0.9  ,
            0.9  , 0.9  ]),
     'varinf_means': array([0.89007742, 0.89393137, 0.89348035, 0.89529006, 0.89632515,
            0.89304711, 0.89470237, 0.89476074, 0.88891055, 0.89065612,
            0.88963285, 0.88927123, 0.89480197, 0.89551207, 0.8947679 ,
            0.90084125, 0.89603184, 0.89083602, 0.89697313, 0.88937301]),
     'varinf_std': array([0.42671129, 0.43096865, 0.43001351, 0.42473237, 0.42611894,
            0.42608757, 0.42984578, 0.4286463 , 0.42915316, 0.42875199,
            0.42553345, 0.43073474, 0.42927409, 0.42736479, 0.42435752,
            0.43133787, 0.42896748, 0.42596593, 0.4269107 , 0.42523812]),
     'nzdir_modes': array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
     'nzdir_means': array([0.91527627, 0.92144451, 0.92166271, 0.92410584, 0.92118734,
            0.91847357, 0.92651638, 0.91452663, 0.92038542, 0.92942564,
            0.93107928, 0.91022267, 0.92142493, 0.92510869, 0.91931933,
            0.92364447, 0.919305  , 0.92659316, 0.91014124, 0.92499533]),
     'nzdir_std': array([0.46761557, 0.47080438, 0.46766624, 0.46515462, 0.46097655,
            0.46587178, 0.46626591, 0.46660589, 0.4646169 , 0.47014536,
            0.47077795, 0.46410977, 0.46431021, 0.46323947, 0.46435023,
            0.47121424, 0.46146407, 0.46880833, 0.46674124, 0.46798514])}



You can also use qp to compute quantities the pdf, cdf, ppf, etc… on any
grid that you want, much of the functionality of scipy.stats
distributions have been inherited by qp ensembles

.. code:: ipython3

    newgrid = np.linspace(0.005,2.995, 35)
    naive_pdf = newens.pdf(newgrid)
    point_cdf = pens.cdf(newgrid)
    var_ppf = vens.ppf(newgrid)

.. code:: ipython3

    plt.plot(newgrid, naive_pdf[0])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f13c4b16a40>]




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_50_1.png


.. code:: ipython3

    plt.plot(newgrid, point_cdf[0])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f13c4bccdf0>]




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_51_1.png


.. code:: ipython3

    plt.plot(newgrid, var_ppf[0])




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f13c4b69fc0>]




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_52_1.png


Shifts
------

If you want to “shift” a PDF, you can just evaluate the PDF on a shifted
grid, for example to shift the PDF by +0.0375 in redshift you could
evaluate on a shifted grid. For now we can just do this “by hand”, we
could easily implement ``shift`` functionality in qp, I think.

.. code:: ipython3

    def_grid = np.linspace(0., 3., 41)
    shift_grid = def_grid - 0.0675
    native_nz = newens.pdf(def_grid)
    shift_nz = newens.pdf(shift_grid)

.. code:: ipython3

    fig=plt.figure(figsize=(12,10))
    plt.plot(def_grid, native_nz[0], label="original")
    plt.plot(def_grid, shift_nz[0], label="shifted +0.0675")
    plt.legend(loc='upper right')




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f13c4c7efe0>




.. image:: ../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_files/../../../docs/rendered/estimation_examples/13_Sampled_Summarizers_55_1.png


You can estimate how much shift you might expect based on the statistics
of our bootstrap samples, say the std dev of the means for the
NZDir-derived distribution:

.. code:: ipython3

    results_dict['nzdir_means']




.. parsed-literal::

    array([0.91527627, 0.92144451, 0.92166271, 0.92410584, 0.92118734,
           0.91847357, 0.92651638, 0.91452663, 0.92038542, 0.92942564,
           0.93107928, 0.91022267, 0.92142493, 0.92510869, 0.91931933,
           0.92364447, 0.919305  , 0.92659316, 0.91014124, 0.92499533])



.. code:: ipython3

    spread = np.std(results_dict['nzdir_means'])

.. code:: ipython3

    spread




.. parsed-literal::

    np.float64(0.0054984088439073335)



Again, not a huge spread in predicted mean redshifts based solely on
bootstraps, even with only ~20,000 galaxies.
