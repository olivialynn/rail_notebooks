Somoclu SOM Summarizer Cluster Demo
===================================

**Author:** Ziang Yan, Sam Schmidt

**Last successfully run:** June 16, 2023

This notebook shows a quick demonstration of the use of the
``SOMocluSummarizer`` summarization module. Algorithmically, this module
is not very different from the NZDir estimator/summarizer. NZDir
operates by finding neighboring photometric points around spectroscopic
objects. SOMocluSummarizer takes a large training set of data in the
``Inform_SOMocluUmmarizer`` stage and trains a self-organized map (SOM)
(using code from the ``somoclu`` package available at:
https://github.com/peterwittek/somoclu/). Once the SOM is set up, the
“best match unit” are determined for both the photometric/unknown data
and a set of spectroscopic data with known redshifts. For each SOM cell,
the algorithm constructs a histogram using the spectroscopic members
mapped to that cell, and weights these by the number of photometric
galaxies in that cell. Both the photometric and spectroscopic datasets
can also employ an optional weight per-galaxy.

The summarizer also identifies SOM cells that contain photometric data
but do not contain and galaxies with a measured spec-z, and thus do not
have an obvious redshift estimate. It writes out the (raveled) SOM cell
indices that contain “uncovered”/uncalibratable data to the file
specified by the ``uncovered_cell_file`` option as a list of integers.
The cellIDs and galaxy/objIDs for all photometric galaxies will be
written out to the file specified by the ``cellid_output`` parameter.
Any galaxies in these cells should really be removed, and thus some
iteration may be necessary in defining bin membership by looking at the
properties of objects in these uncovered cells before a final N(z) is
estimated, as otherwise a bias may be present.

The shape and number of cells used in constructing the SOM affects
performance, as do several tuning parameters. This paper,
http://www.giscience2010.org/pdfs/paper_230.pdf gives a rough guideline
that the number of cells should be of the order ~ 5 x sqrt (number of
data rows x number of column rows), though this is a rough guide. Some
studies have found a 2D SOM that is more elongated in one direction to
be preferential, while others claim that a square layout is optimal, the
user can set the number of cells in each SOM dimension via the
``n_rows`` and ``n_cols`` parameters. For more discussion on SOMs see
the Appendices of this KiDS paper: http://arxiv.org/abs/1909.09632.

As with the other RAIL summarizers, we bootstrap the spectroscopic
sample and return N bootstraps in an ensemble, along with a single
fiducial N(z) estimate.

More specific details of the algorithm’s set up will be described in the
course of this notebook, along with some illustrative plots.

Let’s set up our dependencies:

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pickle
    import rail
    import os
    import qp
    import tables_io
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.utils.path_utils import find_rail_file
    from rail.estimation.algos.somoclu_som import SOMocluInformer, SOMocluSummarizer
    from rail.estimation.algos.somoclu_som import get_bmus, plot_som

Next, let’s set up the Data Store, so that our RAIL module will know
where to fetch data:

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

First, let’s grab some data files. For the SOM, we will want to train on
a fairly large, representative set that encompasses all of our expected
data. We’ll grab a larger data file than we typically use in our demos
to ensure that we construct a meaningful SOM.

This data consists of ~150,000 galaxies from a single healpix pixel of
the comsoDC2 truth catalog with mock 10-year magnitude errors added. It
is cut at a relatively bright i<23.5 magnitudes in order to concentrate
on galaxies with particularly high S/N rates.

.. code:: ipython3

    training_file = "./healpix_10326_bright_data.hdf5"
    
    if not os.path.exists(training_file):
      os.system('curl -O https://portal.nersc.gov/cfs/lsst/PZ/healpix_10326_bright_data.hdf5')

.. code:: ipython3

    # way to get big data file
    training_data = DS.read_file("training_data", TableHandle, training_file)

Now, let’s set up the inform stage for our summarizer

We need to define all of our necessary initialization params, which
includes the following: - ``name`` (str): the name of our estimator, as
utilized by ceci - ``model`` (str): the name for the model file
containing the SOM and associated parameters that will be written by
this stage - ``hdf5_groupname`` (str): name of the hdf5 group (if any)
where the photometric data resides in the training file - ``n_rows``
(int): the number of dimensions in the y-direction for our 2D SOM -
``m_columns`` (int): the number of dimensions in the x-direction for our
2D SOM - ``som_iterations`` (int): the number of iteration steps during
SOM training. SOMs can take a while to converge, so we will use a fairly
large number of 500,000 iterations. - ``std_coeff`` (float): the
“radius” of how far to spread changes in the SOM - ``som_learning_rate``
(float): a number between 0 and 1 that controls how quickly the
weighting function decreases. SOM’s are not guaranteed to converge
mathematically, and so this parameter tunes how the response drops per
iteration. A typical values we might use might be between 0.5 and 0.75.
- ``column_usage`` (str): this value determines what values will be used
to construct the SOM, valid choices are ``colors``, ``magandcolors``,
and ``columns``. If set to ``colors``, the code will take adjacent
columns as specified in ``usecols`` to construct colors and use those as
SOM inputs. If set to ``magandcolors`` it will use the single column
specfied by ``ref_column_name`` and the aforementioned colors to
construct the SOM. If set to ``columns`` then it will simply take each
of the columns in ``usecols`` with no modification. So, if a user wants
to use K magnitudes and L colors, they can precompute the colors and
specify all names in ``usecols``. NOTE: accompanying ``usecols`` you
must have a ``nondetect_val`` dictionary that lists the replacement
values for any non-detection-valued entries for each column, see the
code for an example dictionary. WE will set ``column_usage`` to colors
and use only colors in this example notebook.

.. code:: ipython3

    grid_type = 'hexagonal'
    inform_dict = dict(model='output_SOMoclu_model.pkl', hdf5_groupname='photometry',
                       n_rows=71, n_columns=71, 
                       gridtype = grid_type,
                       std_coeff=12.0, som_learning_rate=0.75,
                       column_usage='colors')

.. code:: ipython3

    inform_som = SOMocluInformer.make_stage(name='inform_som', **inform_dict)

Let’s run our stage, which will write out a file called
``output_SOM_model.pkl``

**NOTE for those using M1 Macs:** you may get an error like
``wrap_train not found`` when running the inform stage in the cell just
below here. If so, this can be solved by reinstalling somoclu from conda
rather than pip with the command:

::

   conda install -c conda-forge somoclu

.. code:: ipython3

    %%time
    inform_som.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  model_inform_som: inprogress_output_SOMoclu_model.pkl, inform_som
    CPU times: user 8min 6s, sys: 390 ms, total: 8min 6s
    Wall time: 2min 4s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f0e6c7aab60>



Running the stage took ~1 minute wall time on a desktop Mac and ~3.5
minutes on NERSC Jupyter lab. Remember, however, that in many production
cases we would likely load a pre-trained SOM specifically tuned to the
given dataset, and this inform stage would not be run each time. Let’s
read in the SOM model file, which contains our som model and several of
the parameters used in constructing the SOM, and needed by our
summarization model.

.. code:: ipython3

    with open("output_SOMoclu_model.pkl", "rb") as f:
        model = pickle.load(f)

To visualize our SOM, let’s calculate the cell occupation of our
training sample, as well as the mean redshift of the galaxies in each
cell. The SOM took colors as inputs, so we will need to construct the
colors for our training set galaxie:

.. code:: ipython3

    bands = ['u','g','r','i','z','y']
    bandnames = [f"mag_{band}_lsst" for band in bands]
    ngal = len(training_data.data['photometry']['mag_i_lsst'])
    colors = np.zeros([5, ngal])
    for i in range(5):
        colors[i] = training_data.data['photometry'][bandnames[i]] - training_data.data['photometry'][bandnames[i+1]]

We can calculate the best SOM cell using the get_bmus() function defined
in somoclu_som.py, which will return the 2D SOM coordinates for each
galaxy. Then we group the SOM cells into 100 hierarchical clusters and
calculate the occupation and mean redshift in each cluster.

.. code:: ipython3

    SOM = model['som']
    bmu_coordinates = get_bmus(SOM, colors.T, 1000).T

.. code:: ipython3

    import sklearn.cluster as sc
    
    n_clusters = 100
    algorithm = sc.AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    SOM.cluster(algorithm)
    som_cluster_inds = SOM.clusters.reshape(-1)
    phot_pixel_coords = np.ravel_multi_index(bmu_coordinates, (71, 71))
    
    phot_som_clusterind = som_cluster_inds[phot_pixel_coords]


First, let’s visualize our hierarchical clusters by plotting the SOM
cells grouped into each cluster number:

.. code:: ipython3

    cellid = np.zeros_like(SOM.umatrix).reshape(-1)
    for i in range(n_clusters):
        cellid[som_cluster_inds==i] = i
    cellid = cellid.reshape(SOM.umatrix.shape)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    plot_som(ax, cellid.T, grid_type=grid_type, colormap=cm.coolwarm, cbar_name='cell ID')



.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_22_0.png


we see variations in number of cells in each grouping and geometry, but
mostly nice contiguous cell chunks. Next, let’s plot the cell occupation
and mean redshift:

.. code:: ipython3

    meanszs = np.zeros_like(SOM.umatrix).reshape(-1)
    cellocc = np.zeros_like(SOM.umatrix).reshape(-1)
    
    for i in range(n_clusters):
        meanszs[som_cluster_inds==i] = np.mean(training_data.data['photometry']['redshift'][phot_som_clusterind==i])
        cellocc[som_cluster_inds==i] = (phot_som_clusterind==i).sum()
    meanszs = meanszs.reshape(SOM.umatrix.shape)
    cellocc = cellocc.reshape(SOM.umatrix.shape)

Here is the cluster occupation distribution:

.. code:: ipython3

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    plot_som(ax, cellocc.T, grid_type=grid_type, colormap=cm.coolwarm, cbar_name='cell occupation')



.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_26_0.png


And here is the mean redshift per cluster:

.. code:: ipython3

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    plot_som(ax, meanszs.T, grid_type=grid_type, colormap=cm.coolwarm, cbar_name='mean redshift')



.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_28_0.png


Now that we have illustrated what exactly we have constructed, let’s use
the SOM to predict the redshift distribution for a set of photometric
objects. We will make a simple cut in spectroscopic redshift to create a
compact redshift bin. In more realistic circumstances we would likely be
using color cuts or photometric redshift estimates to define our test
bin(s). We will cut our photometric sample to only include galaxies in
0.5<specz<0.9.

We will need to trim both our spec-z set to i<23.5 to match our trained
SOM:

.. code:: ipython3

    testfile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    data = tables_io.read(testfile)['photometry']
    mask = ((data['redshift'] > 0.2) & (data['redshift']<0.5))
    brightmask = ((mask) & (data['mag_i_lsst']<23.5))
    trim_data = {}
    bright_data = {}
    for key in data.keys():
        trim_data[key] = data[key][mask]
        bright_data[key] = data[key][brightmask]
    trimdict = dict(photometry=trim_data)
    brightdict = dict(photometry=bright_data)
    # add data to data store
    test_data = DS.add_data("tomo_bin", trimdict, TableHandle)
    bright_data = DS.add_data("bright_bin", brightdict, TableHandle)

.. code:: ipython3

    specfile = find_rail_file("examples_data/testdata/test_dc2_validation_9816.hdf5")
    spec_data = tables_io.read(specfile)['photometry']
    smask = (spec_data['mag_i_lsst'] <23.5)
    trim_spec = {}
    for key in spec_data.keys():
        trim_spec[key] = spec_data[key][smask]
    trim_dict = dict(photometry=trim_spec)
    spec_data = DS.add_data("spec_data", trim_dict, TableHandle)

Note that we have removed the ‘photometry’ group, we will specify the
``phot_groupname`` as "" in the parameters below.

As before, let us specify our initialization params for the
SomocluSOMSummarizer stage, including:

-  ``model``: name of the pickled model that we created, in this case
   “output_SOM_model.pkl”
-  ``hdf5_groupname`` (str): hdf5 group for our photometric data (in our
   case "")
-  ``objid_name`` (str): string specifying the name of the ID column, if
   present photom data, will be written out to cellid_output file
-  ``spec_groupname`` (str): hdf5 group for the spectroscopic data
-  ``nzbins`` (int): number of bins to use in our histogram ensemble
-  ``n_clusters`` (int): number of hierarchical clusters
-  ``nsamples`` (int): number of bootstrap samples to generate
-  ``output`` (str): name of the output qp file with N samples
-  ``single_NZ`` (str): name of the qp file with fiducial distribution
-  ``uncovered_cell_file`` (str): name of hdf5 file containing a list of
   all of the cells with phot data but no spec-z objects: photometric
   objects in these cells will *not* be accounted for in the final N(z),
   and should really be removed from the sample before running the
   summarizer. Note that we return a single integer that is constructed
   from the pairs of SOM cell indices via
   ``np.ravel_multi_index``\ (indices).

Now let’s initialize and run the summarizer. One feature of the SOM: if
any SOM cells contain photometric data but do not contain any redshifts
values in the spectroscopic set, then no reasonable redshift estimate
for those objects is defined, and they are skipped. The method currently
prints the indices of uncovered cells, we may modify the algorithm to
actually output the uncovered galaxies in a separate file in the future.

Let’s open the fiducial N(z) file, plot it, and see how it looks, and
compare it to the true tomographic bin file:

.. code:: ipython3

    summ_dict = dict(model="output_SOMoclu_model.pkl", hdf5_groupname='photometry',
                     spec_groupname='photometry', nzbins=101, nsamples=25,
                     output='SOM_ensemble.hdf5', single_NZ='fiducial_SOMoclu_NZ.hdf5',
                     uncovered_cell_file='all_uncovered_cells.hdf5',
                     objid_name='id',
                     cellid_output='output_cellIDs.hdf5')
    som_summarizer = SOMocluSummarizer.make_stage(name='SOMoclu_summarizer', **summ_dict)    
    som_summarizer.summarize(test_data, spec_data)


.. parsed-literal::

    Inserting handle into data store.  model: output_SOMoclu_model.pkl, SOMoclu_summarizer
    Warning: number of clusters is not provided. The SOM will NOT be grouped into clusters.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(1024), np.int64(4609), np.int64(3074), np.int64(2053), np.int64(4101), np.int64(8), np.int64(2056), np.int64(4619), np.int64(2060), np.int64(3595), np.int64(16), np.int64(2065), np.int64(4114), np.int64(19), np.int64(3603), np.int64(2070), np.int64(4119), np.int64(4632), np.int64(3094), np.int64(1047), np.int64(3613), np.int64(4126), np.int64(1568), np.int64(2081), np.int64(2595), np.int64(4132), np.int64(4647), np.int64(4649), np.int64(4141), np.int64(4655), np.int64(4144), np.int64(4658), np.int64(3636), np.int64(3637), np.int64(2104), np.int64(4153), np.int64(4159), np.int64(4671), np.int64(4673), np.int64(4164), np.int64(3141), np.int64(3653), np.int64(2122), np.int64(4682), np.int64(4172), np.int64(2636), np.int64(3149), np.int64(3664), np.int64(2644), np.int64(4692), np.int64(4693), np.int64(4088), np.int64(2139), np.int64(4699), np.int64(4701), np.int64(3166), np.int64(4193), np.int64(4706), np.int64(4707), np.int64(4196), np.int64(4197), np.int64(4090), np.int64(3688), np.int64(3183), np.int64(3184), np.int64(4721), np.int64(2674), np.int64(2676), np.int64(2678), np.int64(1656), np.int64(1146), np.int64(3707), np.int64(4732), np.int64(4230), np.int64(4744), np.int64(2698), np.int64(3210), np.int64(4749), np.int64(3726), np.int64(4239), np.int64(4241), np.int64(3217), np.int64(2707), np.int64(4244), np.int64(1681), np.int64(1174), np.int64(4759), np.int64(3730), np.int64(1686), np.int64(4762), np.int64(4766), np.int64(2211), np.int64(4260), np.int64(4772), np.int64(1187), np.int64(3240), np.int64(4778), np.int64(4267), np.int64(2730), np.int64(2733), np.int64(3246), np.int64(4274), np.int64(2229), np.int64(3768), np.int64(1721), np.int64(1213), np.int64(3262), np.int64(1726), np.int64(4294), np.int64(4806), np.int64(2760), np.int64(3784), np.int64(206), np.int64(2256), np.int64(2258), np.int64(2259), np.int64(4309), np.int64(4313), np.int64(3803), np.int64(3294), np.int64(3296), np.int64(2273), np.int64(3811), np.int64(3304), np.int64(3306), np.int64(2795), np.int64(3824), np.int64(2801), np.int64(4339), np.int64(3315), np.int64(4854), np.int64(3320), np.int64(3833), np.int64(4858), np.int64(4351), np.int64(3330), np.int64(4355), np.int64(4870), np.int64(2311), np.int64(2824), np.int64(4361), np.int64(1288), np.int64(1289), np.int64(4876), np.int64(4365), np.int64(4878), np.int64(2319), np.int64(2830), np.int64(4369), np.int64(4370), np.int64(2323), np.int64(3855), np.int64(3859), np.int64(4374), np.int64(3861), np.int64(2330), np.int64(2332), np.int64(4895), np.int64(2340), np.int64(3878), np.int64(4906), np.int64(4396), np.int64(4909), np.int64(3884), np.int64(3887), np.int64(1328), np.int64(3377), np.int64(3380), np.int64(4919), np.int64(3385), np.int64(3897), np.int64(2364), np.int64(4415), np.int64(4930), np.int64(3394), np.int64(1347), np.int64(3908), np.int64(3909), np.int64(2376), np.int64(2889), np.int64(2890), np.int64(3400), np.int64(1354), np.int64(4941), np.int64(2382), np.int64(2383), np.int64(4951), np.int64(4440), np.int64(3418), np.int64(2911), np.int64(2913), np.int64(4454), np.int64(4457), np.int64(2921), np.int64(4971), np.int64(2412), np.int64(4972), np.int64(4974), np.int64(3947), np.int64(2929), np.int64(4466), np.int64(3442), np.int64(3444), np.int64(4469), np.int64(3958), np.int64(2935), np.int64(2937), np.int64(3452), np.int64(2941), np.int64(1917), np.int64(3456), np.int64(1921), np.int64(4482), np.int64(3459), np.int64(2436), np.int64(3971), np.int64(3975), np.int64(1928), np.int64(5002), np.int64(4491), np.int64(4492), np.int64(5006), np.int64(4498), np.int64(1426), np.int64(3987), np.int64(5017), np.int64(2460), np.int64(4510), np.int64(2463), np.int64(3999), np.int64(4000), np.int64(4514), np.int64(5026), np.int64(4002), np.int64(1954), np.int64(3849), np.int64(2984), np.int64(4009), np.int64(2986), np.int64(1451), np.int64(4013), np.int64(5039), np.int64(2998), np.int64(2487), np.int64(3000), np.int64(4538), np.int64(3517), np.int64(3520), np.int64(1478), np.int64(4551), np.int64(4552), np.int64(4039), np.int64(3530), np.int64(2507), np.int64(4560), np.int64(3025), np.int64(3536), np.int64(4563), np.int64(4564), np.int64(4049), np.int64(4051), np.int64(4052), np.int64(4568), np.int64(4570), np.int64(1499), np.int64(3550), np.int64(4062), np.int64(2532), np.int64(4070), np.int64(2535), np.int64(4072), np.int64(4585), np.int64(4590), np.int64(3054), np.int64(4080), np.int64(3570), np.int64(2547), np.int64(3572), np.int64(3573), np.int64(3062), np.int64(2551), np.int64(2035), np.int64(4083), np.int64(4086), np.int64(3071)}
    514 out of 5041 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.




.. parsed-literal::

    <rail.core.data.QPHandle at 0x7f0e00a356f0>



.. code:: ipython3

    fid_ens = qp.read("fiducial_SOMoclu_NZ.hdf5")

.. code:: ipython3

    def get_cont_hist(data, bins):
        hist, bin_edge = np.histogram(data, bins=bins, density=True)
        return hist, (bin_edge[1:]+bin_edge[:-1])/2

.. code:: ipython3

    test_nz_hist, zbin = get_cont_hist(test_data.data['photometry']['redshift'], np.linspace(0,3,101))
    som_nz_hist = np.squeeze(fid_ens.pdf(zbin))

Now we try to group SOM cells together with hierarchical clustering
method. To do this, we simply specify ``n_cluster`` in the input dict.
We want to test how many hierarchical clusters are optimal for the
redshift calibration task. We evaluate the performance by three values:
the difference between mean redshifts of the phot and spec catalog; the
difference between standard deviations; the ratio between effective
number density of represented photometric sources and the whole
photometric sample.

.. code:: ipython3

    n_clusterss = np.array([50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 71*71])
    true_full_mean = np.mean(test_data.data['photometry']['redshift'])
    true_full_std = np.std(test_data.data['photometry']['redshift'])
    mu_diff = np.zeros(n_clusterss.size)
    means_diff = np.zeros((n_clusterss.size, 25))
    
    std_diff_mean = np.zeros(n_clusterss.size)
    neff_p_to_neff = np.zeros(n_clusterss.size)
    std_diff = np.zeros((n_clusterss.size, 25))
    for i, n_clusters in enumerate(n_clusterss):
        summ_dict = dict(model="output_SOMoclu_model.pkl", hdf5_groupname='photometry',
                     spec_groupname='photometry', nzbins=101, nsamples=25,
                     output='SOM_ensemble.hdf5', single_NZ='fiducial_SOMoclu_NZ.hdf5',
                     n_clusters=n_clusters,
                     uncovered_cell_file='all_uncovered_cells.hdf5',
                     objid_name='id',
                     cellid_output='output_cellIDs.hdf5')
        som_summarizer = SOMocluSummarizer.make_stage(name='SOMoclu_summarizer', **summ_dict)    
        som_summarizer.summarize(test_data, spec_data)
        
        full_ens = qp.read("SOM_ensemble.hdf5")
        full_means = full_ens.mean().flatten()
        full_stds = full_ens.std().flatten()
        
        # mean and width of bootstraps
        mu_diff[i] = np.mean(full_means) - true_full_mean
        means_diff[i] = full_means - true_full_mean
        
        std_diff_mean[i] = np.mean(full_stds) - true_full_std
        std_diff[i] = full_stds - true_full_std
        neff_p_to_neff[i] = som_summarizer.neff_p_to_neff
        full_sig = np.std(full_means)
        



.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer
    the following clusters contain photometric data but not spectroscopic data:
    set()
    18 out of 50 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer
    the following clusters contain photometric data but not spectroscopic data:
    set()
    29 out of 100 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer
    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(138), np.int64(190)}
    51 out of 200 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(413), np.int64(94)}
    123 out of 500 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(611), np.int64(901), np.int64(648), np.int64(494), np.int64(121), np.int64(698), np.int64(955), np.int64(828)}
    227 out of 1000 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(901), np.int64(1033), np.int64(1289), np.int64(1297), np.int64(1050), np.int64(540), np.int64(1311), np.int64(432), np.int64(1456), np.int64(955), np.int64(828), np.int64(1223), np.int64(1484), np.int64(1231), np.int64(354), np.int64(486), np.int64(494), np.int64(370), np.int64(1398), np.int64(1146), np.int64(1278), np.int64(767)}
    323 out of 1500 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(516), np.int64(1541), np.int64(1289), np.int64(1545), np.int64(269), np.int64(655), np.int64(1297), np.int64(1686), np.int64(1050), np.int64(1567), np.int64(160), np.int64(1571), np.int64(1534), np.int64(45), np.int64(303), np.int64(1456), np.int64(1663), np.int64(1074), np.int64(944), np.int64(1590), np.int64(1975), np.int64(1720), np.int64(698), np.int64(1223), np.int64(1484), np.int64(1229), np.int64(1231), np.int64(1872), np.int64(1999), np.int64(215), np.int64(494), np.int64(242), np.int64(118), np.int64(886), np.int64(1657), np.int64(1146), np.int64(507), np.int64(1661), np.int64(1278), np.int64(1535)}
    388 out of 2000 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(1027), np.int64(1545), np.int64(1546), np.int64(2059), np.int64(2062), np.int64(15), np.int64(2579), np.int64(1049), np.int64(1567), np.int64(2595), np.int64(1571), np.int64(2092), np.int64(2096), np.int64(1094), np.int64(2119), np.int64(2120), np.int64(585), np.int64(1110), np.int64(1628), np.int64(1119), np.int64(2154), np.int64(2669), np.int64(1657), np.int64(1148), np.int64(2172), np.int64(1151), np.int64(1663), np.int64(1153), np.int64(1154), np.int64(2703), np.int64(1686), np.int64(676), np.int64(1195), np.int64(2220), np.int64(1201), np.int64(1720), np.int64(698), np.int64(709), np.int64(1232), np.int64(1234), np.int64(1242), np.int64(1245), np.int64(1758), np.int64(1250), np.int64(741), np.int64(2279), np.int64(2283), np.int64(235), np.int64(2795), np.int64(749), np.int64(1271), np.int64(2808), np.int64(766), np.int64(2304), np.int64(770), np.int64(1795), np.int64(2824), np.int64(2319), np.int64(1296), np.int64(2323), np.int64(2324), np.int64(790), np.int64(281), np.int64(1316), np.int64(2344), np.int64(1322), np.int64(811), np.int64(2356), np.int64(830), np.int64(843), np.int64(1872), np.int64(1362), np.int64(1370), np.int64(2911), np.int64(2913), np.int64(1386), np.int64(876), np.int64(886), np.int64(2935), np.int64(2941), np.int64(1412), np.int64(1925), np.int64(396), np.int64(2447), np.int64(1426), np.int64(1433), np.int64(2459), np.int64(2463), np.int64(415), np.int64(2982), np.int64(2984), np.int64(2986), np.int64(2485), np.int64(1470), np.int64(1478), np.int64(1490), np.int64(2519), np.int64(2523), np.int64(1500), np.int64(1503), np.int64(999), np.int64(2024), np.int64(1002), np.int64(493), np.int64(499), np.int64(1014), np.int64(1535)}
    451 out of 3000 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(3), np.int64(1027), np.int64(2059), np.int64(2062), np.int64(2579), np.int64(3603), np.int64(1045), np.int64(22), np.int64(3094), np.int64(2076), np.int64(3613), np.int64(2079), np.int64(33), np.int64(2595), np.int64(554), np.int64(47), np.int64(2096), np.int64(49), np.int64(48), np.int64(3637), np.int64(2104), np.int64(569), np.int64(61), np.int64(573), np.int64(575), np.int64(64), np.int64(576), np.int64(66), np.int64(3141), np.int64(3653), np.int64(71), np.int64(2119), np.int64(2120), np.int64(74), np.int64(3149), np.int64(3664), np.int64(597), np.int64(2139), np.int64(92), np.int64(1119), np.int64(97), np.int64(100), np.int64(616), np.int64(2154), np.int64(620), np.int64(3183), np.int64(624), np.int64(1139), np.int64(630), np.int64(640), np.int64(642), np.int64(647), np.int64(137), np.int64(3726), np.int64(3730), np.int64(1171), np.int64(675), np.int64(168), np.int64(680), np.int64(3246), np.int64(178), np.int64(692), np.int64(693), np.int64(3768), np.int64(3262), np.int64(705), np.int64(2245), np.int64(1221), np.int64(3784), np.int64(3803), np.int64(734), np.int64(223), np.int64(3296), np.int64(3811), np.int64(3304), np.int64(2283), np.int64(2795), np.int64(1261), np.int64(751), np.int64(3824), np.int64(3315), np.int64(3320), np.int64(3833), np.int64(3836), np.int64(2304), np.int64(2824), np.int64(778), np.int64(2830), np.int64(2319), np.int64(3855), np.int64(3858), np.int64(2323), np.int64(2324), np.int64(3859), np.int64(287), np.int64(2336), np.int64(2340), np.int64(1316), np.int64(1322), np.int64(3884), np.int64(3887), np.int64(2353), np.int64(307), np.int64(2356), np.int64(1332), np.int64(310), np.int64(3380), np.int64(1334), np.int64(314), np.int64(829), np.int64(3394), np.int64(3908), np.int64(3909), np.int64(2374), np.int64(3400), np.int64(842), np.int64(341), np.int64(2903), np.int64(348), np.int64(350), np.int64(2911), np.int64(2913), np.int64(3947), np.int64(3442), np.int64(371), np.int64(3444), np.int64(3958), np.int64(2935), np.int64(2941), np.int64(382), np.int64(385), np.int64(1921), np.int64(3459), np.int64(2436), np.int64(3971), np.int64(390), np.int64(392), np.int64(2447), np.int64(911), np.int64(1426), np.int64(3987), np.int64(406), np.int64(2459), np.int64(414), np.int64(415), np.int64(2463), np.int64(2464), np.int64(418), np.int64(421), np.int64(2470), np.int64(935), np.int64(2984), np.int64(2986), np.int64(2485), np.int64(443), np.int64(3517), np.int64(3520), np.int64(1478), np.int64(3530), np.int64(1490), np.int64(3550), np.int64(2024), np.int64(3054), np.int64(3570), np.int64(3573), np.int64(1014), np.int64(2551), np.int64(3062), np.int64(3071)}
    497 out of 4000 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(1024), np.int64(4609), np.int64(3074), np.int64(2053), np.int64(4101), np.int64(8), np.int64(2056), np.int64(4619), np.int64(2060), np.int64(3595), np.int64(16), np.int64(2065), np.int64(4114), np.int64(19), np.int64(3603), np.int64(2070), np.int64(4119), np.int64(4632), np.int64(3094), np.int64(1047), np.int64(3613), np.int64(4126), np.int64(1568), np.int64(2081), np.int64(2595), np.int64(4132), np.int64(4647), np.int64(4649), np.int64(4141), np.int64(4655), np.int64(4144), np.int64(4658), np.int64(3636), np.int64(3637), np.int64(2104), np.int64(4153), np.int64(4159), np.int64(4671), np.int64(4673), np.int64(4164), np.int64(3141), np.int64(3653), np.int64(2122), np.int64(4682), np.int64(4172), np.int64(2636), np.int64(3149), np.int64(3664), np.int64(2644), np.int64(4692), np.int64(4693), np.int64(4088), np.int64(2139), np.int64(4699), np.int64(4701), np.int64(3166), np.int64(4193), np.int64(4706), np.int64(4707), np.int64(4196), np.int64(4197), np.int64(4090), np.int64(3688), np.int64(3183), np.int64(3184), np.int64(4721), np.int64(2674), np.int64(2676), np.int64(2678), np.int64(1656), np.int64(1146), np.int64(3707), np.int64(4732), np.int64(4230), np.int64(4744), np.int64(2698), np.int64(3210), np.int64(4749), np.int64(3726), np.int64(4239), np.int64(4241), np.int64(3217), np.int64(2707), np.int64(4244), np.int64(1681), np.int64(1174), np.int64(4759), np.int64(3730), np.int64(1686), np.int64(4762), np.int64(4766), np.int64(2211), np.int64(4260), np.int64(4772), np.int64(1187), np.int64(3240), np.int64(4778), np.int64(4267), np.int64(2730), np.int64(2733), np.int64(3246), np.int64(4274), np.int64(2229), np.int64(3768), np.int64(1721), np.int64(1213), np.int64(3262), np.int64(1726), np.int64(4294), np.int64(4806), np.int64(2760), np.int64(3784), np.int64(206), np.int64(2256), np.int64(2258), np.int64(2259), np.int64(4309), np.int64(4313), np.int64(3803), np.int64(3294), np.int64(3296), np.int64(2273), np.int64(3811), np.int64(3304), np.int64(3306), np.int64(2795), np.int64(3824), np.int64(2801), np.int64(4339), np.int64(3315), np.int64(4854), np.int64(3320), np.int64(3833), np.int64(4858), np.int64(4351), np.int64(3330), np.int64(4355), np.int64(4870), np.int64(2311), np.int64(2824), np.int64(4361), np.int64(1288), np.int64(1289), np.int64(4876), np.int64(4365), np.int64(4878), np.int64(2319), np.int64(2830), np.int64(4369), np.int64(4370), np.int64(2323), np.int64(3855), np.int64(3859), np.int64(4374), np.int64(3861), np.int64(2330), np.int64(2332), np.int64(4895), np.int64(2340), np.int64(3878), np.int64(4906), np.int64(4396), np.int64(4909), np.int64(3884), np.int64(3887), np.int64(1328), np.int64(3377), np.int64(3380), np.int64(4919), np.int64(3385), np.int64(3897), np.int64(2364), np.int64(4415), np.int64(4930), np.int64(3394), np.int64(1347), np.int64(3908), np.int64(3909), np.int64(2376), np.int64(2889), np.int64(2890), np.int64(3400), np.int64(1354), np.int64(4941), np.int64(2382), np.int64(2383), np.int64(4951), np.int64(4440), np.int64(3418), np.int64(2911), np.int64(2913), np.int64(4454), np.int64(4457), np.int64(2921), np.int64(4971), np.int64(2412), np.int64(4972), np.int64(4974), np.int64(3947), np.int64(2929), np.int64(4466), np.int64(3442), np.int64(3444), np.int64(4469), np.int64(3958), np.int64(2935), np.int64(2937), np.int64(3452), np.int64(2941), np.int64(1917), np.int64(3456), np.int64(1921), np.int64(4482), np.int64(3459), np.int64(2436), np.int64(3971), np.int64(3975), np.int64(1928), np.int64(5002), np.int64(4491), np.int64(4492), np.int64(5006), np.int64(4498), np.int64(1426), np.int64(3987), np.int64(5017), np.int64(2460), np.int64(4510), np.int64(2463), np.int64(3999), np.int64(4000), np.int64(4514), np.int64(5026), np.int64(4002), np.int64(1954), np.int64(3849), np.int64(2984), np.int64(4009), np.int64(2986), np.int64(1451), np.int64(4013), np.int64(5039), np.int64(2998), np.int64(2487), np.int64(3000), np.int64(4538), np.int64(3517), np.int64(3520), np.int64(1478), np.int64(4551), np.int64(4552), np.int64(4039), np.int64(3530), np.int64(2507), np.int64(4560), np.int64(3025), np.int64(3536), np.int64(4563), np.int64(4564), np.int64(4049), np.int64(4051), np.int64(4052), np.int64(4568), np.int64(4570), np.int64(1499), np.int64(3550), np.int64(4062), np.int64(2532), np.int64(4070), np.int64(2535), np.int64(4072), np.int64(4585), np.int64(4590), np.int64(3054), np.int64(4080), np.int64(3570), np.int64(2547), np.int64(3572), np.int64(3573), np.int64(3062), np.int64(2551), np.int64(2035), np.int64(4083), np.int64(4086), np.int64(3071)}
    514 out of 5041 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. code:: ipython3

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20,5))
    
    for i in range(25):
        axes[0].plot(n_clusterss, means_diff.T[i], lw=0.2, color='C1')
    axes[0].plot(n_clusterss, mu_diff, lw=1, color='k')
    axes[0].axhline(0,1,0)
    axes[0].set_xlabel('Number of clusters')
    axes[0].set_ylabel(r'$\left\langle z \right\rangle - \left\langle z \right\rangle_{\mathrm{true}}$')
    
    for i in range(25):
        axes[1].plot(n_clusterss, std_diff.T[i], lw=0.2, color='C1')
    axes[1].plot(n_clusterss, std_diff_mean, lw=1, color='k')
    axes[1].axhline(0,1,0)
    
    axes[1].set_xlabel('Number of clusters')
    axes[1].set_ylabel(r'$\mathrm{std}(z) - \mathrm{std}(z)_{\mathrm{true}}$')
    
    
    axes[2].plot(n_clusterss, neff_p_to_neff*100, lw=1, color='k')
    
    axes[2].set_xlabel('Number of clusters')
    axes[2].set_ylabel(r'$n_{\mathrm{eff}}\'/n_{\mathrm{eff}}$(%)')




.. parsed-literal::

    Text(0, 0.5, "$n_{\\mathrm{eff}}\\'/n_{\\mathrm{eff}}$(%)")




.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_41_1.png


From the three plots above, we can see that when n_cluster>1000, the
redshift bias is within ~0.01 and the difference in standard deviation
does not change significantly, but the effective number density
continues to decrease. This is because when we have more clusters, the
risk that a cluster does not contain a spectroscopic source becomes
higher. Therefore, we might choose ~1000 clusters for the calibration in
this practice, so that we can keep as many galaxies as possible while
minimize the bias in average and standard deviation of galaxy redshifts.

.. code:: ipython3

    summ_dict = dict(model="output_SOMoclu_model.pkl", hdf5_groupname='photometry',
                     spec_groupname='photometry', nzbins=101, nsamples=25,
                     output='SOM_ensemble.hdf5', single_NZ='fiducial_SOMoclu_NZ.hdf5',
                     n_clusters=1000,
                     uncovered_cell_file='all_uncovered_cells.hdf5',
                     objid_name='id',
                     cellid_output='output_cellIDs.hdf5')
    
    som_summarizer = SOMocluSummarizer.make_stage(name='SOMoclu_summarizer', **summ_dict)
    som_summarizer.summarize(test_data, spec_data)
    
    test_nz_hist, zbin = get_cont_hist(test_data.data['photometry']['redshift'], np.linspace(0,3,101))
    som_nz_hist = np.squeeze(fid_ens.pdf(zbin))


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 1545
    Inserting handle into data store.  cellid_output_SOMoclu_summarizer: inprogress_output_cellIDs.hdf5, SOMoclu_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(611), np.int64(901), np.int64(648), np.int64(494), np.int64(121), np.int64(698), np.int64(955), np.int64(828)}
    227 out of 1000 have usable data
    Inserting handle into data store.  output_SOMoclu_summarizer: inprogress_SOM_ensemble.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  single_NZ_SOMoclu_summarizer: inprogress_fiducial_SOMoclu_NZ.hdf5, SOMoclu_summarizer
    Inserting handle into data store.  uncovered_cluster_file_SOMoclu_summarizer: inprogress_uncovered_cluster_file_SOMoclu_summarizer, SOMoclu_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_SOMoclu_summarizer was not generated.


.. code:: ipython3

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ax.set_xlabel("redshift", fontsize=15)
    ax.set_ylabel("N(z)", fontsize=15)
    ax.plot(zbin, test_nz_hist, label='True N(z)')
    ax.plot(zbin, som_nz_hist, label='SOM N(z)')
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f0e0093ec80>




.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_44_1.png


Seems fine, roughly the correct redshift range for the lower redshift
peak, but a few secondary peaks at large z tail. What if we try the
bright dataset that we made?

.. code:: ipython3

    bright_dict = dict(model="output_SOMoclu_model.pkl", hdf5_groupname='photometry',
                       spec_groupname='photometry', nzbins=101, nsamples=25,
                       output='BRIGHT_SOMoclu_ensemble.hdf5', single_NZ='BRIGHT_fiducial_SOMoclu_NZ.hdf5',
                       uncovered_cell_file="BRIGHT_uncovered_cells.hdf5",
                       n_clusters=1000,
                       objid_name='id',
                       cellid_output='BRIGHT_output_cellIDs.hdf5')
    bright_summarizer = SOMocluSummarizer.make_stage(name='bright_summarizer', **bright_dict)

.. code:: ipython3

    bright_summarizer.summarize(bright_data, spec_data)


.. parsed-literal::

    Process 0 running summarizer on chunk 0 - 645
    Inserting handle into data store.  cellid_output_bright_summarizer: inprogress_BRIGHT_output_cellIDs.hdf5, bright_summarizer


.. parsed-literal::

    the following clusters contain photometric data but not spectroscopic data:
    {np.int64(121), np.int64(955), np.int64(611), np.int64(901)}
    181 out of 1000 have usable data
    Inserting handle into data store.  output_bright_summarizer: inprogress_BRIGHT_SOMoclu_ensemble.hdf5, bright_summarizer
    Inserting handle into data store.  single_NZ_bright_summarizer: inprogress_BRIGHT_fiducial_SOMoclu_NZ.hdf5, bright_summarizer
    Inserting handle into data store.  uncovered_cluster_file_bright_summarizer: inprogress_uncovered_cluster_file_bright_summarizer, bright_summarizer


.. parsed-literal::

    NOTE/WARNING: Expected output file uncovered_cluster_file_bright_summarizer was not generated.




.. parsed-literal::

    <rail.core.data.QPHandle at 0x7f0e00931b70>



.. code:: ipython3

    bright_fid_ens = qp.read("BRIGHT_fiducial_SOMoclu_NZ.hdf5")

.. code:: ipython3

    bright_nz_hist, zbin = get_cont_hist(bright_data.data['photometry']['redshift'], np.linspace(0,3,101))
    bright_som_nz_hist = np.squeeze(bright_fid_ens.pdf(zbin))

.. code:: ipython3

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ax.set_xlabel("redshift", fontsize=15)
    ax.set_ylabel("N(z)", fontsize=15)
    ax.plot(zbin, bright_nz_hist, label='True N(z), bright')
    ax.plot(zbin, bright_som_nz_hist, label='SOM N(z), bright')
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f0e00931720>




.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_50_1.png


Looks better, we’ve eliminated the secondary peak. Now, SOMs are a bit
touchy to train, and are highly dependent on the dataset used to train
them. This demo used a relatively small dataset (~150,000 DC2 galaxies
from one healpix pixel) to train the SOM, and even smaller photometric
and spectroscopic datasets of 10,000 and 20,000 galaxies. We should
expect slightly better results with more data, at least in cells where
the spectroscopic data is representative.

However, there is a caveat that SOMs are not guaranteed to converge, and
are very sensitive to both the input data and tunable parameters of the
model. So, users should do some verification tests before trusting the
SOM is going to give accurate results.

Finally, let’s load up our bootstrap ensembles and overplot N(z) of
bootstrap samples:

.. code:: ipython3

    boot_ens = qp.read("BRIGHT_SOMoclu_ensemble.hdf5")

.. code:: ipython3

    fig, ax=plt.subplots(1,1,figsize=(12, 8))
    ax.set_xlim((0,1))
    ax.set_xlabel("redshift", fontsize=15)
    ax.set_ylabel("bootstrap N(z)", fontsize=15)
    ax.legend(loc='upper right', fontsize=13);
    
    ax.plot(zbin, bright_nz_hist, label='True N(z), bright', color='C1', zorder=1)
    ax.plot(zbin, bright_som_nz_hist, label='SOM mean N(z), bright', color='k', zorder=2)
    
    for i in range(boot_ens.npdf):
        #ax = plt.subplot(2,3,i+1)
        pdf = np.squeeze(boot_ens[i].pdf(zbin))
        if i == 0:        
            ax.plot(zbin, pdf, color='C2',zorder=0, alpha=0.5, label='SOM bootstrap N(z) samples, bright')
        else:
            ax.plot(zbin, pdf, color='C2',zorder=0, alpha=0.5)
        #boot_ens[i].plot_native(axes=ax, label=f'SOM bootstrap {i}')
    plt.legend(fontsize=15)


.. parsed-literal::

    /tmp/ipykernel_6597/4031386170.py:5: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
      ax.legend(loc='upper right', fontsize=13);




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f0e00931270>




.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_54_2.png


Quantitative metrics
--------------------

Let’s look at how we’ve done at estimating the mean redshift and “width”
(via standard deviation) of our tomographic bin compared to the true
redshift and “width” for both our “full” sample and “bright” i<23.5
samples. We will plot the mean and std dev for the full and bright
distributions compared to the true mean and width, and show the Gaussian
uncertainty approximation given the scatter in the bootstraps for the
mean:

.. code:: ipython3

    from scipy.stats import norm

.. code:: ipython3

    full_ens = qp.read("SOM_ensemble.hdf5")
    full_means = full_ens.mean().flatten()
    full_stds = full_ens.std().flatten()
    true_full_mean = np.mean(test_data.data['photometry']['redshift'])
    true_full_std = np.std(test_data.data['photometry']['redshift'])
    # mean and width of bootstraps
    full_mu = np.mean(full_means)
    full_sig = np.std(full_means)
    full_norm = norm(loc=full_mu, scale=full_sig)
    grid = np.linspace(0, .7, 301)
    full_uncert = full_norm.pdf(grid)*2.51*full_sig

Let’s check the accuracy and precision of mean readshift:

.. code:: ipython3

    print("The mean redshift of the SOM ensemble is: "+str(round(np.mean(full_means),4)) + '+-' + str(round(np.std(full_means),4)))
    print("The mean redshift of the real data is: "+str(round(true_full_mean,4)))
    print("The bias of mean redshift is:"+str(round(np.mean(full_means)-true_full_mean,4)) + '+-' + str(round(np.std(full_means),4)))


.. parsed-literal::

    The mean redshift of the SOM ensemble is: 0.3391+-0.0032
    The mean redshift of the real data is: 0.3547
    The bias of mean redshift is:-0.0156+-0.0032


.. code:: ipython3

    bright_means = boot_ens.mean().flatten()
    bright_stds = boot_ens.std().flatten()
    true_bright_mean = np.mean(bright_data.data['photometry']['redshift'])
    true_bright_std = np.std(bright_data.data['photometry']['redshift'])
    bright_uncert = np.std(bright_means)
    # mean and width of bootstraps
    bright_mu = np.mean(bright_means)
    bright_sig = np.std(bright_means)
    bright_norm = norm(loc=bright_mu, scale=bright_sig)
    bright_uncert = bright_norm.pdf(grid)*2.51*bright_sig

.. code:: ipython3

    print("The mean redshift of the SOM ensemble is: "+str(round(np.mean(bright_means),4)) + '+-' + str(round(np.std(bright_means),4)))
    print("The mean redshift of the real data is: "+str(round(true_bright_mean,4)))
    print("The bias of mean redshift is:"+str(round(np.mean(bright_means)-true_bright_mean, 4)) + '+-' + str(round(np.std(bright_means),4)))


.. parsed-literal::

    The mean redshift of the SOM ensemble is: 0.3408+-0.0026
    The mean redshift of the real data is: 0.3493
    The bias of mean redshift is:-0.0086+-0.0026


.. code:: ipython3

    plt.figure(figsize=(12,18))
    ax0 = plt.subplot(2, 1, 1)
    ax0.set_xlim(0.0, 0.7)
    ax0.axvline(true_full_mean, color='r', lw=3, label='true mean full sample')
    ax0.vlines(full_means, ymin=0, ymax=1, color='r', ls='--', lw=1, label='bootstrap means')
    ax0.axvline(true_full_std, color='b', lw=3, label='true std full sample')
    ax0.vlines(full_stds, ymin=0, ymax=1, lw=1, color='b', ls='--', label='bootstrap stds')
    ax0.plot(grid, full_uncert, c='k', label='full mean uncertainty')
    ax0.legend(loc='upper right', fontsize=12)
    ax0.set_xlabel('redshift', fontsize=12)
    ax0.set_title('mean and std for full sample', fontsize=12)
    
    ax1 = plt.subplot(2, 1, 2)
    ax1.set_xlim(0.0, 0.7)
    ax1.axvline(true_bright_mean, color='r', lw=3, label='true mean bright sample')
    ax1.vlines(bright_means, ymin=0, ymax=1, color='r', ls='--', lw=1, label='bootstrap means')
    ax1.axvline(true_bright_std, color='b', lw=3, label='true std bright sample')
    ax1.plot(grid, bright_uncert, c='k', label='bright mean uncertainty')
    ax1.vlines(bright_stds, ymin=0, ymax=1, ls='--', lw=1, color='b', label='bootstrap stds')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_xlabel('redshift', fontsize=12)
    ax1.set_title('mean and std for bright sample', fontsize=12);



.. image:: ../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_files/../../../docs/rendered/estimation_examples/somocluSOMcluster_demo_62_0.png


For both cases, the mean redshifts seem to be pretty precise and
accurate (bright sample seems more precise). For the full sample, the
SOM N(z) are slightly wider, while for the bright sample the widths are
also fairly accurate. For both cases, the errors in mean redshift are at
levels of ~0.005, close to the tolerance for cosmological analysis.
However, we have not consider the photometric error in magnitudes and
colors, as well as additional color selections. Our sample is also
limited. This demo only serves as a preliminary implementation of SOM in
RAIL.
