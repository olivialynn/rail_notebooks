Demo: RAIL Evaluation
=====================

**Authors:** Sam Schmidt, Alex Malz, Julia Gschwend, others…

**Last run successfully:** June 16, 2023 (with qp-prob >= 0.8.2)

The purpose of this notebook is to demonstrate the application of the
metrics scripts to be used on the photo-z PDF catalogs produced by the
PZ working group. The first implementation of the *evaluation* module is
based on the refactoring of the code used in `Schmidt et
al. 2020 <https://arxiv.org/pdf/2001.03621.pdf>`__, available on Github
repository `PZDC1paper <https://github.com/LSSTDESC/PZDC1paper>`__.

To run this notebook, you must install qp and have the notebook in the
same directory as ``utils.py`` (available in RAIL’s examples
directrory). You must also have installed all RAIL dependencies,
particularly for the estimation codes that you want to run, as well as
ceci, qp, tables_io, etc… See the RAIL installation instructions for
more info.

.. code:: ipython3

    import rail
    from rail.evaluation.metrics.cdeloss import *
    from rail.evaluation.evaluator import OldEvaluator
    from rail.core.data import QPHandle, TableHandle, Hdf5Handle
    from rail.core.stage import RailStage
    from utils import plot_pit_qq, ks_plot
    import qp 
    import os
    %matplotlib inline
    %reload_ext autoreload
    %autoreload 2

Let’s set up the data store, for more information on the data store, see
the golden spike example notebook in
``RAIL/examples/goldenspike_examples/goldenspike.ipynb``

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

Data
----

To compute the photo-z metrics of a given test sample, it is necessary
to read the output of a photo-z code containing galaxies’ photo-z PDFs.
Let’s use the toy data available in ``tests/data/``
(**test_dc2_training_9816.hdf5** and **test_dc2_validation_9816.hdf5**)
to generate a small sample of photo-z PDFs using the **FlexZBoost**
algorithm available on RAIL’s *estimation* module.

Photo-z Results
~~~~~~~~~~~~~~~

Run FlexZBoost
^^^^^^^^^^^^^^

If you have run the notebook ``RAIL_estimation_demo.ipynb``, this will
produce a file ``output_fzboost.hdf5``, writen at the location:
``<your_path>/RAIL/examples/estimation_examples/output_fzboost.hdf5``.

Otherwise we can download the file from NERSC

Next we need to set up some paths for the Data Store:

.. code:: ipython3

    from rail.utils.path_utils import find_rail_file
    try:
        pdfs_file = find_rail_file('examples_data/evaluation_data/data/output_fzboost.hdf5')
    except ValueError:
        pdfs_file = 'examples_data/evaluation_data/data/output_fzboost.hdf5'
        try:
            os.makedirs(os.path.dirname(pdfs_file))
        except FileExistsError:
            pass
        curl_com = f"curl -o {pdfs_file} https://portal.nersc.gov/cfs/lsst/PZ/output_fzboost.hdf5"
        os.system(curl_com)
    ztrue_file = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')


.. parsed-literal::

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:02 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:03 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:04 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:05 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:06 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:07 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:08 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:09 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:10 --:--:--     0

.. parsed-literal::

      0     0    0     0    0     0      0      0 --:--:--  0:00:11 --:--:--     0

.. parsed-literal::

      0 47.1M    0 11978    0     0   1013      0 13:32:58  0:00:11 13:32:47  2634

.. parsed-literal::

      0 47.1M    0  159k    0     0  12986      0  1:03:25  0:00:12  1:03:13 37891

.. parsed-literal::

      2 47.1M    2 1147k    0     0  86428      0  0:09:31  0:00:13  0:09:18  265k

.. parsed-literal::

      2 47.1M    2 1388k    0     0  95404      0  0:08:37  0:00:14  0:08:23  300k

.. parsed-literal::

      2 47.1M    2 1402k    0     0  87070      0  0:09:27  0:00:16  0:09:11  268k

.. parsed-literal::

      2 47.1M    2 1411k    0     0  84578      0  0:09:44  0:00:17  0:09:27  265k

.. parsed-literal::

      2 47.1M    2 1420k    0     0  82137      0  0:10:01  0:00:17  0:09:44  246k

.. parsed-literal::

      2 47.1M    2 1434k    0     0  78322      0  0:10:30  0:00:18  0:10:12 56975

.. parsed-literal::

      3 47.1M    3 1461k    0     0  75620      0  0:10:53  0:00:19  0:10:34 15243

.. parsed-literal::

      3 47.1M    3 1490k    0     0  73322      0  0:11:13  0:00:20  0:10:53 20702

.. parsed-literal::

      3 47.1M    3 1490k    0     0  69957      0  0:11:46  0:00:21  0:11:25 16973

.. parsed-literal::

      4 47.1M    4 1963k    0     0  88828      0  0:09:16  0:00:22  0:08:54  110k

.. parsed-literal::

      5 47.1M    5 2887k    0     0   122k      0  0:06:35  0:00:23  0:06:12  297k

.. parsed-literal::

      6 47.1M    6 3002k    0     0   119k      0  0:06:44  0:00:25  0:06:19  285k

.. parsed-literal::

      6 47.1M    6 3002k    0     0   114k      0  0:07:00  0:00:26  0:06:34  281k

.. parsed-literal::

      6 47.1M    6 3002k    0     0   110k      0  0:07:16  0:00:27  0:06:49  281k

.. parsed-literal::

      6 47.1M    6 3002k    0     0   106k      0  0:07:32  0:00:28  0:07:04  187k

.. parsed-literal::

      6 47.1M    6 3002k    0     0   102k      0  0:07:49  0:00:29  0:07:20 21279

.. parsed-literal::

      6 47.1M    6 3002k    0     0    99k      0  0:08:05  0:00:30  0:07:35     0

.. parsed-literal::

      6 47.1M    6 3002k    0     0  98588      0  0:08:21  0:00:31  0:07:50     0

.. parsed-literal::

      6 47.1M    6 3002k    0     0  95521      0  0:08:37  0:00:32  0:08:05     0

.. parsed-literal::

      6 47.1M    6 3002k    0     0  92639      0  0:08:53  0:00:33  0:08:20     0

.. parsed-literal::

      6 47.1M    6 3002k    0     0  89926      0  0:09:09  0:00:34  0:08:35     0

.. parsed-literal::

      6 47.1M    6 3122k    0     0  92385      0  0:08:54  0:00:34  0:08:20 27769

.. parsed-literal::

      8 47.1M    8 4256k    0     0   119k      0  0:06:43  0:00:35  0:06:08  281k

.. parsed-literal::

     10 47.1M   10 4868k    0     0   133k      0  0:06:02  0:00:36  0:05:26  422k

.. parsed-literal::

     12 47.1M   12 5896k    0     0   156k      0  0:05:08  0:00:37  0:04:31  647k

.. parsed-literal::

     15 47.1M   15 7448k    0     0   192k      0  0:04:10  0:00:38  0:03:32  995k

.. parsed-literal::

     16 47.1M   16 8184k    0     0   206k      0  0:03:53  0:00:39  0:03:14 1009k

.. parsed-literal::

     19 47.1M   19 9288k    0     0   228k      0  0:03:30  0:00:40  0:02:50 1010k

.. parsed-literal::

     21 47.1M   21 10.2M    0     0   253k      0  0:03:10  0:00:41  0:02:29 1135k

.. parsed-literal::

     23 47.1M   23 11.2M    0     0   269k      0  0:02:59  0:00:42  0:02:17 1127k

.. parsed-literal::

     25 47.1M   25 11.8M    0     0   277k      0  0:02:54  0:00:43  0:02:11  927k

.. parsed-literal::

     26 47.1M   26 12.4M    0     0   286k      0  0:02:48  0:00:44  0:02:04  902k

.. parsed-literal::

     28 47.1M   28 13.3M    0     0   299k      0  0:02:40  0:00:45  0:01:55  862k

.. parsed-literal::

     29 47.1M   29 14.1M    0     0   310k      0  0:02:35  0:00:46  0:01:49  781k

.. parsed-literal::

     31 47.1M   31 14.6M    0     0   315k      0  0:02:32  0:00:47  0:01:45  708k

.. parsed-literal::

     32 47.1M   32 15.5M    0     0   326k      0  0:02:27  0:00:48  0:01:39  762k

.. parsed-literal::

     35 47.1M   35 16.6M    0     0   344k      0  0:02:20  0:00:49  0:01:31  873k

.. parsed-literal::

     38 47.1M   38 18.2M    0     0   368k      0  0:02:10  0:00:50  0:01:20 1009k

.. parsed-literal::

     43 47.1M   43 20.5M    0     0   408k      0  0:01:58  0:00:51  0:01:07 1321k

.. parsed-literal::

     46 47.1M   46 21.8M    0     0   425k      0  0:01:53  0:00:52  0:01:01 1454k

.. parsed-literal::

     50 47.1M   50 23.9M    0     0   458k      0  0:01:45  0:00:53  0:00:52 1743k

.. parsed-literal::

     53 47.1M   53 25.0M    0     0   469k      0  0:01:42  0:00:54  0:00:48 1722k

.. parsed-literal::

     54 47.1M   54 25.7M    0     0   474k      0  0:01:41  0:00:55  0:00:46 1548k

.. parsed-literal::

     57 47.1M   57 27.1M    0     0   485k      0  0:01:39  0:00:57  0:00:42 1198k

.. parsed-literal::

     57 47.1M   57 27.1M    0     0   479k      0  0:01:40  0:00:58  0:00:42 1010k

.. parsed-literal::

     57 47.1M   57 27.2M    0     0   468k      0  0:01:42  0:00:59  0:00:43  562k

.. parsed-literal::

     57 47.1M   57 27.2M    0     0   460k      0  0:01:44  0:01:00  0:00:44  379k

.. parsed-literal::

     57 47.1M   57 27.2M    0     0   454k      0  0:01:46  0:01:01  0:00:45  264k

.. parsed-literal::

     57 47.1M   57 27.2M    0     0   448k      0  0:01:47  0:01:02  0:00:45 24359

.. parsed-literal::

     57 47.1M   57 27.2M    0     0   446k      0  0:01:48  0:01:02  0:00:46 27256

.. parsed-literal::

     59 47.1M   59 28.1M    0     0   452k      0  0:01:46  0:01:03  0:00:43  222k

.. parsed-literal::

     63 47.1M   63 29.8M    0     0   473k      0  0:01:41  0:01:04  0:00:37  655k

.. parsed-literal::

     65 47.1M   65 31.0M    0     0   481k      0  0:01:40  0:01:06  0:00:34  815k

.. parsed-literal::

     65 47.1M   65 31.0M    0     0   473k      0  0:01:41  0:01:07  0:00:34  812k

.. parsed-literal::

     65 47.1M   65 31.0M    0     0   468k      0  0:01:43  0:01:07  0:00:36  726k

.. parsed-literal::

     68 47.1M   68 32.1M    0     0   479k      0  0:01:40  0:01:08  0:00:32  823k

.. parsed-literal::

     71 47.1M   71 33.7M    0     0   495k      0  0:01:37  0:01:09  0:00:28  775k

.. parsed-literal::

     75 47.1M   75 35.6M    0     0   516k      0  0:01:33  0:01:10  0:00:23 1031k

.. parsed-literal::

     81 47.1M   81 38.3M    0     0   547k      0  0:01:28  0:01:11  0:00:17 1628k

.. parsed-literal::

     85 47.1M   85 40.4M    0     0   570k      0  0:01:24  0:01:12  0:00:12 2041k

.. parsed-literal::

     88 47.1M   88 41.8M    0     0   581k      0  0:01:22  0:01:13  0:00:09 1970k

.. parsed-literal::

     90 47.1M   90 42.6M    0     0   585k      0  0:01:22  0:01:14  0:00:08 1870k

.. parsed-literal::

     92 47.1M   92 43.7M    0     0   592k      0  0:01:21  0:01:15  0:00:06 1697k

.. parsed-literal::

     94 47.1M   94 44.4M    0     0   590k      0  0:01:21  0:01:16  0:00:05 1176k

.. parsed-literal::

     97 47.1M   97 45.7M    0     0   603k      0  0:01:20  0:01:17  0:00:03 1078k

.. parsed-literal::

    100 47.1M  100 47.1M    0     0   614k      0  0:01:18  0:01:18 --:--:-- 1127k


Read the data in, note that the fzdata is a ``qp`` Ensemble, and thus we
should read it in as a ``QPHandle`` type file, while the ztrue_data is
tabular data, and should be read in as a ``Tablehandle`` when adding to
the data store

.. code:: ipython3

    fzdata = DS.read_file('pdfs_data', QPHandle, pdfs_file)
    ztrue_data = DS.read_file('ztrue_data', TableHandle, ztrue_file)

.. code:: ipython3

    ztrue = ztrue_data()['photometry']['redshift']
    zgrid = fzdata().metadata['xvals'].ravel()
    photoz_mode = fzdata().mode(grid=zgrid)

.. code:: ipython3

    truth = DS.add_data('truth', ztrue_data()['photometry'], Hdf5Handle)
    ensemble = DS.add_data('ensemble', fzdata(), QPHandle)

Make an evaulator stage
-----------------------

Now let’s set up the Evaluator stage to compute our metrics for the
FlexZBoost results

.. code:: ipython3

    FZB_eval = OldEvaluator.make_stage(name='FZB_eval', truth=truth)

.. code:: ipython3

    FZB_results = FZB_eval.evaluate(ensemble(), truth)


.. parsed-literal::

    Inserting handle into data store.  input: None, FZB_eval


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/qp/metrics/array_metrics.py:27: UserWarning: p-value floored: true value smaller than 0.001. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
      return stats.anderson_ksamp([p_random_variables, q_random_variables], **kwargs)


.. parsed-literal::

    Inserting handle into data store.  output_FZB_eval: inprogress_output_FZB_eval.hdf5, FZB_eval


We can view the results as a pandas dataframe:

.. code:: ipython3

    import tables_io
    results_df= tables_io.convertObj(FZB_results(), tables_io.types.PD_DATAFRAME)
    results_df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PIT_AD_stat</th>
          <th>PIT_AD_pval</th>
          <th>PIT_AD_significance_level</th>
          <th>PIT_CvM_stat</th>
          <th>PIT_CvM_pval</th>
          <th>PIT_KS_stat</th>
          <th>PIT_KS_pval</th>
          <th>PIT_OutRate_stat</th>
          <th>POINT_SimgaIQR</th>
          <th>POINT_Bias</th>
          <th>POINT_OutlierRate</th>
          <th>POINT_SigmaMAD</th>
          <th>CDE_stat</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>84.956236</td>
          <td>NaN</td>
          <td>0.001</td>
          <td>9.623352</td>
          <td>NaN</td>
          <td>0.03359</td>
          <td>NaN</td>
          <td>0.058738</td>
          <td>0.020859</td>
          <td>0.00027</td>
          <td>0.106167</td>
          <td>0.020891</td>
          <td>-6.74027</td>
        </tr>
      </tbody>
    </table>
    </div>



So, there we have it, a way to generate all of our summary statistics
for FZBoost. And note also that the results file has been written out to
``output_FZB_eval.hdf5``, the name we specified when we ran
``make_stage`` (with output\_ prepended).

As an alternative, and to allow for a little more explanation for each
individual metric, we can calculate the metrics using functions from the
evaluation class separate from the stage infrastructure. Here are some
examples below.

CDF-based Metrics
-----------------

PIT
~~~

The Probability Integral Transform (PIT), is the Cumulative Distribution
Function (CDF) of the photo-z PDF

.. math::  \mathrm{CDF}(f, q)\ =\ \int_{-\infty}^{q}\ f(z)\ dz 

evaluated at the galaxy’s true redshift for every galaxy :math:`i` in
the catalog.

.. math::  \mathrm{PIT}(p_{i}(z);\ z_{i})\ =\ \int_{-\infty}^{z^{true}_{i}}\ p_{i}(z)\ dz 

.. code:: ipython3

    from qp.metrics.pit import PIT
    pitobj = PIT(fzdata(), ztrue)
    quant_ens = pitobj.pit
    metamets = pitobj.calculate_pit_meta_metrics()


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/qp/metrics/array_metrics.py:27: UserWarning: p-value floored: true value smaller than 0.001. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
      return stats.anderson_ksamp([p_random_variables, q_random_variables], **kwargs)


The *evaluate* method PIT class returns two objects, a quantile
distribution based on the full set of PIT values (a frozen distribution
object), and a dictionary of meta metrics associated to PIT (to be
detailed below).

.. code:: ipython3

    quant_ens




.. parsed-literal::

    Ensemble(the_class=quant,shape=(1, 96))



.. code:: ipython3

    metamets




.. parsed-literal::

    {'ad': Anderson_ksampResult(statistic=np.float64(84.95623553609381), critical_values=array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546]), pvalue=np.float64(0.001)),
     'cvm': CramerVonMisesResult(statistic=9.62335199605935, pvalue=9.265039846440004e-10),
     'ks': KstestResult(statistic=np.float64(0.033590049370962216), pvalue=np.float64(1.7621068075751534e-20), statistic_location=np.float64(0.9921210288809627), statistic_sign=np.int8(-1)),
     'outlier_rate': np.float64(0.05873797877466336)}



PIT values

.. code:: ipython3

    pit_vals = np.array(pitobj.pit_samps)
    pit_vals




.. parsed-literal::

    array([0.19392947, 0.36675619, 0.52017547, ..., 1.        , 0.93189232,
           0.4674437 ], shape=(20449,))



PIT outlier rate
~~~~~~~~~~~~~~~~

The PIT outlier rate is a global metric defined as the fraction of
galaxies in the sample with extreme PIT values. The lower and upper
limits for considering a PIT as outlier are optional parameters set at
the Metrics instantiation (default values are: PIT :math:`<10^{-4}` or
PIT :math:`>0.9999`).

.. code:: ipython3

    pit_out_rate = metamets['outlier_rate']
    print(f"PIT outlier rate of this sample: {pit_out_rate:.6f}") 
    pit_out_rate = pitobj.evaluate_PIT_outlier_rate()
    print(f"PIT outlier rate of this sample: {pit_out_rate:.6f}") 


.. parsed-literal::

    PIT outlier rate of this sample: 0.058738
    PIT outlier rate of this sample: 0.058738


PIT-QQ plot
~~~~~~~~~~~

The histogram of PIT values is a useful tool for a qualitative
assessment of PDFs quality. It shows whether the PDFs are: \* biased
(tilted PIT histogram) \* under-dispersed (excess counts close to the
boudaries 0 and 1) \* over-dispersed (lack of counts close the boudaries
0 and 1) \* well-calibrated (flat histogram)

Following the standards in DC1 paper, the PIT histogram is accompanied
by the quantile-quantile (QQ), which can be used to compare
qualitatively the PIT distribution obtained with the PDFs agaist the
ideal case (uniform distribution). The closer the QQ plot is to the
diagonal, the better is the PDFs calibration.

.. code:: ipython3

    pdfs = fzdata.data.objdata['yvals']
    plot_pit_qq(pdfs, zgrid, ztrue, title="PIT-QQ - toy data", code="FlexZBoost",
                    pit_out_rate=pit_out_rate, savefig=False)



.. image:: ../../../docs/rendered/evaluation_examples/00_Single_Evaluation_files/../../../docs/rendered/evaluation_examples/00_Single_Evaluation_32_0.png


The black horizontal line represents the ideal case where the PIT
histogram would behave as a uniform distribution U(0,1).

Summary statistics of CDF-based metrics
---------------------------------------

To evaluate globally the quality of PDFs estimates, ``rail.evaluation``
provides a set of metrics to compare the empirical distributions of PIT
values with the reference uniform distribution, U(0,1).

Kolmogorov-Smirnov
~~~~~~~~~~~~~~~~~~

Let’s start with the traditional Kolmogorov-Smirnov (KS) statistic test,
which is the maximum difference between the empirical and the expected
cumulative distributions of PIT values:

.. math::


   \mathrm{KS} \equiv \max_{PIT} \Big( \left| \ \mathrm{CDF} \small[ \hat{f}, z \small] - \mathrm{CDF} \small[ \tilde{f}, z \small] \  \right| \Big)

Where :math:`\hat{f}` is the PIT distribution and :math:`\tilde{f}` is
U(0,1). Therefore, the smaller value of KS the closer the PIT
distribution is to be uniform. The ``evaluate`` method of the PITKS
class returns a named tuple with the statistic and p-value.

.. code:: ipython3

    ks_stat_and_pval = metamets['ks']
    print(f"PIT KS stat and pval: {ks_stat_and_pval}") 
    ks_stat_and_pval = pitobj.evaluate_PIT_KS()
    print(f"PIT KS stat and pval: {ks_stat_and_pval}") 


.. parsed-literal::

    PIT KS stat and pval: KstestResult(statistic=np.float64(0.033590049370962216), pvalue=np.float64(1.7621068075751534e-20), statistic_location=np.float64(0.9921210288809627), statistic_sign=np.int8(-1))
    PIT KS stat and pval: KstestResult(statistic=np.float64(0.033590049370962216), pvalue=np.float64(1.7621068075751534e-20), statistic_location=np.float64(0.9921210288809627), statistic_sign=np.int8(-1))


Visual interpretation of the KS statistic:

.. code:: ipython3

    ks_plot(pitobj)



.. image:: ../../../docs/rendered/evaluation_examples/00_Single_Evaluation_files/../../../docs/rendered/evaluation_examples/00_Single_Evaluation_38_0.png


.. code:: ipython3

    print(f"KS metric of this sample: {ks_stat_and_pval.statistic:.4f}") 


.. parsed-literal::

    KS metric of this sample: 0.0336


Cramer-von Mises
~~~~~~~~~~~~~~~~

Similarly, let’s calculate the Cramer-von Mises (CvM) test, a variant of
the KS statistic defined as the mean-square difference between the CDFs
of an empirical PDF and the true PDFs:

.. math::  \mathrm{CvM}^2 \equiv \int_{-\infty}^{\infty} \Big( \mathrm{CDF} \small[ \hat{f}, z \small] \ - \ \mathrm{CDF} \small[ \tilde{f}, z \small] \Big)^{2} \mathrm{dCDF}(\tilde{f}, z) 

on the distribution of PIT values, which should be uniform if the PDFs
are perfect.

.. code:: ipython3

    cvm_stat_and_pval = metamets['cvm']
    print(f"PIT CvM stat and pval: {cvm_stat_and_pval}") 
    cvm_stat_and_pval = pitobj.evaluate_PIT_CvM()
    print(f"PIT CvM stat and pval: {cvm_stat_and_pval}")


.. parsed-literal::

    PIT CvM stat and pval: CramerVonMisesResult(statistic=9.62335199605935, pvalue=9.265039846440004e-10)
    PIT CvM stat and pval: CramerVonMisesResult(statistic=9.62335199605935, pvalue=9.265039846440004e-10)


.. code:: ipython3

    print(f"CvM metric of this sample: {cvm_stat_and_pval.statistic:.4f}") 


.. parsed-literal::

    CvM metric of this sample: 9.6234


Anderson-Darling
~~~~~~~~~~~~~~~~

Another variation of the KS statistic is the Anderson-Darling (AD) test,
a weighted mean-squared difference featuring enhanced sensitivity to
discrepancies in the tails of the distribution.

.. math::  \mathrm{AD}^2 \equiv N_{tot} \int_{-\infty}^{\infty} \frac{\big( \mathrm{CDF} \small[ \hat{f}, z \small] \ - \ \mathrm{CDF} \small[ \tilde{f}, z \small] \big)^{2}}{\mathrm{CDF} \small[ \tilde{f}, z \small] \big( 1 \ - \ \mathrm{CDF} \small[ \tilde{f}, z \small] \big)}\mathrm{dCDF}(\tilde{f}, z) 

.. code:: ipython3

    ad_stat_crit_sig = metamets['ad']
    print(f"PIT AD stat and pval: {ad_stat_crit_sig}") 
    ad_stat_crit_sig = pitobj.evaluate_PIT_anderson_ksamp()
    print(f"PIT AD stat and pval: {ad_stat_crit_sig}")


.. parsed-literal::

    PIT AD stat and pval: Anderson_ksampResult(statistic=np.float64(84.95623553609381), critical_values=array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546]), pvalue=np.float64(0.001))
    PIT AD stat and pval: Anderson_ksampResult(statistic=np.float64(84.95623553609381), critical_values=array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546]), pvalue=np.float64(0.001))


.. code:: ipython3

    print(f"AD metric of this sample: {ad_stat_crit_sig.statistic:.4f}") 


.. parsed-literal::

    AD metric of this sample: 84.9562


It is possible to remove catastrophic outliers before calculating the
integral for the sake of preserving numerical instability. For instance,
Schmidt et al. computed the Anderson-Darling statistic within the
interval (0.01, 0.99).

.. code:: ipython3

    ad_stat_crit_sig_cut = pitobj.evaluate_PIT_anderson_ksamp(pit_min=0.01, pit_max=0.99)
    print(f"AD metric of this sample: {ad_stat_crit_sig.statistic:.4f}") 
    print(f"AD metric for 0.01 < PIT < 0.99: {ad_stat_crit_sig_cut.statistic:.4f}") 


.. parsed-literal::

    WARNING:root:Removed 1760 PITs from the sample.


.. parsed-literal::

    AD metric of this sample: 84.9562
    AD metric for 0.01 < PIT < 0.99: 89.9826


CDE Loss
--------

In the absence of true photo-z posteriors, the metric used to evaluate
individual PDFs is the **Conditional Density Estimate (CDE) Loss**, a
metric analogue to the root-mean-squared-error:

.. math::  L(f, \hat{f}) \equiv  \int \int {\big(f(z | x) - \hat{f}(z | x) \big)}^{2} dzdP(x), 

where :math:`f(z | x)` is the true photo-z PDF and
:math:`\hat{f}(z | x)` is the estimated PDF in terms of the photometry
:math:`x`. Since :math:`f(z | x)` is unknown, we estimate the **CDE
Loss** as described in `Izbicki & Lee, 2017
(arXiv:1704.08095) <https://arxiv.org/abs/1704.08095>`__. :

.. math::  \mathrm{CDE} = \mathbb{E}\big(  \int{{\hat{f}(z | X)}^2 dz} \big) - 2{\mathbb{E}}_{X, Z}\big(\hat{f}(Z, X) \big) + K_{f},  

where the first term is the expectation value of photo-z posterior with
respect to the marginal distribution of the covariates X, and the second
term is the expectation value with respect to the joint distribution of
observables X and the space Z of all possible redshifts (in practice,
the centroids of the PDF bins), and the third term is a constant
depending on the true conditional densities :math:`f(z | x)`.

.. code:: ipython3

    cdelossobj = CDELoss(fzdata.data, zgrid, ztrue)

.. code:: ipython3

    cde_stat_and_pval = cdelossobj.evaluate()
    cde_stat_and_pval




.. parsed-literal::

    stat_and_pval(statistic=np.float64(-6.725602928688286), p_value=nan)



.. code:: ipython3

    print(f"CDE loss of this sample: {cde_stat_and_pval.statistic:.2f}") 


.. parsed-literal::

    CDE loss of this sample: -6.73


We note that all of the quantities as run individually are identical to
the quantities in our summary table - a nice check that things have run
properly.

