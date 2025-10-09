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

      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0

.. parsed-literal::

      0 47.1M    0  175k    0     0   124k      0  0:06:28  0:00:01  0:06:27  123k

.. parsed-literal::

      0 47.1M    0  263k    0     0   111k      0  0:07:11  0:00:02  0:07:09  111k

.. parsed-literal::

      1 47.1M    1  491k    0     0   141k      0  0:05:42  0:00:03  0:05:39  141k

.. parsed-literal::

      1 47.1M    1  603k    0     0   137k      0  0:05:50  0:00:04  0:05:46  137k

.. parsed-literal::

      1 47.1M    1  667k    0     0   122k      0  0:06:33  0:00:05  0:06:28  131k

.. parsed-literal::

      1 47.1M    1  755k    0     0   117k      0  0:06:49  0:00:06  0:06:43  116k

.. parsed-literal::

      2 47.1M    2 1079k    0     0   147k      0  0:05:27  0:00:07  0:05:20  163k

.. parsed-literal::

      2 47.1M    2 1283k    0     0   147k      0  0:05:27  0:00:08  0:05:19  151k

.. parsed-literal::

      2 47.1M    2 1379k    0     0   146k      0  0:05:28  0:00:09  0:05:19  155k

.. parsed-literal::

      3 47.1M    3 1715k    0     0   156k      0  0:05:07  0:00:10  0:04:57  190k

.. parsed-literal::

      3 47.1M    3 1847k    0     0   162k      0  0:04:57  0:00:11  0:04:46  219k

.. parsed-literal::

      4 47.1M    4 2039k    0     0   165k      0  0:04:51  0:00:12  0:04:39  191k

.. parsed-literal::

      4 47.1M    4 2263k    0     0   166k      0  0:04:49  0:00:13  0:04:36  201k

.. parsed-literal::

      5 47.1M    5 2439k    0     0   169k      0  0:04:45  0:00:14  0:04:31  211k

.. parsed-literal::

      5 47.1M    5 2843k    0     0   185k      0  0:04:20  0:00:15  0:04:05  256k

.. parsed-literal::

      6 47.1M    6 3171k    0     0   194k      0  0:04:08  0:00:16  0:03:52  266k

.. parsed-literal::

      7 47.1M    7 3443k    0     0   196k      0  0:04:05  0:00:17  0:03:48  272k

.. parsed-literal::

      7 47.1M    7 3743k    0     0   204k      0  0:03:56  0:00:18  0:03:38  309k

.. parsed-literal::

     10 47.1M   10 4887k    0     0   251k      0  0:03:11  0:00:19  0:02:52  487k

.. parsed-literal::

     11 47.1M   11 5371k    0     0   264k      0  0:03:02  0:00:20  0:02:42  505k

.. parsed-literal::

     12 47.1M   12 6103k    0     0   285k      0  0:02:48  0:00:21  0:02:27  584k

.. parsed-literal::

     13 47.1M   13 6619k    0     0   296k      0  0:02:42  0:00:22  0:02:20  652k

.. parsed-literal::

     15 47.1M   15 7291k    0     0   310k      0  0:02:35  0:00:23  0:02:12  693k

.. parsed-literal::

     16 47.1M   16 7795k    0     0   318k      0  0:02:31  0:00:24  0:02:07  578k

.. parsed-literal::

     18 47.1M   18 9063k    0     0   357k      0  0:02:14  0:00:25  0:01:49  737k

.. parsed-literal::

     20 47.1M   20 9719k    0     0   369k      0  0:02:10  0:00:26  0:01:44  725k

.. parsed-literal::

     21 47.1M   21 10.1M    0     0   379k      0  0:02:07  0:00:27  0:01:40  755k

.. parsed-literal::

     21 47.1M   21 10.3M    0     0   373k      0  0:02:09  0:00:28  0:01:41  673k

.. parsed-literal::

     21 47.1M   21 10.3M    0     0   358k      0  0:02:14  0:00:29  0:01:45  548k

.. parsed-literal::

     22 47.1M   22 10.3M    0     0   348k      0  0:02:18  0:00:30  0:01:48  301k

.. parsed-literal::

     22 47.1M   22 10.7M    0     0   350k      0  0:02:17  0:00:31  0:01:46  253k

.. parsed-literal::

     23 47.1M   23 11.2M    0     0   355k      0  0:02:15  0:00:32  0:01:43  223k

.. parsed-literal::

     24 47.1M   24 11.6M    0     0   356k      0  0:02:15  0:00:33  0:01:42  264k

.. parsed-literal::

     25 47.1M   25 12.2M    0     0   362k      0  0:02:13  0:00:34  0:01:39  383k

.. parsed-literal::

     26 47.1M   26 12.4M    0     0   360k      0  0:02:13  0:00:35  0:01:38  437k

.. parsed-literal::

     27 47.1M   27 13.0M    0     0   368k      0  0:02:10  0:00:36  0:01:34  483k

.. parsed-literal::

     28 47.1M   28 13.6M    0     0   372k      0  0:02:09  0:00:37  0:01:32  485k

.. parsed-literal::

     30 47.1M   30 14.1M    0     0   377k      0  0:02:07  0:00:38  0:01:29  519k

.. parsed-literal::

     34 47.1M   34 16.3M    0     0   425k      0  0:01:53  0:00:39  0:01:14  878k

.. parsed-literal::

     36 47.1M   36 17.0M    0     0   430k      0  0:01:52  0:00:40  0:01:12  928k

.. parsed-literal::

     36 47.1M   36 17.1M    0     0   425k      0  0:01:53  0:00:41  0:01:12  835k

.. parsed-literal::

     37 47.1M   37 17.6M    0     0   425k      0  0:01:53  0:00:42  0:01:11  804k

.. parsed-literal::

     37 47.1M   37 17.8M    0     0   421k      0  0:01:54  0:00:43  0:01:11  758k

.. parsed-literal::

     38 47.1M   38 18.1M    0     0   410k      0  0:01:57  0:00:45  0:01:12  305k

.. parsed-literal::

     38 47.1M   38 18.1M    0     0   401k      0  0:02:00  0:00:46  0:01:14  198k

.. parsed-literal::

     39 47.1M   39 18.5M    0     0   403k      0  0:01:59  0:00:47  0:01:12  249k

.. parsed-literal::

     39 47.1M   39 18.6M    0     0   398k      0  0:02:01  0:00:47  0:01:14  188k

.. parsed-literal::

     40 47.1M   40 19.1M    0     0   400k      0  0:02:00  0:00:48  0:01:12  234k

.. parsed-literal::

     40 47.1M   40 19.1M    0     0   394k      0  0:02:02  0:00:49  0:01:13  236k

.. parsed-literal::

     40 47.1M   40 19.1M    0     0   386k      0  0:02:04  0:00:50  0:01:14  236k

.. parsed-literal::

     40 47.1M   40 19.1M    0     0   379k      0  0:02:07  0:00:51  0:01:16  131k

.. parsed-literal::

     41 47.1M   41 19.7M    0     0   386k      0  0:02:04  0:00:52  0:01:12  259k

.. parsed-literal::

     42 47.1M   42 19.8M    0     0   379k      0  0:02:07  0:00:53  0:01:14  160k

.. parsed-literal::

     43 47.1M   43 20.4M    0     0   383k      0  0:02:05  0:00:54  0:01:11  274k

.. parsed-literal::

     43 47.1M   43 20.5M    0     0   379k      0  0:02:07  0:00:55  0:01:12  301k

.. parsed-literal::

     45 47.1M   45 21.5M    0     0   389k      0  0:02:03  0:00:56  0:01:07  501k

.. parsed-literal::

     46 47.1M   46 21.9M    0     0   389k      0  0:02:04  0:00:57  0:01:07  416k

.. parsed-literal::

     47 47.1M   47 22.5M    0     0   394k      0  0:02:02  0:00:58  0:01:04  566k

.. parsed-literal::

     47 47.1M   47 22.5M    0     0   387k      0  0:02:04  0:00:59  0:01:05  427k

.. parsed-literal::

     50 47.1M   50 23.9M    0     0   405k      0  0:01:58  0:01:00  0:00:58  697k

.. parsed-literal::

     51 47.1M   51 24.4M    0     0   405k      0  0:01:58  0:01:01  0:00:57  584k

.. parsed-literal::

     51 47.1M   51 24.4M    0     0   399k      0  0:02:00  0:01:02  0:00:58  525k

.. parsed-literal::

     52 47.1M   52 24.8M    0     0   401k      0  0:02:00  0:01:03  0:00:57  488k

.. parsed-literal::

     53 47.1M   53 25.0M    0     0   398k      0  0:02:01  0:01:04  0:00:57  536k

.. parsed-literal::

     54 47.1M   54 25.7M    0     0   401k      0  0:02:00  0:01:05  0:00:55  353k

.. parsed-literal::

     55 47.1M   55 26.2M    0     0   405k      0  0:01:59  0:01:06  0:00:53  392k

.. parsed-literal::

     56 47.1M   56 26.4M    0     0   402k      0  0:01:59  0:01:07  0:00:52  439k

.. parsed-literal::

     56 47.1M   56 26.6M    0     0   398k      0  0:02:00  0:01:08  0:00:52  362k

.. parsed-literal::

     56 47.1M   56 26.7M    0     0   394k      0  0:02:02  0:01:09  0:00:53  343k

.. parsed-literal::

     57 47.1M   57 27.1M    0     0   395k      0  0:02:02  0:01:10  0:00:52  312k

.. parsed-literal::

     58 47.1M   58 27.4M    0     0   393k      0  0:02:02  0:01:11  0:00:51  242k

.. parsed-literal::

     59 47.1M   59 27.9M    0     0   395k      0  0:02:01  0:01:12  0:00:49  304k

.. parsed-literal::

     60 47.1M   60 28.3M    0     0   395k      0  0:02:02  0:01:13  0:00:49  343k

.. parsed-literal::

     60 47.1M   60 28.3M    0     0   388k      0  0:02:04  0:01:14  0:00:50  311k

.. parsed-literal::

     60 47.1M   60 28.4M    0     0   385k      0  0:02:05  0:01:15  0:00:50  251k

.. parsed-literal::

     61 47.1M   61 28.7M    0     0   386k      0  0:02:04  0:01:16  0:00:48  275k

.. parsed-literal::

     61 47.1M   61 29.1M    0     0   380k      0  0:02:06  0:01:18  0:00:48  197k

.. parsed-literal::

     61 47.1M   61 29.2M    0     0   380k      0  0:02:06  0:01:18  0:00:48  174k

.. parsed-literal::

     62 47.1M   62 29.3M    0     0   377k      0  0:02:07  0:01:19  0:00:48  209k

.. parsed-literal::

     62 47.1M   62 29.4M    0     0   371k      0  0:02:10  0:01:21  0:00:49  180k

.. parsed-literal::

     62 47.1M   62 29.6M    0     0   370k      0  0:02:10  0:01:21  0:00:49  157k

.. parsed-literal::

     62 47.1M   62 29.6M    0     0   366k      0  0:02:11  0:01:22  0:00:49  117k

.. parsed-literal::

     62 47.1M   62 29.6M    0     0   364k      0  0:02:12  0:01:23  0:00:49   98k

.. parsed-literal::

     63 47.1M   63 29.8M    0     0   362k      0  0:02:13  0:01:24  0:00:49  114k

.. parsed-literal::

     63 47.1M   63 29.8M    0     0   358k      0  0:02:14  0:01:25  0:00:49  103k

.. parsed-literal::

     64 47.1M   64 30.2M    0     0   359k      0  0:02:14  0:01:26  0:00:48  149k

.. parsed-literal::

     64 47.1M   64 30.5M    0     0   358k      0  0:02:14  0:01:27  0:00:47  213k

.. parsed-literal::

     64 47.1M   64 30.5M    0     0   354k      0  0:02:16  0:01:28  0:00:48  186k

.. parsed-literal::

     64 47.1M   64 30.5M    0     0   348k      0  0:02:18  0:01:29  0:00:49  128k

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   349k      0  0:02:18  0:01:30  0:00:48  204k

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   345k      0  0:02:19  0:01:31  0:00:48  124k

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   342k      0  0:02:21  0:01:32  0:00:49 71146

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   338k      0  0:02:22  0:01:33  0:00:49 71146

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   334k      0  0:02:24  0:01:34  0:00:50 79746

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   331k      0  0:02:25  0:01:35  0:00:50  1635

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   327k      0  0:02:27  0:01:36  0:00:51     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   324k      0  0:02:28  0:01:37  0:00:51     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   321k      0  0:02:30  0:01:38  0:00:52     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   318k      0  0:02:31  0:01:39  0:00:52     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   314k      0  0:02:33  0:01:40  0:00:53     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   311k      0  0:02:34  0:01:41  0:00:53     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   308k      0  0:02:36  0:01:42  0:00:54     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   305k      0  0:02:37  0:01:43  0:00:54     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   302k      0  0:02:39  0:01:44  0:00:55     0

.. parsed-literal::

     65 47.1M   65 30.9M    0     0   299k      0  0:02:40  0:01:45  0:00:55     0

.. parsed-literal::

     74 47.1M   74 35.3M    0     0   340k      0  0:02:21  0:01:46  0:00:35  952k

.. parsed-literal::

     95 47.1M   95 44.8M    0     0   428k      0  0:01:52  0:01:47  0:00:05 3028k100 47.1M  100 47.1M    0     0   448k      0  0:01:47  0:01:47 --:--:-- 4266k


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

