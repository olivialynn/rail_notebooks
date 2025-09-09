GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.5167667e-01	 3.2299270e-01	-3.4196022e-01	 3.1062745e-01	[-3.1940586e-01]	 4.6523333e-01


.. parsed-literal::

       2	-2.7994127e-01	 3.1216478e-01	-2.5590808e-01	 2.9999144e-01	[-2.2056555e-01]	 2.3390031e-01


.. parsed-literal::

       3	-2.3646480e-01	 2.9200208e-01	-1.9535672e-01	 2.8033145e-01	[-1.4683331e-01]	 2.8080392e-01


.. parsed-literal::

       4	-2.0053138e-01	 2.6856387e-01	-1.6014081e-01	 2.5508482e-01	[-8.6410571e-02]	 2.1552348e-01


.. parsed-literal::

       5	-1.1457080e-01	 2.5991413e-01	-8.0866115e-02	 2.4748615e-01	[-2.4110487e-02]	 2.0642734e-01


.. parsed-literal::

       6	-8.0943104e-02	 2.5439516e-01	-5.1016093e-02	 2.3975733e-01	[ 3.1376950e-05]	 2.1007705e-01
       7	-6.4008251e-02	 2.5181408e-01	-4.0175956e-02	 2.3709786e-01	[ 1.4726824e-02]	 1.9448996e-01


.. parsed-literal::

       8	-5.1523639e-02	 2.4972076e-01	-3.1510248e-02	 2.3475303e-01	[ 2.5756060e-02]	 2.1241045e-01


.. parsed-literal::

       9	-3.8413967e-02	 2.4727791e-01	-2.1175013e-02	 2.3208467e-01	[ 3.7613083e-02]	 2.0485425e-01


.. parsed-literal::

      10	-2.7790401e-02	 2.4534997e-01	-1.2657692e-02	 2.3088081e-01	[ 4.5252885e-02]	 2.0383692e-01


.. parsed-literal::

      11	-2.2285876e-02	 2.4435406e-01	-8.1236172e-03	 2.3058508e-01	  4.4554969e-02 	 2.0337272e-01
      12	-1.9892777e-02	 2.4386230e-01	-5.8859435e-03	 2.3019813e-01	[ 4.8033141e-02]	 1.9905329e-01


.. parsed-literal::

      13	-1.5397894e-02	 2.4301734e-01	-1.4379168e-03	 2.2951882e-01	[ 5.0787973e-02]	 2.1804976e-01


.. parsed-literal::

      14	 1.3093325e-01	 2.2621372e-01	 1.5789738e-01	 2.1151255e-01	[ 2.1881103e-01]	 4.4148469e-01


.. parsed-literal::

      15	 1.6428603e-01	 2.2502571e-01	 1.9289743e-01	 2.1245724e-01	  2.1090755e-01 	 2.1200538e-01


.. parsed-literal::

      16	 2.6904927e-01	 2.1717956e-01	 2.9969043e-01	 2.0526095e-01	[ 3.2552919e-01]	 2.2137547e-01


.. parsed-literal::

      17	 3.2472039e-01	 2.1302592e-01	 3.5769347e-01	 2.0164843e-01	[ 3.8962133e-01]	 3.2778072e-01


.. parsed-literal::

      18	 3.6131943e-01	 2.0777941e-01	 3.9599339e-01	 1.9651023e-01	[ 4.2898517e-01]	 2.1519566e-01
      19	 4.1850078e-01	 2.0351193e-01	 4.5283280e-01	 1.9257121e-01	[ 4.8288612e-01]	 1.7869473e-01


.. parsed-literal::

      20	 5.2487303e-01	 1.9491981e-01	 5.6131615e-01	 1.8421277e-01	[ 5.9638341e-01]	 2.1473646e-01


.. parsed-literal::

      21	 5.9590085e-01	 1.9248535e-01	 6.3456955e-01	 1.7735222e-01	[ 6.6679966e-01]	 2.0512342e-01


.. parsed-literal::

      22	 6.3625487e-01	 1.8714295e-01	 6.7477116e-01	 1.7425967e-01	[ 6.9810199e-01]	 2.0222735e-01


.. parsed-literal::

      23	 6.7047764e-01	 1.8324981e-01	 7.1006677e-01	 1.7188066e-01	[ 7.3274877e-01]	 2.0693111e-01


.. parsed-literal::

      24	 7.2038025e-01	 1.7872842e-01	 7.5882227e-01	 1.6926648e-01	[ 7.8825755e-01]	 2.0316577e-01
      25	 7.5641634e-01	 1.8153315e-01	 7.9551605e-01	 1.7082328e-01	[ 8.1652727e-01]	 1.8125510e-01


.. parsed-literal::

      26	 7.9018249e-01	 1.8056820e-01	 8.3021935e-01	 1.6961830e-01	[ 8.4614364e-01]	 2.0191956e-01


.. parsed-literal::

      27	 8.1277725e-01	 1.8687220e-01	 8.5398354e-01	 1.7379132e-01	[ 8.5620898e-01]	 2.0267391e-01


.. parsed-literal::

      28	 8.4812621e-01	 1.8217113e-01	 8.8924386e-01	 1.6892867e-01	[ 8.9746601e-01]	 2.2013736e-01


.. parsed-literal::

      29	 8.6984361e-01	 1.8187257e-01	 9.1065300e-01	 1.6872522e-01	[ 9.2059739e-01]	 2.1614909e-01
      30	 8.9171101e-01	 1.8029536e-01	 9.3372726e-01	 1.6679130e-01	[ 9.4018410e-01]	 1.9955039e-01


.. parsed-literal::

      31	 9.1224668e-01	 1.7696809e-01	 9.5453263e-01	 1.6449517e-01	[ 9.6646929e-01]	 2.0115757e-01


.. parsed-literal::

      32	 9.3492723e-01	 1.6820263e-01	 9.7759871e-01	 1.5922400e-01	[ 9.8285927e-01]	 2.0143867e-01


.. parsed-literal::

      33	 9.5275129e-01	 1.6505544e-01	 9.9563652e-01	 1.5790179e-01	[ 1.0007375e+00]	 2.1113372e-01


.. parsed-literal::

      34	 9.7128829e-01	 1.6231181e-01	 1.0148434e+00	 1.5739922e-01	[ 1.0113481e+00]	 2.1948314e-01


.. parsed-literal::

      35	 9.8939126e-01	 1.5862797e-01	 1.0341590e+00	 1.5402367e-01	[ 1.0228325e+00]	 2.1796250e-01


.. parsed-literal::

      36	 1.0110155e+00	 1.5730391e-01	 1.0563266e+00	 1.5182559e-01	[ 1.0421789e+00]	 2.1761131e-01
      37	 1.0253834e+00	 1.5648338e-01	 1.0710672e+00	 1.5032443e-01	[ 1.0610020e+00]	 1.9587183e-01


.. parsed-literal::

      38	 1.0406837e+00	 1.5521059e-01	 1.0866100e+00	 1.4821371e-01	[ 1.0815135e+00]	 1.8821836e-01


.. parsed-literal::

      39	 1.0624672e+00	 1.5287440e-01	 1.1087081e+00	 1.4533212e-01	[ 1.1154940e+00]	 2.1761179e-01


.. parsed-literal::

      40	 1.0836190e+00	 1.4963399e-01	 1.1301102e+00	 1.4113693e-01	[ 1.1371837e+00]	 2.1297288e-01


.. parsed-literal::

      41	 1.0979326e+00	 1.4826334e-01	 1.1442626e+00	 1.3947043e-01	[ 1.1557612e+00]	 2.0404005e-01


.. parsed-literal::

      42	 1.1048999e+00	 1.4758708e-01	 1.1509867e+00	 1.3929597e-01	[ 1.1593844e+00]	 2.1260691e-01
      43	 1.1156524e+00	 1.4611558e-01	 1.1619924e+00	 1.3843647e-01	[ 1.1658460e+00]	 1.7070436e-01


.. parsed-literal::

      44	 1.1265895e+00	 1.4436876e-01	 1.1733504e+00	 1.3694527e-01	[ 1.1738033e+00]	 2.0903516e-01


.. parsed-literal::

      45	 1.1386424e+00	 1.4261875e-01	 1.1855953e+00	 1.3528201e-01	[ 1.1813231e+00]	 2.0269799e-01


.. parsed-literal::

      46	 1.1496387e+00	 1.4115198e-01	 1.1964411e+00	 1.3389902e-01	[ 1.1919844e+00]	 2.1691895e-01


.. parsed-literal::

      47	 1.1675506e+00	 1.3951482e-01	 1.2141308e+00	 1.3253368e-01	[ 1.1980504e+00]	 2.1027875e-01


.. parsed-literal::

      48	 1.1745670e+00	 1.3794149e-01	 1.2212867e+00	 1.3206928e-01	  1.1893231e+00 	 2.0866346e-01


.. parsed-literal::

      49	 1.1858131e+00	 1.3763103e-01	 1.2323156e+00	 1.3166159e-01	[ 1.2072840e+00]	 2.1728158e-01


.. parsed-literal::

      50	 1.1924961e+00	 1.3723549e-01	 1.2391793e+00	 1.3129141e-01	[ 1.2107538e+00]	 2.1300602e-01


.. parsed-literal::

      51	 1.2019562e+00	 1.3662843e-01	 1.2490403e+00	 1.3089232e-01	[ 1.2123190e+00]	 2.1916080e-01
      52	 1.2150475e+00	 1.3530759e-01	 1.2625928e+00	 1.2972705e-01	  1.2107916e+00 	 2.0682168e-01


.. parsed-literal::

      53	 1.2235088e+00	 1.3478184e-01	 1.2718879e+00	 1.2895416e-01	  1.1997073e+00 	 2.1366143e-01


.. parsed-literal::

      54	 1.2342913e+00	 1.3443740e-01	 1.2823975e+00	 1.2854200e-01	  1.2095762e+00 	 2.0509171e-01


.. parsed-literal::

      55	 1.2390967e+00	 1.3417661e-01	 1.2869573e+00	 1.2832308e-01	[ 1.2192347e+00]	 2.1339846e-01


.. parsed-literal::

      56	 1.2483133e+00	 1.3353053e-01	 1.2962430e+00	 1.2767675e-01	[ 1.2317193e+00]	 2.1349859e-01


.. parsed-literal::

      57	 1.2564165e+00	 1.3313001e-01	 1.3046459e+00	 1.2704011e-01	[ 1.2346446e+00]	 2.1125913e-01


.. parsed-literal::

      58	 1.2635041e+00	 1.3295130e-01	 1.3118309e+00	 1.2715798e-01	[ 1.2452575e+00]	 2.0224905e-01


.. parsed-literal::

      59	 1.2702363e+00	 1.3277547e-01	 1.3186718e+00	 1.2706039e-01	[ 1.2540032e+00]	 2.1231031e-01
      60	 1.2801833e+00	 1.3264197e-01	 1.3287948e+00	 1.2689302e-01	[ 1.2610656e+00]	 1.8465042e-01


.. parsed-literal::

      61	 1.2880647e+00	 1.3228969e-01	 1.3371474e+00	 1.2616818e-01	[ 1.2727090e+00]	 2.0570016e-01


.. parsed-literal::

      62	 1.2972605e+00	 1.3202421e-01	 1.3460861e+00	 1.2591915e-01	[ 1.2782171e+00]	 2.1024156e-01
      63	 1.3016122e+00	 1.3182922e-01	 1.3504280e+00	 1.2567552e-01	[ 1.2793646e+00]	 1.7220974e-01


.. parsed-literal::

      64	 1.3081287e+00	 1.3140692e-01	 1.3571715e+00	 1.2531827e-01	[ 1.2793961e+00]	 1.9206309e-01


.. parsed-literal::

      65	 1.3138726e+00	 1.3123920e-01	 1.3631305e+00	 1.2465566e-01	[ 1.2838963e+00]	 2.2653079e-01


.. parsed-literal::

      66	 1.3219631e+00	 1.3105171e-01	 1.3711829e+00	 1.2441271e-01	[ 1.2927072e+00]	 2.0816708e-01


.. parsed-literal::

      67	 1.3273356e+00	 1.3082212e-01	 1.3766130e+00	 1.2431441e-01	[ 1.2990588e+00]	 2.0228934e-01


.. parsed-literal::

      68	 1.3316707e+00	 1.3050642e-01	 1.3809939e+00	 1.2431047e-01	[ 1.3023539e+00]	 2.1339202e-01


.. parsed-literal::

      69	 1.3359069e+00	 1.2966319e-01	 1.3855427e+00	 1.2450556e-01	[ 1.3060068e+00]	 2.1589422e-01


.. parsed-literal::

      70	 1.3422642e+00	 1.2952994e-01	 1.3917862e+00	 1.2418419e-01	[ 1.3090427e+00]	 2.1110916e-01


.. parsed-literal::

      71	 1.3459141e+00	 1.2934926e-01	 1.3955163e+00	 1.2400628e-01	  1.3086062e+00 	 2.1188188e-01


.. parsed-literal::

      72	 1.3518415e+00	 1.2902482e-01	 1.4016167e+00	 1.2384630e-01	  1.3072724e+00 	 2.1200252e-01


.. parsed-literal::

      73	 1.3616257e+00	 1.2842932e-01	 1.4117305e+00	 1.2407750e-01	  1.3025513e+00 	 2.0654893e-01


.. parsed-literal::

      74	 1.3658290e+00	 1.2812819e-01	 1.4161003e+00	 1.2422499e-01	  1.3012797e+00 	 3.3092928e-01


.. parsed-literal::

      75	 1.3705892e+00	 1.2774688e-01	 1.4208888e+00	 1.2433627e-01	  1.3020072e+00 	 2.2107530e-01


.. parsed-literal::

      76	 1.3748122e+00	 1.2732540e-01	 1.4251212e+00	 1.2451516e-01	  1.3049349e+00 	 2.0424509e-01
      77	 1.3784674e+00	 1.2679785e-01	 1.4288742e+00	 1.2448821e-01	  1.3031422e+00 	 1.9923830e-01


.. parsed-literal::

      78	 1.3822938e+00	 1.2647187e-01	 1.4327201e+00	 1.2440904e-01	  1.3042438e+00 	 2.1483922e-01


.. parsed-literal::

      79	 1.3896358e+00	 1.2569442e-01	 1.4402600e+00	 1.2444960e-01	  1.3009587e+00 	 2.2182035e-01


.. parsed-literal::

      80	 1.3937790e+00	 1.2549995e-01	 1.4443875e+00	 1.2444736e-01	  1.3041700e+00 	 2.0405769e-01
      81	 1.3992804e+00	 1.2505897e-01	 1.4499322e+00	 1.2512223e-01	  1.3042920e+00 	 1.7788768e-01


.. parsed-literal::

      82	 1.4032582e+00	 1.2497888e-01	 1.4538328e+00	 1.2520552e-01	  1.3087437e+00 	 2.1789479e-01
      83	 1.4072980e+00	 1.2483252e-01	 1.4579033e+00	 1.2553339e-01	[ 1.3127844e+00]	 1.7811751e-01


.. parsed-literal::

      84	 1.4114677e+00	 1.2477634e-01	 1.4621592e+00	 1.2571660e-01	  1.3126593e+00 	 2.1557403e-01
      85	 1.4153166e+00	 1.2455241e-01	 1.4660942e+00	 1.2566990e-01	[ 1.3138297e+00]	 2.0310664e-01


.. parsed-literal::

      86	 1.4189319e+00	 1.2435093e-01	 1.4697739e+00	 1.2543077e-01	  1.3136272e+00 	 2.0790625e-01


.. parsed-literal::

      87	 1.4236223e+00	 1.2411148e-01	 1.4745809e+00	 1.2501897e-01	  1.3123086e+00 	 2.0765567e-01
      88	 1.4265492e+00	 1.2383619e-01	 1.4776129e+00	 1.2459226e-01	  1.3080426e+00 	 1.9834137e-01


.. parsed-literal::

      89	 1.4292496e+00	 1.2381924e-01	 1.4801860e+00	 1.2466001e-01	[ 1.3146643e+00]	 2.0538855e-01


.. parsed-literal::

      90	 1.4312102e+00	 1.2374973e-01	 1.4821375e+00	 1.2479757e-01	[ 1.3164996e+00]	 2.0737839e-01


.. parsed-literal::

      91	 1.4340107e+00	 1.2359945e-01	 1.4850189e+00	 1.2489093e-01	  1.3152068e+00 	 2.0973110e-01
      92	 1.4366886e+00	 1.2327972e-01	 1.4878935e+00	 1.2486524e-01	  1.3075787e+00 	 1.9948864e-01


.. parsed-literal::

      93	 1.4400585e+00	 1.2311825e-01	 1.4912902e+00	 1.2487540e-01	  1.3064149e+00 	 2.0975304e-01
      94	 1.4421105e+00	 1.2297058e-01	 1.4933694e+00	 1.2476712e-01	  1.3047886e+00 	 1.9182944e-01


.. parsed-literal::

      95	 1.4446351e+00	 1.2278435e-01	 1.4959536e+00	 1.2472755e-01	  1.3024511e+00 	 2.0187092e-01
      96	 1.4477156e+00	 1.2236560e-01	 1.4991501e+00	 1.2491204e-01	  1.2979176e+00 	 1.8089557e-01


.. parsed-literal::

      97	 1.4508106e+00	 1.2220256e-01	 1.5022845e+00	 1.2513034e-01	  1.2985000e+00 	 2.1108341e-01
      98	 1.4526056e+00	 1.2211155e-01	 1.5040182e+00	 1.2536017e-01	  1.3017491e+00 	 1.7927146e-01


.. parsed-literal::

      99	 1.4549329e+00	 1.2197926e-01	 1.5063309e+00	 1.2569598e-01	  1.3041947e+00 	 1.7666817e-01


.. parsed-literal::

     100	 1.4570391e+00	 1.2178668e-01	 1.5084760e+00	 1.2613943e-01	  1.3016355e+00 	 2.1699452e-01


.. parsed-literal::

     101	 1.4597158e+00	 1.2162856e-01	 1.5111856e+00	 1.2625500e-01	  1.2990704e+00 	 2.0902443e-01


.. parsed-literal::

     102	 1.4627001e+00	 1.2116537e-01	 1.5142326e+00	 1.2584362e-01	  1.2853410e+00 	 2.0810485e-01


.. parsed-literal::

     103	 1.4653496e+00	 1.2098441e-01	 1.5169100e+00	 1.2548709e-01	  1.2791609e+00 	 2.1931362e-01


.. parsed-literal::

     104	 1.4668878e+00	 1.2085290e-01	 1.5184046e+00	 1.2515479e-01	  1.2751658e+00 	 2.1114135e-01


.. parsed-literal::

     105	 1.4691960e+00	 1.2058597e-01	 1.5207124e+00	 1.2470676e-01	  1.2674472e+00 	 2.0628905e-01


.. parsed-literal::

     106	 1.4705558e+00	 1.2019040e-01	 1.5222280e+00	 1.2424024e-01	  1.2464541e+00 	 2.2553992e-01


.. parsed-literal::

     107	 1.4727266e+00	 1.2019764e-01	 1.5243150e+00	 1.2424492e-01	  1.2551116e+00 	 2.1146059e-01


.. parsed-literal::

     108	 1.4740230e+00	 1.2015669e-01	 1.5256226e+00	 1.2420694e-01	  1.2578007e+00 	 2.1832705e-01
     109	 1.4752537e+00	 1.2010053e-01	 1.5268752e+00	 1.2409744e-01	  1.2588040e+00 	 1.8206120e-01


.. parsed-literal::

     110	 1.4779309e+00	 1.1995644e-01	 1.5295604e+00	 1.2378654e-01	  1.2587133e+00 	 2.1289110e-01
     111	 1.4799052e+00	 1.1962562e-01	 1.5316568e+00	 1.2322919e-01	  1.2562182e+00 	 1.9580317e-01


.. parsed-literal::

     112	 1.4831553e+00	 1.1965084e-01	 1.5348054e+00	 1.2306207e-01	  1.2513409e+00 	 2.1220589e-01
     113	 1.4845638e+00	 1.1965226e-01	 1.5361616e+00	 1.2303921e-01	  1.2491194e+00 	 2.0444059e-01


.. parsed-literal::

     114	 1.4859817e+00	 1.1959488e-01	 1.5375960e+00	 1.2291463e-01	  1.2386768e+00 	 2.0104098e-01


.. parsed-literal::

     115	 1.4874679e+00	 1.1959115e-01	 1.5391250e+00	 1.2275900e-01	  1.2324858e+00 	 2.1624970e-01


.. parsed-literal::

     116	 1.4892419e+00	 1.1955954e-01	 1.5409684e+00	 1.2256095e-01	  1.2235214e+00 	 2.1974277e-01


.. parsed-literal::

     117	 1.4903686e+00	 1.1964526e-01	 1.5422489e+00	 1.2226966e-01	  1.2036657e+00 	 2.0236945e-01


.. parsed-literal::

     118	 1.4921696e+00	 1.1958148e-01	 1.5440344e+00	 1.2224881e-01	  1.2034866e+00 	 2.1453547e-01
     119	 1.4929125e+00	 1.1955241e-01	 1.5447543e+00	 1.2230502e-01	  1.2043215e+00 	 1.8727946e-01


.. parsed-literal::

     120	 1.4948436e+00	 1.1955811e-01	 1.5467004e+00	 1.2246287e-01	  1.1974008e+00 	 2.1890807e-01


.. parsed-literal::

     121	 1.4968359e+00	 1.1959698e-01	 1.5487618e+00	 1.2256194e-01	  1.1832302e+00 	 2.1763563e-01


.. parsed-literal::

     122	 1.4979829e+00	 1.1947401e-01	 1.5501593e+00	 1.2248576e-01	  1.1577822e+00 	 2.1748328e-01


.. parsed-literal::

     123	 1.5005898e+00	 1.1952671e-01	 1.5526972e+00	 1.2229205e-01	  1.1553206e+00 	 2.1699929e-01


.. parsed-literal::

     124	 1.5013124e+00	 1.1941431e-01	 1.5533931e+00	 1.2213242e-01	  1.1580362e+00 	 2.1414232e-01


.. parsed-literal::

     125	 1.5024641e+00	 1.1918701e-01	 1.5546023e+00	 1.2187384e-01	  1.1602670e+00 	 2.1200538e-01


.. parsed-literal::

     126	 1.5037513e+00	 1.1903075e-01	 1.5559763e+00	 1.2174948e-01	  1.1539697e+00 	 2.0427871e-01


.. parsed-literal::

     127	 1.5047396e+00	 1.1895686e-01	 1.5569639e+00	 1.2180997e-01	  1.1578639e+00 	 2.1850276e-01


.. parsed-literal::

     128	 1.5066923e+00	 1.1879031e-01	 1.5589758e+00	 1.2188548e-01	  1.1616571e+00 	 2.1347547e-01


.. parsed-literal::

     129	 1.5070660e+00	 1.1872430e-01	 1.5594217e+00	 1.2191312e-01	  1.1706702e+00 	 2.2095656e-01


.. parsed-literal::

     130	 1.5083315e+00	 1.1866665e-01	 1.5606109e+00	 1.2183887e-01	  1.1668930e+00 	 2.2466898e-01


.. parsed-literal::

     131	 1.5092119e+00	 1.1857899e-01	 1.5614745e+00	 1.2172434e-01	  1.1629484e+00 	 2.1265650e-01
     132	 1.5104356e+00	 1.1843540e-01	 1.5626817e+00	 1.2154744e-01	  1.1599414e+00 	 2.0420861e-01


.. parsed-literal::

     133	 1.5120063e+00	 1.1827735e-01	 1.5642390e+00	 1.2120456e-01	  1.1591858e+00 	 2.1231961e-01


.. parsed-literal::

     134	 1.5132899e+00	 1.1811941e-01	 1.5655483e+00	 1.2090195e-01	  1.1642083e+00 	 2.1131659e-01


.. parsed-literal::

     135	 1.5143091e+00	 1.1809072e-01	 1.5665676e+00	 1.2080231e-01	  1.1661410e+00 	 2.0838141e-01


.. parsed-literal::

     136	 1.5150459e+00	 1.1807489e-01	 1.5673369e+00	 1.2070791e-01	  1.1678412e+00 	 2.1232748e-01


.. parsed-literal::

     137	 1.5158883e+00	 1.1800511e-01	 1.5682233e+00	 1.2061521e-01	  1.1681409e+00 	 2.0688057e-01


.. parsed-literal::

     138	 1.5167296e+00	 1.1796257e-01	 1.5691282e+00	 1.2042010e-01	  1.1684631e+00 	 3.1442809e-01


.. parsed-literal::

     139	 1.5178961e+00	 1.1784532e-01	 1.5703218e+00	 1.2033354e-01	  1.1658170e+00 	 2.1146226e-01


.. parsed-literal::

     140	 1.5189161e+00	 1.1774619e-01	 1.5713396e+00	 1.2032556e-01	  1.1629808e+00 	 2.1742940e-01


.. parsed-literal::

     141	 1.5200174e+00	 1.1769892e-01	 1.5724283e+00	 1.2032806e-01	  1.1582510e+00 	 2.1745491e-01
     142	 1.5204355e+00	 1.1757429e-01	 1.5728658e+00	 1.2043017e-01	  1.1523859e+00 	 1.9112492e-01


.. parsed-literal::

     143	 1.5215797e+00	 1.1762945e-01	 1.5739510e+00	 1.2039117e-01	  1.1539856e+00 	 2.1445060e-01


.. parsed-literal::

     144	 1.5221153e+00	 1.1765430e-01	 1.5744663e+00	 1.2039011e-01	  1.1545057e+00 	 2.1941900e-01


.. parsed-literal::

     145	 1.5228265e+00	 1.1766156e-01	 1.5751638e+00	 1.2041865e-01	  1.1541711e+00 	 2.0774674e-01


.. parsed-literal::

     146	 1.5238451e+00	 1.1763720e-01	 1.5761736e+00	 1.2048413e-01	  1.1535425e+00 	 2.1448636e-01


.. parsed-literal::

     147	 1.5245013e+00	 1.1756062e-01	 1.5768603e+00	 1.2038714e-01	  1.1499385e+00 	 3.3729935e-01


.. parsed-literal::

     148	 1.5256688e+00	 1.1749210e-01	 1.5780341e+00	 1.2042205e-01	  1.1483941e+00 	 2.0532966e-01
     149	 1.5267356e+00	 1.1738802e-01	 1.5791357e+00	 1.2036066e-01	  1.1482423e+00 	 1.9347787e-01


.. parsed-literal::

     150	 1.5280930e+00	 1.1724139e-01	 1.5805667e+00	 1.2014844e-01	  1.1455584e+00 	 2.1304226e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.27 s, total: 2min 9s
    Wall time: 32.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3d08e4d840>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.1 s, sys: 33 ms, total: 2.13 s
    Wall time: 636 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

