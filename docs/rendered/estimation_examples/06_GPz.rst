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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.5634448e-01	 3.2443849e-01	-3.4658408e-01	 3.0490949e-01	[-3.1019734e-01]	 4.6419573e-01


.. parsed-literal::

       2	-2.8324550e-01	 3.1288818e-01	-2.5841642e-01	 2.9543314e-01	[-2.0461359e-01]	 2.2929645e-01


.. parsed-literal::

       3	-2.4027619e-01	 2.9298697e-01	-1.9861118e-01	 2.7837655e-01	[-1.3489181e-01]	 2.7620602e-01
       4	-2.0374356e-01	 2.6891458e-01	-1.6277133e-01	 2.5466774e-01	[-8.0466717e-02]	 1.8547177e-01


.. parsed-literal::

       5	-1.1543024e-01	 2.5949856e-01	-8.0406051e-02	 2.5019314e-01	[-2.4617101e-02]	 2.1315241e-01


.. parsed-literal::

       6	-8.2289632e-02	 2.5470857e-01	-5.1544361e-02	 2.4029402e-01	[-4.6291776e-04]	 2.0588374e-01
       7	-6.3991455e-02	 2.5165943e-01	-3.9646498e-02	 2.3833263e-01	[ 1.2631220e-02]	 1.9871330e-01


.. parsed-literal::

       8	-5.2630966e-02	 2.4980815e-01	-3.2078896e-02	 2.3661381e-01	[ 2.1807245e-02]	 2.0526552e-01
       9	-4.0442327e-02	 2.4761119e-01	-2.2792668e-02	 2.3399798e-01	[ 3.3480998e-02]	 1.8008852e-01


.. parsed-literal::

      10	-2.7269903e-02	 2.4496519e-01	-1.1576497e-02	 2.3137516e-01	[ 4.4960148e-02]	 2.1635175e-01


.. parsed-literal::

      11	-2.5343947e-02	 2.4488915e-01	-1.1180610e-02	 2.3123375e-01	  3.9520351e-02 	 2.1251869e-01
      12	-1.9626265e-02	 2.4382598e-01	-5.8818114e-03	 2.3084999e-01	[ 4.9102326e-02]	 1.9711471e-01


.. parsed-literal::

      13	-1.6717449e-02	 2.4324024e-01	-3.0909709e-03	 2.3078473e-01	  4.8474189e-02 	 2.1283889e-01


.. parsed-literal::

      14	-1.3893268e-02	 2.4262022e-01	-5.5557181e-05	 2.3025109e-01	[ 5.1609180e-02]	 2.1841598e-01


.. parsed-literal::

      15	 8.2300272e-02	 2.2736142e-01	 1.0267837e-01	 2.2163289e-01	[ 1.3948281e-01]	 3.1485057e-01


.. parsed-literal::

      16	 1.0795146e-01	 2.2595030e-01	 1.2954178e-01	 2.2401661e-01	[ 1.4837570e-01]	 3.1001115e-01


.. parsed-literal::

      17	 1.7415774e-01	 2.1990917e-01	 1.9633628e-01	 2.1518257e-01	[ 2.1427662e-01]	 2.0367670e-01
      18	 3.1152969e-01	 2.1586715e-01	 3.4205739e-01	 2.0739621e-01	[ 3.7212787e-01]	 2.0335054e-01


.. parsed-literal::

      19	 3.5305495e-01	 2.1197023e-01	 3.8517223e-01	 2.0730039e-01	[ 4.1071748e-01]	 2.2102618e-01


.. parsed-literal::

      20	 4.0614345e-01	 2.0876196e-01	 4.3788113e-01	 1.9994862e-01	[ 4.6403639e-01]	 2.1936274e-01
      21	 4.5099858e-01	 2.0552910e-01	 4.8369943e-01	 1.9951763e-01	[ 5.0624048e-01]	 1.9030333e-01


.. parsed-literal::

      22	 5.2610588e-01	 2.0246054e-01	 5.6057609e-01	 1.9648458e-01	[ 5.7778186e-01]	 2.1453214e-01


.. parsed-literal::

      23	 5.6483868e-01	 2.0803600e-01	 6.0424308e-01	 1.9858255e-01	[ 6.2948548e-01]	 2.1922994e-01


.. parsed-literal::

      24	 6.3144050e-01	 2.0061893e-01	 6.6873821e-01	 1.9067443e-01	[ 6.8917970e-01]	 2.2542191e-01


.. parsed-literal::

      25	 6.6199987e-01	 1.9751709e-01	 6.9860257e-01	 1.8670380e-01	[ 7.2437748e-01]	 2.1144199e-01


.. parsed-literal::

      26	 6.8063405e-01	 2.0372458e-01	 7.1494418e-01	 1.9006559e-01	[ 7.4125612e-01]	 2.0485187e-01
      27	 7.1303399e-01	 1.9838499e-01	 7.4829428e-01	 1.9136067e-01	[ 7.6659177e-01]	 1.7972827e-01


.. parsed-literal::

      28	 7.4138403e-01	 1.9184457e-01	 7.7764812e-01	 1.8417411e-01	[ 7.9594252e-01]	 2.1194434e-01


.. parsed-literal::

      29	 7.6403830e-01	 1.8973360e-01	 8.0133102e-01	 1.8010064e-01	[ 8.1955220e-01]	 2.2928548e-01


.. parsed-literal::

      30	 7.8852851e-01	 1.8746650e-01	 8.2629986e-01	 1.7545964e-01	[ 8.4312875e-01]	 2.0978332e-01
      31	 8.1882370e-01	 1.8604400e-01	 8.5771851e-01	 1.7221723e-01	[ 8.6527161e-01]	 1.9127750e-01


.. parsed-literal::

      32	 8.4221389e-01	 1.8542465e-01	 8.8104581e-01	 1.7159150e-01	[ 8.8301208e-01]	 2.1503758e-01


.. parsed-literal::

      33	 8.6553677e-01	 1.8626432e-01	 9.0511584e-01	 1.7350220e-01	[ 8.9720968e-01]	 2.1462154e-01
      34	 8.9335747e-01	 1.8799889e-01	 9.3406317e-01	 1.7530874e-01	[ 9.1702687e-01]	 2.0379233e-01


.. parsed-literal::

      35	 9.1318750e-01	 1.9101355e-01	 9.5558138e-01	 1.8119224e-01	[ 9.2856555e-01]	 2.0651126e-01


.. parsed-literal::

      36	 9.2966809e-01	 1.9102999e-01	 9.7188979e-01	 1.7893298e-01	[ 9.4611421e-01]	 2.0194936e-01


.. parsed-literal::

      37	 9.4597849e-01	 1.8890034e-01	 9.8856167e-01	 1.7563623e-01	[ 9.6346389e-01]	 2.0922422e-01


.. parsed-literal::

      38	 9.7522630e-01	 1.8460392e-01	 1.0190051e+00	 1.7109430e-01	[ 9.9594806e-01]	 2.0524979e-01


.. parsed-literal::

      39	 9.8984881e-01	 1.8129808e-01	 1.0339914e+00	 1.6287841e-01	[ 1.0268738e+00]	 2.1117926e-01


.. parsed-literal::

      40	 1.0022682e+00	 1.7880369e-01	 1.0461531e+00	 1.6162469e-01	[ 1.0343892e+00]	 2.0133710e-01


.. parsed-literal::

      41	 1.0099507e+00	 1.7758732e-01	 1.0540501e+00	 1.6037579e-01	[ 1.0405143e+00]	 2.1351361e-01


.. parsed-literal::

      42	 1.0247801e+00	 1.7428085e-01	 1.0696155e+00	 1.5628064e-01	[ 1.0537035e+00]	 2.0848012e-01


.. parsed-literal::

      43	 1.0373277e+00	 1.7322956e-01	 1.0826844e+00	 1.5408564e-01	[ 1.0675367e+00]	 2.1714640e-01


.. parsed-literal::

      44	 1.0480960e+00	 1.7135220e-01	 1.0937779e+00	 1.5132510e-01	[ 1.0755030e+00]	 2.1794939e-01


.. parsed-literal::

      45	 1.0579956e+00	 1.6970679e-01	 1.1039072e+00	 1.5039446e-01	[ 1.0803038e+00]	 2.0224166e-01
      46	 1.0667378e+00	 1.6965862e-01	 1.1126090e+00	 1.4964949e-01	[ 1.0878333e+00]	 1.8700385e-01


.. parsed-literal::

      47	 1.0756118e+00	 1.6856162e-01	 1.1214247e+00	 1.4919019e-01	[ 1.0941090e+00]	 1.7925882e-01


.. parsed-literal::

      48	 1.0852439e+00	 1.6689332e-01	 1.1313929e+00	 1.4879405e-01	[ 1.1044693e+00]	 2.0859432e-01
      49	 1.0928467e+00	 1.6509269e-01	 1.1390608e+00	 1.4670079e-01	[ 1.1137678e+00]	 1.9968700e-01


.. parsed-literal::

      50	 1.1005535e+00	 1.6358938e-01	 1.1469286e+00	 1.4484742e-01	[ 1.1222635e+00]	 2.0749569e-01


.. parsed-literal::

      51	 1.1140722e+00	 1.6053550e-01	 1.1611680e+00	 1.4239498e-01	[ 1.1338563e+00]	 2.0725703e-01


.. parsed-literal::

      52	 1.1229110e+00	 1.5863195e-01	 1.1705093e+00	 1.4267250e-01	[ 1.1346657e+00]	 2.1085358e-01
      53	 1.1306364e+00	 1.5732142e-01	 1.1782018e+00	 1.4237904e-01	[ 1.1400830e+00]	 1.8403745e-01


.. parsed-literal::

      54	 1.1382626e+00	 1.5589471e-01	 1.1860859e+00	 1.4261946e-01	[ 1.1420875e+00]	 1.9759870e-01
      55	 1.1445979e+00	 1.5473576e-01	 1.1926948e+00	 1.4217555e-01	[ 1.1448685e+00]	 1.7353463e-01


.. parsed-literal::

      56	 1.1542048e+00	 1.5300274e-01	 1.2029111e+00	 1.4241447e-01	[ 1.1501802e+00]	 2.0773792e-01


.. parsed-literal::

      57	 1.1616201e+00	 1.5224808e-01	 1.2103782e+00	 1.4140683e-01	[ 1.1559727e+00]	 2.0896697e-01
      58	 1.1672317e+00	 1.5204069e-01	 1.2155934e+00	 1.4016220e-01	[ 1.1653804e+00]	 1.7883945e-01


.. parsed-literal::

      59	 1.1756907e+00	 1.5098370e-01	 1.2242666e+00	 1.3917803e-01	[ 1.1745456e+00]	 2.1504831e-01
      60	 1.1829792e+00	 1.5035277e-01	 1.2318338e+00	 1.3908548e-01	[ 1.1830077e+00]	 1.8107867e-01


.. parsed-literal::

      61	 1.1898861e+00	 1.4961717e-01	 1.2388758e+00	 1.3907563e-01	[ 1.1880557e+00]	 2.0202327e-01


.. parsed-literal::

      62	 1.1973316e+00	 1.4889773e-01	 1.2463156e+00	 1.3864903e-01	[ 1.1954286e+00]	 2.1248555e-01


.. parsed-literal::

      63	 1.2050542e+00	 1.4837501e-01	 1.2542759e+00	 1.3855474e-01	[ 1.2004889e+00]	 2.0266294e-01


.. parsed-literal::

      64	 1.2140956e+00	 1.4756520e-01	 1.2636459e+00	 1.3845801e-01	[ 1.2029290e+00]	 2.0685768e-01
      65	 1.2221616e+00	 1.4703101e-01	 1.2718763e+00	 1.3763692e-01	[ 1.2080065e+00]	 1.8423629e-01


.. parsed-literal::

      66	 1.2297258e+00	 1.4609720e-01	 1.2797286e+00	 1.3784433e-01	  1.2079241e+00 	 2.0197392e-01


.. parsed-literal::

      67	 1.2353112e+00	 1.4547820e-01	 1.2857440e+00	 1.3669029e-01	  1.2063025e+00 	 2.2680330e-01
      68	 1.2416528e+00	 1.4501797e-01	 1.2919768e+00	 1.3645755e-01	[ 1.2142227e+00]	 2.0306468e-01


.. parsed-literal::

      69	 1.2461829e+00	 1.4474709e-01	 1.2966629e+00	 1.3654381e-01	[ 1.2180793e+00]	 2.0995498e-01


.. parsed-literal::

      70	 1.2509956e+00	 1.4476490e-01	 1.3015306e+00	 1.3632214e-01	[ 1.2284032e+00]	 2.1333170e-01


.. parsed-literal::

      71	 1.2568637e+00	 1.4462622e-01	 1.3074220e+00	 1.3607058e-01	[ 1.2332237e+00]	 2.1051836e-01


.. parsed-literal::

      72	 1.2632203e+00	 1.4439791e-01	 1.3139749e+00	 1.3607920e-01	[ 1.2355372e+00]	 2.1841145e-01


.. parsed-literal::

      73	 1.2699161e+00	 1.4425857e-01	 1.3207685e+00	 1.3595678e-01	[ 1.2435474e+00]	 2.0922375e-01
      74	 1.2759022e+00	 1.4391440e-01	 1.3266887e+00	 1.3607844e-01	[ 1.2461192e+00]	 1.8539548e-01


.. parsed-literal::

      75	 1.2798485e+00	 1.4336609e-01	 1.3305244e+00	 1.3594355e-01	[ 1.2521392e+00]	 2.1666026e-01


.. parsed-literal::

      76	 1.2852476e+00	 1.4271759e-01	 1.3359804e+00	 1.3576825e-01	[ 1.2576038e+00]	 2.0674586e-01


.. parsed-literal::

      77	 1.2893616e+00	 1.4214052e-01	 1.3401661e+00	 1.3602363e-01	[ 1.2596964e+00]	 2.1518421e-01


.. parsed-literal::

      78	 1.2934109e+00	 1.4204745e-01	 1.3442280e+00	 1.3581601e-01	[ 1.2623414e+00]	 2.1897840e-01


.. parsed-literal::

      79	 1.3013863e+00	 1.4170967e-01	 1.3524994e+00	 1.3533411e-01	[ 1.2627494e+00]	 2.0434284e-01
      80	 1.3061894e+00	 1.4169826e-01	 1.3574362e+00	 1.3471744e-01	[ 1.2662180e+00]	 1.9211578e-01


.. parsed-literal::

      81	 1.3118609e+00	 1.4129193e-01	 1.3632586e+00	 1.3406692e-01	[ 1.2686074e+00]	 1.8248296e-01


.. parsed-literal::

      82	 1.3194075e+00	 1.4051281e-01	 1.3714873e+00	 1.3318758e-01	  1.2675106e+00 	 2.1658254e-01


.. parsed-literal::

      83	 1.3226595e+00	 1.4110266e-01	 1.3747965e+00	 1.3219474e-01	[ 1.2696565e+00]	 2.0803738e-01


.. parsed-literal::

      84	 1.3280038e+00	 1.4080626e-01	 1.3800895e+00	 1.3222561e-01	[ 1.2773667e+00]	 2.1869183e-01
      85	 1.3320234e+00	 1.4059906e-01	 1.3843340e+00	 1.3207888e-01	[ 1.2798036e+00]	 1.9763923e-01


.. parsed-literal::

      86	 1.3360891e+00	 1.4078659e-01	 1.3884683e+00	 1.3152134e-01	[ 1.2846561e+00]	 2.0564008e-01
      87	 1.3405587e+00	 1.4062465e-01	 1.3931566e+00	 1.3133782e-01	[ 1.2863416e+00]	 1.8473911e-01


.. parsed-literal::

      88	 1.3450768e+00	 1.4047595e-01	 1.3977514e+00	 1.3105416e-01	  1.2862991e+00 	 2.0690751e-01


.. parsed-literal::

      89	 1.3481760e+00	 1.3995887e-01	 1.4012141e+00	 1.3156934e-01	  1.2839038e+00 	 2.2909307e-01


.. parsed-literal::

      90	 1.3511472e+00	 1.3956187e-01	 1.4039119e+00	 1.3164092e-01	  1.2859745e+00 	 2.1438336e-01
      91	 1.3538208e+00	 1.3917331e-01	 1.4065945e+00	 1.3156791e-01	[ 1.2872827e+00]	 1.9885230e-01


.. parsed-literal::

      92	 1.3588160e+00	 1.3855249e-01	 1.4116802e+00	 1.3175988e-01	[ 1.2891043e+00]	 2.1727777e-01


.. parsed-literal::

      93	 1.3609245e+00	 1.3827705e-01	 1.4142063e+00	 1.3346496e-01	  1.2865038e+00 	 2.2319770e-01
      94	 1.3663208e+00	 1.3813178e-01	 1.4194199e+00	 1.3260689e-01	[ 1.2911025e+00]	 2.0154428e-01


.. parsed-literal::

      95	 1.3686816e+00	 1.3807320e-01	 1.4217674e+00	 1.3233463e-01	[ 1.2922045e+00]	 2.1455503e-01


.. parsed-literal::

      96	 1.3712377e+00	 1.3782629e-01	 1.4243574e+00	 1.3206083e-01	[ 1.2930813e+00]	 2.1706367e-01


.. parsed-literal::

      97	 1.3752209e+00	 1.3719024e-01	 1.4285031e+00	 1.3146090e-01	  1.2899976e+00 	 2.1143055e-01


.. parsed-literal::

      98	 1.3777552e+00	 1.3664210e-01	 1.4311507e+00	 1.3090605e-01	  1.2927486e+00 	 3.2745433e-01
      99	 1.3805788e+00	 1.3613983e-01	 1.4339907e+00	 1.3046913e-01	  1.2910352e+00 	 2.0062709e-01


.. parsed-literal::

     100	 1.3831149e+00	 1.3565459e-01	 1.4365568e+00	 1.3017270e-01	  1.2900457e+00 	 1.8178177e-01


.. parsed-literal::

     101	 1.3861953e+00	 1.3531795e-01	 1.4396462e+00	 1.2957452e-01	  1.2899018e+00 	 2.0951104e-01
     102	 1.3881102e+00	 1.3430160e-01	 1.4418150e+00	 1.2915369e-01	  1.2832915e+00 	 1.9906068e-01


.. parsed-literal::

     103	 1.3923437e+00	 1.3455389e-01	 1.4458617e+00	 1.2860492e-01	  1.2873760e+00 	 2.0903254e-01


.. parsed-literal::

     104	 1.3942020e+00	 1.3460383e-01	 1.4476985e+00	 1.2840802e-01	  1.2878280e+00 	 2.0434713e-01
     105	 1.3970991e+00	 1.3446070e-01	 1.4506221e+00	 1.2807922e-01	  1.2853574e+00 	 2.0576096e-01


.. parsed-literal::

     106	 1.4015978e+00	 1.3400331e-01	 1.4551872e+00	 1.2781448e-01	  1.2842669e+00 	 2.1201205e-01
     107	 1.4028974e+00	 1.3312127e-01	 1.4568858e+00	 1.2729750e-01	  1.2652628e+00 	 1.8502021e-01


.. parsed-literal::

     108	 1.4071819e+00	 1.3303926e-01	 1.4608939e+00	 1.2743709e-01	  1.2769044e+00 	 1.9466114e-01
     109	 1.4084866e+00	 1.3293378e-01	 1.4621720e+00	 1.2737506e-01	  1.2799520e+00 	 2.0269203e-01


.. parsed-literal::

     110	 1.4110789e+00	 1.3256962e-01	 1.4648284e+00	 1.2710607e-01	  1.2847092e+00 	 2.0837116e-01


.. parsed-literal::

     111	 1.4115113e+00	 1.3242120e-01	 1.4654225e+00	 1.2706315e-01	  1.2772671e+00 	 2.1955180e-01
     112	 1.4142409e+00	 1.3236385e-01	 1.4680707e+00	 1.2688115e-01	  1.2825870e+00 	 1.6845703e-01


.. parsed-literal::

     113	 1.4156760e+00	 1.3232850e-01	 1.4695298e+00	 1.2672318e-01	  1.2838042e+00 	 2.0848656e-01


.. parsed-literal::

     114	 1.4172551e+00	 1.3231801e-01	 1.4711603e+00	 1.2656634e-01	  1.2844666e+00 	 2.1328974e-01
     115	 1.4198636e+00	 1.3225845e-01	 1.4737991e+00	 1.2650851e-01	  1.2865445e+00 	 1.8395472e-01


.. parsed-literal::

     116	 1.4215938e+00	 1.3239459e-01	 1.4755896e+00	 1.2625246e-01	  1.2865350e+00 	 3.2389092e-01


.. parsed-literal::

     117	 1.4241119e+00	 1.3217953e-01	 1.4781746e+00	 1.2643495e-01	  1.2856653e+00 	 2.2419643e-01
     118	 1.4259191e+00	 1.3208059e-01	 1.4799827e+00	 1.2648971e-01	  1.2822140e+00 	 2.0201898e-01


.. parsed-literal::

     119	 1.4287629e+00	 1.3190731e-01	 1.4829122e+00	 1.2647191e-01	  1.2746069e+00 	 1.8885469e-01
     120	 1.4306376e+00	 1.3175185e-01	 1.4848917e+00	 1.2638397e-01	  1.2644747e+00 	 2.0103431e-01


.. parsed-literal::

     121	 1.4332086e+00	 1.3178141e-01	 1.4874476e+00	 1.2616797e-01	  1.2605738e+00 	 2.0709729e-01


.. parsed-literal::

     122	 1.4351741e+00	 1.3179282e-01	 1.4894402e+00	 1.2603614e-01	  1.2594067e+00 	 2.2402167e-01
     123	 1.4365966e+00	 1.3181806e-01	 1.4908768e+00	 1.2603134e-01	  1.2576688e+00 	 2.0947695e-01


.. parsed-literal::

     124	 1.4380041e+00	 1.3179082e-01	 1.4923073e+00	 1.2610488e-01	  1.2597229e+00 	 3.2563138e-01
     125	 1.4400233e+00	 1.3181874e-01	 1.4943089e+00	 1.2623816e-01	  1.2573150e+00 	 1.9137979e-01


.. parsed-literal::

     126	 1.4418047e+00	 1.3180624e-01	 1.4960854e+00	 1.2636953e-01	  1.2554816e+00 	 2.1114206e-01
     127	 1.4436896e+00	 1.3178836e-01	 1.4979777e+00	 1.2630650e-01	  1.2521428e+00 	 2.0697474e-01


.. parsed-literal::

     128	 1.4453049e+00	 1.3174573e-01	 1.4996490e+00	 1.2626962e-01	  1.2544486e+00 	 2.1265721e-01


.. parsed-literal::

     129	 1.4470325e+00	 1.3175162e-01	 1.5013485e+00	 1.2592856e-01	  1.2530670e+00 	 2.1923232e-01


.. parsed-literal::

     130	 1.4485223e+00	 1.3173725e-01	 1.5028545e+00	 1.2564979e-01	  1.2515314e+00 	 2.0539689e-01
     131	 1.4498630e+00	 1.3170105e-01	 1.5042293e+00	 1.2547389e-01	  1.2506841e+00 	 1.9986463e-01


.. parsed-literal::

     132	 1.4522123e+00	 1.3155173e-01	 1.5067278e+00	 1.2523631e-01	  1.2485328e+00 	 2.0604253e-01


.. parsed-literal::

     133	 1.4542673e+00	 1.3148415e-01	 1.5089260e+00	 1.2474855e-01	  1.2481503e+00 	 2.0495033e-01


.. parsed-literal::

     134	 1.4559124e+00	 1.3136669e-01	 1.5105195e+00	 1.2474056e-01	  1.2490065e+00 	 2.1583772e-01


.. parsed-literal::

     135	 1.4581529e+00	 1.3123382e-01	 1.5127716e+00	 1.2460685e-01	  1.2457739e+00 	 2.1032262e-01


.. parsed-literal::

     136	 1.4591685e+00	 1.3112368e-01	 1.5138265e+00	 1.2446917e-01	  1.2436965e+00 	 2.2927284e-01


.. parsed-literal::

     137	 1.4606474e+00	 1.3111359e-01	 1.5152691e+00	 1.2429467e-01	  1.2400587e+00 	 2.0727277e-01
     138	 1.4625385e+00	 1.3107568e-01	 1.5171806e+00	 1.2398294e-01	  1.2316773e+00 	 1.9494224e-01


.. parsed-literal::

     139	 1.4638776e+00	 1.3097972e-01	 1.5185365e+00	 1.2385040e-01	  1.2256105e+00 	 2.0736289e-01
     140	 1.4659821e+00	 1.3092571e-01	 1.5207709e+00	 1.2361819e-01	  1.2093825e+00 	 1.9804740e-01


.. parsed-literal::

     141	 1.4676657e+00	 1.3080297e-01	 1.5224831e+00	 1.2364658e-01	  1.1987851e+00 	 2.1520710e-01


.. parsed-literal::

     142	 1.4688822e+00	 1.3073120e-01	 1.5236666e+00	 1.2363743e-01	  1.2065494e+00 	 2.0199466e-01
     143	 1.4704311e+00	 1.3070233e-01	 1.5252506e+00	 1.2361139e-01	  1.2126474e+00 	 2.0228219e-01


.. parsed-literal::

     144	 1.4720140e+00	 1.3063740e-01	 1.5269196e+00	 1.2351873e-01	  1.2149590e+00 	 2.1334577e-01
     145	 1.4735519e+00	 1.3068818e-01	 1.5286483e+00	 1.2347894e-01	  1.2148107e+00 	 1.9487691e-01


.. parsed-literal::

     146	 1.4750179e+00	 1.3064826e-01	 1.5301019e+00	 1.2334054e-01	  1.2096830e+00 	 2.1177459e-01


.. parsed-literal::

     147	 1.4758695e+00	 1.3060870e-01	 1.5309397e+00	 1.2327764e-01	  1.2059770e+00 	 2.1142912e-01
     148	 1.4772952e+00	 1.3058353e-01	 1.5324171e+00	 1.2318049e-01	  1.2014010e+00 	 2.0429564e-01


.. parsed-literal::

     149	 1.4781738e+00	 1.3052781e-01	 1.5333825e+00	 1.2312396e-01	  1.1973512e+00 	 3.1553388e-01


.. parsed-literal::

     150	 1.4793921e+00	 1.3047506e-01	 1.5346555e+00	 1.2311395e-01	  1.1977653e+00 	 2.0999289e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.17 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4da8d3c910>



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
    CPU times: user 1.73 s, sys: 46 ms, total: 1.77 s
    Wall time: 547 ms


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

