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
       1	-3.4689221e-01	 3.2168963e-01	-3.3719252e-01	 3.1576124e-01	[-3.2654538e-01]	 4.7122312e-01


.. parsed-literal::

       2	-2.7664861e-01	 3.1124831e-01	-2.5300348e-01	 3.0622430e-01	[-2.3829812e-01]	 2.3173451e-01


.. parsed-literal::

       3	-2.3464901e-01	 2.9165650e-01	-1.9386266e-01	 2.8600696e-01	[-1.7277956e-01]	 3.0412078e-01
       4	-1.9184542e-01	 2.6775456e-01	-1.4947751e-01	 2.6171885e-01	[-1.1715887e-01]	 1.9766545e-01


.. parsed-literal::

       5	-1.1127636e-01	 2.5832475e-01	-7.9229220e-02	 2.5238942e-01	[-5.0566137e-02]	 2.0520616e-01


.. parsed-literal::

       6	-7.3822589e-02	 2.5254425e-01	-4.4514615e-02	 2.4638423e-01	[-2.1840586e-02]	 2.0825315e-01


.. parsed-literal::

       7	-5.7551015e-02	 2.5006664e-01	-3.3414938e-02	 2.4404092e-01	[-9.1812991e-03]	 2.0920157e-01


.. parsed-literal::

       8	-4.1797364e-02	 2.4730043e-01	-2.1914682e-02	 2.4189265e-01	[ 1.2553218e-03]	 2.1000838e-01
       9	-2.9529937e-02	 2.4498238e-01	-1.2236136e-02	 2.4046031e-01	[ 8.9998497e-03]	 1.7237902e-01


.. parsed-literal::

      10	-2.1998349e-02	 2.4393191e-01	-6.8568980e-03	 2.4053885e-01	[ 1.0307031e-02]	 2.1390843e-01
      11	-1.5660971e-02	 2.4251923e-01	-1.1905057e-03	 2.3877956e-01	[ 1.8048860e-02]	 1.9202614e-01


.. parsed-literal::

      12	-1.3292379e-02	 2.4211421e-01	 8.7111485e-04	 2.3788021e-01	[ 2.1611163e-02]	 1.9484711e-01


.. parsed-literal::

      13	-8.4386534e-03	 2.4121135e-01	 5.7187959e-03	 2.3608922e-01	[ 2.9864961e-02]	 2.0490360e-01


.. parsed-literal::

      14	 1.1031264e-02	 2.3694052e-01	 2.6828408e-02	 2.2895467e-01	[ 5.6638586e-02]	 2.1063066e-01


.. parsed-literal::

      15	 3.3407865e-02	 2.3146722e-01	 4.9017344e-02	 2.2207070e-01	[ 7.6222317e-02]	 3.1977630e-01


.. parsed-literal::

      16	 7.6136667e-02	 2.2567636e-01	 9.4109384e-02	 2.1553392e-01	[ 1.2204311e-01]	 2.1902347e-01


.. parsed-literal::

      17	 1.7491744e-01	 2.2405726e-01	 1.9763216e-01	 2.1500411e-01	[ 2.2978608e-01]	 2.0547867e-01
      18	 2.5942808e-01	 2.2125553e-01	 2.9072007e-01	 2.1639936e-01	[ 3.2476888e-01]	 2.0017099e-01


.. parsed-literal::

      19	 3.0125830e-01	 2.2016906e-01	 3.3315348e-01	 2.1201315e-01	[ 3.7097889e-01]	 2.0364666e-01
      20	 3.4911098e-01	 2.1625210e-01	 3.8161478e-01	 2.0470909e-01	[ 4.2445961e-01]	 1.9429874e-01


.. parsed-literal::

      21	 3.9080327e-01	 2.1273458e-01	 4.2384675e-01	 2.0028479e-01	[ 4.6840790e-01]	 2.0610976e-01
      22	 4.4552203e-01	 2.0919404e-01	 4.7891547e-01	 1.9714800e-01	[ 5.2300869e-01]	 1.9206142e-01


.. parsed-literal::

      23	 5.5967966e-01	 2.0584821e-01	 5.9582329e-01	 1.9333789e-01	[ 6.2902863e-01]	 2.1377420e-01


.. parsed-literal::

      24	 5.9966827e-01	 2.1013818e-01	 6.4093398e-01	 1.9993295e-01	[ 6.4104016e-01]	 2.1135402e-01


.. parsed-literal::

      25	 6.3271443e-01	 2.0506120e-01	 6.7123559e-01	 1.9425032e-01	[ 6.7754629e-01]	 3.2272434e-01


.. parsed-literal::

      26	 6.6135290e-01	 2.0266688e-01	 6.9935337e-01	 1.9064853e-01	[ 7.0779401e-01]	 2.1761823e-01


.. parsed-literal::

      27	 6.9287708e-01	 1.9990865e-01	 7.2995468e-01	 1.8654895e-01	[ 7.4543555e-01]	 3.2406163e-01


.. parsed-literal::

      28	 7.2104981e-01	 1.9621279e-01	 7.5793358e-01	 1.8216831e-01	[ 7.8147852e-01]	 2.0967197e-01


.. parsed-literal::

      29	 7.7056926e-01	 1.9212568e-01	 8.0776824e-01	 1.7552133e-01	[ 8.3135143e-01]	 2.0792246e-01


.. parsed-literal::

      30	 8.0087808e-01	 1.9088894e-01	 8.3916881e-01	 1.7295898e-01	[ 8.6346048e-01]	 2.0998168e-01


.. parsed-literal::

      31	 8.2009902e-01	 1.9155455e-01	 8.6036371e-01	 1.7523119e-01	[ 8.8154571e-01]	 2.1085238e-01
      32	 8.4958779e-01	 1.8860413e-01	 8.9010943e-01	 1.7154090e-01	[ 9.1112207e-01]	 1.8487811e-01


.. parsed-literal::

      33	 8.7012016e-01	 1.8580519e-01	 9.1128263e-01	 1.6796057e-01	[ 9.3384406e-01]	 1.9247389e-01


.. parsed-literal::

      34	 8.9883560e-01	 1.8119958e-01	 9.4106337e-01	 1.6225939e-01	[ 9.7039925e-01]	 2.1539950e-01
      35	 9.1476022e-01	 1.7851074e-01	 9.5795181e-01	 1.5905534e-01	[ 9.8842827e-01]	 1.7390466e-01


.. parsed-literal::

      36	 9.3676629e-01	 1.7675977e-01	 9.7951436e-01	 1.5677777e-01	[ 1.0175065e+00]	 1.9514179e-01


.. parsed-literal::

      37	 9.4874834e-01	 1.7587799e-01	 9.9147655e-01	 1.5563313e-01	[ 1.0304931e+00]	 2.1156335e-01


.. parsed-literal::

      38	 9.7041148e-01	 1.7362412e-01	 1.0137667e+00	 1.5362359e-01	[ 1.0481006e+00]	 2.0710325e-01


.. parsed-literal::

      39	 9.9003092e-01	 1.6975814e-01	 1.0350409e+00	 1.5123904e-01	[ 1.0643574e+00]	 2.1284914e-01
      40	 1.0019506e+00	 1.6842255e-01	 1.0472508e+00	 1.4967543e-01	[ 1.0670199e+00]	 1.7479110e-01


.. parsed-literal::

      41	 1.0101254e+00	 1.6793127e-01	 1.0549347e+00	 1.4895772e-01	[ 1.0761501e+00]	 2.1434379e-01
      42	 1.0263038e+00	 1.6577964e-01	 1.0713409e+00	 1.4571130e-01	[ 1.0932562e+00]	 1.7816997e-01


.. parsed-literal::

      43	 1.0420994e+00	 1.6346035e-01	 1.0874429e+00	 1.4319651e-01	[ 1.1034063e+00]	 2.1369886e-01


.. parsed-literal::

      44	 1.0606503e+00	 1.5971986e-01	 1.1068304e+00	 1.3945587e-01	[ 1.1056888e+00]	 2.1263027e-01


.. parsed-literal::

      45	 1.0736655e+00	 1.5844627e-01	 1.1201173e+00	 1.3725776e-01	[ 1.1182101e+00]	 2.1273088e-01


.. parsed-literal::

      46	 1.0850574e+00	 1.5686657e-01	 1.1314018e+00	 1.3646352e-01	[ 1.1273965e+00]	 2.0960903e-01


.. parsed-literal::

      47	 1.0976886e+00	 1.5454646e-01	 1.1442835e+00	 1.3482285e-01	[ 1.1356798e+00]	 2.0924258e-01
      48	 1.1127366e+00	 1.5212212e-01	 1.1598159e+00	 1.3311453e-01	[ 1.1471399e+00]	 2.0210266e-01


.. parsed-literal::

      49	 1.1218141e+00	 1.5098826e-01	 1.1695138e+00	 1.3308977e-01	  1.1442590e+00 	 2.0874786e-01


.. parsed-literal::

      50	 1.1327465e+00	 1.4961669e-01	 1.1803804e+00	 1.3205208e-01	[ 1.1566205e+00]	 2.1057081e-01


.. parsed-literal::

      51	 1.1394887e+00	 1.4855851e-01	 1.1869449e+00	 1.3125743e-01	[ 1.1640239e+00]	 2.1205735e-01
      52	 1.1500095e+00	 1.4698154e-01	 1.1975892e+00	 1.3042316e-01	[ 1.1720186e+00]	 1.9397306e-01


.. parsed-literal::

      53	 1.1560953e+00	 1.4616019e-01	 1.2042512e+00	 1.3030750e-01	[ 1.1760943e+00]	 1.9809413e-01


.. parsed-literal::

      54	 1.1677348e+00	 1.4495235e-01	 1.2155234e+00	 1.2972801e-01	[ 1.1877543e+00]	 2.1938038e-01


.. parsed-literal::

      55	 1.1733456e+00	 1.4438566e-01	 1.2211677e+00	 1.2974193e-01	[ 1.1918569e+00]	 2.1070719e-01
      56	 1.1808989e+00	 1.4348289e-01	 1.2289843e+00	 1.2969636e-01	[ 1.1945361e+00]	 1.8106771e-01


.. parsed-literal::

      57	 1.1885798e+00	 1.4258334e-01	 1.2370569e+00	 1.3011933e-01	  1.1886566e+00 	 2.1137261e-01


.. parsed-literal::

      58	 1.1962678e+00	 1.4193224e-01	 1.2450311e+00	 1.3018169e-01	  1.1836400e+00 	 2.0136976e-01


.. parsed-literal::

      59	 1.2008777e+00	 1.4164865e-01	 1.2495712e+00	 1.2960291e-01	  1.1855477e+00 	 2.0493984e-01
      60	 1.2124181e+00	 1.4085024e-01	 1.2614212e+00	 1.2862721e-01	  1.1819053e+00 	 1.8773413e-01


.. parsed-literal::

      61	 1.2231663e+00	 1.4015821e-01	 1.2727050e+00	 1.2750629e-01	  1.1632423e+00 	 1.7451692e-01
      62	 1.2328503e+00	 1.3990212e-01	 1.2824547e+00	 1.2742896e-01	  1.1620584e+00 	 1.9587827e-01


.. parsed-literal::

      63	 1.2387317e+00	 1.3971358e-01	 1.2881562e+00	 1.2727525e-01	  1.1715171e+00 	 1.8539548e-01


.. parsed-literal::

      64	 1.2488897e+00	 1.3954838e-01	 1.2986932e+00	 1.2756670e-01	  1.1716729e+00 	 2.1516967e-01


.. parsed-literal::

      65	 1.2562131e+00	 1.3946806e-01	 1.3064294e+00	 1.2703199e-01	  1.1723425e+00 	 2.0318365e-01
      66	 1.2647671e+00	 1.3914226e-01	 1.3149241e+00	 1.2676256e-01	  1.1754968e+00 	 2.0240664e-01


.. parsed-literal::

      67	 1.2702569e+00	 1.3850378e-01	 1.3204793e+00	 1.2616362e-01	  1.1733717e+00 	 1.7874837e-01
      68	 1.2760011e+00	 1.3787254e-01	 1.3263123e+00	 1.2548165e-01	  1.1715916e+00 	 1.8200397e-01


.. parsed-literal::

      69	 1.2813959e+00	 1.3650158e-01	 1.3322728e+00	 1.2435393e-01	  1.1671508e+00 	 2.1368456e-01


.. parsed-literal::

      70	 1.2888834e+00	 1.3624325e-01	 1.3396237e+00	 1.2406042e-01	  1.1759258e+00 	 2.1320605e-01


.. parsed-literal::

      71	 1.2925717e+00	 1.3604806e-01	 1.3433075e+00	 1.2399664e-01	  1.1811160e+00 	 2.2099805e-01


.. parsed-literal::

      72	 1.3013533e+00	 1.3526285e-01	 1.3525195e+00	 1.2375227e-01	  1.1863237e+00 	 2.0950675e-01
      73	 1.3047398e+00	 1.3516619e-01	 1.3563562e+00	 1.2378415e-01	[ 1.2015494e+00]	 1.7672586e-01


.. parsed-literal::

      74	 1.3115968e+00	 1.3469765e-01	 1.3629634e+00	 1.2343512e-01	[ 1.2057835e+00]	 2.1767426e-01


.. parsed-literal::

      75	 1.3154302e+00	 1.3450159e-01	 1.3668322e+00	 1.2328297e-01	[ 1.2108384e+00]	 2.0880985e-01


.. parsed-literal::

      76	 1.3211737e+00	 1.3440670e-01	 1.3726710e+00	 1.2310701e-01	[ 1.2214958e+00]	 2.1057653e-01
      77	 1.3301428e+00	 1.3446386e-01	 1.3819705e+00	 1.2286828e-01	[ 1.2371977e+00]	 1.8751884e-01


.. parsed-literal::

      78	 1.3368869e+00	 1.3486100e-01	 1.3893634e+00	 1.2328098e-01	[ 1.2483471e+00]	 1.7688227e-01


.. parsed-literal::

      79	 1.3437454e+00	 1.3454000e-01	 1.3958955e+00	 1.2300862e-01	[ 1.2547754e+00]	 2.6021886e-01


.. parsed-literal::

      80	 1.3493794e+00	 1.3422289e-01	 1.4014234e+00	 1.2266057e-01	[ 1.2547927e+00]	 2.1847630e-01


.. parsed-literal::

      81	 1.3550449e+00	 1.3374082e-01	 1.4073960e+00	 1.2220140e-01	  1.2507086e+00 	 2.1895671e-01


.. parsed-literal::

      82	 1.3578164e+00	 1.3341283e-01	 1.4103936e+00	 1.2207216e-01	  1.2390484e+00 	 2.1256709e-01
      83	 1.3630445e+00	 1.3313778e-01	 1.4155518e+00	 1.2162006e-01	  1.2479624e+00 	 1.8667364e-01


.. parsed-literal::

      84	 1.3664178e+00	 1.3280655e-01	 1.4188580e+00	 1.2121649e-01	  1.2536079e+00 	 1.8630075e-01
      85	 1.3714005e+00	 1.3242296e-01	 1.4239188e+00	 1.2069231e-01	  1.2543573e+00 	 1.9787216e-01


.. parsed-literal::

      86	 1.3763561e+00	 1.3170439e-01	 1.4291785e+00	 1.2011655e-01	  1.2457109e+00 	 2.0900702e-01


.. parsed-literal::

      87	 1.3813721e+00	 1.3184110e-01	 1.4342982e+00	 1.2011708e-01	  1.2486224e+00 	 2.1084595e-01


.. parsed-literal::

      88	 1.3845934e+00	 1.3168084e-01	 1.4374022e+00	 1.1999127e-01	  1.2504112e+00 	 2.1253538e-01
      89	 1.3884686e+00	 1.3132474e-01	 1.4413920e+00	 1.1975022e-01	  1.2477883e+00 	 1.7703629e-01


.. parsed-literal::

      90	 1.3931052e+00	 1.3139961e-01	 1.4463281e+00	 1.1984121e-01	  1.2507773e+00 	 2.1391034e-01
      91	 1.3979891e+00	 1.3083453e-01	 1.4512065e+00	 1.1931884e-01	[ 1.2596187e+00]	 1.8030953e-01


.. parsed-literal::

      92	 1.4017887e+00	 1.3068671e-01	 1.4549881e+00	 1.1925418e-01	[ 1.2644470e+00]	 2.1880698e-01


.. parsed-literal::

      93	 1.4057411e+00	 1.3040931e-01	 1.4589061e+00	 1.1924607e-01	[ 1.2710172e+00]	 2.2357798e-01


.. parsed-literal::

      94	 1.4090386e+00	 1.3003924e-01	 1.4623705e+00	 1.1915039e-01	[ 1.2761508e+00]	 2.2540116e-01


.. parsed-literal::

      95	 1.4137020e+00	 1.2939214e-01	 1.4669620e+00	 1.1869326e-01	[ 1.2824679e+00]	 2.1728683e-01
      96	 1.4169472e+00	 1.2889100e-01	 1.4702545e+00	 1.1825997e-01	[ 1.2856366e+00]	 1.8712616e-01


.. parsed-literal::

      97	 1.4200757e+00	 1.2861426e-01	 1.4733865e+00	 1.1788573e-01	[ 1.2866307e+00]	 2.0794654e-01
      98	 1.4252157e+00	 1.2828417e-01	 1.4787522e+00	 1.1717303e-01	[ 1.2884442e+00]	 1.7765141e-01


.. parsed-literal::

      99	 1.4269813e+00	 1.2818226e-01	 1.4806022e+00	 1.1693554e-01	  1.2732659e+00 	 2.2183490e-01


.. parsed-literal::

     100	 1.4307768e+00	 1.2811997e-01	 1.4841817e+00	 1.1685902e-01	  1.2849589e+00 	 2.1473122e-01


.. parsed-literal::

     101	 1.4324572e+00	 1.2804136e-01	 1.4858808e+00	 1.1681668e-01	[ 1.2897823e+00]	 2.1252084e-01


.. parsed-literal::

     102	 1.4352967e+00	 1.2789462e-01	 1.4887737e+00	 1.1660686e-01	[ 1.2929632e+00]	 2.1229410e-01


.. parsed-literal::

     103	 1.4386508e+00	 1.2764412e-01	 1.4922273e+00	 1.1635274e-01	  1.2914620e+00 	 2.1537042e-01


.. parsed-literal::

     104	 1.4419976e+00	 1.2746981e-01	 1.4956175e+00	 1.1594291e-01	  1.2906676e+00 	 2.0935607e-01


.. parsed-literal::

     105	 1.4449689e+00	 1.2735269e-01	 1.4986158e+00	 1.1562526e-01	  1.2870176e+00 	 2.2150683e-01


.. parsed-literal::

     106	 1.4477607e+00	 1.2718892e-01	 1.5014789e+00	 1.1542053e-01	  1.2859981e+00 	 2.1623111e-01


.. parsed-literal::

     107	 1.4500503e+00	 1.2734953e-01	 1.5039465e+00	 1.1522200e-01	  1.2805335e+00 	 2.1612883e-01


.. parsed-literal::

     108	 1.4527529e+00	 1.2717493e-01	 1.5065822e+00	 1.1518134e-01	  1.2849497e+00 	 2.1467352e-01
     109	 1.4551005e+00	 1.2715783e-01	 1.5089223e+00	 1.1520486e-01	  1.2884059e+00 	 1.8558216e-01


.. parsed-literal::

     110	 1.4576949e+00	 1.2710978e-01	 1.5115455e+00	 1.1506453e-01	  1.2904603e+00 	 2.0503759e-01


.. parsed-literal::

     111	 1.4614402e+00	 1.2692596e-01	 1.5155248e+00	 1.1434921e-01	  1.2909377e+00 	 2.0952010e-01


.. parsed-literal::

     112	 1.4632836e+00	 1.2667610e-01	 1.5174261e+00	 1.1391604e-01	[ 1.2974454e+00]	 2.0710635e-01


.. parsed-literal::

     113	 1.4651637e+00	 1.2658244e-01	 1.5191791e+00	 1.1387606e-01	[ 1.2974470e+00]	 2.1042800e-01


.. parsed-literal::

     114	 1.4663258e+00	 1.2649668e-01	 1.5203527e+00	 1.1369371e-01	  1.2964127e+00 	 2.0503139e-01
     115	 1.4685283e+00	 1.2624768e-01	 1.5226122e+00	 1.1321456e-01	  1.2956285e+00 	 1.7435360e-01


.. parsed-literal::

     116	 1.4704520e+00	 1.2596447e-01	 1.5246526e+00	 1.1267665e-01	  1.2953526e+00 	 1.8726802e-01


.. parsed-literal::

     117	 1.4725179e+00	 1.2584206e-01	 1.5267213e+00	 1.1237578e-01	[ 1.2984019e+00]	 2.0916247e-01


.. parsed-literal::

     118	 1.4748372e+00	 1.2570936e-01	 1.5290638e+00	 1.1208183e-01	[ 1.3001818e+00]	 2.1077991e-01


.. parsed-literal::

     119	 1.4760650e+00	 1.2565129e-01	 1.5302991e+00	 1.1190840e-01	[ 1.3045762e+00]	 2.1528316e-01


.. parsed-literal::

     120	 1.4778930e+00	 1.2556575e-01	 1.5320939e+00	 1.1183454e-01	  1.3041247e+00 	 2.0400643e-01


.. parsed-literal::

     121	 1.4809483e+00	 1.2544680e-01	 1.5351418e+00	 1.1165341e-01	  1.3022399e+00 	 2.0995021e-01


.. parsed-literal::

     122	 1.4824841e+00	 1.2523098e-01	 1.5366785e+00	 1.1160058e-01	[ 1.3046256e+00]	 3.1709409e-01
     123	 1.4840988e+00	 1.2515909e-01	 1.5383122e+00	 1.1152844e-01	  1.3017924e+00 	 1.9928098e-01


.. parsed-literal::

     124	 1.4857573e+00	 1.2505800e-01	 1.5400087e+00	 1.1151025e-01	  1.3008575e+00 	 2.0924735e-01
     125	 1.4879126e+00	 1.2491846e-01	 1.5422173e+00	 1.1154793e-01	  1.2976025e+00 	 1.8621778e-01


.. parsed-literal::

     126	 1.4899143e+00	 1.2480921e-01	 1.5442868e+00	 1.1187033e-01	  1.2973261e+00 	 1.8533230e-01


.. parsed-literal::

     127	 1.4916344e+00	 1.2480024e-01	 1.5459731e+00	 1.1190337e-01	  1.2970084e+00 	 2.1700454e-01


.. parsed-literal::

     128	 1.4932301e+00	 1.2480053e-01	 1.5475375e+00	 1.1194295e-01	  1.2975928e+00 	 2.1549034e-01
     129	 1.4948009e+00	 1.2484690e-01	 1.5491715e+00	 1.1205991e-01	  1.2973621e+00 	 1.9716454e-01


.. parsed-literal::

     130	 1.4966998e+00	 1.2485476e-01	 1.5511502e+00	 1.1210752e-01	  1.2917996e+00 	 1.7293739e-01


.. parsed-literal::

     131	 1.4981699e+00	 1.2487301e-01	 1.5526922e+00	 1.1211221e-01	  1.2909821e+00 	 2.0828867e-01


.. parsed-literal::

     132	 1.4997836e+00	 1.2488666e-01	 1.5544298e+00	 1.1215969e-01	  1.2806642e+00 	 2.0083570e-01


.. parsed-literal::

     133	 1.5013417e+00	 1.2498421e-01	 1.5560595e+00	 1.1223477e-01	  1.2804879e+00 	 2.0900941e-01


.. parsed-literal::

     134	 1.5027500e+00	 1.2494375e-01	 1.5574686e+00	 1.1229098e-01	  1.2789859e+00 	 2.0856452e-01


.. parsed-literal::

     135	 1.5045788e+00	 1.2490739e-01	 1.5592970e+00	 1.1244068e-01	  1.2721903e+00 	 2.1495938e-01


.. parsed-literal::

     136	 1.5058325e+00	 1.2483681e-01	 1.5605843e+00	 1.1267343e-01	  1.2720696e+00 	 2.1683073e-01


.. parsed-literal::

     137	 1.5074340e+00	 1.2482014e-01	 1.5621614e+00	 1.1279564e-01	  1.2699367e+00 	 2.0969772e-01


.. parsed-literal::

     138	 1.5096612e+00	 1.2487445e-01	 1.5644349e+00	 1.1302231e-01	  1.2639947e+00 	 2.1379042e-01


.. parsed-literal::

     139	 1.5107605e+00	 1.2489953e-01	 1.5655403e+00	 1.1303098e-01	  1.2636330e+00 	 2.1134043e-01


.. parsed-literal::

     140	 1.5118015e+00	 1.2483832e-01	 1.5665741e+00	 1.1290888e-01	  1.2637025e+00 	 2.1504641e-01
     141	 1.5142007e+00	 1.2463066e-01	 1.5690249e+00	 1.1261261e-01	  1.2610109e+00 	 1.8440723e-01


.. parsed-literal::

     142	 1.5151787e+00	 1.2449963e-01	 1.5700044e+00	 1.1249397e-01	  1.2602994e+00 	 2.0536780e-01


.. parsed-literal::

     143	 1.5164280e+00	 1.2441814e-01	 1.5712492e+00	 1.1246718e-01	  1.2574671e+00 	 2.0989799e-01


.. parsed-literal::

     144	 1.5179519e+00	 1.2435866e-01	 1.5727694e+00	 1.1256888e-01	  1.2533010e+00 	 2.0851207e-01
     145	 1.5189806e+00	 1.2432561e-01	 1.5737825e+00	 1.1253455e-01	  1.2492592e+00 	 1.9016886e-01


.. parsed-literal::

     146	 1.5202258e+00	 1.2433289e-01	 1.5750066e+00	 1.1251723e-01	  1.2503750e+00 	 2.0793629e-01


.. parsed-literal::

     147	 1.5209199e+00	 1.2433980e-01	 1.5758584e+00	 1.1253891e-01	  1.2372804e+00 	 2.1256495e-01
     148	 1.5227788e+00	 1.2431279e-01	 1.5776108e+00	 1.1241117e-01	  1.2494845e+00 	 1.7157412e-01


.. parsed-literal::

     149	 1.5232372e+00	 1.2427820e-01	 1.5780676e+00	 1.1239795e-01	  1.2493504e+00 	 2.0858479e-01
     150	 1.5251359e+00	 1.2418217e-01	 1.5800397e+00	 1.1248912e-01	  1.2418790e+00 	 1.9770861e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.07 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f68e4f2dc00>



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
    CPU times: user 2.07 s, sys: 37.9 ms, total: 2.11 s
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

