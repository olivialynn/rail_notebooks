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
       1	-3.4472197e-01	 3.2094618e-01	-3.3504172e-01	 3.1899999e-01	[-3.3128129e-01]	 4.6726918e-01


.. parsed-literal::

       2	-2.7476705e-01	 3.1072883e-01	-2.5105434e-01	 3.0777779e-01	[-2.4236911e-01]	 2.3601198e-01


.. parsed-literal::

       3	-2.2972071e-01	 2.8908019e-01	-1.8658038e-01	 2.8743197e-01	[-1.7633998e-01]	 2.8098655e-01


.. parsed-literal::

       4	-1.9241969e-01	 2.6607717e-01	-1.5043635e-01	 2.6323157e-01	[-1.3034302e-01]	 2.0392418e-01


.. parsed-literal::

       5	-1.0404672e-01	 2.5775146e-01	-7.1111321e-02	 2.5225401e-01	[-4.4432035e-02]	 2.0957160e-01
       6	-7.4007692e-02	 2.5290438e-01	-4.5118138e-02	 2.4596533e-01	[-1.9926677e-02]	 1.9066620e-01


.. parsed-literal::

       7	-5.7144580e-02	 2.5026245e-01	-3.3994932e-02	 2.4312692e-01	[-6.4410918e-03]	 2.1503353e-01
       8	-4.4904372e-02	 2.4818005e-01	-2.5362485e-02	 2.4074869e-01	[ 3.9618004e-03]	 1.8370008e-01


.. parsed-literal::

       9	-3.1576731e-02	 2.4565840e-01	-1.4637945e-02	 2.3816442e-01	[ 1.5022048e-02]	 2.0692182e-01


.. parsed-literal::

      10	-2.0937713e-02	 2.4352878e-01	-5.9117038e-03	 2.3667310e-01	[ 2.3289382e-02]	 2.1433949e-01


.. parsed-literal::

      11	-1.5470209e-02	 2.4270077e-01	-1.4794534e-03	 2.3703882e-01	  2.1217005e-02 	 2.0839381e-01


.. parsed-literal::

      12	-1.2805057e-02	 2.4213530e-01	 1.1237064e-03	 2.3661132e-01	[ 2.4048642e-02]	 2.1235704e-01


.. parsed-literal::

      13	-8.0316493e-03	 2.4109504e-01	 6.2586296e-03	 2.3559564e-01	[ 2.9521606e-02]	 2.0577455e-01


.. parsed-literal::

      14	 1.2366400e-01	 2.2527966e-01	 1.5325041e-01	 2.2121435e-01	[ 1.6918248e-01]	 4.3474793e-01


.. parsed-literal::

      15	 1.3627411e-01	 2.1835663e-01	 1.6456076e-01	 2.1829834e-01	  1.6201683e-01 	 2.1631980e-01


.. parsed-literal::

      16	 2.7465250e-01	 2.1227180e-01	 3.0621183e-01	 2.1259362e-01	[ 3.0251532e-01]	 2.0909071e-01


.. parsed-literal::

      17	 3.3080969e-01	 2.0919537e-01	 3.6471794e-01	 2.0990987e-01	[ 3.5800993e-01]	 3.1699848e-01


.. parsed-literal::

      18	 3.6726438e-01	 2.0278419e-01	 4.0297297e-01	 2.0361145e-01	[ 3.9126678e-01]	 3.3072472e-01
      19	 4.1490829e-01	 1.9613672e-01	 4.5084154e-01	 1.9897095e-01	[ 4.2968940e-01]	 1.7756772e-01


.. parsed-literal::

      20	 5.1057404e-01	 1.8622342e-01	 5.4780558e-01	 1.9258467e-01	[ 5.1379131e-01]	 2.1197891e-01


.. parsed-literal::

      21	 5.7696150e-01	 1.8176645e-01	 6.1642404e-01	 1.8828457e-01	[ 5.7111390e-01]	 2.0686889e-01


.. parsed-literal::

      22	 6.2425392e-01	 1.7838399e-01	 6.6539413e-01	 1.8441285e-01	[ 6.2065988e-01]	 2.1180224e-01
      23	 6.5442004e-01	 1.7494607e-01	 6.9704411e-01	 1.8097751e-01	[ 6.4018203e-01]	 1.9812536e-01


.. parsed-literal::

      24	 7.0734844e-01	 1.7127336e-01	 7.4836294e-01	 1.7657577e-01	[ 6.9550354e-01]	 2.1050715e-01


.. parsed-literal::

      25	 7.5408171e-01	 1.6780224e-01	 7.9377609e-01	 1.7414659e-01	[ 7.4774481e-01]	 2.0892906e-01


.. parsed-literal::

      26	 7.8591027e-01	 1.6944904e-01	 8.2528859e-01	 1.7505448e-01	[ 7.9013143e-01]	 2.1036053e-01
      27	 8.2423104e-01	 1.7176924e-01	 8.6394756e-01	 1.7759865e-01	[ 8.4241010e-01]	 1.7632318e-01


.. parsed-literal::

      28	 8.6329604e-01	 1.7599156e-01	 9.0414174e-01	 1.8057565e-01	[ 8.9101999e-01]	 2.1087027e-01


.. parsed-literal::

      29	 8.9603497e-01	 1.7588849e-01	 9.3724378e-01	 1.8210466e-01	[ 9.2073747e-01]	 2.1320844e-01
      30	 9.2374078e-01	 1.6963530e-01	 9.6554835e-01	 1.7836323e-01	[ 9.4291849e-01]	 1.8770742e-01


.. parsed-literal::

      31	 9.5285019e-01	 1.6219118e-01	 9.9537785e-01	 1.7302104e-01	[ 9.6955304e-01]	 2.0947957e-01
      32	 9.7154137e-01	 1.5825240e-01	 1.0145412e+00	 1.6916816e-01	[ 9.8516262e-01]	 1.9116473e-01


.. parsed-literal::

      33	 9.8855404e-01	 1.5480585e-01	 1.0317259e+00	 1.6551721e-01	[ 1.0067245e+00]	 1.8439674e-01


.. parsed-literal::

      34	 1.0043518e+00	 1.5145026e-01	 1.0477069e+00	 1.6257308e-01	[ 1.0266028e+00]	 2.1352553e-01
      35	 1.0193206e+00	 1.4828186e-01	 1.0635073e+00	 1.5838296e-01	[ 1.0463439e+00]	 1.9312572e-01


.. parsed-literal::

      36	 1.0306868e+00	 1.4770360e-01	 1.0746750e+00	 1.5764516e-01	[ 1.0529804e+00]	 1.9451690e-01
      37	 1.0478308e+00	 1.4735713e-01	 1.0921723e+00	 1.5730287e-01	[ 1.0595988e+00]	 1.8437386e-01


.. parsed-literal::

      38	 1.0659262e+00	 1.4732903e-01	 1.1105598e+00	 1.5660146e-01	[ 1.0706273e+00]	 2.1562624e-01
      39	 1.0828963e+00	 1.4787131e-01	 1.1283395e+00	 1.5777286e-01	  1.0665518e+00 	 2.0364738e-01


.. parsed-literal::

      40	 1.1006719e+00	 1.4593387e-01	 1.1462900e+00	 1.5556662e-01	[ 1.0830748e+00]	 2.1291137e-01
      41	 1.1163985e+00	 1.4362206e-01	 1.1622162e+00	 1.5363535e-01	[ 1.0971941e+00]	 1.9923019e-01


.. parsed-literal::

      42	 1.1314589e+00	 1.4159610e-01	 1.1775029e+00	 1.5235545e-01	[ 1.1115677e+00]	 2.1787190e-01


.. parsed-literal::

      43	 1.1383624e+00	 1.4111828e-01	 1.1845330e+00	 1.5340824e-01	[ 1.1165263e+00]	 2.1263480e-01
      44	 1.1479949e+00	 1.4053379e-01	 1.1939135e+00	 1.5322257e-01	[ 1.1214741e+00]	 1.8821812e-01


.. parsed-literal::

      45	 1.1555742e+00	 1.3986093e-01	 1.2016265e+00	 1.5350738e-01	  1.1199044e+00 	 1.9374299e-01


.. parsed-literal::

      46	 1.1647564e+00	 1.3921279e-01	 1.2109523e+00	 1.5442286e-01	  1.1203427e+00 	 2.1160269e-01


.. parsed-literal::

      47	 1.1807501e+00	 1.3844876e-01	 1.2270765e+00	 1.5592439e-01	[ 1.1295932e+00]	 2.1403337e-01
      48	 1.1953318e+00	 1.3865831e-01	 1.2419375e+00	 1.5973927e-01	[ 1.1346269e+00]	 1.9682574e-01


.. parsed-literal::

      49	 1.2083127e+00	 1.3787742e-01	 1.2548176e+00	 1.5908690e-01	[ 1.1505815e+00]	 2.1145940e-01


.. parsed-literal::

      50	 1.2152759e+00	 1.3715004e-01	 1.2617662e+00	 1.5738008e-01	[ 1.1584076e+00]	 2.0856619e-01


.. parsed-literal::

      51	 1.2291044e+00	 1.3643930e-01	 1.2760418e+00	 1.5589494e-01	[ 1.1701542e+00]	 2.1588373e-01


.. parsed-literal::

      52	 1.2360821e+00	 1.3630391e-01	 1.2832421e+00	 1.5554210e-01	[ 1.1722632e+00]	 2.0593262e-01


.. parsed-literal::

      53	 1.2473224e+00	 1.3588503e-01	 1.2944871e+00	 1.5537733e-01	[ 1.1855610e+00]	 2.1105385e-01


.. parsed-literal::

      54	 1.2576900e+00	 1.3570289e-01	 1.3051691e+00	 1.5622218e-01	[ 1.1967157e+00]	 2.1062756e-01


.. parsed-literal::

      55	 1.2656433e+00	 1.3540447e-01	 1.3133578e+00	 1.5630433e-01	[ 1.2044504e+00]	 2.0900512e-01


.. parsed-literal::

      56	 1.2761023e+00	 1.3528181e-01	 1.3242539e+00	 1.5707949e-01	[ 1.2115728e+00]	 2.1181154e-01


.. parsed-literal::

      57	 1.2861941e+00	 1.3540757e-01	 1.3345811e+00	 1.5624997e-01	[ 1.2253997e+00]	 2.0407510e-01


.. parsed-literal::

      58	 1.2930884e+00	 1.3496850e-01	 1.3413243e+00	 1.5563744e-01	[ 1.2294368e+00]	 2.1367717e-01
      59	 1.3068057e+00	 1.3433145e-01	 1.3553798e+00	 1.5434094e-01	[ 1.2357816e+00]	 1.8240499e-01


.. parsed-literal::

      60	 1.3154421e+00	 1.3442061e-01	 1.3643917e+00	 1.5429483e-01	  1.2351134e+00 	 2.0975733e-01


.. parsed-literal::

      61	 1.3236338e+00	 1.3405300e-01	 1.3725067e+00	 1.5303325e-01	[ 1.2456477e+00]	 2.1692872e-01
      62	 1.3301181e+00	 1.3350052e-01	 1.3789377e+00	 1.5186822e-01	[ 1.2571324e+00]	 1.9568872e-01


.. parsed-literal::

      63	 1.3360580e+00	 1.3301097e-01	 1.3850886e+00	 1.5087644e-01	[ 1.2598834e+00]	 2.4215198e-01
      64	 1.3432042e+00	 1.3294934e-01	 1.3923625e+00	 1.5093184e-01	[ 1.2621632e+00]	 1.7677450e-01


.. parsed-literal::

      65	 1.3505170e+00	 1.3247431e-01	 1.3997543e+00	 1.5012362e-01	[ 1.2629335e+00]	 2.0856142e-01


.. parsed-literal::

      66	 1.3577828e+00	 1.3229847e-01	 1.4069241e+00	 1.5011766e-01	[ 1.2639824e+00]	 2.1392632e-01
      67	 1.3644743e+00	 1.3203562e-01	 1.4136645e+00	 1.4965257e-01	[ 1.2715271e+00]	 1.7792583e-01


.. parsed-literal::

      68	 1.3708806e+00	 1.3176084e-01	 1.4201141e+00	 1.4919680e-01	[ 1.2780658e+00]	 2.0160866e-01


.. parsed-literal::

      69	 1.3768086e+00	 1.3151257e-01	 1.4261097e+00	 1.4899301e-01	[ 1.2826431e+00]	 2.1216917e-01
      70	 1.3834825e+00	 1.3042136e-01	 1.4331021e+00	 1.4779500e-01	[ 1.2828459e+00]	 1.7622232e-01


.. parsed-literal::

      71	 1.3882827e+00	 1.3042098e-01	 1.4380612e+00	 1.4790951e-01	[ 1.2853717e+00]	 2.2037411e-01


.. parsed-literal::

      72	 1.3913868e+00	 1.3000935e-01	 1.4410995e+00	 1.4749020e-01	[ 1.2880755e+00]	 2.0672107e-01


.. parsed-literal::

      73	 1.3950282e+00	 1.2944486e-01	 1.4448386e+00	 1.4693580e-01	  1.2871251e+00 	 2.2050095e-01


.. parsed-literal::

      74	 1.3992257e+00	 1.2887609e-01	 1.4491624e+00	 1.4643095e-01	[ 1.2906256e+00]	 2.1031022e-01


.. parsed-literal::

      75	 1.4044413e+00	 1.2816597e-01	 1.4545820e+00	 1.4569264e-01	  1.2895673e+00 	 2.1689200e-01


.. parsed-literal::

      76	 1.4098229e+00	 1.2777180e-01	 1.4602148e+00	 1.4527045e-01	  1.2865484e+00 	 2.1067691e-01
      77	 1.4125830e+00	 1.2739177e-01	 1.4631614e+00	 1.4471654e-01	  1.2830401e+00 	 1.9491673e-01


.. parsed-literal::

      78	 1.4165081e+00	 1.2737723e-01	 1.4669195e+00	 1.4466992e-01	  1.2865536e+00 	 2.1397591e-01


.. parsed-literal::

      79	 1.4201441e+00	 1.2714892e-01	 1.4705462e+00	 1.4444789e-01	  1.2865947e+00 	 2.1739244e-01
      80	 1.4240838e+00	 1.2688915e-01	 1.4744631e+00	 1.4424347e-01	  1.2895449e+00 	 1.9501472e-01


.. parsed-literal::

      81	 1.4277011e+00	 1.2634662e-01	 1.4781806e+00	 1.4431500e-01	  1.2897628e+00 	 2.1298289e-01
      82	 1.4319800e+00	 1.2628724e-01	 1.4824104e+00	 1.4422297e-01	[ 1.2941401e+00]	 1.9907045e-01


.. parsed-literal::

      83	 1.4337485e+00	 1.2627743e-01	 1.4842089e+00	 1.4423564e-01	[ 1.2961511e+00]	 2.0138788e-01


.. parsed-literal::

      84	 1.4369372e+00	 1.2606477e-01	 1.4875463e+00	 1.4416255e-01	[ 1.2969558e+00]	 2.1165681e-01
      85	 1.4379016e+00	 1.2649577e-01	 1.4887022e+00	 1.4454479e-01	  1.2966793e+00 	 1.8478179e-01


.. parsed-literal::

      86	 1.4421467e+00	 1.2612256e-01	 1.4927875e+00	 1.4419713e-01	[ 1.2997250e+00]	 2.1651173e-01


.. parsed-literal::

      87	 1.4443606e+00	 1.2595990e-01	 1.4948991e+00	 1.4401221e-01	[ 1.3008035e+00]	 2.1472096e-01
      88	 1.4467245e+00	 1.2584680e-01	 1.4971649e+00	 1.4383506e-01	[ 1.3034355e+00]	 1.9231629e-01


.. parsed-literal::

      89	 1.4500930e+00	 1.2559330e-01	 1.5004777e+00	 1.4364005e-01	[ 1.3054398e+00]	 1.9409561e-01


.. parsed-literal::

      90	 1.4510484e+00	 1.2554960e-01	 1.5015433e+00	 1.4362033e-01	[ 1.3131416e+00]	 2.2050905e-01


.. parsed-literal::

      91	 1.4550465e+00	 1.2541653e-01	 1.5054774e+00	 1.4358341e-01	  1.3115502e+00 	 2.0867705e-01


.. parsed-literal::

      92	 1.4566970e+00	 1.2535897e-01	 1.5071881e+00	 1.4361559e-01	  1.3104544e+00 	 2.1855140e-01
      93	 1.4591795e+00	 1.2527900e-01	 1.5097937e+00	 1.4369357e-01	  1.3096764e+00 	 1.9803429e-01


.. parsed-literal::

      94	 1.4623337e+00	 1.2529617e-01	 1.5130675e+00	 1.4382437e-01	  1.3074958e+00 	 2.0791054e-01


.. parsed-literal::

      95	 1.4653398e+00	 1.2533091e-01	 1.5162084e+00	 1.4410498e-01	  1.3058725e+00 	 2.2018766e-01


.. parsed-literal::

      96	 1.4675354e+00	 1.2527349e-01	 1.5182928e+00	 1.4404243e-01	  1.3079149e+00 	 2.2037578e-01


.. parsed-literal::

      97	 1.4701569e+00	 1.2512982e-01	 1.5208787e+00	 1.4388221e-01	  1.3068125e+00 	 2.0995760e-01
      98	 1.4716657e+00	 1.2489028e-01	 1.5225240e+00	 1.4390458e-01	  1.3012179e+00 	 1.8248510e-01


.. parsed-literal::

      99	 1.4739128e+00	 1.2473906e-01	 1.5247916e+00	 1.4374632e-01	  1.3007130e+00 	 2.1033955e-01
     100	 1.4767903e+00	 1.2438941e-01	 1.5278393e+00	 1.4358262e-01	  1.2964370e+00 	 1.8431044e-01


.. parsed-literal::

     101	 1.4788280e+00	 1.2406747e-01	 1.5299347e+00	 1.4349995e-01	  1.2959347e+00 	 1.9978833e-01


.. parsed-literal::

     102	 1.4829673e+00	 1.2342298e-01	 1.5340404e+00	 1.4347129e-01	  1.2985524e+00 	 2.0715189e-01


.. parsed-literal::

     103	 1.4850083e+00	 1.2270844e-01	 1.5361217e+00	 1.4360163e-01	  1.3010685e+00 	 2.1249962e-01


.. parsed-literal::

     104	 1.4880473e+00	 1.2274949e-01	 1.5389617e+00	 1.4353755e-01	  1.3032657e+00 	 2.1156216e-01


.. parsed-literal::

     105	 1.4899657e+00	 1.2271441e-01	 1.5407844e+00	 1.4350621e-01	  1.3051490e+00 	 2.1188760e-01
     106	 1.4926957e+00	 1.2249332e-01	 1.5434863e+00	 1.4341096e-01	  1.3059499e+00 	 1.9565558e-01


.. parsed-literal::

     107	 1.4952197e+00	 1.2190859e-01	 1.5461294e+00	 1.4341378e-01	  1.3072898e+00 	 1.8810058e-01
     108	 1.4985116e+00	 1.2156505e-01	 1.5494404e+00	 1.4320285e-01	  1.3029611e+00 	 1.8530321e-01


.. parsed-literal::

     109	 1.5005902e+00	 1.2130311e-01	 1.5515773e+00	 1.4313164e-01	  1.2987369e+00 	 1.8698502e-01


.. parsed-literal::

     110	 1.5031673e+00	 1.2083647e-01	 1.5543083e+00	 1.4304816e-01	  1.2915213e+00 	 2.1148133e-01


.. parsed-literal::

     111	 1.5048726e+00	 1.2038877e-01	 1.5562224e+00	 1.4324912e-01	  1.2792343e+00 	 2.1242666e-01
     112	 1.5072820e+00	 1.2038401e-01	 1.5585430e+00	 1.4308931e-01	  1.2824348e+00 	 1.9897532e-01


.. parsed-literal::

     113	 1.5095151e+00	 1.2026337e-01	 1.5607371e+00	 1.4299147e-01	  1.2827316e+00 	 2.0940328e-01


.. parsed-literal::

     114	 1.5115012e+00	 1.2010410e-01	 1.5627533e+00	 1.4288931e-01	  1.2809702e+00 	 2.0396566e-01
     115	 1.5123480e+00	 1.1945794e-01	 1.5638358e+00	 1.4248806e-01	  1.2751056e+00 	 1.9374037e-01


.. parsed-literal::

     116	 1.5159709e+00	 1.1945410e-01	 1.5673596e+00	 1.4251076e-01	  1.2767838e+00 	 2.1034908e-01


.. parsed-literal::

     117	 1.5171506e+00	 1.1933721e-01	 1.5685715e+00	 1.4247377e-01	  1.2760242e+00 	 2.1257401e-01
     118	 1.5190396e+00	 1.1909535e-01	 1.5705681e+00	 1.4236138e-01	  1.2744467e+00 	 1.9714403e-01


.. parsed-literal::

     119	 1.5215472e+00	 1.1890417e-01	 1.5731818e+00	 1.4226064e-01	  1.2721517e+00 	 2.0749760e-01


.. parsed-literal::

     120	 1.5234892e+00	 1.1861473e-01	 1.5752832e+00	 1.4228768e-01	  1.2647937e+00 	 2.0929003e-01


.. parsed-literal::

     121	 1.5258341e+00	 1.1862573e-01	 1.5774859e+00	 1.4215146e-01	  1.2748813e+00 	 2.0684314e-01


.. parsed-literal::

     122	 1.5269683e+00	 1.1865071e-01	 1.5785282e+00	 1.4206284e-01	  1.2779439e+00 	 2.0988774e-01
     123	 1.5285771e+00	 1.1850385e-01	 1.5801032e+00	 1.4188105e-01	  1.2834499e+00 	 1.8199110e-01


.. parsed-literal::

     124	 1.5304888e+00	 1.1841272e-01	 1.5820277e+00	 1.4164965e-01	  1.2817611e+00 	 2.1011138e-01


.. parsed-literal::

     125	 1.5320641e+00	 1.1811963e-01	 1.5836615e+00	 1.4144234e-01	  1.2798717e+00 	 2.1600509e-01


.. parsed-literal::

     126	 1.5339175e+00	 1.1767566e-01	 1.5856844e+00	 1.4121587e-01	  1.2737446e+00 	 2.1241498e-01


.. parsed-literal::

     127	 1.5353044e+00	 1.1746401e-01	 1.5871610e+00	 1.4108350e-01	  1.2703285e+00 	 2.1033573e-01


.. parsed-literal::

     128	 1.5369307e+00	 1.1731328e-01	 1.5888591e+00	 1.4104366e-01	  1.2692294e+00 	 2.1738577e-01


.. parsed-literal::

     129	 1.5387034e+00	 1.1718438e-01	 1.5907401e+00	 1.4093182e-01	  1.2660430e+00 	 2.1129155e-01


.. parsed-literal::

     130	 1.5404377e+00	 1.1705149e-01	 1.5924826e+00	 1.4082660e-01	  1.2669103e+00 	 2.1415615e-01
     131	 1.5413899e+00	 1.1704104e-01	 1.5933658e+00	 1.4076395e-01	  1.2675616e+00 	 2.0037889e-01


.. parsed-literal::

     132	 1.5434963e+00	 1.1687247e-01	 1.5954557e+00	 1.4035486e-01	  1.2621463e+00 	 1.8801022e-01


.. parsed-literal::

     133	 1.5445655e+00	 1.1672126e-01	 1.5965905e+00	 1.4029553e-01	  1.2517440e+00 	 3.1950855e-01
     134	 1.5458731e+00	 1.1657123e-01	 1.5979542e+00	 1.4009672e-01	  1.2454610e+00 	 1.8940687e-01


.. parsed-literal::

     135	 1.5485401e+00	 1.1622006e-01	 1.6007805e+00	 1.3977934e-01	  1.2255495e+00 	 2.2026682e-01


.. parsed-literal::

     136	 1.5498831e+00	 1.1604820e-01	 1.6021709e+00	 1.3976667e-01	  1.2180802e+00 	 2.1532273e-01


.. parsed-literal::

     137	 1.5514604e+00	 1.1594913e-01	 1.6037143e+00	 1.3984834e-01	  1.2176843e+00 	 2.1481609e-01


.. parsed-literal::

     138	 1.5531586e+00	 1.1585655e-01	 1.6053171e+00	 1.3999968e-01	  1.2183966e+00 	 2.1332693e-01


.. parsed-literal::

     139	 1.5538969e+00	 1.1574649e-01	 1.6059951e+00	 1.4012797e-01	  1.2265005e+00 	 2.2136021e-01


.. parsed-literal::

     140	 1.5546408e+00	 1.1574009e-01	 1.6067076e+00	 1.4009927e-01	  1.2278524e+00 	 2.1373820e-01


.. parsed-literal::

     141	 1.5561659e+00	 1.1561658e-01	 1.6082224e+00	 1.4005326e-01	  1.2272237e+00 	 2.0622993e-01


.. parsed-literal::

     142	 1.5571713e+00	 1.1550759e-01	 1.6092541e+00	 1.4009205e-01	  1.2265100e+00 	 2.1026802e-01


.. parsed-literal::

     143	 1.5588789e+00	 1.1534304e-01	 1.6110759e+00	 1.4026757e-01	  1.2269362e+00 	 2.1367931e-01


.. parsed-literal::

     144	 1.5601397e+00	 1.1518726e-01	 1.6124098e+00	 1.4036219e-01	  1.2270444e+00 	 2.1279240e-01


.. parsed-literal::

     145	 1.5611815e+00	 1.1523504e-01	 1.6134027e+00	 1.4034606e-01	  1.2262862e+00 	 2.0252848e-01


.. parsed-literal::

     146	 1.5621705e+00	 1.1523827e-01	 1.6143874e+00	 1.4034384e-01	  1.2262952e+00 	 2.2129560e-01


.. parsed-literal::

     147	 1.5634872e+00	 1.1516894e-01	 1.6157091e+00	 1.4034970e-01	  1.2242043e+00 	 2.0612836e-01


.. parsed-literal::

     148	 1.5647838e+00	 1.1500417e-01	 1.6170180e+00	 1.4026118e-01	  1.2269646e+00 	 2.1333170e-01


.. parsed-literal::

     149	 1.5663019e+00	 1.1488782e-01	 1.6185028e+00	 1.4031102e-01	  1.2242305e+00 	 2.0425534e-01


.. parsed-literal::

     150	 1.5671396e+00	 1.1481678e-01	 1.6193126e+00	 1.4027928e-01	  1.2250416e+00 	 2.1139693e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.18 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fac4c9feb30>



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
    CPU times: user 2.15 s, sys: 48.9 ms, total: 2.2 s
    Wall time: 676 ms


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

