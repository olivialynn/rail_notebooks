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
       1	-3.4673637e-01	 3.2117176e-01	-3.3709851e-01	 3.1813481e-01	[-3.3087064e-01]	 4.5933867e-01


.. parsed-literal::

       2	-2.7406709e-01	 3.1025941e-01	-2.4993040e-01	 3.0825554e-01	[-2.4100991e-01]	 2.3554945e-01


.. parsed-literal::

       3	-2.2717776e-01	 2.8818155e-01	-1.8346910e-01	 2.8812323e-01	[-1.7934840e-01]	 2.9248786e-01


.. parsed-literal::

       4	-1.8785884e-01	 2.7074472e-01	-1.3940188e-01	 2.7234258e-01	[-1.4104004e-01]	 3.1732416e-01


.. parsed-literal::

       5	-1.2356900e-01	 2.5854550e-01	-9.2989185e-02	 2.5679209e-01	[-7.1330795e-02]	 2.0575333e-01
       6	-7.6252849e-02	 2.5400380e-01	-4.9210682e-02	 2.4867915e-01	[-2.8398441e-02]	 2.0960522e-01


.. parsed-literal::

       7	-5.9121393e-02	 2.5056794e-01	-3.6028952e-02	 2.4537706e-01	[-1.3158586e-02]	 1.8536139e-01
       8	-4.7533163e-02	 2.4870739e-01	-2.7937945e-02	 2.4273886e-01	[ 1.2052985e-04]	 1.9235587e-01


.. parsed-literal::

       9	-3.5914072e-02	 2.4639992e-01	-1.8475993e-02	 2.4046721e-01	[ 1.1051689e-02]	 2.1643853e-01
      10	-2.5201301e-02	 2.4444776e-01	-9.6674002e-03	 2.3666310e-01	[ 2.6774446e-02]	 1.9821453e-01


.. parsed-literal::

      11	-2.0600083e-02	 2.4352277e-01	-5.6778175e-03	 2.3582578e-01	[ 3.0850206e-02]	 2.2060966e-01
      12	-1.5253648e-02	 2.4261605e-01	-1.0454523e-03	 2.3492777e-01	[ 3.5209739e-02]	 1.8321252e-01


.. parsed-literal::

      13	-9.7640285e-03	 2.4157167e-01	 4.3527729e-03	 2.3480566e-01	[ 3.7557083e-02]	 2.0995903e-01


.. parsed-literal::

      14	 3.8785236e-02	 2.3269120e-01	 5.5347006e-02	 2.2786855e-01	[ 8.7723314e-02]	 3.2962871e-01


.. parsed-literal::

      15	 4.6791713e-02	 2.3024010e-01	 6.6891539e-02	 2.2486025e-01	  8.5436972e-02 	 3.3176208e-01


.. parsed-literal::

      16	 1.1425464e-01	 2.2587293e-01	 1.3659118e-01	 2.1679609e-01	[ 1.5820178e-01]	 2.1801114e-01
      17	 2.3379653e-01	 2.1908569e-01	 2.6559607e-01	 2.1779508e-01	[ 2.8485656e-01]	 1.9414949e-01


.. parsed-literal::

      18	 2.9883397e-01	 2.1237379e-01	 3.3198403e-01	 2.1475283e-01	[ 3.3897402e-01]	 2.1789074e-01


.. parsed-literal::

      19	 3.4326488e-01	 2.0842893e-01	 3.7628083e-01	 2.0551586e-01	[ 3.8319212e-01]	 2.1902752e-01


.. parsed-literal::

      20	 3.9286710e-01	 2.0250228e-01	 4.2641577e-01	 2.0053563e-01	[ 4.2640219e-01]	 2.1066642e-01
      21	 5.0322737e-01	 1.9542531e-01	 5.3843142e-01	 1.9464722e-01	[ 5.2621197e-01]	 1.8927693e-01


.. parsed-literal::

      22	 5.8058503e-01	 1.9317413e-01	 6.1917064e-01	 1.8706745e-01	[ 6.0457721e-01]	 1.9552445e-01


.. parsed-literal::

      23	 6.3324621e-01	 1.9393964e-01	 6.7210442e-01	 1.8908171e-01	[ 6.3720695e-01]	 2.1915603e-01


.. parsed-literal::

      24	 6.7484813e-01	 1.8849628e-01	 7.1397520e-01	 1.8283704e-01	[ 6.7231695e-01]	 2.2079086e-01


.. parsed-literal::

      25	 7.0800661e-01	 1.8628525e-01	 7.4602808e-01	 1.7981878e-01	[ 7.0591168e-01]	 2.1198916e-01


.. parsed-literal::

      26	 7.4579497e-01	 1.8390599e-01	 7.8435097e-01	 1.7752531e-01	[ 7.3355250e-01]	 2.0946860e-01
      27	 7.7514146e-01	 1.8742912e-01	 8.1459908e-01	 1.7788377e-01	[ 7.5939486e-01]	 1.8728876e-01


.. parsed-literal::

      28	 8.0406533e-01	 1.9310645e-01	 8.4319790e-01	 1.8320537e-01	[ 7.9668868e-01]	 1.8240285e-01


.. parsed-literal::

      29	 8.2677389e-01	 1.8920428e-01	 8.6630710e-01	 1.7826000e-01	[ 8.1781945e-01]	 2.0498395e-01


.. parsed-literal::

      30	 8.5847667e-01	 1.8938758e-01	 8.9886888e-01	 1.7752316e-01	[ 8.5448562e-01]	 2.1162200e-01
      31	 8.7976329e-01	 1.8686390e-01	 9.2266080e-01	 1.7722634e-01	[ 8.7407764e-01]	 1.8445158e-01


.. parsed-literal::

      32	 9.1160892e-01	 1.8365478e-01	 9.5345884e-01	 1.7441053e-01	[ 9.1138898e-01]	 2.1451473e-01


.. parsed-literal::

      33	 9.2739063e-01	 1.8099803e-01	 9.6934301e-01	 1.7234190e-01	[ 9.2699586e-01]	 2.1595669e-01


.. parsed-literal::

      34	 9.4712330e-01	 1.7771736e-01	 9.8959443e-01	 1.7062338e-01	[ 9.4524412e-01]	 2.0513463e-01


.. parsed-literal::

      35	 9.6776345e-01	 1.7411971e-01	 1.0106730e+00	 1.6889732e-01	[ 9.6455699e-01]	 2.1074700e-01


.. parsed-literal::

      36	 9.8335568e-01	 1.7233297e-01	 1.0267981e+00	 1.6934297e-01	[ 9.7653829e-01]	 2.1484804e-01
      37	 9.9805475e-01	 1.7061426e-01	 1.0414343e+00	 1.6647171e-01	[ 9.9456333e-01]	 1.7292285e-01


.. parsed-literal::

      38	 1.0206710e+00	 1.6523422e-01	 1.0651610e+00	 1.6048271e-01	[ 1.0179986e+00]	 2.0612979e-01


.. parsed-literal::

      39	 1.0379158e+00	 1.6273893e-01	 1.0831371e+00	 1.5776300e-01	[ 1.0284929e+00]	 2.1428347e-01
      40	 1.0561772e+00	 1.5931730e-01	 1.1028568e+00	 1.5465703e-01	[ 1.0446813e+00]	 1.8272877e-01


.. parsed-literal::

      41	 1.0711410e+00	 1.5737642e-01	 1.1177420e+00	 1.5313378e-01	[ 1.0544416e+00]	 2.1881294e-01
      42	 1.0842953e+00	 1.5599128e-01	 1.1303325e+00	 1.5227616e-01	[ 1.0658938e+00]	 1.8022370e-01


.. parsed-literal::

      43	 1.1019090e+00	 1.5167358e-01	 1.1481522e+00	 1.4894840e-01	[ 1.0749990e+00]	 1.7330050e-01


.. parsed-literal::

      44	 1.1154510e+00	 1.4822618e-01	 1.1616742e+00	 1.4470699e-01	[ 1.0811977e+00]	 2.1454048e-01


.. parsed-literal::

      45	 1.1287845e+00	 1.4714579e-01	 1.1747107e+00	 1.4336700e-01	[ 1.0945264e+00]	 2.0570493e-01


.. parsed-literal::

      46	 1.1423966e+00	 1.4506775e-01	 1.1886441e+00	 1.4101120e-01	[ 1.1019819e+00]	 2.1062255e-01
      47	 1.1556933e+00	 1.4425490e-01	 1.2025456e+00	 1.4000174e-01	[ 1.1036665e+00]	 2.0076108e-01


.. parsed-literal::

      48	 1.1676142e+00	 1.4225808e-01	 1.2151374e+00	 1.3787202e-01	[ 1.1118625e+00]	 1.9048762e-01


.. parsed-literal::

      49	 1.1799092e+00	 1.4147454e-01	 1.2274956e+00	 1.3719939e-01	[ 1.1225288e+00]	 2.1196556e-01


.. parsed-literal::

      50	 1.1907578e+00	 1.4069028e-01	 1.2385461e+00	 1.3680192e-01	[ 1.1287655e+00]	 2.0739317e-01


.. parsed-literal::

      51	 1.2011221e+00	 1.3955470e-01	 1.2492350e+00	 1.3572760e-01	[ 1.1313161e+00]	 2.0956421e-01


.. parsed-literal::

      52	 1.2104369e+00	 1.3858222e-01	 1.2588288e+00	 1.3411027e-01	[ 1.1345532e+00]	 2.0278335e-01
      53	 1.2204311e+00	 1.3824130e-01	 1.2688039e+00	 1.3308910e-01	[ 1.1361550e+00]	 2.0518541e-01


.. parsed-literal::

      54	 1.2354971e+00	 1.3813061e-01	 1.2842443e+00	 1.3217319e-01	  1.1226781e+00 	 2.0000911e-01
      55	 1.2436494e+00	 1.3868112e-01	 1.2928534e+00	 1.3106333e-01	  1.0949625e+00 	 1.8549681e-01


.. parsed-literal::

      56	 1.2523285e+00	 1.3800234e-01	 1.3011392e+00	 1.3107756e-01	  1.0975000e+00 	 1.7954278e-01


.. parsed-literal::

      57	 1.2592835e+00	 1.3723690e-01	 1.3081526e+00	 1.3122845e-01	  1.0969981e+00 	 2.1580791e-01


.. parsed-literal::

      58	 1.2697782e+00	 1.3632345e-01	 1.3189523e+00	 1.3106173e-01	  1.0798909e+00 	 2.1728063e-01


.. parsed-literal::

      59	 1.2816519e+00	 1.3531480e-01	 1.3310701e+00	 1.3005329e-01	  1.0624582e+00 	 2.0515728e-01


.. parsed-literal::

      60	 1.2913357e+00	 1.3489704e-01	 1.3409013e+00	 1.2889230e-01	  1.0376465e+00 	 2.2157192e-01
      61	 1.2977737e+00	 1.3474472e-01	 1.3473548e+00	 1.2814308e-01	  1.0451544e+00 	 1.8710279e-01


.. parsed-literal::

      62	 1.3102340e+00	 1.3423369e-01	 1.3605427e+00	 1.2685997e-01	  1.0378471e+00 	 2.1066785e-01


.. parsed-literal::

      63	 1.3183616e+00	 1.3463253e-01	 1.3689042e+00	 1.2688887e-01	  1.0487626e+00 	 2.0892167e-01


.. parsed-literal::

      64	 1.3265387e+00	 1.3378446e-01	 1.3770164e+00	 1.2638422e-01	  1.0690725e+00 	 2.2069716e-01


.. parsed-literal::

      65	 1.3347082e+00	 1.3302063e-01	 1.3853868e+00	 1.2606940e-01	  1.0846341e+00 	 2.0134997e-01
      66	 1.3421707e+00	 1.3257763e-01	 1.3928782e+00	 1.2545770e-01	  1.1054803e+00 	 1.9269753e-01


.. parsed-literal::

      67	 1.3507716e+00	 1.3239527e-01	 1.4018801e+00	 1.2542303e-01	[ 1.1382252e+00]	 1.9730735e-01
      68	 1.3593110e+00	 1.3232567e-01	 1.4101900e+00	 1.2458292e-01	[ 1.1496708e+00]	 1.9669580e-01


.. parsed-literal::

      69	 1.3634257e+00	 1.3210206e-01	 1.4141911e+00	 1.2452186e-01	  1.1466116e+00 	 1.8265128e-01


.. parsed-literal::

      70	 1.3709318e+00	 1.3190367e-01	 1.4219720e+00	 1.2447638e-01	  1.1306495e+00 	 2.0449138e-01
      71	 1.3773327e+00	 1.3172237e-01	 1.4287371e+00	 1.2475296e-01	  1.1088247e+00 	 1.8282628e-01


.. parsed-literal::

      72	 1.3836816e+00	 1.3160778e-01	 1.4351784e+00	 1.2488510e-01	  1.1106850e+00 	 2.1746135e-01


.. parsed-literal::

      73	 1.3890340e+00	 1.3142546e-01	 1.4406947e+00	 1.2492411e-01	  1.1093082e+00 	 2.1716237e-01


.. parsed-literal::

      74	 1.3943450e+00	 1.3113124e-01	 1.4462052e+00	 1.2492397e-01	  1.1090501e+00 	 2.2061801e-01
      75	 1.3993644e+00	 1.3070249e-01	 1.4516133e+00	 1.2496521e-01	  1.0924624e+00 	 2.0438671e-01


.. parsed-literal::

      76	 1.4054758e+00	 1.3015999e-01	 1.4575126e+00	 1.2451130e-01	  1.0998885e+00 	 2.1398640e-01
      77	 1.4090568e+00	 1.2972748e-01	 1.4610192e+00	 1.2413543e-01	  1.1021193e+00 	 1.9452858e-01


.. parsed-literal::

      78	 1.4137038e+00	 1.2922161e-01	 1.4655621e+00	 1.2352457e-01	  1.1082667e+00 	 2.1079922e-01


.. parsed-literal::

      79	 1.4188097e+00	 1.2864677e-01	 1.4707247e+00	 1.2301451e-01	  1.1059918e+00 	 2.1271920e-01
      80	 1.4237170e+00	 1.2861621e-01	 1.4756064e+00	 1.2257338e-01	  1.1125159e+00 	 2.0465112e-01


.. parsed-literal::

      81	 1.4269705e+00	 1.2855511e-01	 1.4788101e+00	 1.2218353e-01	  1.1217804e+00 	 2.0170879e-01


.. parsed-literal::

      82	 1.4317516e+00	 1.2847119e-01	 1.4839257e+00	 1.2189667e-01	  1.1258573e+00 	 2.0715475e-01


.. parsed-literal::

      83	 1.4348598e+00	 1.2800612e-01	 1.4872882e+00	 1.2106329e-01	  1.1318689e+00 	 2.0148540e-01
      84	 1.4381518e+00	 1.2781602e-01	 1.4905357e+00	 1.2090613e-01	  1.1327585e+00 	 1.7474151e-01


.. parsed-literal::

      85	 1.4425541e+00	 1.2730260e-01	 1.4949980e+00	 1.2061907e-01	  1.1336919e+00 	 1.7654657e-01


.. parsed-literal::

      86	 1.4458214e+00	 1.2689304e-01	 1.4982379e+00	 1.2050578e-01	  1.1332048e+00 	 2.1305656e-01
      87	 1.4491310e+00	 1.2602064e-01	 1.5017487e+00	 1.2055291e-01	  1.1278030e+00 	 1.8977904e-01


.. parsed-literal::

      88	 1.4536306e+00	 1.2586977e-01	 1.5062568e+00	 1.2042933e-01	  1.1211507e+00 	 2.1579480e-01


.. parsed-literal::

      89	 1.4558057e+00	 1.2584572e-01	 1.5084275e+00	 1.2040415e-01	  1.1174991e+00 	 2.1056342e-01


.. parsed-literal::

      90	 1.4604853e+00	 1.2564928e-01	 1.5132910e+00	 1.2045166e-01	  1.1032781e+00 	 2.1370769e-01


.. parsed-literal::

      91	 1.4628885e+00	 1.2555637e-01	 1.5158502e+00	 1.2074369e-01	  1.0808834e+00 	 2.1473813e-01


.. parsed-literal::

      92	 1.4666502e+00	 1.2545547e-01	 1.5195417e+00	 1.2068843e-01	  1.0845345e+00 	 2.0185399e-01


.. parsed-literal::

      93	 1.4685628e+00	 1.2528687e-01	 1.5213936e+00	 1.2048601e-01	  1.0900032e+00 	 2.0405579e-01


.. parsed-literal::

      94	 1.4715004e+00	 1.2513414e-01	 1.5243941e+00	 1.2021830e-01	  1.0857154e+00 	 2.0574307e-01


.. parsed-literal::

      95	 1.4728495e+00	 1.2482150e-01	 1.5260174e+00	 1.1979236e-01	  1.0776336e+00 	 2.0465612e-01


.. parsed-literal::

      96	 1.4758278e+00	 1.2484322e-01	 1.5289108e+00	 1.1973849e-01	  1.0787035e+00 	 2.0645475e-01
      97	 1.4773212e+00	 1.2481463e-01	 1.5304687e+00	 1.1965576e-01	  1.0738298e+00 	 2.0068693e-01


.. parsed-literal::

      98	 1.4794810e+00	 1.2470932e-01	 1.5327524e+00	 1.1942280e-01	  1.0698342e+00 	 2.0881486e-01
      99	 1.4823990e+00	 1.2444557e-01	 1.5358362e+00	 1.1904850e-01	  1.0730942e+00 	 1.7614698e-01


.. parsed-literal::

     100	 1.4850677e+00	 1.2422795e-01	 1.5386315e+00	 1.1875129e-01	  1.0716567e+00 	 2.0225763e-01


.. parsed-literal::

     101	 1.4868017e+00	 1.2415144e-01	 1.5402864e+00	 1.1870364e-01	  1.0798059e+00 	 2.0673680e-01


.. parsed-literal::

     102	 1.4889157e+00	 1.2406774e-01	 1.5423672e+00	 1.1875177e-01	  1.0870234e+00 	 2.0937800e-01


.. parsed-literal::

     103	 1.4910830e+00	 1.2400693e-01	 1.5445455e+00	 1.1876433e-01	  1.0909802e+00 	 2.1920443e-01


.. parsed-literal::

     104	 1.4930280e+00	 1.2396716e-01	 1.5464922e+00	 1.1878330e-01	  1.0967845e+00 	 3.2863259e-01


.. parsed-literal::

     105	 1.4953842e+00	 1.2392239e-01	 1.5488718e+00	 1.1872353e-01	  1.0976323e+00 	 2.1122789e-01
     106	 1.4968623e+00	 1.2387511e-01	 1.5503597e+00	 1.1858766e-01	  1.0970412e+00 	 1.9526482e-01


.. parsed-literal::

     107	 1.4993250e+00	 1.2384167e-01	 1.5528492e+00	 1.1840362e-01	  1.0957394e+00 	 2.1710849e-01


.. parsed-literal::

     108	 1.5008884e+00	 1.2373113e-01	 1.5545220e+00	 1.1821226e-01	  1.0928796e+00 	 2.1159744e-01
     109	 1.5030316e+00	 1.2374397e-01	 1.5565823e+00	 1.1819636e-01	  1.0977760e+00 	 1.9763970e-01


.. parsed-literal::

     110	 1.5051881e+00	 1.2376406e-01	 1.5587093e+00	 1.1825329e-01	  1.1031657e+00 	 2.1440864e-01
     111	 1.5067469e+00	 1.2382489e-01	 1.5602797e+00	 1.1828776e-01	  1.1053291e+00 	 1.8806434e-01


.. parsed-literal::

     112	 1.5083999e+00	 1.2381538e-01	 1.5620085e+00	 1.1830345e-01	  1.1087145e+00 	 3.0548358e-01


.. parsed-literal::

     113	 1.5104969e+00	 1.2397664e-01	 1.5641577e+00	 1.1836030e-01	  1.1107393e+00 	 2.1542978e-01


.. parsed-literal::

     114	 1.5117611e+00	 1.2398269e-01	 1.5654486e+00	 1.1837603e-01	  1.1099072e+00 	 2.1354938e-01


.. parsed-literal::

     115	 1.5136015e+00	 1.2390462e-01	 1.5673744e+00	 1.1834625e-01	  1.1087033e+00 	 2.1206903e-01


.. parsed-literal::

     116	 1.5142675e+00	 1.2378170e-01	 1.5681177e+00	 1.1860963e-01	  1.1015771e+00 	 2.0939970e-01


.. parsed-literal::

     117	 1.5157653e+00	 1.2369931e-01	 1.5695500e+00	 1.1849140e-01	  1.1046428e+00 	 2.0433569e-01
     118	 1.5167787e+00	 1.2360206e-01	 1.5705652e+00	 1.1850371e-01	  1.1051154e+00 	 1.8858719e-01


.. parsed-literal::

     119	 1.5178275e+00	 1.2347602e-01	 1.5716596e+00	 1.1859762e-01	  1.1025983e+00 	 2.0893097e-01


.. parsed-literal::

     120	 1.5199535e+00	 1.2328153e-01	 1.5738991e+00	 1.1882968e-01	  1.0952634e+00 	 2.1916175e-01


.. parsed-literal::

     121	 1.5212054e+00	 1.2309595e-01	 1.5752903e+00	 1.1910797e-01	  1.0856662e+00 	 3.1905508e-01


.. parsed-literal::

     122	 1.5228925e+00	 1.2306223e-01	 1.5770703e+00	 1.1937279e-01	  1.0794379e+00 	 2.1514082e-01


.. parsed-literal::

     123	 1.5238873e+00	 1.2304664e-01	 1.5781084e+00	 1.1938889e-01	  1.0763477e+00 	 2.1199346e-01
     124	 1.5255582e+00	 1.2298553e-01	 1.5798467e+00	 1.1940351e-01	  1.0693039e+00 	 1.7453885e-01


.. parsed-literal::

     125	 1.5262176e+00	 1.2286520e-01	 1.5806781e+00	 1.1940895e-01	  1.0544802e+00 	 1.9063282e-01


.. parsed-literal::

     126	 1.5281833e+00	 1.2279841e-01	 1.5825448e+00	 1.1934312e-01	  1.0547978e+00 	 2.1283984e-01
     127	 1.5292906e+00	 1.2269947e-01	 1.5836271e+00	 1.1932730e-01	  1.0528453e+00 	 1.8891311e-01


.. parsed-literal::

     128	 1.5306074e+00	 1.2258508e-01	 1.5849475e+00	 1.1926502e-01	  1.0484680e+00 	 2.0872617e-01
     129	 1.5317046e+00	 1.2244786e-01	 1.5861516e+00	 1.1936568e-01	  1.0362369e+00 	 1.8957686e-01


.. parsed-literal::

     130	 1.5333515e+00	 1.2243715e-01	 1.5877324e+00	 1.1922247e-01	  1.0362139e+00 	 2.2129965e-01
     131	 1.5343474e+00	 1.2241024e-01	 1.5887823e+00	 1.1930830e-01	  1.0299155e+00 	 2.0360208e-01


.. parsed-literal::

     132	 1.5354801e+00	 1.2234857e-01	 1.5899688e+00	 1.1937187e-01	  1.0228317e+00 	 1.7856336e-01
     133	 1.5366188e+00	 1.2222704e-01	 1.5912013e+00	 1.1933842e-01	  1.0108278e+00 	 1.9898677e-01


.. parsed-literal::

     134	 1.5377220e+00	 1.2212907e-01	 1.5923229e+00	 1.1937984e-01	  1.0016062e+00 	 1.7527509e-01
     135	 1.5387717e+00	 1.2197521e-01	 1.5933906e+00	 1.1929646e-01	  9.9715320e-01 	 1.7040992e-01


.. parsed-literal::

     136	 1.5398266e+00	 1.2186370e-01	 1.5944839e+00	 1.1927664e-01	  9.9270112e-01 	 1.8200064e-01


.. parsed-literal::

     137	 1.5410545e+00	 1.2172655e-01	 1.5957266e+00	 1.1926109e-01	  9.9040608e-01 	 2.1104503e-01


.. parsed-literal::

     138	 1.5421414e+00	 1.2166049e-01	 1.5968158e+00	 1.1930629e-01	  9.8902804e-01 	 2.1719933e-01
     139	 1.5433080e+00	 1.2163251e-01	 1.5980275e+00	 1.1947832e-01	  9.8261759e-01 	 1.8192649e-01


.. parsed-literal::

     140	 1.5443234e+00	 1.2158293e-01	 1.5990906e+00	 1.1955305e-01	  9.7400469e-01 	 2.1247983e-01


.. parsed-literal::

     141	 1.5453737e+00	 1.2153541e-01	 1.6001519e+00	 1.1955821e-01	  9.7116838e-01 	 2.1567106e-01


.. parsed-literal::

     142	 1.5467463e+00	 1.2148863e-01	 1.6015877e+00	 1.1949698e-01	  9.6357875e-01 	 2.1241069e-01


.. parsed-literal::

     143	 1.5475611e+00	 1.2139803e-01	 1.6024385e+00	 1.1931368e-01	  9.6802811e-01 	 2.1054149e-01


.. parsed-literal::

     144	 1.5485106e+00	 1.2139647e-01	 1.6033391e+00	 1.1921287e-01	  9.7219634e-01 	 2.1189213e-01


.. parsed-literal::

     145	 1.5493715e+00	 1.2137201e-01	 1.6041838e+00	 1.1906341e-01	  9.7709290e-01 	 2.1159601e-01


.. parsed-literal::

     146	 1.5499829e+00	 1.2135106e-01	 1.6047875e+00	 1.1898387e-01	  9.8001860e-01 	 2.1783662e-01
     147	 1.5512552e+00	 1.2128020e-01	 1.6060928e+00	 1.1874142e-01	  9.8513169e-01 	 1.7734551e-01


.. parsed-literal::

     148	 1.5524167e+00	 1.2123378e-01	 1.6072557e+00	 1.1876448e-01	  9.8709538e-01 	 2.1508479e-01
     149	 1.5531193e+00	 1.2119254e-01	 1.6079547e+00	 1.1878179e-01	  9.8346261e-01 	 1.8837786e-01


.. parsed-literal::

     150	 1.5540697e+00	 1.2109788e-01	 1.6089740e+00	 1.1877870e-01	  9.7480331e-01 	 2.1919107e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.13 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fef10465420>



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
    CPU times: user 1.78 s, sys: 56.9 ms, total: 1.84 s
    Wall time: 603 ms


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

