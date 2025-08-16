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
       1	-3.4570294e-01	 3.2185162e-01	-3.3592117e-01	 3.1660834e-01	[-3.2412614e-01]	 4.5428252e-01


.. parsed-literal::

       2	-2.7940847e-01	 3.1248182e-01	-2.5643171e-01	 3.0584808e-01	[-2.3582869e-01]	 2.3189354e-01


.. parsed-literal::

       3	-2.3605732e-01	 2.9146021e-01	-1.9469996e-01	 2.8602000e-01	[-1.6772640e-01]	 2.8815675e-01
       4	-1.9322323e-01	 2.6871572e-01	-1.4939103e-01	 2.6538353e-01	[-1.2367851e-01]	 1.9101906e-01


.. parsed-literal::

       5	-1.0794087e-01	 2.5790364e-01	-7.6293962e-02	 2.5511487e-01	[-6.2023834e-02]	 1.8875742e-01


.. parsed-literal::

       6	-7.4282824e-02	 2.5302400e-01	-4.5890108e-02	 2.4964395e-01	[-3.4656247e-02]	 2.1911144e-01


.. parsed-literal::

       7	-5.7081780e-02	 2.5005463e-01	-3.3139324e-02	 2.4716714e-01	[-2.2887975e-02]	 2.0724463e-01
       8	-4.3157186e-02	 2.4768399e-01	-2.3171884e-02	 2.4445334e-01	[-1.0960128e-02]	 1.9765425e-01


.. parsed-literal::

       9	-3.1008912e-02	 2.4539659e-01	-1.3394039e-02	 2.4177131e-01	[ 1.1124217e-06]	 2.0078182e-01


.. parsed-literal::

      10	-2.3143939e-02	 2.4314073e-01	-7.9252346e-03	 2.3978178e-01	[ 6.5909634e-03]	 2.0810103e-01


.. parsed-literal::

      11	-1.3310425e-02	 2.4217060e-01	 1.1647081e-03	 2.3816555e-01	[ 1.6493967e-02]	 2.0532799e-01


.. parsed-literal::

      12	-1.0587407e-02	 2.4166034e-01	 3.6029902e-03	 2.3769294e-01	[ 1.8516686e-02]	 2.1490359e-01


.. parsed-literal::

      13	-4.5132284e-03	 2.4059128e-01	 9.3879092e-03	 2.3659754e-01	[ 2.4101403e-02]	 2.1373534e-01


.. parsed-literal::

      14	 8.3589498e-02	 2.2869729e-01	 1.0096592e-01	 2.2455730e-01	[ 1.0945929e-01]	 3.1572318e-01


.. parsed-literal::

      15	 1.2812343e-01	 2.2001233e-01	 1.5024725e-01	 2.1427197e-01	[ 1.5847265e-01]	 3.2300520e-01
      16	 1.7217148e-01	 2.1797228e-01	 1.9409031e-01	 2.1285891e-01	[ 2.0084196e-01]	 1.8345189e-01


.. parsed-literal::

      17	 2.7431591e-01	 2.1474959e-01	 3.0010421e-01	 2.1064511e-01	[ 3.0680609e-01]	 2.0679784e-01


.. parsed-literal::

      18	 3.2521845e-01	 2.1001534e-01	 3.5497560e-01	 2.0751931e-01	[ 3.5188472e-01]	 2.0875692e-01


.. parsed-literal::

      19	 3.8400680e-01	 2.0544068e-01	 4.1602615e-01	 2.0502182e-01	[ 3.9738890e-01]	 2.1998215e-01
      20	 4.7520698e-01	 2.0261784e-01	 5.0867261e-01	 2.0090207e-01	[ 4.9016659e-01]	 2.0481277e-01


.. parsed-literal::

      21	 5.3360999e-01	 2.0698444e-01	 5.6752095e-01	 2.0268027e-01	[ 5.5240191e-01]	 1.9662690e-01


.. parsed-literal::

      22	 6.0171155e-01	 2.0003256e-01	 6.3667352e-01	 1.9644274e-01	[ 6.1748586e-01]	 2.0886755e-01


.. parsed-literal::

      23	 6.4920857e-01	 1.9635461e-01	 6.8729787e-01	 1.9097062e-01	[ 6.4677534e-01]	 2.0705271e-01


.. parsed-literal::

      24	 6.9190010e-01	 1.9093438e-01	 7.2998397e-01	 1.8889372e-01	[ 6.7516478e-01]	 2.1587133e-01


.. parsed-literal::

      25	 7.0999334e-01	 1.8791359e-01	 7.4737024e-01	 1.8593112e-01	[ 6.9833647e-01]	 2.1330976e-01


.. parsed-literal::

      26	 7.2871716e-01	 1.8681008e-01	 7.6570386e-01	 1.8350603e-01	[ 7.2046947e-01]	 2.1049500e-01


.. parsed-literal::

      27	 7.5267603e-01	 1.8913084e-01	 7.8946595e-01	 1.8053259e-01	[ 7.6026462e-01]	 2.0699215e-01


.. parsed-literal::

      28	 7.7321618e-01	 1.8817859e-01	 8.0996444e-01	 1.8099249e-01	[ 7.8084883e-01]	 2.0826054e-01


.. parsed-literal::

      29	 7.9722504e-01	 1.8336566e-01	 8.3494937e-01	 1.7865398e-01	[ 8.0377289e-01]	 2.1362758e-01


.. parsed-literal::

      30	 8.3042858e-01	 1.7776639e-01	 8.6882228e-01	 1.7557765e-01	[ 8.3810251e-01]	 2.0585370e-01


.. parsed-literal::

      31	 8.4799469e-01	 1.7517263e-01	 8.8719627e-01	 1.7322481e-01	[ 8.5315853e-01]	 2.1722627e-01
      32	 8.7311836e-01	 1.7315548e-01	 9.1201909e-01	 1.7269658e-01	[ 8.7580665e-01]	 1.9599915e-01


.. parsed-literal::

      33	 8.9144433e-01	 1.7224002e-01	 9.3032332e-01	 1.7149259e-01	[ 8.9493437e-01]	 2.0254302e-01


.. parsed-literal::

      34	 9.1226664e-01	 1.7019377e-01	 9.5216103e-01	 1.7106954e-01	[ 9.1606387e-01]	 2.1273494e-01


.. parsed-literal::

      35	 9.2465870e-01	 1.7044069e-01	 9.6525415e-01	 1.6914161e-01	[ 9.3162666e-01]	 2.1486497e-01
      36	 9.3767038e-01	 1.6830410e-01	 9.7867972e-01	 1.6962068e-01	[ 9.4334155e-01]	 1.9913530e-01


.. parsed-literal::

      37	 9.4917862e-01	 1.6618527e-01	 9.9056476e-01	 1.6915354e-01	[ 9.6132589e-01]	 2.0372319e-01
      38	 9.6035710e-01	 1.6474505e-01	 1.0020592e+00	 1.6731945e-01	[ 9.7323756e-01]	 1.7309046e-01


.. parsed-literal::

      39	 9.7923938e-01	 1.6249158e-01	 1.0215411e+00	 1.6180702e-01	[ 9.9250256e-01]	 2.0969272e-01


.. parsed-literal::

      40	 9.9419038e-01	 1.6142596e-01	 1.0368231e+00	 1.6002044e-01	[ 1.0113451e+00]	 2.0628619e-01


.. parsed-literal::

      41	 1.0092585e+00	 1.6007284e-01	 1.0522096e+00	 1.5724158e-01	[ 1.0277948e+00]	 2.0533228e-01


.. parsed-literal::

      42	 1.0194914e+00	 1.5928971e-01	 1.0627267e+00	 1.5584073e-01	[ 1.0401568e+00]	 2.1389270e-01


.. parsed-literal::

      43	 1.0289265e+00	 1.5814740e-01	 1.0724366e+00	 1.5531525e-01	[ 1.0474745e+00]	 2.1038246e-01


.. parsed-literal::

      44	 1.0434518e+00	 1.5622653e-01	 1.0873843e+00	 1.5493393e-01	[ 1.0587773e+00]	 2.0549035e-01


.. parsed-literal::

      45	 1.0572577e+00	 1.5495323e-01	 1.1016221e+00	 1.5383472e-01	[ 1.0650137e+00]	 2.0895576e-01


.. parsed-literal::

      46	 1.0674910e+00	 1.5443849e-01	 1.1118396e+00	 1.5310541e-01	[ 1.0769749e+00]	 2.0711923e-01


.. parsed-literal::

      47	 1.0772551e+00	 1.5345665e-01	 1.1219060e+00	 1.5195964e-01	[ 1.0884515e+00]	 2.1387720e-01
      48	 1.0869824e+00	 1.5267964e-01	 1.1321011e+00	 1.5100778e-01	[ 1.0980890e+00]	 1.9557357e-01


.. parsed-literal::

      49	 1.0952937e+00	 1.5111385e-01	 1.1409895e+00	 1.5053691e-01	[ 1.1064045e+00]	 2.1023631e-01
      50	 1.1025071e+00	 1.5082280e-01	 1.1482004e+00	 1.4983962e-01	[ 1.1131554e+00]	 1.7630053e-01


.. parsed-literal::

      51	 1.1101633e+00	 1.5041249e-01	 1.1559185e+00	 1.4910685e-01	[ 1.1196128e+00]	 2.0727444e-01
      52	 1.1173892e+00	 1.4958121e-01	 1.1633251e+00	 1.4836307e-01	[ 1.1262626e+00]	 2.0413995e-01


.. parsed-literal::

      53	 1.1268201e+00	 1.4813267e-01	 1.1733182e+00	 1.4699468e-01	[ 1.1356325e+00]	 2.0316243e-01


.. parsed-literal::

      54	 1.1340991e+00	 1.4708033e-01	 1.1808885e+00	 1.4566685e-01	[ 1.1450717e+00]	 2.0272350e-01


.. parsed-literal::

      55	 1.1398604e+00	 1.4676982e-01	 1.1864901e+00	 1.4539613e-01	[ 1.1505157e+00]	 2.0175481e-01


.. parsed-literal::

      56	 1.1485890e+00	 1.4599763e-01	 1.1954147e+00	 1.4485598e-01	[ 1.1572101e+00]	 2.0983005e-01
      57	 1.1580827e+00	 1.4517176e-01	 1.2050014e+00	 1.4400800e-01	[ 1.1651635e+00]	 1.7824268e-01


.. parsed-literal::

      58	 1.1642371e+00	 1.4330218e-01	 1.2119219e+00	 1.4236069e-01	[ 1.1681514e+00]	 2.0995164e-01
      59	 1.1772929e+00	 1.4295440e-01	 1.2246214e+00	 1.4201553e-01	[ 1.1778905e+00]	 2.0003748e-01


.. parsed-literal::

      60	 1.1835722e+00	 1.4262787e-01	 1.2309016e+00	 1.4194170e-01	[ 1.1821674e+00]	 2.0779300e-01


.. parsed-literal::

      61	 1.1955112e+00	 1.4155988e-01	 1.2430538e+00	 1.4107860e-01	[ 1.1917820e+00]	 2.1617246e-01


.. parsed-literal::

      62	 1.2063921e+00	 1.4081879e-01	 1.2544500e+00	 1.4064310e-01	[ 1.2006133e+00]	 2.1036124e-01


.. parsed-literal::

      63	 1.2170267e+00	 1.4008004e-01	 1.2651143e+00	 1.3971577e-01	[ 1.2137130e+00]	 2.1424723e-01
      64	 1.2231163e+00	 1.3965952e-01	 1.2712715e+00	 1.3936285e-01	[ 1.2186989e+00]	 1.8972588e-01


.. parsed-literal::

      65	 1.2331557e+00	 1.3926136e-01	 1.2816036e+00	 1.3863199e-01	[ 1.2257879e+00]	 1.9961238e-01


.. parsed-literal::

      66	 1.2389973e+00	 1.3884489e-01	 1.2877153e+00	 1.3871274e-01	[ 1.2265101e+00]	 2.0885253e-01
      67	 1.2474195e+00	 1.3875775e-01	 1.2960234e+00	 1.3888446e-01	[ 1.2349469e+00]	 1.8613815e-01


.. parsed-literal::

      68	 1.2549414e+00	 1.3872818e-01	 1.3037329e+00	 1.3948704e-01	[ 1.2408365e+00]	 2.1418595e-01


.. parsed-literal::

      69	 1.2625523e+00	 1.3856963e-01	 1.3116438e+00	 1.3989786e-01	[ 1.2475716e+00]	 2.1376324e-01


.. parsed-literal::

      70	 1.2690416e+00	 1.3872228e-01	 1.3189757e+00	 1.4210220e-01	  1.2435878e+00 	 2.1064448e-01
      71	 1.2813288e+00	 1.3817658e-01	 1.3311495e+00	 1.4126857e-01	[ 1.2595723e+00]	 1.7450142e-01


.. parsed-literal::

      72	 1.2877526e+00	 1.3777815e-01	 1.3376227e+00	 1.4061171e-01	[ 1.2655102e+00]	 2.1716809e-01


.. parsed-literal::

      73	 1.2963649e+00	 1.3735655e-01	 1.3464092e+00	 1.4014649e-01	[ 1.2704042e+00]	 2.0780873e-01


.. parsed-literal::

      74	 1.3062915e+00	 1.3707799e-01	 1.3567013e+00	 1.4011910e-01	[ 1.2708356e+00]	 2.0630765e-01


.. parsed-literal::

      75	 1.3095741e+00	 1.3673826e-01	 1.3605120e+00	 1.4110590e-01	  1.2623854e+00 	 2.0680857e-01
      76	 1.3189531e+00	 1.3638316e-01	 1.3695007e+00	 1.4066894e-01	[ 1.2743540e+00]	 1.7760849e-01


.. parsed-literal::

      77	 1.3230588e+00	 1.3626968e-01	 1.3736806e+00	 1.4107908e-01	[ 1.2778364e+00]	 2.1918869e-01
      78	 1.3297839e+00	 1.3580921e-01	 1.3806078e+00	 1.4164535e-01	[ 1.2814175e+00]	 1.7496705e-01


.. parsed-literal::

      79	 1.3354929e+00	 1.3535963e-01	 1.3866087e+00	 1.4322523e-01	  1.2804313e+00 	 2.0590663e-01
      80	 1.3426832e+00	 1.3496512e-01	 1.3939338e+00	 1.4322954e-01	  1.2813959e+00 	 2.0587206e-01


.. parsed-literal::

      81	 1.3490697e+00	 1.3448735e-01	 1.4004462e+00	 1.4286124e-01	[ 1.2815496e+00]	 2.1004725e-01


.. parsed-literal::

      82	 1.3554612e+00	 1.3419159e-01	 1.4070080e+00	 1.4274493e-01	[ 1.2829578e+00]	 2.0948935e-01


.. parsed-literal::

      83	 1.3596971e+00	 1.3370674e-01	 1.4115463e+00	 1.4253155e-01	[ 1.2858303e+00]	 2.0089960e-01
      84	 1.3679012e+00	 1.3377100e-01	 1.4196126e+00	 1.4265890e-01	[ 1.2948595e+00]	 1.8759084e-01


.. parsed-literal::

      85	 1.3714770e+00	 1.3379090e-01	 1.4231437e+00	 1.4273597e-01	[ 1.2994196e+00]	 2.1036315e-01


.. parsed-literal::

      86	 1.3768121e+00	 1.3388482e-01	 1.4286988e+00	 1.4299819e-01	[ 1.3017693e+00]	 2.0217681e-01
      87	 1.3832448e+00	 1.3398128e-01	 1.4352889e+00	 1.4307532e-01	[ 1.3078397e+00]	 2.0286536e-01


.. parsed-literal::

      88	 1.3898719e+00	 1.3369341e-01	 1.4420144e+00	 1.4255444e-01	  1.3071183e+00 	 2.0812845e-01
      89	 1.3966760e+00	 1.3345960e-01	 1.4491164e+00	 1.4213159e-01	  1.3011169e+00 	 1.8780971e-01


.. parsed-literal::

      90	 1.4019227e+00	 1.3272067e-01	 1.4544341e+00	 1.4117378e-01	  1.2967858e+00 	 2.0755887e-01
      91	 1.4073302e+00	 1.3207594e-01	 1.4598636e+00	 1.4032566e-01	  1.2952620e+00 	 1.8223596e-01


.. parsed-literal::

      92	 1.4120412e+00	 1.3160155e-01	 1.4646471e+00	 1.3991827e-01	  1.2958756e+00 	 2.0870876e-01


.. parsed-literal::

      93	 1.4165382e+00	 1.3077927e-01	 1.4692893e+00	 1.3916245e-01	  1.2961550e+00 	 2.1972847e-01


.. parsed-literal::

      94	 1.4207744e+00	 1.3039883e-01	 1.4735199e+00	 1.3882917e-01	  1.2987887e+00 	 2.1980190e-01


.. parsed-literal::

      95	 1.4237866e+00	 1.3030159e-01	 1.4766056e+00	 1.3873775e-01	  1.2989590e+00 	 2.2107124e-01


.. parsed-literal::

      96	 1.4291075e+00	 1.2967152e-01	 1.4820584e+00	 1.3792828e-01	  1.2977905e+00 	 2.1174598e-01


.. parsed-literal::

      97	 1.4317886e+00	 1.2941166e-01	 1.4849468e+00	 1.3745951e-01	  1.2921249e+00 	 2.1102524e-01


.. parsed-literal::

      98	 1.4353571e+00	 1.2913381e-01	 1.4883768e+00	 1.3709733e-01	  1.2970688e+00 	 2.1268821e-01
      99	 1.4381939e+00	 1.2875685e-01	 1.4912693e+00	 1.3660992e-01	  1.2960071e+00 	 1.8857861e-01


.. parsed-literal::

     100	 1.4409802e+00	 1.2850569e-01	 1.4941102e+00	 1.3617251e-01	  1.3002842e+00 	 1.7667818e-01
     101	 1.4429286e+00	 1.2810238e-01	 1.4962577e+00	 1.3554100e-01	  1.2958452e+00 	 1.8679643e-01


.. parsed-literal::

     102	 1.4461816e+00	 1.2799804e-01	 1.4994648e+00	 1.3551054e-01	  1.3046188e+00 	 1.7255640e-01


.. parsed-literal::

     103	 1.4475696e+00	 1.2793567e-01	 1.5008472e+00	 1.3551887e-01	  1.3071026e+00 	 2.0965528e-01


.. parsed-literal::

     104	 1.4501507e+00	 1.2769840e-01	 1.5034975e+00	 1.3539361e-01	[ 1.3083759e+00]	 2.0652962e-01


.. parsed-literal::

     105	 1.4539134e+00	 1.2719468e-01	 1.5073502e+00	 1.3504192e-01	  1.3074308e+00 	 2.2309661e-01


.. parsed-literal::

     106	 1.4561420e+00	 1.2679103e-01	 1.5097212e+00	 1.3464542e-01	  1.3057004e+00 	 3.1315637e-01


.. parsed-literal::

     107	 1.4585927e+00	 1.2647184e-01	 1.5121781e+00	 1.3431135e-01	  1.3021274e+00 	 2.1090770e-01
     108	 1.4608807e+00	 1.2613762e-01	 1.5144816e+00	 1.3389034e-01	  1.2967397e+00 	 2.0088911e-01


.. parsed-literal::

     109	 1.4627676e+00	 1.2578058e-01	 1.5164363e+00	 1.3328932e-01	  1.2873993e+00 	 1.8608189e-01
     110	 1.4649936e+00	 1.2550898e-01	 1.5186541e+00	 1.3297612e-01	  1.2855349e+00 	 1.7563343e-01


.. parsed-literal::

     111	 1.4672110e+00	 1.2516869e-01	 1.5209067e+00	 1.3262250e-01	  1.2847883e+00 	 2.0981479e-01


.. parsed-literal::

     112	 1.4693157e+00	 1.2490260e-01	 1.5229934e+00	 1.3235976e-01	  1.2844774e+00 	 2.0456314e-01


.. parsed-literal::

     113	 1.4711568e+00	 1.2441939e-01	 1.5249445e+00	 1.3192301e-01	  1.2810986e+00 	 2.0360756e-01
     114	 1.4743874e+00	 1.2429826e-01	 1.5280515e+00	 1.3184604e-01	  1.2819047e+00 	 1.9930625e-01


.. parsed-literal::

     115	 1.4758099e+00	 1.2417117e-01	 1.5294631e+00	 1.3178615e-01	  1.2810871e+00 	 2.0703793e-01


.. parsed-literal::

     116	 1.4780343e+00	 1.2400972e-01	 1.5317269e+00	 1.3169117e-01	  1.2793641e+00 	 2.0522833e-01


.. parsed-literal::

     117	 1.4795071e+00	 1.2373672e-01	 1.5333911e+00	 1.3145010e-01	  1.2793004e+00 	 2.1105957e-01


.. parsed-literal::

     118	 1.4821507e+00	 1.2378366e-01	 1.5359668e+00	 1.3145310e-01	  1.2804643e+00 	 2.0082951e-01
     119	 1.4837242e+00	 1.2375529e-01	 1.5375655e+00	 1.3137737e-01	  1.2785514e+00 	 1.7442632e-01


.. parsed-literal::

     120	 1.4853082e+00	 1.2370559e-01	 1.5391948e+00	 1.3127872e-01	  1.2774180e+00 	 1.9006205e-01
     121	 1.4862041e+00	 1.2379199e-01	 1.5402805e+00	 1.3114932e-01	  1.2652752e+00 	 1.8497133e-01


.. parsed-literal::

     122	 1.4885621e+00	 1.2362843e-01	 1.5425608e+00	 1.3104336e-01	  1.2715559e+00 	 1.9807458e-01


.. parsed-literal::

     123	 1.4898199e+00	 1.2354867e-01	 1.5438166e+00	 1.3096195e-01	  1.2723241e+00 	 2.0649958e-01


.. parsed-literal::

     124	 1.4913490e+00	 1.2349698e-01	 1.5454148e+00	 1.3087044e-01	  1.2705052e+00 	 2.0198607e-01
     125	 1.4937697e+00	 1.2346674e-01	 1.5479571e+00	 1.3078312e-01	  1.2662582e+00 	 1.9264078e-01


.. parsed-literal::

     126	 1.4950569e+00	 1.2354854e-01	 1.5495358e+00	 1.3090157e-01	  1.2597944e+00 	 2.4346781e-01


.. parsed-literal::

     127	 1.4970948e+00	 1.2352045e-01	 1.5514227e+00	 1.3088642e-01	  1.2616335e+00 	 2.2982430e-01
     128	 1.4978832e+00	 1.2351167e-01	 1.5521601e+00	 1.3089883e-01	  1.2619533e+00 	 1.8013859e-01


.. parsed-literal::

     129	 1.4992341e+00	 1.2347717e-01	 1.5535064e+00	 1.3097417e-01	  1.2626160e+00 	 1.6641808e-01
     130	 1.5006844e+00	 1.2342560e-01	 1.5550257e+00	 1.3107277e-01	  1.2569152e+00 	 1.9755077e-01


.. parsed-literal::

     131	 1.5018764e+00	 1.2336250e-01	 1.5562368e+00	 1.3108025e-01	  1.2572953e+00 	 2.0217609e-01


.. parsed-literal::

     132	 1.5032225e+00	 1.2329354e-01	 1.5576556e+00	 1.3111568e-01	  1.2548803e+00 	 2.1615863e-01


.. parsed-literal::

     133	 1.5045581e+00	 1.2323695e-01	 1.5590250e+00	 1.3112851e-01	  1.2524473e+00 	 2.0536542e-01


.. parsed-literal::

     134	 1.5072705e+00	 1.2311728e-01	 1.5618732e+00	 1.3111715e-01	  1.2464437e+00 	 2.1302128e-01


.. parsed-literal::

     135	 1.5084540e+00	 1.2316383e-01	 1.5631628e+00	 1.3127625e-01	  1.2339191e+00 	 2.1059823e-01


.. parsed-literal::

     136	 1.5099544e+00	 1.2297761e-01	 1.5645147e+00	 1.3098936e-01	  1.2456740e+00 	 2.1219707e-01


.. parsed-literal::

     137	 1.5107600e+00	 1.2291763e-01	 1.5653006e+00	 1.3089248e-01	  1.2481057e+00 	 2.1280336e-01


.. parsed-literal::

     138	 1.5123494e+00	 1.2271874e-01	 1.5669481e+00	 1.3068931e-01	  1.2518327e+00 	 2.0780087e-01


.. parsed-literal::

     139	 1.5139209e+00	 1.2258171e-01	 1.5686066e+00	 1.3054032e-01	  1.2505983e+00 	 2.1866798e-01


.. parsed-literal::

     140	 1.5154545e+00	 1.2241983e-01	 1.5701827e+00	 1.3047401e-01	  1.2510151e+00 	 2.0751548e-01
     141	 1.5164672e+00	 1.2236020e-01	 1.5712279e+00	 1.3047835e-01	  1.2498847e+00 	 1.9972301e-01


.. parsed-literal::

     142	 1.5175567e+00	 1.2233039e-01	 1.5723497e+00	 1.3049536e-01	  1.2475471e+00 	 1.7124987e-01


.. parsed-literal::

     143	 1.5186646e+00	 1.2228939e-01	 1.5735336e+00	 1.3044861e-01	  1.2464580e+00 	 2.0747590e-01
     144	 1.5201603e+00	 1.2229540e-01	 1.5750267e+00	 1.3043574e-01	  1.2459676e+00 	 1.9387603e-01


.. parsed-literal::

     145	 1.5211610e+00	 1.2227951e-01	 1.5760278e+00	 1.3035402e-01	  1.2481235e+00 	 2.1063185e-01
     146	 1.5219982e+00	 1.2222138e-01	 1.5768858e+00	 1.3024498e-01	  1.2511993e+00 	 1.8647933e-01


.. parsed-literal::

     147	 1.5238944e+00	 1.2204332e-01	 1.5788341e+00	 1.2999232e-01	  1.2570075e+00 	 2.0500231e-01


.. parsed-literal::

     148	 1.5247828e+00	 1.2185536e-01	 1.5797592e+00	 1.2977886e-01	  1.2629154e+00 	 3.2256222e-01


.. parsed-literal::

     149	 1.5262315e+00	 1.2169992e-01	 1.5812330e+00	 1.2963928e-01	  1.2643867e+00 	 2.0841050e-01


.. parsed-literal::

     150	 1.5275797e+00	 1.2152431e-01	 1.5825682e+00	 1.2951117e-01	  1.2659073e+00 	 2.0679855e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.26 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3f9822b7c0>



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
    CPU times: user 1.98 s, sys: 39 ms, total: 2.02 s
    Wall time: 597 ms


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

