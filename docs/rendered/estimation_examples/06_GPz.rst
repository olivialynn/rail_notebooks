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
       1	-3.5036562e-01	 3.2308243e-01	-3.4073619e-01	 3.1019800e-01	[-3.1572717e-01]	 4.7775984e-01


.. parsed-literal::

       2	-2.8450567e-01	 3.1424833e-01	-2.6226464e-01	 3.0062423e-01	[-2.1821747e-01]	 2.3643541e-01


.. parsed-literal::

       3	-2.3933987e-01	 2.9288056e-01	-1.9569556e-01	 2.7796618e-01	[-1.3513124e-01]	 2.9808140e-01


.. parsed-literal::

       4	-2.0662568e-01	 2.7619257e-01	-1.5783509e-01	 2.6420650e-01	[-9.6695705e-02]	 3.0454087e-01


.. parsed-literal::

       5	-1.4170672e-01	 2.6019296e-01	-1.0445352e-01	 2.5272443e-01	[-5.7605217e-02]	 2.1038651e-01


.. parsed-literal::

       6	-7.9484789e-02	 2.5411156e-01	-4.9574322e-02	 2.4579267e-01	[-1.4471282e-02]	 2.0826578e-01


.. parsed-literal::

       7	-5.8071194e-02	 2.5050266e-01	-3.4708406e-02	 2.4238505e-01	[ 1.9076926e-03]	 2.1957755e-01
       8	-4.4114615e-02	 2.4790812e-01	-2.5252881e-02	 2.3991428e-01	[ 1.0535103e-02]	 1.9412136e-01


.. parsed-literal::

       9	-3.5768978e-02	 2.4651130e-01	-1.8450988e-02	 2.3860344e-01	[ 1.4519106e-02]	 2.1764946e-01


.. parsed-literal::

      10	-2.6946921e-02	 2.4480575e-01	-1.1044187e-02	 2.3640730e-01	[ 2.3367223e-02]	 2.1064520e-01


.. parsed-literal::

      11	-1.8841227e-02	 2.4343370e-01	-4.5148164e-03	 2.3744620e-01	  2.1233290e-02 	 2.1277523e-01


.. parsed-literal::

      12	-1.3600293e-02	 2.4239551e-01	 4.3540333e-04	 2.3632750e-01	[ 2.7267782e-02]	 2.0769167e-01


.. parsed-literal::

      13	-1.0192069e-02	 2.4166104e-01	 3.9725265e-03	 2.3521592e-01	[ 3.2893424e-02]	 2.0566201e-01


.. parsed-literal::

      14	 1.1018577e-01	 2.2475215e-01	 1.3103476e-01	 2.2546493e-01	[ 1.5387077e-01]	 3.3192539e-01


.. parsed-literal::

      15	 1.4338474e-01	 2.2605853e-01	 1.6777238e-01	 2.2125424e-01	[ 1.7877028e-01]	 2.0396733e-01
      16	 2.0255678e-01	 2.2026827e-01	 2.2871195e-01	 2.1642317e-01	[ 2.3744707e-01]	 1.9817829e-01


.. parsed-literal::

      17	 2.9474113e-01	 2.1453022e-01	 3.2497827e-01	 2.0957486e-01	[ 3.3681946e-01]	 2.0909071e-01


.. parsed-literal::

      18	 3.5689828e-01	 2.0904519e-01	 3.8892869e-01	 2.1058093e-01	[ 3.7851862e-01]	 2.1513700e-01
      19	 4.0677054e-01	 2.0717141e-01	 4.3959232e-01	 2.0847345e-01	[ 4.2639587e-01]	 2.1091652e-01


.. parsed-literal::

      20	 4.6032417e-01	 2.0296427e-01	 4.9444437e-01	 2.0658741e-01	[ 4.7172335e-01]	 2.0568991e-01


.. parsed-literal::

      21	 5.3232456e-01	 2.0269717e-01	 5.7012961e-01	 2.0504956e-01	[ 5.4033593e-01]	 2.0406342e-01


.. parsed-literal::

      22	 5.7969643e-01	 1.9893307e-01	 6.2019504e-01	 1.9971992e-01	[ 5.7689226e-01]	 2.1416092e-01
      23	 6.2097496e-01	 1.9582436e-01	 6.6120093e-01	 1.9577242e-01	[ 6.2120958e-01]	 1.9886804e-01


.. parsed-literal::

      24	 6.7885936e-01	 1.9517564e-01	 7.1754077e-01	 1.9454491e-01	[ 6.8704487e-01]	 2.1552992e-01


.. parsed-literal::

      25	 7.0373162e-01	 1.9608314e-01	 7.3906146e-01	 1.9222576e-01	[ 7.3364690e-01]	 2.1269488e-01


.. parsed-literal::

      26	 7.4169381e-01	 1.9391387e-01	 7.7933542e-01	 1.9151325e-01	[ 7.5688500e-01]	 2.2003675e-01
      27	 7.6788376e-01	 1.9438553e-01	 8.0712319e-01	 1.9198565e-01	[ 7.7051544e-01]	 2.0355225e-01


.. parsed-literal::

      28	 8.0204053e-01	 1.9426797e-01	 8.4217627e-01	 1.9314491e-01	[ 7.9619368e-01]	 2.1314120e-01
      29	 8.2333696e-01	 1.9923950e-01	 8.6254710e-01	 1.9236459e-01	[ 8.3151793e-01]	 1.7669702e-01


.. parsed-literal::

      30	 8.4143502e-01	 2.0108134e-01	 8.8216931e-01	 1.9397693e-01	[ 8.5619666e-01]	 2.1362138e-01


.. parsed-literal::

      31	 8.6046876e-01	 1.9814973e-01	 9.0058095e-01	 1.9626133e-01	[ 8.6502487e-01]	 2.1564913e-01


.. parsed-literal::

      32	 8.7327810e-01	 1.9746649e-01	 9.1345299e-01	 1.9446809e-01	[ 8.7852201e-01]	 2.1805811e-01


.. parsed-literal::

      33	 8.9941806e-01	 1.9528375e-01	 9.4058592e-01	 1.8922092e-01	[ 9.0898115e-01]	 2.1361446e-01


.. parsed-literal::

      34	 9.1486698e-01	 1.9478944e-01	 9.5634585e-01	 1.8923855e-01	[ 9.2990117e-01]	 2.1330810e-01
      35	 9.3124439e-01	 1.9370508e-01	 9.7300518e-01	 1.8792805e-01	[ 9.5290244e-01]	 1.7553329e-01


.. parsed-literal::

      36	 9.4970804e-01	 1.8948100e-01	 9.9195545e-01	 1.8417816e-01	[ 9.7347625e-01]	 2.1606064e-01


.. parsed-literal::

      37	 9.7217956e-01	 1.8259237e-01	 1.0157618e+00	 1.7804933e-01	[ 9.8914563e-01]	 2.2269869e-01


.. parsed-literal::

      38	 9.8791056e-01	 1.7538482e-01	 1.0322779e+00	 1.6958770e-01	[ 1.0053124e+00]	 2.1897936e-01


.. parsed-literal::

      39	 9.9943755e-01	 1.7472132e-01	 1.0435848e+00	 1.6859189e-01	[ 1.0094794e+00]	 2.0145822e-01


.. parsed-literal::

      40	 1.0107060e+00	 1.7206273e-01	 1.0552283e+00	 1.6665529e-01	[ 1.0158148e+00]	 2.2164106e-01


.. parsed-literal::

      41	 1.0207094e+00	 1.6927035e-01	 1.0655190e+00	 1.6413544e-01	[ 1.0220169e+00]	 2.1227312e-01
      42	 1.0337292e+00	 1.6374403e-01	 1.0790595e+00	 1.5920389e-01	[ 1.0411366e+00]	 1.9368672e-01


.. parsed-literal::

      43	 1.0450881e+00	 1.6206213e-01	 1.0906191e+00	 1.5747412e-01	[ 1.0416417e+00]	 2.1511221e-01


.. parsed-literal::

      44	 1.0534049e+00	 1.6172126e-01	 1.0987555e+00	 1.5693636e-01	[ 1.0456484e+00]	 2.1569562e-01


.. parsed-literal::

      45	 1.0664800e+00	 1.5897821e-01	 1.1125213e+00	 1.5295377e-01	[ 1.0552824e+00]	 2.1310616e-01


.. parsed-literal::

      46	 1.0773082e+00	 1.5741458e-01	 1.1236498e+00	 1.5134353e-01	[ 1.0687356e+00]	 2.0927835e-01


.. parsed-literal::

      47	 1.0840103e+00	 1.5660283e-01	 1.1303819e+00	 1.4994553e-01	[ 1.0756466e+00]	 2.0987892e-01


.. parsed-literal::

      48	 1.0913717e+00	 1.5551072e-01	 1.1378044e+00	 1.4832981e-01	[ 1.0820469e+00]	 2.1298432e-01
      49	 1.1001682e+00	 1.5414224e-01	 1.1467578e+00	 1.4615726e-01	[ 1.0889325e+00]	 1.8575883e-01


.. parsed-literal::

      50	 1.1099636e+00	 1.5230683e-01	 1.1567246e+00	 1.4309038e-01	[ 1.0967084e+00]	 2.1341372e-01


.. parsed-literal::

      51	 1.1178254e+00	 1.5173240e-01	 1.1645257e+00	 1.4233325e-01	[ 1.0994647e+00]	 2.0433879e-01
      52	 1.1329192e+00	 1.5069065e-01	 1.1801303e+00	 1.3976717e-01	[ 1.1037927e+00]	 1.9237947e-01


.. parsed-literal::

      53	 1.1422048e+00	 1.5064680e-01	 1.1899796e+00	 1.3932325e-01	  1.0866111e+00 	 2.0476365e-01


.. parsed-literal::

      54	 1.1508137e+00	 1.4966277e-01	 1.1985863e+00	 1.3832013e-01	  1.0979684e+00 	 2.1113539e-01
      55	 1.1603884e+00	 1.4817224e-01	 1.2083869e+00	 1.3646117e-01	[ 1.1121898e+00]	 1.7983723e-01


.. parsed-literal::

      56	 1.1685581e+00	 1.4689230e-01	 1.2167496e+00	 1.3533318e-01	[ 1.1165471e+00]	 2.1379185e-01


.. parsed-literal::

      57	 1.1759519e+00	 1.4563484e-01	 1.2248888e+00	 1.3362194e-01	[ 1.1254519e+00]	 2.2088099e-01


.. parsed-literal::

      58	 1.1855807e+00	 1.4475154e-01	 1.2341548e+00	 1.3344751e-01	[ 1.1266586e+00]	 2.1550441e-01
      59	 1.1916315e+00	 1.4451032e-01	 1.2401217e+00	 1.3314169e-01	  1.1264989e+00 	 1.8569779e-01


.. parsed-literal::

      60	 1.2036414e+00	 1.4385040e-01	 1.2523728e+00	 1.3304884e-01	  1.1186455e+00 	 2.1135283e-01
      61	 1.2139362e+00	 1.4330634e-01	 1.2631558e+00	 1.3132993e-01	  1.0909320e+00 	 1.8332267e-01


.. parsed-literal::

      62	 1.2257604e+00	 1.4306599e-01	 1.2748919e+00	 1.3181131e-01	  1.0989264e+00 	 2.1146917e-01
      63	 1.2335501e+00	 1.4237629e-01	 1.2828059e+00	 1.3116242e-01	  1.1030992e+00 	 1.7813110e-01


.. parsed-literal::

      64	 1.2438025e+00	 1.4130818e-01	 1.2936689e+00	 1.2955150e-01	  1.0971363e+00 	 1.7813540e-01


.. parsed-literal::

      65	 1.2520868e+00	 1.4063108e-01	 1.3021506e+00	 1.2796629e-01	  1.0929748e+00 	 2.1838903e-01


.. parsed-literal::

      66	 1.2621643e+00	 1.3980252e-01	 1.3123165e+00	 1.2670209e-01	  1.0963285e+00 	 2.0996284e-01


.. parsed-literal::

      67	 1.2715403e+00	 1.3888539e-01	 1.3220687e+00	 1.2548328e-01	  1.0893282e+00 	 2.1690917e-01


.. parsed-literal::

      68	 1.2815114e+00	 1.3839503e-01	 1.3321257e+00	 1.2528792e-01	  1.0973255e+00 	 2.1339965e-01
      69	 1.2878210e+00	 1.3826580e-01	 1.3385846e+00	 1.2480500e-01	  1.0959689e+00 	 1.9016242e-01


.. parsed-literal::

      70	 1.2941172e+00	 1.3799338e-01	 1.3447294e+00	 1.2471543e-01	  1.1011570e+00 	 2.1057987e-01
      71	 1.3032137e+00	 1.3769858e-01	 1.3540815e+00	 1.2467305e-01	  1.1129461e+00 	 1.9512177e-01


.. parsed-literal::

      72	 1.3093825e+00	 1.3747117e-01	 1.3602788e+00	 1.2457657e-01	  1.1252661e+00 	 2.1958804e-01


.. parsed-literal::

      73	 1.3193531e+00	 1.3771423e-01	 1.3708596e+00	 1.2490684e-01	[ 1.1647724e+00]	 2.1544147e-01


.. parsed-literal::

      74	 1.3280704e+00	 1.3685084e-01	 1.3794599e+00	 1.2421179e-01	[ 1.1837123e+00]	 2.1440840e-01


.. parsed-literal::

      75	 1.3333408e+00	 1.3673661e-01	 1.3846407e+00	 1.2414257e-01	[ 1.1926298e+00]	 2.1872973e-01


.. parsed-literal::

      76	 1.3417608e+00	 1.3628158e-01	 1.3932492e+00	 1.2373491e-01	[ 1.2022986e+00]	 2.0822430e-01


.. parsed-literal::

      77	 1.3482092e+00	 1.3629740e-01	 1.3998893e+00	 1.2362265e-01	[ 1.2086926e+00]	 2.1227217e-01


.. parsed-literal::

      78	 1.3560927e+00	 1.3597916e-01	 1.4080201e+00	 1.2341638e-01	[ 1.2223312e+00]	 2.1339703e-01


.. parsed-literal::

      79	 1.3620523e+00	 1.3585034e-01	 1.4140913e+00	 1.2303461e-01	  1.2217650e+00 	 2.2657895e-01


.. parsed-literal::

      80	 1.3660068e+00	 1.3558696e-01	 1.4179734e+00	 1.2289134e-01	  1.2208913e+00 	 2.0637083e-01


.. parsed-literal::

      81	 1.3736558e+00	 1.3503189e-01	 1.4261768e+00	 1.2323942e-01	  1.2099502e+00 	 2.1769881e-01


.. parsed-literal::

      82	 1.3797641e+00	 1.3444852e-01	 1.4323742e+00	 1.2262528e-01	  1.1921959e+00 	 2.0727205e-01
      83	 1.3856753e+00	 1.3411911e-01	 1.4382748e+00	 1.2248753e-01	  1.1877402e+00 	 1.9790101e-01


.. parsed-literal::

      84	 1.3911026e+00	 1.3399525e-01	 1.4440102e+00	 1.2261244e-01	  1.1706678e+00 	 2.1477222e-01


.. parsed-literal::

      85	 1.3951181e+00	 1.3400998e-01	 1.4484653e+00	 1.2252842e-01	  1.1431429e+00 	 2.0984244e-01
      86	 1.4000327e+00	 1.3382811e-01	 1.4531728e+00	 1.2224299e-01	  1.1489615e+00 	 1.9999576e-01


.. parsed-literal::

      87	 1.4048248e+00	 1.3379163e-01	 1.4580183e+00	 1.2207350e-01	  1.1440075e+00 	 2.1984482e-01


.. parsed-literal::

      88	 1.4093633e+00	 1.3376016e-01	 1.4624405e+00	 1.2186268e-01	  1.1493269e+00 	 2.2394228e-01


.. parsed-literal::

      89	 1.4147478e+00	 1.3396905e-01	 1.4680328e+00	 1.2163739e-01	  1.1406226e+00 	 2.1548915e-01


.. parsed-literal::

      90	 1.4207606e+00	 1.3384399e-01	 1.4737970e+00	 1.2149084e-01	  1.1569404e+00 	 2.2345138e-01


.. parsed-literal::

      91	 1.4239318e+00	 1.3358588e-01	 1.4769211e+00	 1.2114540e-01	  1.1610985e+00 	 2.1338391e-01


.. parsed-literal::

      92	 1.4281309e+00	 1.3343905e-01	 1.4812909e+00	 1.2077434e-01	  1.1591534e+00 	 2.1493912e-01


.. parsed-literal::

      93	 1.4310827e+00	 1.3362948e-01	 1.4843881e+00	 1.2041806e-01	  1.1494064e+00 	 2.1166849e-01


.. parsed-literal::

      94	 1.4336635e+00	 1.3349636e-01	 1.4868824e+00	 1.2010787e-01	  1.1536434e+00 	 2.1227694e-01


.. parsed-literal::

      95	 1.4377411e+00	 1.3355548e-01	 1.4909838e+00	 1.1960441e-01	  1.1567292e+00 	 2.0633006e-01


.. parsed-literal::

      96	 1.4410463e+00	 1.3342019e-01	 1.4942936e+00	 1.1917135e-01	  1.1559801e+00 	 2.1836448e-01


.. parsed-literal::

      97	 1.4449075e+00	 1.3327280e-01	 1.4981727e+00	 1.1853686e-01	  1.1644032e+00 	 2.0584774e-01


.. parsed-literal::

      98	 1.4491614e+00	 1.3316189e-01	 1.5024750e+00	 1.1785264e-01	  1.1703514e+00 	 2.2107339e-01


.. parsed-literal::

      99	 1.4524745e+00	 1.3264131e-01	 1.5058780e+00	 1.1726485e-01	  1.1736782e+00 	 2.2207904e-01


.. parsed-literal::

     100	 1.4554483e+00	 1.3247685e-01	 1.5088477e+00	 1.1711101e-01	  1.1660788e+00 	 2.1204424e-01


.. parsed-literal::

     101	 1.4590183e+00	 1.3231678e-01	 1.5124932e+00	 1.1716381e-01	  1.1541373e+00 	 2.2901273e-01


.. parsed-literal::

     102	 1.4625237e+00	 1.3212311e-01	 1.5160998e+00	 1.1711607e-01	  1.1437665e+00 	 2.0818257e-01


.. parsed-literal::

     103	 1.4655631e+00	 1.3185593e-01	 1.5193598e+00	 1.1751677e-01	  1.1194250e+00 	 2.1215105e-01


.. parsed-literal::

     104	 1.4688188e+00	 1.3171277e-01	 1.5225231e+00	 1.1753545e-01	  1.1240873e+00 	 2.1601892e-01


.. parsed-literal::

     105	 1.4710467e+00	 1.3155089e-01	 1.5247424e+00	 1.1742929e-01	  1.1257292e+00 	 2.1351361e-01


.. parsed-literal::

     106	 1.4742337e+00	 1.3127677e-01	 1.5280275e+00	 1.1729992e-01	  1.1154306e+00 	 2.1877575e-01


.. parsed-literal::

     107	 1.4765567e+00	 1.3088713e-01	 1.5305076e+00	 1.1702971e-01	  1.1025883e+00 	 3.4404826e-01


.. parsed-literal::

     108	 1.4790636e+00	 1.3064893e-01	 1.5331086e+00	 1.1676526e-01	  1.0908714e+00 	 2.1336031e-01


.. parsed-literal::

     109	 1.4815761e+00	 1.3039662e-01	 1.5356851e+00	 1.1632291e-01	  1.0812213e+00 	 2.1778941e-01
     110	 1.4840333e+00	 1.3026996e-01	 1.5382014e+00	 1.1599382e-01	  1.0710504e+00 	 1.8323922e-01


.. parsed-literal::

     111	 1.4865258e+00	 1.3013403e-01	 1.5406751e+00	 1.1563755e-01	  1.0662009e+00 	 2.0461130e-01


.. parsed-literal::

     112	 1.4888757e+00	 1.3010940e-01	 1.5429621e+00	 1.1562429e-01	  1.0640728e+00 	 2.0958209e-01


.. parsed-literal::

     113	 1.4914886e+00	 1.3006867e-01	 1.5455625e+00	 1.1569037e-01	  1.0626149e+00 	 2.1471906e-01


.. parsed-literal::

     114	 1.4941468e+00	 1.3003450e-01	 1.5482470e+00	 1.1564186e-01	  1.0530368e+00 	 2.1216941e-01
     115	 1.4972884e+00	 1.3004672e-01	 1.5514460e+00	 1.1553382e-01	  1.0494429e+00 	 1.7709398e-01


.. parsed-literal::

     116	 1.5003653e+00	 1.2989912e-01	 1.5546220e+00	 1.1515673e-01	  1.0325727e+00 	 2.0966268e-01
     117	 1.5026644e+00	 1.2976593e-01	 1.5570241e+00	 1.1499600e-01	  1.0269038e+00 	 2.0229268e-01


.. parsed-literal::

     118	 1.5045223e+00	 1.2960458e-01	 1.5588601e+00	 1.1484435e-01	  1.0233286e+00 	 2.1279216e-01
     119	 1.5059592e+00	 1.2940633e-01	 1.5602923e+00	 1.1470218e-01	  1.0114946e+00 	 1.8257165e-01


.. parsed-literal::

     120	 1.5074573e+00	 1.2924135e-01	 1.5618073e+00	 1.1461367e-01	  9.9742232e-01 	 2.0926571e-01


.. parsed-literal::

     121	 1.5094630e+00	 1.2900841e-01	 1.5639090e+00	 1.1440249e-01	  9.7669761e-01 	 2.1683908e-01


.. parsed-literal::

     122	 1.5112940e+00	 1.2892270e-01	 1.5657375e+00	 1.1423462e-01	  9.7190371e-01 	 2.0416689e-01


.. parsed-literal::

     123	 1.5134909e+00	 1.2888531e-01	 1.5679573e+00	 1.1399381e-01	  9.6478618e-01 	 2.0209694e-01


.. parsed-literal::

     124	 1.5145540e+00	 1.2876060e-01	 1.5690973e+00	 1.1372241e-01	  9.5757866e-01 	 2.1275735e-01
     125	 1.5159958e+00	 1.2875199e-01	 1.5705160e+00	 1.1372360e-01	  9.5099807e-01 	 2.1019959e-01


.. parsed-literal::

     126	 1.5176826e+00	 1.2871189e-01	 1.5722883e+00	 1.1368220e-01	  9.2631571e-01 	 2.1066761e-01


.. parsed-literal::

     127	 1.5189575e+00	 1.2864623e-01	 1.5736005e+00	 1.1374669e-01	  9.0951009e-01 	 2.2117615e-01


.. parsed-literal::

     128	 1.5216758e+00	 1.2841424e-01	 1.5764273e+00	 1.1406036e-01	  8.5274315e-01 	 2.0406246e-01


.. parsed-literal::

     129	 1.5227337e+00	 1.2835959e-01	 1.5776589e+00	 1.1472620e-01	  8.1552770e-01 	 2.1907377e-01


.. parsed-literal::

     130	 1.5248313e+00	 1.2825292e-01	 1.5796449e+00	 1.1460741e-01	  8.1858636e-01 	 2.1167254e-01
     131	 1.5259760e+00	 1.2820672e-01	 1.5807724e+00	 1.1462060e-01	  8.1358571e-01 	 1.8178844e-01


.. parsed-literal::

     132	 1.5276120e+00	 1.2812989e-01	 1.5824711e+00	 1.1477737e-01	  7.8942308e-01 	 2.0261121e-01


.. parsed-literal::

     133	 1.5280677e+00	 1.2808419e-01	 1.5831180e+00	 1.1489547e-01	  7.5624418e-01 	 2.1366549e-01
     134	 1.5302997e+00	 1.2801346e-01	 1.5852911e+00	 1.1493612e-01	  7.4497891e-01 	 1.7168713e-01


.. parsed-literal::

     135	 1.5310476e+00	 1.2796970e-01	 1.5860661e+00	 1.1495046e-01	  7.3121565e-01 	 2.1750212e-01
     136	 1.5322813e+00	 1.2785720e-01	 1.5873540e+00	 1.1491625e-01	  7.0575164e-01 	 1.9035530e-01


.. parsed-literal::

     137	 1.5334174e+00	 1.2771665e-01	 1.5885668e+00	 1.1489170e-01	  6.7359258e-01 	 2.1555591e-01


.. parsed-literal::

     138	 1.5349739e+00	 1.2756292e-01	 1.5901090e+00	 1.1476928e-01	  6.5660193e-01 	 2.0933938e-01


.. parsed-literal::

     139	 1.5361315e+00	 1.2746684e-01	 1.5912284e+00	 1.1467765e-01	  6.5425727e-01 	 2.1348691e-01
     140	 1.5373364e+00	 1.2733297e-01	 1.5924032e+00	 1.1460850e-01	  6.4871505e-01 	 1.9457698e-01


.. parsed-literal::

     141	 1.5378903e+00	 1.2729316e-01	 1.5930023e+00	 1.1474439e-01	  6.3037730e-01 	 2.2308064e-01


.. parsed-literal::

     142	 1.5397364e+00	 1.2710505e-01	 1.5947759e+00	 1.1467123e-01	  6.2241111e-01 	 2.1712470e-01
     143	 1.5407348e+00	 1.2701255e-01	 1.5957726e+00	 1.1470753e-01	  6.1013695e-01 	 1.7785645e-01


.. parsed-literal::

     144	 1.5419998e+00	 1.2685651e-01	 1.5970638e+00	 1.1481783e-01	  5.8273255e-01 	 1.8251300e-01
     145	 1.5439241e+00	 1.2668659e-01	 1.5990197e+00	 1.1502583e-01	  5.6017623e-01 	 1.8988442e-01


.. parsed-literal::

     146	 1.5448844e+00	 1.2628867e-01	 1.6000857e+00	 1.1537519e-01	  4.9579585e-01 	 2.0515132e-01


.. parsed-literal::

     147	 1.5463341e+00	 1.2642551e-01	 1.6014275e+00	 1.1524278e-01	  5.2229073e-01 	 2.1603680e-01


.. parsed-literal::

     148	 1.5468189e+00	 1.2643324e-01	 1.6018879e+00	 1.1520076e-01	  5.2743763e-01 	 2.0805597e-01
     149	 1.5479970e+00	 1.2640541e-01	 1.6030486e+00	 1.1514931e-01	  5.2456039e-01 	 1.9807529e-01


.. parsed-literal::

     150	 1.5488787e+00	 1.2624339e-01	 1.6040011e+00	 1.1504552e-01	  4.9403965e-01 	 1.9753242e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.26 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa08815f520>



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
    CPU times: user 1.87 s, sys: 40 ms, total: 1.91 s
    Wall time: 624 ms


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

