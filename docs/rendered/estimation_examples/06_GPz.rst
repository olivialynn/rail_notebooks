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
       1	-3.4168815e-01	 3.1978754e-01	-3.3193704e-01	 3.2328726e-01	[-3.3891475e-01]	 4.6893501e-01


.. parsed-literal::

       2	-2.6851880e-01	 3.0834166e-01	-2.4383964e-01	 3.1275287e-01	[-2.5752355e-01]	 2.3591113e-01


.. parsed-literal::

       3	-2.2459569e-01	 2.8861050e-01	-1.8326912e-01	 2.9379885e-01	[-2.0610189e-01]	 2.7922511e-01


.. parsed-literal::

       4	-1.9335179e-01	 2.6410818e-01	-1.5326215e-01	 2.6964758e-01	[-1.9014839e-01]	 2.0460010e-01


.. parsed-literal::

       5	-9.9326145e-02	 2.5523975e-01	-6.2699091e-02	 2.6328619e-01	[-1.0229447e-01]	 2.0677090e-01


.. parsed-literal::

       6	-6.5820283e-02	 2.5020959e-01	-3.3785464e-02	 2.5878107e-01	[-6.7625376e-02]	 2.0591283e-01


.. parsed-literal::

       7	-4.6520297e-02	 2.4719019e-01	-2.1718798e-02	 2.5609258e-01	[-5.8064146e-02]	 2.0436192e-01
       8	-3.5498301e-02	 2.4544814e-01	-1.4457205e-02	 2.5411183e-01	[-5.0189477e-02]	 1.7355466e-01


.. parsed-literal::

       9	-2.2347776e-02	 2.4311980e-01	-4.4756412e-03	 2.5123558e-01	[-3.8298288e-02]	 2.1131945e-01


.. parsed-literal::

      10	-9.1535070e-03	 2.4055082e-01	 6.6484387e-03	 2.4890585e-01	[-2.9229527e-02]	 2.1001220e-01


.. parsed-literal::

      11	-5.6364753e-03	 2.4016504e-01	 8.7570784e-03	 2.4968079e-01	 -2.9410933e-02 	 2.1352339e-01


.. parsed-literal::

      12	-1.3170665e-03	 2.3933067e-01	 1.2734413e-02	 2.4854619e-01	[-2.7414355e-02]	 2.1661019e-01


.. parsed-literal::

      13	 1.3682851e-03	 2.3877961e-01	 1.5381929e-02	 2.4778551e-01	[-2.2644847e-02]	 2.1182108e-01


.. parsed-literal::

      14	 4.9415357e-03	 2.3807986e-01	 1.9096549e-02	 2.4735720e-01	[-2.0831261e-02]	 2.0360255e-01


.. parsed-literal::

      15	 9.5790281e-02	 2.2091021e-01	 1.1684914e-01	 2.2676397e-01	[ 1.0047128e-01]	 4.4819283e-01


.. parsed-literal::

      16	 2.0021237e-01	 2.1442486e-01	 2.2452237e-01	 2.2062386e-01	[ 2.0175337e-01]	 2.0811701e-01
      17	 2.7074787e-01	 2.0812790e-01	 3.0305134e-01	 2.1455647e-01	[ 2.5922406e-01]	 1.9811606e-01


.. parsed-literal::

      18	 3.3571101e-01	 2.1317994e-01	 3.6514665e-01	 2.1985734e-01	[ 3.2458167e-01]	 2.0508814e-01


.. parsed-literal::

      19	 3.7168145e-01	 2.0764669e-01	 4.0113031e-01	 2.1473763e-01	[ 3.6452367e-01]	 2.1126390e-01
      20	 4.1400697e-01	 2.0364224e-01	 4.4460431e-01	 2.1086509e-01	[ 4.0647178e-01]	 1.9055295e-01


.. parsed-literal::

      21	 5.1748341e-01	 2.0013636e-01	 5.5169728e-01	 2.0855633e-01	[ 4.9980424e-01]	 2.0969057e-01


.. parsed-literal::

      22	 5.8165030e-01	 1.9881112e-01	 6.2015896e-01	 2.0634180e-01	[ 5.5902114e-01]	 2.1244836e-01


.. parsed-literal::

      23	 6.3614839e-01	 1.9598304e-01	 6.7333826e-01	 2.0321250e-01	[ 6.1613691e-01]	 2.1818280e-01


.. parsed-literal::

      24	 6.6833071e-01	 1.9178753e-01	 7.0541907e-01	 1.9864931e-01	[ 6.4672403e-01]	 2.1136880e-01


.. parsed-literal::

      25	 7.1749035e-01	 1.9206374e-01	 7.5375135e-01	 1.9938882e-01	[ 6.9791720e-01]	 2.2237802e-01


.. parsed-literal::

      26	 7.2740390e-01	 1.9709105e-01	 7.6207587e-01	 2.0510475e-01	[ 7.0811269e-01]	 2.0503020e-01
      27	 7.6344034e-01	 1.9047014e-01	 7.9894426e-01	 1.9730215e-01	[ 7.4473874e-01]	 1.8146110e-01


.. parsed-literal::

      28	 7.8671667e-01	 1.8978758e-01	 8.2262419e-01	 1.9608278e-01	[ 7.6606235e-01]	 2.0931411e-01


.. parsed-literal::

      29	 8.0855890e-01	 1.8759122e-01	 8.4593836e-01	 1.9303520e-01	[ 7.8972279e-01]	 2.1484017e-01


.. parsed-literal::

      30	 8.3825318e-01	 1.8328313e-01	 8.7544775e-01	 1.8816563e-01	[ 8.2455762e-01]	 2.0709014e-01


.. parsed-literal::

      31	 8.6804469e-01	 1.7961012e-01	 9.0587041e-01	 1.8334928e-01	[ 8.6109998e-01]	 2.2141027e-01


.. parsed-literal::

      32	 8.8756472e-01	 1.7883046e-01	 9.2608855e-01	 1.8425817e-01	[ 8.8000786e-01]	 2.0327330e-01


.. parsed-literal::

      33	 9.0747459e-01	 1.7894179e-01	 9.4662020e-01	 1.8427492e-01	[ 9.0410939e-01]	 2.1584654e-01


.. parsed-literal::

      34	 9.2319271e-01	 1.7787132e-01	 9.6301497e-01	 1.8392653e-01	[ 9.1685625e-01]	 2.1258068e-01


.. parsed-literal::

      35	 9.3799040e-01	 1.7800058e-01	 9.7853805e-01	 1.8501619e-01	[ 9.2501783e-01]	 2.1123886e-01
      36	 9.5318028e-01	 1.7905798e-01	 9.9428488e-01	 1.8653973e-01	[ 9.4126396e-01]	 1.9771147e-01


.. parsed-literal::

      37	 9.6780343e-01	 1.7869509e-01	 1.0093608e+00	 1.8725301e-01	[ 9.5699381e-01]	 2.1890163e-01


.. parsed-literal::

      38	 9.8043453e-01	 1.7800320e-01	 1.0221516e+00	 1.8715381e-01	[ 9.7162462e-01]	 2.0229840e-01


.. parsed-literal::

      39	 9.9896906e-01	 1.7612154e-01	 1.0415276e+00	 1.8596960e-01	[ 9.9534342e-01]	 2.2325563e-01
      40	 1.0157916e+00	 1.7390276e-01	 1.0598782e+00	 1.8442943e-01	[ 1.0006255e+00]	 1.8878579e-01


.. parsed-literal::

      41	 1.0288264e+00	 1.7184592e-01	 1.0733697e+00	 1.8261770e-01	[ 1.0229968e+00]	 2.1051002e-01


.. parsed-literal::

      42	 1.0390495e+00	 1.7077383e-01	 1.0836330e+00	 1.8178688e-01	[ 1.0305122e+00]	 2.1168017e-01


.. parsed-literal::

      43	 1.0525513e+00	 1.6964213e-01	 1.0974422e+00	 1.8062609e-01	[ 1.0373190e+00]	 2.1241856e-01
      44	 1.0611460e+00	 1.6894468e-01	 1.1057236e+00	 1.8025369e-01	[ 1.0511045e+00]	 1.9944072e-01


.. parsed-literal::

      45	 1.0681092e+00	 1.6775441e-01	 1.1128211e+00	 1.7958949e-01	[ 1.0578307e+00]	 2.1452308e-01


.. parsed-literal::

      46	 1.0775510e+00	 1.6619657e-01	 1.1226262e+00	 1.7842077e-01	[ 1.0694999e+00]	 2.1176744e-01


.. parsed-literal::

      47	 1.0871634e+00	 1.6424460e-01	 1.1328486e+00	 1.7732181e-01	[ 1.0723262e+00]	 2.2123408e-01


.. parsed-literal::

      48	 1.0966747e+00	 1.6297728e-01	 1.1429087e+00	 1.7608054e-01	[ 1.0787065e+00]	 3.5600519e-01


.. parsed-literal::

      49	 1.1035332e+00	 1.6223374e-01	 1.1498938e+00	 1.7539490e-01	[ 1.0827714e+00]	 2.2276330e-01
      50	 1.1123333e+00	 1.6125492e-01	 1.1590255e+00	 1.7478768e-01	[ 1.0856514e+00]	 1.9331264e-01


.. parsed-literal::

      51	 1.1196726e+00	 1.6028297e-01	 1.1663983e+00	 1.7379546e-01	[ 1.0952191e+00]	 2.0915461e-01


.. parsed-literal::

      52	 1.1290581e+00	 1.5912661e-01	 1.1757945e+00	 1.7310291e-01	[ 1.1034013e+00]	 2.1171498e-01


.. parsed-literal::

      53	 1.1387246e+00	 1.5830428e-01	 1.1858139e+00	 1.7306989e-01	[ 1.1068074e+00]	 2.1447611e-01


.. parsed-literal::

      54	 1.1467378e+00	 1.5673008e-01	 1.1939377e+00	 1.7217918e-01	[ 1.1171431e+00]	 2.0995736e-01
      55	 1.1546822e+00	 1.5593562e-01	 1.2018660e+00	 1.7218927e-01	[ 1.1241626e+00]	 1.7255807e-01


.. parsed-literal::

      56	 1.1638551e+00	 1.5475831e-01	 1.2115008e+00	 1.7203461e-01	[ 1.1297745e+00]	 2.1389461e-01


.. parsed-literal::

      57	 1.1716103e+00	 1.5357752e-01	 1.2198129e+00	 1.7139670e-01	[ 1.1359676e+00]	 2.0690680e-01
      58	 1.1795396e+00	 1.5232971e-01	 1.2281541e+00	 1.7105646e-01	  1.1337835e+00 	 2.0582986e-01


.. parsed-literal::

      59	 1.1864646e+00	 1.5174480e-01	 1.2349494e+00	 1.7027059e-01	[ 1.1420945e+00]	 1.9034743e-01
      60	 1.1943477e+00	 1.5099632e-01	 1.2429972e+00	 1.6945997e-01	[ 1.1453141e+00]	 1.9984102e-01


.. parsed-literal::

      61	 1.2005003e+00	 1.5017711e-01	 1.2496972e+00	 1.6889895e-01	[ 1.1488953e+00]	 1.8908429e-01
      62	 1.2079295e+00	 1.4936936e-01	 1.2570830e+00	 1.6821603e-01	[ 1.1550391e+00]	 2.0041895e-01


.. parsed-literal::

      63	 1.2131244e+00	 1.4889945e-01	 1.2623848e+00	 1.6833168e-01	[ 1.1573725e+00]	 2.1160889e-01


.. parsed-literal::

      64	 1.2202992e+00	 1.4839788e-01	 1.2698203e+00	 1.6855285e-01	[ 1.1617404e+00]	 2.1145415e-01
      65	 1.2274684e+00	 1.4786638e-01	 1.2772272e+00	 1.6844170e-01	[ 1.1700885e+00]	 2.0216131e-01


.. parsed-literal::

      66	 1.2354181e+00	 1.4787535e-01	 1.2851618e+00	 1.6869549e-01	[ 1.1769698e+00]	 2.1465158e-01


.. parsed-literal::

      67	 1.2413215e+00	 1.4796987e-01	 1.2909631e+00	 1.6865987e-01	[ 1.1812471e+00]	 2.1025896e-01


.. parsed-literal::

      68	 1.2499119e+00	 1.4768399e-01	 1.2997729e+00	 1.6857564e-01	[ 1.1831990e+00]	 2.0994568e-01


.. parsed-literal::

      69	 1.2536475e+00	 1.4679975e-01	 1.3040696e+00	 1.6869814e-01	  1.1766708e+00 	 2.0675397e-01


.. parsed-literal::

      70	 1.2623466e+00	 1.4631930e-01	 1.3124740e+00	 1.6820249e-01	[ 1.1865063e+00]	 2.1284866e-01


.. parsed-literal::

      71	 1.2661702e+00	 1.4574098e-01	 1.3163836e+00	 1.6788746e-01	[ 1.1895395e+00]	 2.1718693e-01


.. parsed-literal::

      72	 1.2715456e+00	 1.4499595e-01	 1.3218831e+00	 1.6742285e-01	[ 1.1922339e+00]	 2.1252561e-01
      73	 1.2793578e+00	 1.4398553e-01	 1.3300839e+00	 1.6659077e-01	[ 1.1958502e+00]	 1.8627834e-01


.. parsed-literal::

      74	 1.2841580e+00	 1.4357885e-01	 1.3353464e+00	 1.6631695e-01	  1.1933658e+00 	 2.0339465e-01
      75	 1.2899750e+00	 1.4359103e-01	 1.3409560e+00	 1.6628129e-01	[ 1.1992974e+00]	 1.8907356e-01


.. parsed-literal::

      76	 1.2939817e+00	 1.4330166e-01	 1.3449977e+00	 1.6599992e-01	[ 1.2001630e+00]	 1.9036222e-01


.. parsed-literal::

      77	 1.2991839e+00	 1.4253581e-01	 1.3504323e+00	 1.6523482e-01	  1.1967573e+00 	 2.0313859e-01


.. parsed-literal::

      78	 1.3027590e+00	 1.4100149e-01	 1.3545058e+00	 1.6384343e-01	  1.1917775e+00 	 2.1105671e-01
      79	 1.3084371e+00	 1.4066121e-01	 1.3600752e+00	 1.6337863e-01	  1.1960758e+00 	 2.0365000e-01


.. parsed-literal::

      80	 1.3111936e+00	 1.4049317e-01	 1.3627905e+00	 1.6306405e-01	[ 1.2003303e+00]	 2.1156240e-01
      81	 1.3171480e+00	 1.4010975e-01	 1.3688781e+00	 1.6218674e-01	[ 1.2069250e+00]	 2.0061302e-01


.. parsed-literal::

      82	 1.3239413e+00	 1.3996799e-01	 1.3759349e+00	 1.6128770e-01	[ 1.2121778e+00]	 2.0938301e-01


.. parsed-literal::

      83	 1.3275111e+00	 1.3972603e-01	 1.3797740e+00	 1.6062664e-01	  1.2121428e+00 	 2.1945763e-01


.. parsed-literal::

      84	 1.3321354e+00	 1.3962708e-01	 1.3841425e+00	 1.6064657e-01	[ 1.2177179e+00]	 2.2127867e-01


.. parsed-literal::

      85	 1.3345841e+00	 1.3937717e-01	 1.3865688e+00	 1.6056755e-01	[ 1.2187484e+00]	 2.2027016e-01


.. parsed-literal::

      86	 1.3389966e+00	 1.3883717e-01	 1.3910245e+00	 1.6035034e-01	[ 1.2227816e+00]	 2.1373963e-01


.. parsed-literal::

      87	 1.3421567e+00	 1.3833842e-01	 1.3942980e+00	 1.5988724e-01	[ 1.2253348e+00]	 3.3165026e-01


.. parsed-literal::

      88	 1.3465396e+00	 1.3780256e-01	 1.3987891e+00	 1.5949278e-01	[ 1.2322884e+00]	 2.1563864e-01


.. parsed-literal::

      89	 1.3501266e+00	 1.3743126e-01	 1.4024257e+00	 1.5920989e-01	[ 1.2373163e+00]	 2.0875978e-01


.. parsed-literal::

      90	 1.3542633e+00	 1.3689574e-01	 1.4066937e+00	 1.5834597e-01	[ 1.2430098e+00]	 2.1861005e-01
      91	 1.3582224e+00	 1.3646312e-01	 1.4106925e+00	 1.5812884e-01	  1.2414154e+00 	 1.9208407e-01


.. parsed-literal::

      92	 1.3620043e+00	 1.3605752e-01	 1.4145289e+00	 1.5747901e-01	[ 1.2434617e+00]	 2.1478319e-01
      93	 1.3659087e+00	 1.3556912e-01	 1.4185616e+00	 1.5676896e-01	  1.2426473e+00 	 1.8388271e-01


.. parsed-literal::

      94	 1.3690740e+00	 1.3538015e-01	 1.4218201e+00	 1.5639183e-01	  1.2409315e+00 	 2.1135879e-01


.. parsed-literal::

      95	 1.3734690e+00	 1.3477991e-01	 1.4263473e+00	 1.5556689e-01	[ 1.2446796e+00]	 2.1968961e-01


.. parsed-literal::

      96	 1.3778838e+00	 1.3453274e-01	 1.4308694e+00	 1.5511623e-01	[ 1.2456264e+00]	 2.1530795e-01


.. parsed-literal::

      97	 1.3811204e+00	 1.3427023e-01	 1.4341029e+00	 1.5490588e-01	[ 1.2476068e+00]	 2.1660495e-01


.. parsed-literal::

      98	 1.3836682e+00	 1.3433733e-01	 1.4365777e+00	 1.5484319e-01	[ 1.2508263e+00]	 2.1211886e-01
      99	 1.3868442e+00	 1.3443401e-01	 1.4397505e+00	 1.5481853e-01	  1.2506858e+00 	 2.0016932e-01


.. parsed-literal::

     100	 1.3899211e+00	 1.3422569e-01	 1.4429607e+00	 1.5434859e-01	  1.2480752e+00 	 2.1466303e-01


.. parsed-literal::

     101	 1.3933223e+00	 1.3394336e-01	 1.4464641e+00	 1.5391849e-01	  1.2466614e+00 	 2.1019363e-01


.. parsed-literal::

     102	 1.3965408e+00	 1.3364359e-01	 1.4497972e+00	 1.5340755e-01	  1.2475906e+00 	 2.1115136e-01


.. parsed-literal::

     103	 1.3994034e+00	 1.3322153e-01	 1.4526816e+00	 1.5300484e-01	  1.2465911e+00 	 2.1532845e-01


.. parsed-literal::

     104	 1.4025261e+00	 1.3289374e-01	 1.4557460e+00	 1.5271655e-01	  1.2486820e+00 	 2.1949005e-01


.. parsed-literal::

     105	 1.4063546e+00	 1.3226360e-01	 1.4597310e+00	 1.5204376e-01	  1.2476460e+00 	 2.1320152e-01


.. parsed-literal::

     106	 1.4095067e+00	 1.3187144e-01	 1.4630293e+00	 1.5131254e-01	  1.2498728e+00 	 2.0880127e-01


.. parsed-literal::

     107	 1.4119355e+00	 1.3167179e-01	 1.4654428e+00	 1.5100614e-01	[ 1.2529389e+00]	 2.0958042e-01


.. parsed-literal::

     108	 1.4155585e+00	 1.3129558e-01	 1.4691563e+00	 1.5038596e-01	[ 1.2575929e+00]	 2.2643518e-01


.. parsed-literal::

     109	 1.4183600e+00	 1.3073424e-01	 1.4721760e+00	 1.4962198e-01	[ 1.2580058e+00]	 2.1411943e-01


.. parsed-literal::

     110	 1.4218566e+00	 1.3041160e-01	 1.4757025e+00	 1.4920746e-01	[ 1.2608632e+00]	 2.1247864e-01


.. parsed-literal::

     111	 1.4255084e+00	 1.3002325e-01	 1.4795066e+00	 1.4888082e-01	  1.2595578e+00 	 2.1543765e-01


.. parsed-literal::

     112	 1.4277928e+00	 1.2985196e-01	 1.4818794e+00	 1.4846472e-01	[ 1.2638061e+00]	 2.1549106e-01
     113	 1.4302015e+00	 1.2967668e-01	 1.4843136e+00	 1.4844906e-01	[ 1.2656022e+00]	 1.9355845e-01


.. parsed-literal::

     114	 1.4330707e+00	 1.2949362e-01	 1.4872504e+00	 1.4834155e-01	[ 1.2687155e+00]	 2.0700550e-01


.. parsed-literal::

     115	 1.4352439e+00	 1.2930306e-01	 1.4894368e+00	 1.4830922e-01	[ 1.2725636e+00]	 2.1682858e-01


.. parsed-literal::

     116	 1.4374299e+00	 1.2912893e-01	 1.4915936e+00	 1.4828029e-01	[ 1.2764062e+00]	 2.1307969e-01
     117	 1.4418925e+00	 1.2881881e-01	 1.4961089e+00	 1.4817447e-01	[ 1.2849746e+00]	 1.8171334e-01


.. parsed-literal::

     118	 1.4437244e+00	 1.2857769e-01	 1.4979830e+00	 1.4835768e-01	[ 1.2851016e+00]	 3.2791376e-01


.. parsed-literal::

     119	 1.4461557e+00	 1.2851051e-01	 1.5004263e+00	 1.4830815e-01	[ 1.2889072e+00]	 2.1433234e-01
     120	 1.4482167e+00	 1.2840230e-01	 1.5025540e+00	 1.4810933e-01	[ 1.2896956e+00]	 1.7390084e-01


.. parsed-literal::

     121	 1.4506660e+00	 1.2825550e-01	 1.5050931e+00	 1.4794123e-01	[ 1.2923080e+00]	 2.2085428e-01


.. parsed-literal::

     122	 1.4528113e+00	 1.2812386e-01	 1.5072597e+00	 1.4769986e-01	  1.2916165e+00 	 2.0435143e-01


.. parsed-literal::

     123	 1.4551484e+00	 1.2790811e-01	 1.5096556e+00	 1.4747221e-01	[ 1.2945109e+00]	 2.1044564e-01


.. parsed-literal::

     124	 1.4566968e+00	 1.2781227e-01	 1.5112913e+00	 1.4749674e-01	  1.2910876e+00 	 2.0483255e-01


.. parsed-literal::

     125	 1.4585621e+00	 1.2758325e-01	 1.5131619e+00	 1.4738271e-01	[ 1.2945193e+00]	 2.2012830e-01
     126	 1.4607210e+00	 1.2731023e-01	 1.5154136e+00	 1.4739868e-01	[ 1.2973103e+00]	 1.7551780e-01


.. parsed-literal::

     127	 1.4624712e+00	 1.2713003e-01	 1.5172432e+00	 1.4748952e-01	  1.2969859e+00 	 2.1330500e-01


.. parsed-literal::

     128	 1.4652869e+00	 1.2681152e-01	 1.5202456e+00	 1.4771069e-01	  1.2897111e+00 	 2.1075106e-01
     129	 1.4671492e+00	 1.2658993e-01	 1.5221940e+00	 1.4807025e-01	  1.2857308e+00 	 2.0690203e-01


.. parsed-literal::

     130	 1.4688408e+00	 1.2658860e-01	 1.5237766e+00	 1.4790029e-01	  1.2868857e+00 	 2.0878983e-01


.. parsed-literal::

     131	 1.4703773e+00	 1.2656523e-01	 1.5252851e+00	 1.4779011e-01	  1.2851706e+00 	 2.0468688e-01


.. parsed-literal::

     132	 1.4724091e+00	 1.2649063e-01	 1.5273725e+00	 1.4781614e-01	  1.2796734e+00 	 2.0420289e-01


.. parsed-literal::

     133	 1.4740793e+00	 1.2645831e-01	 1.5291966e+00	 1.4751385e-01	  1.2710904e+00 	 2.1732187e-01
     134	 1.4759778e+00	 1.2635961e-01	 1.5311168e+00	 1.4763888e-01	  1.2681700e+00 	 1.8355894e-01


.. parsed-literal::

     135	 1.4773130e+00	 1.2627655e-01	 1.5324805e+00	 1.4770333e-01	  1.2662289e+00 	 2.1170068e-01
     136	 1.4788039e+00	 1.2619454e-01	 1.5340211e+00	 1.4762747e-01	  1.2625309e+00 	 1.9899464e-01


.. parsed-literal::

     137	 1.4800297e+00	 1.2610809e-01	 1.5352824e+00	 1.4760819e-01	  1.2531553e+00 	 3.2329011e-01


.. parsed-literal::

     138	 1.4814730e+00	 1.2602203e-01	 1.5367074e+00	 1.4740413e-01	  1.2511909e+00 	 2.1391821e-01


.. parsed-literal::

     139	 1.4828313e+00	 1.2595564e-01	 1.5380480e+00	 1.4722463e-01	  1.2479011e+00 	 2.2569990e-01
     140	 1.4844927e+00	 1.2582421e-01	 1.5397138e+00	 1.4712206e-01	  1.2430632e+00 	 1.9558978e-01


.. parsed-literal::

     141	 1.4861735e+00	 1.2581227e-01	 1.5413997e+00	 1.4705417e-01	  1.2395782e+00 	 2.1226859e-01
     142	 1.4877114e+00	 1.2580433e-01	 1.5429656e+00	 1.4704704e-01	  1.2351113e+00 	 1.7776179e-01


.. parsed-literal::

     143	 1.4890899e+00	 1.2573045e-01	 1.5444359e+00	 1.4705374e-01	  1.2333527e+00 	 2.0712399e-01


.. parsed-literal::

     144	 1.4903692e+00	 1.2567913e-01	 1.5457415e+00	 1.4703215e-01	  1.2295470e+00 	 2.1410060e-01


.. parsed-literal::

     145	 1.4915965e+00	 1.2560252e-01	 1.5469884e+00	 1.4692373e-01	  1.2271220e+00 	 2.0623326e-01


.. parsed-literal::

     146	 1.4931732e+00	 1.2544817e-01	 1.5486031e+00	 1.4679419e-01	  1.2242781e+00 	 2.1145487e-01


.. parsed-literal::

     147	 1.4941147e+00	 1.2533624e-01	 1.5496198e+00	 1.4635432e-01	  1.2198741e+00 	 2.1421051e-01


.. parsed-literal::

     148	 1.4956464e+00	 1.2528068e-01	 1.5510955e+00	 1.4641962e-01	  1.2236458e+00 	 2.0588684e-01


.. parsed-literal::

     149	 1.4964914e+00	 1.2526190e-01	 1.5519258e+00	 1.4643296e-01	  1.2251837e+00 	 2.0296669e-01
     150	 1.4973911e+00	 1.2523502e-01	 1.5528331e+00	 1.4636898e-01	  1.2262550e+00 	 1.7827916e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.27 s, total: 2min 9s
    Wall time: 32.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3b28829030>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.2 s, sys: 56 ms, total: 2.26 s
    Wall time: 717 ms


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

