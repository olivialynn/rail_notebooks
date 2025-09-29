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
       1	-3.4853009e-01	 3.2224052e-01	-3.3887990e-01	 3.1401020e-01	[-3.2393235e-01]	 6.1036539e-01


.. parsed-literal::

       2	-2.7848924e-01	 3.1186396e-01	-2.5488014e-01	 3.0512845e-01	[-2.3503985e-01]	 3.0373740e-01


.. parsed-literal::

       3	-2.3373030e-01	 2.9060448e-01	-1.9145649e-01	 2.8466544e-01	[-1.6975536e-01]	 3.4362960e-01


.. parsed-literal::

       4	-1.9618136e-01	 2.6649734e-01	-1.5442442e-01	 2.6279457e-01	[-1.3572408e-01]	 2.2455716e-01


.. parsed-literal::

       5	-1.0414466e-01	 2.5706028e-01	-7.0447412e-02	 2.5576659e-01	[-6.6266722e-02]	 2.2732067e-01


.. parsed-literal::

       6	-7.1025478e-02	 2.5194558e-01	-4.1239335e-02	 2.5132019e-01	[-3.9507905e-02]	 2.3670650e-01


.. parsed-literal::

       7	-5.2843012e-02	 2.4908456e-01	-2.9276187e-02	 2.4903497e-01	[-2.9208421e-02]	 2.3512268e-01


.. parsed-literal::

       8	-4.0001126e-02	 2.4687901e-01	-2.0105449e-02	 2.4668701e-01	[-1.9533099e-02]	 2.2836351e-01


.. parsed-literal::

       9	-2.6795572e-02	 2.4440598e-01	-9.5830627e-03	 2.4429769e-01	[-9.5404072e-03]	 2.3955107e-01


.. parsed-literal::

      10	-1.5767302e-02	 2.4223124e-01	-2.5638827e-04	 2.4334211e-01	[-5.2086586e-03]	 2.3421955e-01


.. parsed-literal::

      11	-1.2274599e-02	 2.4153226e-01	 2.0225386e-03	 2.4362105e-01	 -8.0376369e-03 	 2.3702621e-01


.. parsed-literal::

      12	-8.5363506e-03	 2.4100735e-01	 5.7163902e-03	 2.4288300e-01	[-2.6202711e-03]	 2.1812820e-01


.. parsed-literal::

      13	-6.1722973e-03	 2.4049898e-01	 8.0052745e-03	 2.4224596e-01	[ 3.0905031e-04]	 2.2355461e-01


.. parsed-literal::

      14	-2.5709747e-03	 2.3979800e-01	 1.1640824e-02	 2.4134517e-01	[ 4.7589477e-03]	 2.2827363e-01


.. parsed-literal::

      15	 9.4742149e-02	 2.2492467e-01	 1.1422780e-01	 2.2623225e-01	[ 1.0569854e-01]	 3.7636614e-01


.. parsed-literal::

      16	 1.1109260e-01	 2.2465981e-01	 1.3309078e-01	 2.2501422e-01	[ 1.2476034e-01]	 3.8661003e-01


.. parsed-literal::

      17	 1.8875592e-01	 2.1816525e-01	 2.1264937e-01	 2.1774985e-01	[ 2.0834780e-01]	 2.2372937e-01


.. parsed-literal::

      18	 2.6166836e-01	 2.1253613e-01	 2.9332491e-01	 2.1300000e-01	[ 2.8357222e-01]	 2.3482370e-01


.. parsed-literal::

      19	 3.0163891e-01	 2.1195844e-01	 3.3160282e-01	 2.1198868e-01	[ 3.2773552e-01]	 2.2428703e-01


.. parsed-literal::

      20	 3.3313896e-01	 2.0881207e-01	 3.6236599e-01	 2.1074345e-01	[ 3.5477194e-01]	 2.2907615e-01


.. parsed-literal::

      21	 3.9286971e-01	 2.0651110e-01	 4.2239614e-01	 2.0850906e-01	[ 4.1657464e-01]	 2.1371818e-01


.. parsed-literal::

      22	 5.2335723e-01	 2.0759196e-01	 5.5658093e-01	 2.0814973e-01	[ 5.5393864e-01]	 2.1732497e-01


.. parsed-literal::

      23	 5.5449813e-01	 2.1402894e-01	 5.9575921e-01	 2.1120994e-01	[ 6.0296561e-01]	 2.2372055e-01


.. parsed-literal::

      24	 6.1408719e-01	 2.0552018e-01	 6.5145276e-01	 2.0645577e-01	[ 6.4438007e-01]	 2.2180486e-01


.. parsed-literal::

      25	 6.4628518e-01	 2.0255714e-01	 6.8236648e-01	 2.0280658e-01	[ 6.7682424e-01]	 2.3130751e-01


.. parsed-literal::

      26	 6.6931933e-01	 1.9810587e-01	 7.0499313e-01	 1.9911856e-01	[ 6.9946718e-01]	 3.8300967e-01


.. parsed-literal::

      27	 6.8742111e-01	 1.9561103e-01	 7.2340846e-01	 1.9572308e-01	[ 7.2506042e-01]	 2.3137498e-01


.. parsed-literal::

      28	 7.1438379e-01	 1.9525728e-01	 7.5043337e-01	 1.9543632e-01	[ 7.4745435e-01]	 2.0967627e-01


.. parsed-literal::

      29	 7.3083309e-01	 1.9578571e-01	 7.6715142e-01	 1.9569153e-01	[ 7.6317957e-01]	 2.2789001e-01


.. parsed-literal::

      30	 7.4295263e-01	 1.9565942e-01	 7.7904358e-01	 1.9573116e-01	[ 7.7157801e-01]	 2.2354579e-01
      31	 7.6254256e-01	 1.9756085e-01	 7.9857566e-01	 1.9743430e-01	[ 7.8674410e-01]	 2.0257926e-01


.. parsed-literal::

      32	 7.8201509e-01	 2.0111846e-01	 8.1913607e-01	 2.0039028e-01	[ 7.9454313e-01]	 2.3271489e-01


.. parsed-literal::

      33	 8.0558262e-01	 2.0051714e-01	 8.4308286e-01	 1.9838634e-01	[ 8.1446947e-01]	 2.3034954e-01


.. parsed-literal::

      34	 8.2632159e-01	 1.9812799e-01	 8.6461822e-01	 1.9470686e-01	[ 8.3728945e-01]	 2.3211408e-01


.. parsed-literal::

      35	 8.5058523e-01	 1.9716782e-01	 8.8978507e-01	 1.9266484e-01	[ 8.6504693e-01]	 2.1156335e-01


.. parsed-literal::

      36	 8.6847622e-01	 1.9616389e-01	 9.0933901e-01	 1.9149441e-01	[ 8.8359931e-01]	 2.2567534e-01


.. parsed-literal::

      37	 8.8592696e-01	 1.9591287e-01	 9.2635336e-01	 1.9195692e-01	[ 9.0242166e-01]	 2.2199965e-01


.. parsed-literal::

      38	 8.9848893e-01	 1.9506389e-01	 9.3834716e-01	 1.9027891e-01	[ 9.1908467e-01]	 2.2919059e-01


.. parsed-literal::

      39	 9.0833570e-01	 1.9481887e-01	 9.4854669e-01	 1.8975005e-01	[ 9.2574519e-01]	 2.2887611e-01


.. parsed-literal::

      40	 9.2150875e-01	 1.9513856e-01	 9.6314745e-01	 1.8950136e-01	[ 9.3434210e-01]	 2.2765851e-01


.. parsed-literal::

      41	 9.3633849e-01	 1.9532032e-01	 9.7863926e-01	 1.8913594e-01	[ 9.5002074e-01]	 2.2852969e-01


.. parsed-literal::

      42	 9.5516929e-01	 1.9594886e-01	 9.9871335e-01	 1.8889946e-01	[ 9.6765175e-01]	 2.3544669e-01


.. parsed-literal::

      43	 9.6855760e-01	 1.9447663e-01	 1.0123476e+00	 1.8764277e-01	[ 9.8352339e-01]	 2.2158289e-01


.. parsed-literal::

      44	 9.7805144e-01	 1.9378549e-01	 1.0214770e+00	 1.8696161e-01	[ 9.9271247e-01]	 2.2642493e-01


.. parsed-literal::

      45	 9.8702740e-01	 1.9187200e-01	 1.0301974e+00	 1.8535849e-01	[ 9.9992274e-01]	 2.2502804e-01


.. parsed-literal::

      46	 9.9996230e-01	 1.9014937e-01	 1.0436187e+00	 1.8361057e-01	[ 1.0069172e+00]	 2.2192788e-01


.. parsed-literal::

      47	 1.0139597e+00	 1.8745423e-01	 1.0581973e+00	 1.8134138e-01	  9.9985555e-01 	 2.3097634e-01


.. parsed-literal::

      48	 1.0211122e+00	 1.8739956e-01	 1.0656494e+00	 1.8139752e-01	  9.9915275e-01 	 2.3010182e-01


.. parsed-literal::

      49	 1.0267125e+00	 1.8729823e-01	 1.0713375e+00	 1.8138004e-01	  9.9782045e-01 	 2.2443795e-01


.. parsed-literal::

      50	 1.0341030e+00	 1.8728003e-01	 1.0787625e+00	 1.8145213e-01	  1.0017925e+00 	 2.2459483e-01


.. parsed-literal::

      51	 1.0439030e+00	 1.8690221e-01	 1.0887987e+00	 1.8114131e-01	  1.0016475e+00 	 2.3316622e-01


.. parsed-literal::

      52	 1.0522739e+00	 1.8674984e-01	 1.0974572e+00	 1.8151730e-01	  1.0013804e+00 	 2.0583677e-01


.. parsed-literal::

      53	 1.0584708e+00	 1.8627728e-01	 1.1035625e+00	 1.8066393e-01	[ 1.0118980e+00]	 2.1386504e-01


.. parsed-literal::

      54	 1.0630824e+00	 1.8534582e-01	 1.1082219e+00	 1.7968092e-01	[ 1.0153797e+00]	 2.3124504e-01


.. parsed-literal::

      55	 1.0707221e+00	 1.8307085e-01	 1.1164222e+00	 1.7769644e-01	[ 1.0160057e+00]	 2.3731828e-01


.. parsed-literal::

      56	 1.0771789e+00	 1.8145182e-01	 1.1234498e+00	 1.7545699e-01	[ 1.0276496e+00]	 2.2548342e-01


.. parsed-literal::

      57	 1.0848754e+00	 1.8047417e-01	 1.1312186e+00	 1.7472398e-01	[ 1.0363345e+00]	 2.2247171e-01


.. parsed-literal::

      58	 1.0914287e+00	 1.7915337e-01	 1.1379586e+00	 1.7346095e-01	[ 1.0402373e+00]	 2.1562958e-01


.. parsed-literal::

      59	 1.0966011e+00	 1.7860006e-01	 1.1433139e+00	 1.7285959e-01	[ 1.0419507e+00]	 2.1882415e-01


.. parsed-literal::

      60	 1.1047380e+00	 1.7663060e-01	 1.1520557e+00	 1.7098618e-01	  1.0367616e+00 	 2.2838545e-01


.. parsed-literal::

      61	 1.1117901e+00	 1.7670248e-01	 1.1593109e+00	 1.7128659e-01	  1.0358216e+00 	 2.2158432e-01


.. parsed-literal::

      62	 1.1169045e+00	 1.7611797e-01	 1.1643934e+00	 1.7087548e-01	  1.0350819e+00 	 2.3058248e-01


.. parsed-literal::

      63	 1.1278856e+00	 1.7439426e-01	 1.1759481e+00	 1.6973395e-01	  1.0263461e+00 	 2.3144412e-01


.. parsed-literal::

      64	 1.1341687e+00	 1.7211667e-01	 1.1824704e+00	 1.6835023e-01	  1.0121934e+00 	 2.2869349e-01


.. parsed-literal::

      65	 1.1419199e+00	 1.7194415e-01	 1.1899316e+00	 1.6819712e-01	  1.0229021e+00 	 2.2299981e-01


.. parsed-literal::

      66	 1.1488708e+00	 1.7160682e-01	 1.1970383e+00	 1.6817953e-01	  1.0260627e+00 	 2.3712635e-01


.. parsed-literal::

      67	 1.1565296e+00	 1.7109152e-01	 1.2048950e+00	 1.6768997e-01	  1.0213128e+00 	 2.2567964e-01


.. parsed-literal::

      68	 1.1639698e+00	 1.7063405e-01	 1.2128731e+00	 1.6719958e-01	  1.0198311e+00 	 2.3586655e-01


.. parsed-literal::

      69	 1.1720828e+00	 1.7021278e-01	 1.2209182e+00	 1.6671195e-01	  1.0103431e+00 	 2.3212218e-01


.. parsed-literal::

      70	 1.1767754e+00	 1.6938313e-01	 1.2256943e+00	 1.6591625e-01	  1.0008612e+00 	 2.3334861e-01


.. parsed-literal::

      71	 1.1854248e+00	 1.6794465e-01	 1.2347497e+00	 1.6460237e-01	  9.8276542e-01 	 2.1656585e-01


.. parsed-literal::

      72	 1.1899849e+00	 1.6671060e-01	 1.2398348e+00	 1.6313512e-01	  9.6651353e-01 	 2.3028064e-01


.. parsed-literal::

      73	 1.1971806e+00	 1.6649003e-01	 1.2468883e+00	 1.6332148e-01	  9.7801026e-01 	 2.5081849e-01


.. parsed-literal::

      74	 1.2008319e+00	 1.6616552e-01	 1.2505520e+00	 1.6336851e-01	  9.8301698e-01 	 2.2211289e-01


.. parsed-literal::

      75	 1.2072363e+00	 1.6508562e-01	 1.2572879e+00	 1.6302484e-01	  9.7747496e-01 	 2.1754837e-01


.. parsed-literal::

      76	 1.2155925e+00	 1.6369056e-01	 1.2659998e+00	 1.6242549e-01	  9.7045729e-01 	 2.2316861e-01


.. parsed-literal::

      77	 1.2213535e+00	 1.6226619e-01	 1.2720714e+00	 1.6184917e-01	  9.5303808e-01 	 3.7860298e-01


.. parsed-literal::

      78	 1.2293680e+00	 1.6161484e-01	 1.2801644e+00	 1.6127543e-01	  9.5302775e-01 	 2.2542787e-01


.. parsed-literal::

      79	 1.2358912e+00	 1.6107456e-01	 1.2869403e+00	 1.6041445e-01	  9.4473149e-01 	 2.2041011e-01


.. parsed-literal::

      80	 1.2400590e+00	 1.6130160e-01	 1.2910259e+00	 1.6032780e-01	  9.4960589e-01 	 2.3661971e-01


.. parsed-literal::

      81	 1.2452296e+00	 1.6062823e-01	 1.2962700e+00	 1.5983009e-01	  9.4625252e-01 	 2.2419214e-01


.. parsed-literal::

      82	 1.2533077e+00	 1.5923527e-01	 1.3047119e+00	 1.5888735e-01	  9.3899585e-01 	 2.1981859e-01


.. parsed-literal::

      83	 1.2577262e+00	 1.5899008e-01	 1.3091620e+00	 1.5845112e-01	  9.4108357e-01 	 2.3083377e-01


.. parsed-literal::

      84	 1.2629239e+00	 1.5855317e-01	 1.3143361e+00	 1.5773955e-01	  9.4603523e-01 	 2.2263598e-01


.. parsed-literal::

      85	 1.2692565e+00	 1.5805505e-01	 1.3209752e+00	 1.5722505e-01	  9.3447691e-01 	 2.3555517e-01


.. parsed-literal::

      86	 1.2746437e+00	 1.5675962e-01	 1.3266615e+00	 1.5562803e-01	  9.0977923e-01 	 2.2059941e-01


.. parsed-literal::

      87	 1.2799361e+00	 1.5579632e-01	 1.3320683e+00	 1.5505340e-01	  8.9094216e-01 	 2.2653317e-01


.. parsed-literal::

      88	 1.2870851e+00	 1.5456454e-01	 1.3394871e+00	 1.5435458e-01	  8.4704215e-01 	 2.3383665e-01


.. parsed-literal::

      89	 1.2897700e+00	 1.5427845e-01	 1.3424290e+00	 1.5416476e-01	  8.0837127e-01 	 2.4036551e-01


.. parsed-literal::

      90	 1.2940757e+00	 1.5385232e-01	 1.3465498e+00	 1.5381049e-01	  8.1863762e-01 	 2.2810864e-01


.. parsed-literal::

      91	 1.2972893e+00	 1.5342573e-01	 1.3497352e+00	 1.5344666e-01	  8.1901613e-01 	 2.2417068e-01


.. parsed-literal::

      92	 1.3006572e+00	 1.5310303e-01	 1.3530897e+00	 1.5316166e-01	  8.1470819e-01 	 2.2582793e-01


.. parsed-literal::

      93	 1.3073244e+00	 1.5271065e-01	 1.3598714e+00	 1.5255961e-01	  7.9365763e-01 	 2.2308111e-01


.. parsed-literal::

      94	 1.3084831e+00	 1.5230976e-01	 1.3613909e+00	 1.5191176e-01	  7.5174243e-01 	 2.4072957e-01


.. parsed-literal::

      95	 1.3181414e+00	 1.5197663e-01	 1.3707972e+00	 1.5135391e-01	  7.5259132e-01 	 2.1435738e-01


.. parsed-literal::

      96	 1.3210414e+00	 1.5195731e-01	 1.3737061e+00	 1.5124235e-01	  7.4644664e-01 	 2.1476960e-01


.. parsed-literal::

      97	 1.3256462e+00	 1.5160411e-01	 1.3784963e+00	 1.5060470e-01	  7.2334953e-01 	 2.0709038e-01


.. parsed-literal::

      98	 1.3296029e+00	 1.5257869e-01	 1.3827059e+00	 1.5140996e-01	  7.0775425e-01 	 2.1039915e-01


.. parsed-literal::

      99	 1.3343501e+00	 1.5164482e-01	 1.3875012e+00	 1.5034720e-01	  7.0227270e-01 	 2.0531678e-01


.. parsed-literal::

     100	 1.3381255e+00	 1.5078168e-01	 1.3913009e+00	 1.4957987e-01	  7.0791400e-01 	 2.1755958e-01


.. parsed-literal::

     101	 1.3415576e+00	 1.5018453e-01	 1.3948124e+00	 1.4910682e-01	  7.0186833e-01 	 2.2746134e-01


.. parsed-literal::

     102	 1.3439810e+00	 1.5019591e-01	 1.3974584e+00	 1.4917314e-01	  7.2049452e-01 	 2.1782470e-01


.. parsed-literal::

     103	 1.3472875e+00	 1.4962493e-01	 1.4007210e+00	 1.4868627e-01	  7.1382019e-01 	 2.3241663e-01


.. parsed-literal::

     104	 1.3489444e+00	 1.4950526e-01	 1.4023530e+00	 1.4855685e-01	  7.1288562e-01 	 2.2812557e-01


.. parsed-literal::

     105	 1.3520599e+00	 1.4889262e-01	 1.4055287e+00	 1.4803084e-01	  7.1615036e-01 	 2.2067809e-01


.. parsed-literal::

     106	 1.3576219e+00	 1.4781051e-01	 1.4111178e+00	 1.4727006e-01	  7.2535779e-01 	 2.2702956e-01


.. parsed-literal::

     107	 1.3599045e+00	 1.4567873e-01	 1.4136619e+00	 1.4563966e-01	  7.6794586e-01 	 2.2743392e-01


.. parsed-literal::

     108	 1.3638913e+00	 1.4586637e-01	 1.4174211e+00	 1.4582618e-01	  7.6325624e-01 	 2.3755622e-01


.. parsed-literal::

     109	 1.3658182e+00	 1.4587454e-01	 1.4192839e+00	 1.4585871e-01	  7.6547928e-01 	 2.3776054e-01


.. parsed-literal::

     110	 1.3691530e+00	 1.4538548e-01	 1.4227610e+00	 1.4533444e-01	  7.7465934e-01 	 2.2019100e-01


.. parsed-literal::

     111	 1.3715573e+00	 1.4564241e-01	 1.4253857e+00	 1.4553717e-01	  7.9076029e-01 	 2.2870827e-01


.. parsed-literal::

     112	 1.3749392e+00	 1.4494121e-01	 1.4287154e+00	 1.4470104e-01	  8.0465284e-01 	 2.1463799e-01


.. parsed-literal::

     113	 1.3774601e+00	 1.4434032e-01	 1.4312367e+00	 1.4401075e-01	  8.1503210e-01 	 2.1540332e-01


.. parsed-literal::

     114	 1.3803763e+00	 1.4392005e-01	 1.4341810e+00	 1.4355438e-01	  8.2109183e-01 	 2.2131705e-01


.. parsed-literal::

     115	 1.3854967e+00	 1.4274086e-01	 1.4395014e+00	 1.4253666e-01	  8.2830142e-01 	 2.2106361e-01


.. parsed-literal::

     116	 1.3870191e+00	 1.4283538e-01	 1.4413236e+00	 1.4283776e-01	  8.4005717e-01 	 2.1175599e-01


.. parsed-literal::

     117	 1.3918997e+00	 1.4244352e-01	 1.4458864e+00	 1.4244645e-01	  8.4021228e-01 	 2.2620058e-01


.. parsed-literal::

     118	 1.3935842e+00	 1.4222866e-01	 1.4475656e+00	 1.4229298e-01	  8.3911238e-01 	 2.3153973e-01


.. parsed-literal::

     119	 1.3962776e+00	 1.4185363e-01	 1.4503415e+00	 1.4196422e-01	  8.3910873e-01 	 2.2043896e-01


.. parsed-literal::

     120	 1.3981395e+00	 1.4104238e-01	 1.4523242e+00	 1.4135826e-01	  8.1121892e-01 	 2.3307586e-01


.. parsed-literal::

     121	 1.4005634e+00	 1.4097963e-01	 1.4546976e+00	 1.4121238e-01	  8.2706872e-01 	 2.3168206e-01


.. parsed-literal::

     122	 1.4029174e+00	 1.4060404e-01	 1.4570695e+00	 1.4078930e-01	  8.3476812e-01 	 2.3169088e-01


.. parsed-literal::

     123	 1.4049151e+00	 1.4023510e-01	 1.4590876e+00	 1.4045868e-01	  8.2692568e-01 	 2.2687960e-01


.. parsed-literal::

     124	 1.4076771e+00	 1.3921022e-01	 1.4620402e+00	 1.3946532e-01	  7.7565363e-01 	 2.3521209e-01


.. parsed-literal::

     125	 1.4107520e+00	 1.3878632e-01	 1.4650851e+00	 1.3916147e-01	  7.4676097e-01 	 2.0385814e-01


.. parsed-literal::

     126	 1.4124679e+00	 1.3879260e-01	 1.4667516e+00	 1.3922346e-01	  7.3975313e-01 	 2.1814275e-01


.. parsed-literal::

     127	 1.4159054e+00	 1.3837178e-01	 1.4703210e+00	 1.3900828e-01	  6.8322120e-01 	 2.3612237e-01


.. parsed-literal::

     128	 1.4166433e+00	 1.3846304e-01	 1.4714234e+00	 1.3913319e-01	  6.0895343e-01 	 2.2337580e-01


.. parsed-literal::

     129	 1.4194259e+00	 1.3823993e-01	 1.4739536e+00	 1.3903008e-01	  6.3754817e-01 	 2.1842074e-01


.. parsed-literal::

     130	 1.4206705e+00	 1.3800926e-01	 1.4752157e+00	 1.3888955e-01	  6.2616574e-01 	 2.2234559e-01


.. parsed-literal::

     131	 1.4224779e+00	 1.3776849e-01	 1.4770779e+00	 1.3877639e-01	  5.9898260e-01 	 2.2516727e-01


.. parsed-literal::

     132	 1.4249330e+00	 1.3752336e-01	 1.4796329e+00	 1.3867654e-01	  5.5146074e-01 	 2.2737002e-01


.. parsed-literal::

     133	 1.4269392e+00	 1.3731614e-01	 1.4818090e+00	 1.3847303e-01	  4.6939609e-01 	 2.2535372e-01


.. parsed-literal::

     134	 1.4293850e+00	 1.3732266e-01	 1.4841887e+00	 1.3847695e-01	  4.6896337e-01 	 2.0722198e-01


.. parsed-literal::

     135	 1.4309167e+00	 1.3737112e-01	 1.4857445e+00	 1.3846122e-01	  4.6533164e-01 	 2.1653175e-01


.. parsed-literal::

     136	 1.4325308e+00	 1.3730093e-01	 1.4874438e+00	 1.3831168e-01	  4.4407331e-01 	 2.2544241e-01


.. parsed-literal::

     137	 1.4355570e+00	 1.3693739e-01	 1.4907191e+00	 1.3787934e-01	  4.0045363e-01 	 2.2780013e-01


.. parsed-literal::

     138	 1.4373836e+00	 1.3664653e-01	 1.4927443e+00	 1.3717874e-01	  3.0249238e-01 	 2.1541142e-01


.. parsed-literal::

     139	 1.4392827e+00	 1.3661148e-01	 1.4944831e+00	 1.3724155e-01	  3.3390474e-01 	 2.1423697e-01


.. parsed-literal::

     140	 1.4406061e+00	 1.3647654e-01	 1.4957576e+00	 1.3709534e-01	  3.3850640e-01 	 2.2758484e-01


.. parsed-literal::

     141	 1.4424047e+00	 1.3636463e-01	 1.4975193e+00	 1.3682807e-01	  3.4068065e-01 	 2.2111630e-01


.. parsed-literal::

     142	 1.4436824e+00	 1.3603096e-01	 1.4989939e+00	 1.3626034e-01	  3.1322870e-01 	 2.0846272e-01


.. parsed-literal::

     143	 1.4466204e+00	 1.3607996e-01	 1.5017816e+00	 1.3614132e-01	  3.1730646e-01 	 2.2738695e-01
     144	 1.4477242e+00	 1.3608898e-01	 1.5029275e+00	 1.3609876e-01	  3.0534870e-01 	 2.0354915e-01


.. parsed-literal::

     145	 1.4492095e+00	 1.3603460e-01	 1.5045111e+00	 1.3605371e-01	  2.8560740e-01 	 2.2397208e-01
     146	 1.4514812e+00	 1.3568578e-01	 1.5068807e+00	 1.3587030e-01	  2.4942331e-01 	 1.9527841e-01


.. parsed-literal::

     147	 1.4531146e+00	 1.3556363e-01	 1.5086669e+00	 1.3618339e-01	  2.2326795e-01 	 3.6033702e-01


.. parsed-literal::

     148	 1.4551167e+00	 1.3503745e-01	 1.5106606e+00	 1.3597798e-01	  1.8420875e-01 	 2.0422316e-01


.. parsed-literal::

     149	 1.4562835e+00	 1.3475311e-01	 1.5117891e+00	 1.3592749e-01	  1.5863972e-01 	 2.1399355e-01


.. parsed-literal::

     150	 1.4580436e+00	 1.3444303e-01	 1.5135162e+00	 1.3605946e-01	  1.0749816e-01 	 2.2089124e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 19s, sys: 1.18 s, total: 2min 20s
    Wall time: 35.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f83c8f30df0>



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
    CPU times: user 1.8 s, sys: 66.9 ms, total: 1.87 s
    Wall time: 621 ms


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

