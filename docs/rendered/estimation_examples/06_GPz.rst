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
       1	-3.4260593e-01	 3.2049306e-01	-3.3287163e-01	 3.2131392e-01	[-3.3354918e-01]	 4.6516919e-01


.. parsed-literal::

       2	-2.7195145e-01	 3.0962726e-01	-2.4744317e-01	 3.0977469e-01	[-2.4827934e-01]	 2.3219419e-01


.. parsed-literal::

       3	-2.2736874e-01	 2.8870687e-01	-1.8400824e-01	 2.9196308e-01	[-1.9603632e-01]	 2.9166555e-01
       4	-1.9953258e-01	 2.6549737e-01	-1.5727980e-01	 2.7389426e-01	 -2.0058236e-01 	 1.8888831e-01


.. parsed-literal::

       5	-1.0857429e-01	 2.5795835e-01	-7.2879225e-02	 2.6146084e-01	[-8.9894739e-02]	 2.0832992e-01


.. parsed-literal::

       6	-7.1410792e-02	 2.5134863e-01	-3.9290705e-02	 2.5625318e-01	[-5.7319180e-02]	 2.0837188e-01


.. parsed-literal::

       7	-5.3461287e-02	 2.4881217e-01	-2.8202436e-02	 2.5307114e-01	[-4.5137882e-02]	 2.0645118e-01


.. parsed-literal::

       8	-3.8925785e-02	 2.4634800e-01	-1.7927994e-02	 2.5015140e-01	[-3.3640188e-02]	 2.1327639e-01


.. parsed-literal::

       9	-2.2880492e-02	 2.4331719e-01	-4.9805066e-03	 2.4658491e-01	[-1.8748945e-02]	 2.0219517e-01
      10	-1.1102466e-02	 2.4114116e-01	 4.4916834e-03	 2.4529619e-01	[-1.3580366e-02]	 1.8008184e-01


.. parsed-literal::

      11	-6.0493188e-03	 2.4049459e-01	 8.2147786e-03	 2.4478556e-01	[-1.0131492e-02]	 2.1434355e-01
      12	-3.4490214e-03	 2.3994794e-01	 1.0549315e-02	 2.4439978e-01	[-8.5302198e-03]	 2.0013833e-01


.. parsed-literal::

      13	-1.3979043e-04	 2.3940132e-01	 1.3603258e-02	 2.4405605e-01	[-6.9252700e-03]	 2.0240545e-01
      14	 6.4563483e-03	 2.3807087e-01	 2.0864157e-02	 2.4296676e-01	[-1.8412198e-04]	 1.7789268e-01


.. parsed-literal::

      15	 8.1715336e-02	 2.2526487e-01	 1.0017015e-01	 2.3054220e-01	[ 8.3326512e-02]	 4.5268774e-01


.. parsed-literal::

      16	 1.3339263e-01	 2.1891233e-01	 1.5463324e-01	 2.2898111e-01	[ 1.3346829e-01]	 2.0969582e-01


.. parsed-literal::

      17	 2.6991850e-01	 2.1583032e-01	 2.9675227e-01	 2.2691932e-01	[ 2.6109247e-01]	 2.1172237e-01


.. parsed-literal::

      18	 3.2958720e-01	 2.1012529e-01	 3.6157781e-01	 2.1539917e-01	[ 3.2021638e-01]	 2.1651196e-01


.. parsed-literal::

      19	 3.7665433e-01	 2.0487714e-01	 4.1035380e-01	 2.1106026e-01	[ 3.6597765e-01]	 2.1484280e-01


.. parsed-literal::

      20	 4.2836283e-01	 2.0248945e-01	 4.6149139e-01	 2.1110788e-01	[ 4.1370632e-01]	 2.0619297e-01
      21	 4.7242841e-01	 1.9900750e-01	 5.0494141e-01	 2.0673573e-01	[ 4.6268142e-01]	 1.9245577e-01


.. parsed-literal::

      22	 5.6974410e-01	 1.9669052e-01	 6.0287323e-01	 2.0362131e-01	[ 5.6112578e-01]	 1.7008090e-01


.. parsed-literal::

      23	 6.0274644e-01	 2.0338255e-01	 6.4402995e-01	 2.0949713e-01	[ 5.8277333e-01]	 2.1156549e-01


.. parsed-literal::

      24	 6.8331477e-01	 1.9446325e-01	 7.2070741e-01	 1.9982744e-01	[ 6.8294955e-01]	 2.0851374e-01
      25	 7.1067060e-01	 1.9246306e-01	 7.4717102e-01	 1.9716142e-01	[ 7.0716526e-01]	 1.9130611e-01


.. parsed-literal::

      26	 7.2802617e-01	 1.9298834e-01	 7.6414401e-01	 1.9730433e-01	[ 7.1554170e-01]	 1.8033147e-01
      27	 7.5654834e-01	 1.9063121e-01	 7.9291648e-01	 1.9323596e-01	[ 7.4478117e-01]	 1.9158268e-01


.. parsed-literal::

      28	 7.8361631e-01	 1.8813029e-01	 8.2080763e-01	 1.9037367e-01	[ 7.7236821e-01]	 2.1210766e-01
      29	 8.0736866e-01	 1.8896175e-01	 8.4582287e-01	 1.8994456e-01	[ 8.0091019e-01]	 1.9352388e-01


.. parsed-literal::

      30	 8.2921578e-01	 1.8429734e-01	 8.6865470e-01	 1.8768331e-01	[ 8.1469624e-01]	 2.1380949e-01
      31	 8.5681246e-01	 1.8458763e-01	 8.9595138e-01	 1.8754454e-01	[ 8.3457919e-01]	 1.8308330e-01


.. parsed-literal::

      32	 8.7453173e-01	 1.8404177e-01	 9.1398691e-01	 1.8727588e-01	[ 8.4855027e-01]	 1.8714809e-01
      33	 8.9148114e-01	 1.8230463e-01	 9.3128569e-01	 1.8574330e-01	[ 8.6450645e-01]	 1.9166040e-01


.. parsed-literal::

      34	 9.0640351e-01	 1.8056471e-01	 9.4693771e-01	 1.8350038e-01	[ 8.7990948e-01]	 1.7245412e-01


.. parsed-literal::

      35	 9.2245465e-01	 1.7945897e-01	 9.6344817e-01	 1.8222610e-01	[ 8.9960348e-01]	 2.2493124e-01


.. parsed-literal::

      36	 9.3735612e-01	 1.7970932e-01	 9.7899368e-01	 1.8219159e-01	[ 9.1943855e-01]	 2.0995545e-01
      37	 9.5362328e-01	 1.7777861e-01	 9.9489511e-01	 1.8072261e-01	[ 9.3283275e-01]	 1.8797779e-01


.. parsed-literal::

      38	 9.6302589e-01	 1.7668331e-01	 1.0042131e+00	 1.7960008e-01	[ 9.3931657e-01]	 2.0787859e-01


.. parsed-literal::

      39	 9.8232079e-01	 1.7480117e-01	 1.0240107e+00	 1.7797904e-01	[ 9.5408358e-01]	 2.0659375e-01


.. parsed-literal::

      40	 9.9005203e-01	 1.7373428e-01	 1.0328996e+00	 1.7616688e-01	  9.5060352e-01 	 2.1289325e-01


.. parsed-literal::

      41	 1.0059753e+00	 1.7220335e-01	 1.0486847e+00	 1.7479456e-01	[ 9.7082540e-01]	 2.1906948e-01


.. parsed-literal::

      42	 1.0126561e+00	 1.7128952e-01	 1.0553665e+00	 1.7388358e-01	[ 9.7883962e-01]	 2.1961737e-01
      43	 1.0263185e+00	 1.6895072e-01	 1.0696420e+00	 1.7149066e-01	[ 9.9427796e-01]	 1.9390345e-01


.. parsed-literal::

      44	 1.0378890e+00	 1.6715350e-01	 1.0815833e+00	 1.7047905e-01	[ 1.0006453e+00]	 2.1320486e-01


.. parsed-literal::

      45	 1.0506089e+00	 1.6530722e-01	 1.0946849e+00	 1.6940154e-01	[ 1.0133213e+00]	 2.1364141e-01


.. parsed-literal::

      46	 1.0627158e+00	 1.6377012e-01	 1.1075699e+00	 1.6874058e-01	[ 1.0207335e+00]	 2.1271849e-01


.. parsed-literal::

      47	 1.0746474e+00	 1.6251708e-01	 1.1200148e+00	 1.6771384e-01	[ 1.0310216e+00]	 2.0782208e-01


.. parsed-literal::

      48	 1.0828364e+00	 1.6098534e-01	 1.1291328e+00	 1.6516139e-01	[ 1.0358815e+00]	 2.1365261e-01
      49	 1.0928210e+00	 1.6084467e-01	 1.1385724e+00	 1.6495077e-01	[ 1.0494279e+00]	 1.7360282e-01


.. parsed-literal::

      50	 1.0979853e+00	 1.6016802e-01	 1.1436977e+00	 1.6414095e-01	[ 1.0544716e+00]	 2.1259737e-01


.. parsed-literal::

      51	 1.1055775e+00	 1.5903999e-01	 1.1513164e+00	 1.6247464e-01	[ 1.0615152e+00]	 2.1483111e-01


.. parsed-literal::

      52	 1.1149796e+00	 1.5801258e-01	 1.1610931e+00	 1.6066999e-01	[ 1.0666973e+00]	 2.1169209e-01
      53	 1.1214735e+00	 1.5678847e-01	 1.1679470e+00	 1.5855890e-01	[ 1.0708637e+00]	 1.9244075e-01


.. parsed-literal::

      54	 1.1299567e+00	 1.5672839e-01	 1.1763439e+00	 1.5845705e-01	[ 1.0768369e+00]	 2.2433662e-01


.. parsed-literal::

      55	 1.1358794e+00	 1.5636391e-01	 1.1822258e+00	 1.5815520e-01	[ 1.0793914e+00]	 2.1847177e-01


.. parsed-literal::

      56	 1.1437094e+00	 1.5537199e-01	 1.1900807e+00	 1.5730214e-01	[ 1.0820808e+00]	 2.1039534e-01
      57	 1.1552616e+00	 1.5314541e-01	 1.2018717e+00	 1.5606190e-01	[ 1.0846808e+00]	 1.8163109e-01


.. parsed-literal::

      58	 1.1571500e+00	 1.5159657e-01	 1.2041449e+00	 1.5559270e-01	  1.0790912e+00 	 2.1470904e-01


.. parsed-literal::

      59	 1.1663417e+00	 1.5116776e-01	 1.2129970e+00	 1.5499830e-01	[ 1.0903869e+00]	 2.1380401e-01


.. parsed-literal::

      60	 1.1698996e+00	 1.5074317e-01	 1.2165955e+00	 1.5464082e-01	[ 1.0926126e+00]	 2.1532059e-01


.. parsed-literal::

      61	 1.1754577e+00	 1.4980887e-01	 1.2222219e+00	 1.5373951e-01	[ 1.0954923e+00]	 2.1529508e-01


.. parsed-literal::

      62	 1.1797530e+00	 1.4873626e-01	 1.2268514e+00	 1.5272769e-01	  1.0915480e+00 	 2.0953727e-01


.. parsed-literal::

      63	 1.1869976e+00	 1.4811710e-01	 1.2339948e+00	 1.5218724e-01	  1.0940783e+00 	 2.2072625e-01


.. parsed-literal::

      64	 1.1909270e+00	 1.4762673e-01	 1.2379575e+00	 1.5176912e-01	  1.0932118e+00 	 2.1327829e-01


.. parsed-literal::

      65	 1.1952184e+00	 1.4692833e-01	 1.2423874e+00	 1.5122182e-01	  1.0930997e+00 	 2.1732879e-01
      66	 1.1985431e+00	 1.4570653e-01	 1.2462410e+00	 1.5049110e-01	  1.0844585e+00 	 1.7606640e-01


.. parsed-literal::

      67	 1.2047333e+00	 1.4523974e-01	 1.2523132e+00	 1.4984164e-01	  1.0936197e+00 	 2.0617938e-01


.. parsed-literal::

      68	 1.2086523e+00	 1.4482311e-01	 1.2562626e+00	 1.4934546e-01	[ 1.0987119e+00]	 2.0961976e-01


.. parsed-literal::

      69	 1.2135873e+00	 1.4414213e-01	 1.2613234e+00	 1.4859278e-01	[ 1.1039484e+00]	 2.1991205e-01


.. parsed-literal::

      70	 1.2184299e+00	 1.4332891e-01	 1.2664484e+00	 1.4791002e-01	  1.1037068e+00 	 2.1894169e-01


.. parsed-literal::

      71	 1.2241562e+00	 1.4274913e-01	 1.2723154e+00	 1.4741752e-01	[ 1.1070162e+00]	 2.1242404e-01


.. parsed-literal::

      72	 1.2279223e+00	 1.4250108e-01	 1.2761096e+00	 1.4734747e-01	[ 1.1084046e+00]	 2.1024466e-01
      73	 1.2327593e+00	 1.4223192e-01	 1.2811504e+00	 1.4757179e-01	  1.1058128e+00 	 1.7605114e-01


.. parsed-literal::

      74	 1.2345277e+00	 1.4182405e-01	 1.2834857e+00	 1.4752507e-01	  1.1041848e+00 	 2.1570539e-01


.. parsed-literal::

      75	 1.2410581e+00	 1.4188121e-01	 1.2898534e+00	 1.4771689e-01	  1.1080400e+00 	 2.1634126e-01


.. parsed-literal::

      76	 1.2435072e+00	 1.4174529e-01	 1.2923844e+00	 1.4766999e-01	  1.1076563e+00 	 2.0427728e-01
      77	 1.2465150e+00	 1.4158246e-01	 1.2955169e+00	 1.4755343e-01	  1.1080502e+00 	 1.8447852e-01


.. parsed-literal::

      78	 1.2515944e+00	 1.4149064e-01	 1.3008289e+00	 1.4745513e-01	[ 1.1089748e+00]	 2.0807719e-01


.. parsed-literal::

      79	 1.2564513e+00	 1.4145150e-01	 1.3062687e+00	 1.4714733e-01	  1.1080854e+00 	 2.0978665e-01


.. parsed-literal::

      80	 1.2639118e+00	 1.4153043e-01	 1.3136392e+00	 1.4721228e-01	[ 1.1150073e+00]	 2.1273971e-01
      81	 1.2674540e+00	 1.4153157e-01	 1.3171255e+00	 1.4702710e-01	[ 1.1202686e+00]	 1.6823363e-01


.. parsed-literal::

      82	 1.2722555e+00	 1.4142306e-01	 1.3221217e+00	 1.4682614e-01	[ 1.1232883e+00]	 1.9166350e-01
      83	 1.2742982e+00	 1.4130546e-01	 1.3245061e+00	 1.4618176e-01	[ 1.1248599e+00]	 2.0057201e-01


.. parsed-literal::

      84	 1.2811856e+00	 1.4104503e-01	 1.3314151e+00	 1.4603916e-01	[ 1.1288015e+00]	 1.6672111e-01


.. parsed-literal::

      85	 1.2837725e+00	 1.4068897e-01	 1.3339328e+00	 1.4576840e-01	[ 1.1310108e+00]	 2.1002817e-01
      86	 1.2895405e+00	 1.3995223e-01	 1.3396985e+00	 1.4500255e-01	[ 1.1344244e+00]	 1.8234897e-01


.. parsed-literal::

      87	 1.2955750e+00	 1.3960536e-01	 1.3457293e+00	 1.4444720e-01	[ 1.1388359e+00]	 2.0781446e-01


.. parsed-literal::

      88	 1.3002072e+00	 1.3903452e-01	 1.3504374e+00	 1.4388118e-01	  1.1379718e+00 	 2.1777558e-01


.. parsed-literal::

      89	 1.3041568e+00	 1.3909118e-01	 1.3542500e+00	 1.4394371e-01	[ 1.1429176e+00]	 2.1559072e-01


.. parsed-literal::

      90	 1.3073608e+00	 1.3911327e-01	 1.3573553e+00	 1.4404003e-01	[ 1.1467043e+00]	 2.0534086e-01


.. parsed-literal::

      91	 1.3113951e+00	 1.3885574e-01	 1.3615356e+00	 1.4428368e-01	  1.1465992e+00 	 2.1163011e-01
      92	 1.3154498e+00	 1.3870897e-01	 1.3657115e+00	 1.4432069e-01	[ 1.1468184e+00]	 1.7497993e-01


.. parsed-literal::

      93	 1.3185367e+00	 1.3848424e-01	 1.3689343e+00	 1.4439740e-01	  1.1464935e+00 	 2.1300006e-01


.. parsed-literal::

      94	 1.3224436e+00	 1.3852723e-01	 1.3730795e+00	 1.4485425e-01	  1.1456057e+00 	 2.1898389e-01


.. parsed-literal::

      95	 1.3249826e+00	 1.3846477e-01	 1.3758015e+00	 1.4509197e-01	  1.1458160e+00 	 2.0692825e-01


.. parsed-literal::

      96	 1.3279339e+00	 1.3854430e-01	 1.3786743e+00	 1.4495816e-01	[ 1.1507471e+00]	 2.1605468e-01


.. parsed-literal::

      97	 1.3311262e+00	 1.3868174e-01	 1.3818814e+00	 1.4495801e-01	[ 1.1551608e+00]	 2.0096946e-01
      98	 1.3341460e+00	 1.3860484e-01	 1.3849960e+00	 1.4487867e-01	[ 1.1576969e+00]	 1.9776082e-01


.. parsed-literal::

      99	 1.3370948e+00	 1.3837112e-01	 1.3883680e+00	 1.4481339e-01	  1.1572626e+00 	 2.1107197e-01


.. parsed-literal::

     100	 1.3419877e+00	 1.3806267e-01	 1.3933416e+00	 1.4488832e-01	  1.1559721e+00 	 2.1893477e-01


.. parsed-literal::

     101	 1.3444520e+00	 1.3777368e-01	 1.3958432e+00	 1.4482100e-01	  1.1545464e+00 	 2.1409941e-01


.. parsed-literal::

     102	 1.3475971e+00	 1.3746486e-01	 1.3991772e+00	 1.4484044e-01	  1.1503541e+00 	 2.3304701e-01


.. parsed-literal::

     103	 1.3517025e+00	 1.3708025e-01	 1.4033349e+00	 1.4482951e-01	  1.1477299e+00 	 2.1306491e-01


.. parsed-literal::

     104	 1.3555248e+00	 1.3678098e-01	 1.4073120e+00	 1.4467315e-01	  1.1474343e+00 	 2.1734452e-01
     105	 1.3582564e+00	 1.3673236e-01	 1.4098838e+00	 1.4451222e-01	  1.1510630e+00 	 2.0561743e-01


.. parsed-literal::

     106	 1.3613165e+00	 1.3658294e-01	 1.4128828e+00	 1.4419275e-01	  1.1564343e+00 	 2.1283603e-01


.. parsed-literal::

     107	 1.3633259e+00	 1.3629381e-01	 1.4149915e+00	 1.4396666e-01	  1.1560739e+00 	 2.1116543e-01


.. parsed-literal::

     108	 1.3661036e+00	 1.3610904e-01	 1.4178078e+00	 1.4378507e-01	[ 1.1580505e+00]	 2.1462750e-01
     109	 1.3693177e+00	 1.3577371e-01	 1.4211705e+00	 1.4356092e-01	[ 1.1584735e+00]	 1.8375659e-01


.. parsed-literal::

     110	 1.3714255e+00	 1.3560017e-01	 1.4233325e+00	 1.4340485e-01	[ 1.1603905e+00]	 2.1851349e-01
     111	 1.3769162e+00	 1.3522340e-01	 1.4289517e+00	 1.4313183e-01	[ 1.1650160e+00]	 1.9061255e-01


.. parsed-literal::

     112	 1.3791527e+00	 1.3505330e-01	 1.4312438e+00	 1.4274162e-01	[ 1.1688190e+00]	 3.3785939e-01


.. parsed-literal::

     113	 1.3818115e+00	 1.3493820e-01	 1.4338401e+00	 1.4254345e-01	[ 1.1722936e+00]	 2.2006059e-01


.. parsed-literal::

     114	 1.3849092e+00	 1.3481529e-01	 1.4369282e+00	 1.4228181e-01	[ 1.1742487e+00]	 2.1341181e-01
     115	 1.3880749e+00	 1.3454551e-01	 1.4401690e+00	 1.4172767e-01	  1.1732188e+00 	 1.9763541e-01


.. parsed-literal::

     116	 1.3915106e+00	 1.3423662e-01	 1.4437921e+00	 1.4119957e-01	  1.1720272e+00 	 2.1018052e-01


.. parsed-literal::

     117	 1.3948463e+00	 1.3406316e-01	 1.4472355e+00	 1.4084902e-01	  1.1685646e+00 	 2.0998836e-01


.. parsed-literal::

     118	 1.3976115e+00	 1.3390127e-01	 1.4500703e+00	 1.4059343e-01	  1.1676131e+00 	 2.2058511e-01


.. parsed-literal::

     119	 1.4010676e+00	 1.3373023e-01	 1.4536789e+00	 1.4042026e-01	  1.1637530e+00 	 2.0476747e-01


.. parsed-literal::

     120	 1.4038494e+00	 1.3361892e-01	 1.4566773e+00	 1.3998252e-01	  1.1602539e+00 	 2.4606633e-01
     121	 1.4066025e+00	 1.3362250e-01	 1.4592925e+00	 1.4008910e-01	  1.1651453e+00 	 1.8789148e-01


.. parsed-literal::

     122	 1.4082403e+00	 1.3360902e-01	 1.4608729e+00	 1.4000570e-01	  1.1687027e+00 	 2.1083713e-01


.. parsed-literal::

     123	 1.4103175e+00	 1.3362384e-01	 1.4629482e+00	 1.3982285e-01	  1.1693281e+00 	 2.2956753e-01


.. parsed-literal::

     124	 1.4128553e+00	 1.3358455e-01	 1.4655217e+00	 1.3968876e-01	  1.1728503e+00 	 2.0513463e-01


.. parsed-literal::

     125	 1.4154164e+00	 1.3355997e-01	 1.4681753e+00	 1.3950884e-01	  1.1713587e+00 	 2.0915461e-01
     126	 1.4179246e+00	 1.3354570e-01	 1.4709206e+00	 1.3940904e-01	  1.1677265e+00 	 1.7409611e-01


.. parsed-literal::

     127	 1.4204296e+00	 1.3354945e-01	 1.4734563e+00	 1.3927451e-01	  1.1670809e+00 	 2.1806931e-01


.. parsed-literal::

     128	 1.4221560e+00	 1.3353133e-01	 1.4751547e+00	 1.3920195e-01	  1.1689385e+00 	 2.1965361e-01


.. parsed-literal::

     129	 1.4249116e+00	 1.3347244e-01	 1.4779905e+00	 1.3912377e-01	  1.1716739e+00 	 2.1523690e-01


.. parsed-literal::

     130	 1.4266265e+00	 1.3339234e-01	 1.4797745e+00	 1.3864818e-01	  1.1719255e+00 	 2.0983076e-01


.. parsed-literal::

     131	 1.4285210e+00	 1.3336361e-01	 1.4816465e+00	 1.3862598e-01	  1.1734664e+00 	 2.0647597e-01


.. parsed-literal::

     132	 1.4306512e+00	 1.3333502e-01	 1.4838063e+00	 1.3852901e-01	  1.1733163e+00 	 2.2243762e-01


.. parsed-literal::

     133	 1.4326546e+00	 1.3329471e-01	 1.4858681e+00	 1.3828002e-01	  1.1727208e+00 	 2.2042918e-01


.. parsed-literal::

     134	 1.4345013e+00	 1.3345333e-01	 1.4880555e+00	 1.3817086e-01	  1.1667594e+00 	 2.0934129e-01
     135	 1.4371938e+00	 1.3332377e-01	 1.4906662e+00	 1.3783282e-01	  1.1699683e+00 	 1.9081593e-01


.. parsed-literal::

     136	 1.4384659e+00	 1.3325741e-01	 1.4919206e+00	 1.3772575e-01	  1.1733477e+00 	 1.8065619e-01
     137	 1.4408672e+00	 1.3324163e-01	 1.4943720e+00	 1.3762262e-01	[ 1.1758621e+00]	 1.8834138e-01


.. parsed-literal::

     138	 1.4430401e+00	 1.3326854e-01	 1.4966838e+00	 1.3760477e-01	[ 1.1832041e+00]	 2.1118450e-01


.. parsed-literal::

     139	 1.4451026e+00	 1.3329803e-01	 1.4987138e+00	 1.3765740e-01	  1.1828706e+00 	 2.2158027e-01


.. parsed-literal::

     140	 1.4470376e+00	 1.3335558e-01	 1.5006577e+00	 1.3774336e-01	  1.1830701e+00 	 2.1869707e-01


.. parsed-literal::

     141	 1.4488603e+00	 1.3336287e-01	 1.5024700e+00	 1.3772251e-01	[ 1.1847297e+00]	 2.0966673e-01
     142	 1.4514863e+00	 1.3336619e-01	 1.5052340e+00	 1.3777550e-01	  1.1823240e+00 	 1.9647884e-01


.. parsed-literal::

     143	 1.4540755e+00	 1.3326231e-01	 1.5078027e+00	 1.3776493e-01	[ 1.1857428e+00]	 2.1647501e-01


.. parsed-literal::

     144	 1.4553708e+00	 1.3315925e-01	 1.5090531e+00	 1.3767578e-01	  1.1856372e+00 	 2.1064830e-01


.. parsed-literal::

     145	 1.4579118e+00	 1.3291526e-01	 1.5116681e+00	 1.3772323e-01	  1.1820928e+00 	 2.0867658e-01


.. parsed-literal::

     146	 1.4598430e+00	 1.3273025e-01	 1.5136928e+00	 1.3778076e-01	  1.1763627e+00 	 2.1468949e-01


.. parsed-literal::

     147	 1.4622064e+00	 1.3260786e-01	 1.5161303e+00	 1.3803079e-01	  1.1722112e+00 	 2.0765400e-01


.. parsed-literal::

     148	 1.4645870e+00	 1.3253490e-01	 1.5185603e+00	 1.3826059e-01	  1.1705075e+00 	 2.1218777e-01


.. parsed-literal::

     149	 1.4661277e+00	 1.3243484e-01	 1.5200694e+00	 1.3816908e-01	  1.1716793e+00 	 2.1393895e-01


.. parsed-literal::

     150	 1.4674304e+00	 1.3237353e-01	 1.5212952e+00	 1.3797494e-01	  1.1756823e+00 	 2.0818353e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.1 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa554313e80>



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
    CPU times: user 1.87 s, sys: 44 ms, total: 1.91 s
    Wall time: 650 ms


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

