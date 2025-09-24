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
       1	-3.3810417e-01	 3.1907916e-01	-3.2844616e-01	 3.2623626e-01	[-3.4255785e-01]	 4.6880817e-01


.. parsed-literal::

       2	-2.6681115e-01	 3.0788556e-01	-2.4197591e-01	 3.1518299e-01	[-2.6492076e-01]	 2.3650861e-01


.. parsed-literal::

       3	-2.2285932e-01	 2.8739945e-01	-1.7959899e-01	 2.9442042e-01	[-2.1163666e-01]	 2.9929423e-01
       4	-1.8949924e-01	 2.6297381e-01	-1.4718821e-01	 2.7463736e-01	 -2.1919851e-01 	 1.7166519e-01


.. parsed-literal::

       5	-9.6377611e-02	 2.5457546e-01	-6.0055987e-02	 2.6602362e-01	[-1.1631338e-01]	 2.1324682e-01


.. parsed-literal::

       6	-6.0455174e-02	 2.4896215e-01	-2.9119656e-02	 2.6070120e-01	[-7.4286533e-02]	 2.1516418e-01


.. parsed-literal::

       7	-4.1702389e-02	 2.4603105e-01	-1.6969801e-02	 2.5867662e-01	[-6.8308418e-02]	 2.0201206e-01


.. parsed-literal::

       8	-2.8810317e-02	 2.4390166e-01	-7.9795823e-03	 2.5748630e-01	[-6.4148481e-02]	 2.1795368e-01


.. parsed-literal::

       9	-1.4440249e-02	 2.4122207e-01	 3.4978861e-03	 2.5649198e-01	[-5.9648201e-02]	 2.0353246e-01


.. parsed-literal::

      10	-1.6812195e-03	 2.3867574e-01	 1.4203340e-02	 2.5519863e-01	[-5.6631474e-02]	 2.1732378e-01


.. parsed-literal::

      11	 1.4671112e-03	 2.3811950e-01	 1.6143326e-02	 2.5613812e-01	[-5.3272305e-02]	 2.1173620e-01


.. parsed-literal::

      12	 6.4554418e-03	 2.3739368e-01	 2.0712524e-02	 2.5554047e-01	 -5.4740242e-02 	 2.1187901e-01


.. parsed-literal::

      13	 9.2405141e-03	 2.3682843e-01	 2.3404501e-02	 2.5484458e-01	[-5.2326311e-02]	 2.1182489e-01
      14	 1.3194782e-02	 2.3596069e-01	 2.7669155e-02	 2.5387588e-01	[-4.9612318e-02]	 1.9915748e-01


.. parsed-literal::

      15	 7.8529268e-02	 2.2700602e-01	 9.6497417e-02	 2.4026269e-01	[ 4.8631788e-02]	 3.2717729e-01


.. parsed-literal::

      16	 1.0740400e-01	 2.2316009e-01	 1.2821521e-01	 2.3421810e-01	[ 9.5771333e-02]	 3.3543777e-01


.. parsed-literal::

      17	 1.5190104e-01	 2.1673704e-01	 1.7302643e-01	 2.3188889e-01	[ 1.3451869e-01]	 2.0834064e-01


.. parsed-literal::

      18	 2.5447783e-01	 2.1458383e-01	 2.7968639e-01	 2.2997868e-01	[ 2.3087324e-01]	 2.1116018e-01


.. parsed-literal::

      19	 3.2938380e-01	 2.1286291e-01	 3.6221838e-01	 2.2835994e-01	[ 2.8231020e-01]	 2.0422482e-01


.. parsed-literal::

      20	 3.6629698e-01	 2.1258170e-01	 3.9878575e-01	 2.2697758e-01	[ 3.1960221e-01]	 2.1264410e-01


.. parsed-literal::

      21	 4.1100082e-01	 2.0946289e-01	 4.4409458e-01	 2.2447802e-01	[ 3.6800305e-01]	 2.0691371e-01


.. parsed-literal::

      22	 4.6230996e-01	 2.0508961e-01	 4.9582066e-01	 2.2197191e-01	[ 4.1798192e-01]	 2.1277022e-01


.. parsed-literal::

      23	 5.4415673e-01	 2.0484233e-01	 5.7881959e-01	 2.1998629e-01	[ 4.9882783e-01]	 2.1595144e-01


.. parsed-literal::

      24	 5.9348229e-01	 2.0593432e-01	 6.3255130e-01	 2.2240592e-01	[ 5.3448027e-01]	 2.0995355e-01


.. parsed-literal::

      25	 6.3261444e-01	 2.0285786e-01	 6.7111963e-01	 2.1964191e-01	[ 5.7715440e-01]	 2.1033502e-01


.. parsed-literal::

      26	 6.5704821e-01	 2.0182562e-01	 6.9476032e-01	 2.1844635e-01	[ 6.0660517e-01]	 2.1310520e-01


.. parsed-literal::

      27	 7.1007626e-01	 2.0215968e-01	 7.4737709e-01	 2.1725877e-01	[ 6.5983403e-01]	 2.0126772e-01


.. parsed-literal::

      28	 7.4105301e-01	 1.9994593e-01	 7.7808010e-01	 2.1554039e-01	[ 6.9565294e-01]	 3.2682943e-01


.. parsed-literal::

      29	 7.7171175e-01	 1.9880012e-01	 8.0987900e-01	 2.1335083e-01	[ 7.2712750e-01]	 2.0710135e-01
      30	 7.8846384e-01	 1.9922124e-01	 8.2717292e-01	 2.1179991e-01	[ 7.5030011e-01]	 1.9769430e-01


.. parsed-literal::

      31	 8.0208741e-01	 1.9834497e-01	 8.4044252e-01	 2.1100008e-01	[ 7.6674388e-01]	 2.1368694e-01


.. parsed-literal::

      32	 8.3865945e-01	 1.9789266e-01	 8.7776441e-01	 2.1023600e-01	[ 8.0552112e-01]	 2.0661402e-01


.. parsed-literal::

      33	 8.5890612e-01	 1.9687380e-01	 8.9812107e-01	 2.0845849e-01	[ 8.3372148e-01]	 2.0692205e-01


.. parsed-literal::

      34	 8.7587235e-01	 1.9489639e-01	 9.1742075e-01	 2.0460332e-01	[ 8.4673554e-01]	 2.0483303e-01


.. parsed-literal::

      35	 8.9538571e-01	 1.9192961e-01	 9.3705283e-01	 2.0303668e-01	[ 8.6798877e-01]	 2.1846867e-01


.. parsed-literal::

      36	 9.1503207e-01	 1.9107539e-01	 9.5671632e-01	 2.0184058e-01	[ 8.9134317e-01]	 2.1335626e-01
      37	 9.3333099e-01	 1.9048180e-01	 9.7552778e-01	 2.0012617e-01	[ 9.1448560e-01]	 1.8665290e-01


.. parsed-literal::

      38	 9.5080567e-01	 1.8921931e-01	 9.9441761e-01	 1.9878286e-01	[ 9.2579466e-01]	 2.1312809e-01


.. parsed-literal::

      39	 9.6342042e-01	 1.8849504e-01	 1.0071837e+00	 1.9820184e-01	[ 9.3407276e-01]	 2.1329188e-01


.. parsed-literal::

      40	 9.7214336e-01	 1.8744282e-01	 1.0156334e+00	 1.9740259e-01	[ 9.4546638e-01]	 2.1033406e-01


.. parsed-literal::

      41	 9.8360211e-01	 1.8552962e-01	 1.0275265e+00	 1.9626126e-01	[ 9.5683062e-01]	 2.1369553e-01


.. parsed-literal::

      42	 9.9834229e-01	 1.8382525e-01	 1.0431548e+00	 1.9561061e-01	[ 9.7140799e-01]	 2.0675159e-01


.. parsed-literal::

      43	 1.0188766e+00	 1.8246084e-01	 1.0648438e+00	 1.9529850e-01	[ 9.8456853e-01]	 2.0146346e-01


.. parsed-literal::

      44	 1.0295810e+00	 1.8189715e-01	 1.0767496e+00	 1.9476398e-01	[ 9.9506049e-01]	 2.1365833e-01


.. parsed-literal::

      45	 1.0416767e+00	 1.8169209e-01	 1.0884602e+00	 1.9417570e-01	[ 1.0090398e+00]	 2.1258307e-01


.. parsed-literal::

      46	 1.0501129e+00	 1.8069530e-01	 1.0963935e+00	 1.9320492e-01	[ 1.0161157e+00]	 2.1155858e-01


.. parsed-literal::

      47	 1.0643370e+00	 1.7906288e-01	 1.1105047e+00	 1.9138168e-01	[ 1.0224800e+00]	 2.1425343e-01


.. parsed-literal::

      48	 1.0761451e+00	 1.7653013e-01	 1.1225201e+00	 1.8795780e-01	[ 1.0270225e+00]	 2.1580791e-01


.. parsed-literal::

      49	 1.0853615e+00	 1.7603940e-01	 1.1318062e+00	 1.8727939e-01	[ 1.0379419e+00]	 2.1768236e-01
      50	 1.0977820e+00	 1.7528465e-01	 1.1449049e+00	 1.8635349e-01	[ 1.0476330e+00]	 2.1200633e-01


.. parsed-literal::

      51	 1.1101031e+00	 1.7298031e-01	 1.1574893e+00	 1.8410132e-01	[ 1.0609473e+00]	 2.1034288e-01


.. parsed-literal::

      52	 1.1208795e+00	 1.7051927e-01	 1.1691662e+00	 1.8203213e-01	[ 1.0661171e+00]	 2.1189117e-01


.. parsed-literal::

      53	 1.1291630e+00	 1.6908485e-01	 1.1777300e+00	 1.8082678e-01	[ 1.0694357e+00]	 2.2184682e-01


.. parsed-literal::

      54	 1.1383963e+00	 1.6786095e-01	 1.1870603e+00	 1.8001603e-01	[ 1.0747499e+00]	 2.2346473e-01


.. parsed-literal::

      55	 1.1499442e+00	 1.6630069e-01	 1.1988283e+00	 1.7912191e-01	[ 1.0829300e+00]	 2.1727920e-01


.. parsed-literal::

      56	 1.1593231e+00	 1.6389378e-01	 1.2083338e+00	 1.7697005e-01	[ 1.0955550e+00]	 2.1509719e-01
      57	 1.1677670e+00	 1.6239476e-01	 1.2168743e+00	 1.7505446e-01	[ 1.1033025e+00]	 1.8116045e-01


.. parsed-literal::

      58	 1.1757761e+00	 1.6102074e-01	 1.2251976e+00	 1.7359442e-01	[ 1.1118734e+00]	 1.9690132e-01
      59	 1.1849863e+00	 1.5860612e-01	 1.2348212e+00	 1.7171677e-01	[ 1.1186171e+00]	 1.7755771e-01


.. parsed-literal::

      60	 1.1936834e+00	 1.5759793e-01	 1.2437428e+00	 1.7097146e-01	[ 1.1226637e+00]	 2.1393919e-01
      61	 1.2037383e+00	 1.5660183e-01	 1.2534846e+00	 1.7030020e-01	[ 1.1332583e+00]	 1.8061852e-01


.. parsed-literal::

      62	 1.2130145e+00	 1.5533512e-01	 1.2627588e+00	 1.6955452e-01	[ 1.1371922e+00]	 2.0913720e-01
      63	 1.2207632e+00	 1.5391765e-01	 1.2707321e+00	 1.6812397e-01	[ 1.1372656e+00]	 1.8394852e-01


.. parsed-literal::

      64	 1.2289984e+00	 1.5288972e-01	 1.2789467e+00	 1.6721637e-01	[ 1.1425551e+00]	 1.9790554e-01
      65	 1.2357475e+00	 1.5190360e-01	 1.2859011e+00	 1.6645160e-01	[ 1.1467765e+00]	 1.7667127e-01


.. parsed-literal::

      66	 1.2445871e+00	 1.5068770e-01	 1.2951536e+00	 1.6539975e-01	[ 1.1499498e+00]	 2.0977306e-01


.. parsed-literal::

      67	 1.2520601e+00	 1.4956270e-01	 1.3029306e+00	 1.6460016e-01	[ 1.1582382e+00]	 2.0687151e-01


.. parsed-literal::

      68	 1.2588564e+00	 1.4900047e-01	 1.3096863e+00	 1.6409313e-01	[ 1.1667669e+00]	 2.0522642e-01


.. parsed-literal::

      69	 1.2655820e+00	 1.4843003e-01	 1.3165260e+00	 1.6358792e-01	[ 1.1740284e+00]	 2.2046852e-01


.. parsed-literal::

      70	 1.2715502e+00	 1.4813195e-01	 1.3225919e+00	 1.6354556e-01	[ 1.1822204e+00]	 2.1359062e-01


.. parsed-literal::

      71	 1.2803631e+00	 1.4781756e-01	 1.3319719e+00	 1.6360138e-01	[ 1.1877117e+00]	 2.0436168e-01


.. parsed-literal::

      72	 1.2881824e+00	 1.4743997e-01	 1.3398094e+00	 1.6340322e-01	[ 1.1980531e+00]	 2.1486044e-01


.. parsed-literal::

      73	 1.2931539e+00	 1.4715729e-01	 1.3447378e+00	 1.6331716e-01	[ 1.2024428e+00]	 2.0859385e-01


.. parsed-literal::

      74	 1.3009450e+00	 1.4602158e-01	 1.3529184e+00	 1.6241833e-01	[ 1.2029118e+00]	 2.1565843e-01


.. parsed-literal::

      75	 1.3056439e+00	 1.4604236e-01	 1.3578712e+00	 1.6234695e-01	[ 1.2064623e+00]	 2.2613144e-01
      76	 1.3116100e+00	 1.4572508e-01	 1.3636669e+00	 1.6209331e-01	[ 1.2116399e+00]	 1.7344403e-01


.. parsed-literal::

      77	 1.3173815e+00	 1.4549543e-01	 1.3695613e+00	 1.6180257e-01	[ 1.2142608e+00]	 1.8327785e-01


.. parsed-literal::

      78	 1.3219574e+00	 1.4536801e-01	 1.3741114e+00	 1.6149578e-01	[ 1.2184512e+00]	 2.0853901e-01


.. parsed-literal::

      79	 1.3295977e+00	 1.4508456e-01	 1.3820577e+00	 1.6052209e-01	[ 1.2226110e+00]	 2.1431541e-01


.. parsed-literal::

      80	 1.3334271e+00	 1.4493403e-01	 1.3860718e+00	 1.5940875e-01	[ 1.2342466e+00]	 2.1138120e-01
      81	 1.3382806e+00	 1.4445239e-01	 1.3907162e+00	 1.5905778e-01	[ 1.2373372e+00]	 1.8580079e-01


.. parsed-literal::

      82	 1.3420332e+00	 1.4392967e-01	 1.3945207e+00	 1.5847721e-01	[ 1.2397769e+00]	 2.1035337e-01


.. parsed-literal::

      83	 1.3473362e+00	 1.4316086e-01	 1.3999495e+00	 1.5751124e-01	[ 1.2441189e+00]	 2.1123433e-01
      84	 1.3517164e+00	 1.4224595e-01	 1.4045333e+00	 1.5668963e-01	[ 1.2481866e+00]	 1.8327999e-01


.. parsed-literal::

      85	 1.3576954e+00	 1.4149905e-01	 1.4104461e+00	 1.5586369e-01	[ 1.2564981e+00]	 2.1318650e-01


.. parsed-literal::

      86	 1.3618224e+00	 1.4121886e-01	 1.4145391e+00	 1.5556871e-01	[ 1.2617556e+00]	 2.0619512e-01
      87	 1.3667472e+00	 1.4051473e-01	 1.4195958e+00	 1.5497883e-01	[ 1.2676748e+00]	 1.8204093e-01


.. parsed-literal::

      88	 1.3700279e+00	 1.4025609e-01	 1.4230104e+00	 1.5468117e-01	[ 1.2685831e+00]	 3.3131099e-01


.. parsed-literal::

      89	 1.3741306e+00	 1.3965866e-01	 1.4272126e+00	 1.5421764e-01	[ 1.2743462e+00]	 2.0131898e-01


.. parsed-literal::

      90	 1.3778854e+00	 1.3912040e-01	 1.4310150e+00	 1.5378740e-01	[ 1.2791228e+00]	 2.1200752e-01


.. parsed-literal::

      91	 1.3815095e+00	 1.3884231e-01	 1.4346462e+00	 1.5349535e-01	[ 1.2896711e+00]	 2.1553707e-01


.. parsed-literal::

      92	 1.3848245e+00	 1.3837743e-01	 1.4379277e+00	 1.5310449e-01	[ 1.2927825e+00]	 2.0682287e-01


.. parsed-literal::

      93	 1.3876644e+00	 1.3813761e-01	 1.4407288e+00	 1.5290724e-01	[ 1.2969567e+00]	 2.0266962e-01


.. parsed-literal::

      94	 1.3919951e+00	 1.3756727e-01	 1.4451550e+00	 1.5235792e-01	[ 1.2974030e+00]	 2.0861006e-01
      95	 1.3935972e+00	 1.3727685e-01	 1.4470369e+00	 1.5200941e-01	  1.2969827e+00 	 1.9174552e-01


.. parsed-literal::

      96	 1.3976055e+00	 1.3706820e-01	 1.4509612e+00	 1.5174649e-01	  1.2969108e+00 	 2.0952654e-01


.. parsed-literal::

      97	 1.3999476e+00	 1.3677848e-01	 1.4533692e+00	 1.5136677e-01	  1.2955297e+00 	 2.1469450e-01


.. parsed-literal::

      98	 1.4027743e+00	 1.3643919e-01	 1.4562776e+00	 1.5086714e-01	  1.2943783e+00 	 2.2106600e-01
      99	 1.4075100e+00	 1.3623603e-01	 1.4611051e+00	 1.5038903e-01	  1.2927107e+00 	 1.8583131e-01


.. parsed-literal::

     100	 1.4097994e+00	 1.3585071e-01	 1.4634293e+00	 1.4978767e-01	  1.2973415e+00 	 3.2576394e-01


.. parsed-literal::

     101	 1.4127920e+00	 1.3575647e-01	 1.4664137e+00	 1.4959880e-01	[ 1.2976274e+00]	 2.0402527e-01


.. parsed-literal::

     102	 1.4152680e+00	 1.3560497e-01	 1.4688440e+00	 1.4947707e-01	[ 1.3006667e+00]	 2.0991850e-01
     103	 1.4182654e+00	 1.3518735e-01	 1.4719334e+00	 1.4909248e-01	  1.2993465e+00 	 1.9800305e-01


.. parsed-literal::

     104	 1.4210803e+00	 1.3489513e-01	 1.4748323e+00	 1.4888785e-01	[ 1.3011712e+00]	 2.0736217e-01


.. parsed-literal::

     105	 1.4233035e+00	 1.3461279e-01	 1.4771156e+00	 1.4865272e-01	  1.2996587e+00 	 2.1767974e-01
     106	 1.4258206e+00	 1.3424186e-01	 1.4797787e+00	 1.4840815e-01	  1.2938789e+00 	 1.7590666e-01


.. parsed-literal::

     107	 1.4278197e+00	 1.3406577e-01	 1.4818070e+00	 1.4834662e-01	  1.2910852e+00 	 2.0558476e-01
     108	 1.4299363e+00	 1.3398340e-01	 1.4838492e+00	 1.4840987e-01	  1.2911021e+00 	 2.0109820e-01


.. parsed-literal::

     109	 1.4334122e+00	 1.3358033e-01	 1.4872241e+00	 1.4864993e-01	  1.2864955e+00 	 2.0088553e-01


.. parsed-literal::

     110	 1.4357038e+00	 1.3352933e-01	 1.4894464e+00	 1.4885123e-01	  1.2842690e+00 	 2.1578240e-01


.. parsed-literal::

     111	 1.4374068e+00	 1.3336007e-01	 1.4911359e+00	 1.4884332e-01	  1.2829457e+00 	 2.0598459e-01


.. parsed-literal::

     112	 1.4408852e+00	 1.3280274e-01	 1.4947372e+00	 1.4900688e-01	  1.2731052e+00 	 2.1779847e-01
     113	 1.4423174e+00	 1.3238910e-01	 1.4962985e+00	 1.4868652e-01	  1.2721713e+00 	 1.9410968e-01


.. parsed-literal::

     114	 1.4445114e+00	 1.3240821e-01	 1.4984175e+00	 1.4868802e-01	  1.2738007e+00 	 2.1217155e-01


.. parsed-literal::

     115	 1.4463435e+00	 1.3230314e-01	 1.5002716e+00	 1.4866801e-01	  1.2732657e+00 	 2.0653725e-01


.. parsed-literal::

     116	 1.4482242e+00	 1.3211529e-01	 1.5022053e+00	 1.4857458e-01	  1.2727646e+00 	 2.1769762e-01


.. parsed-literal::

     117	 1.4511081e+00	 1.3184594e-01	 1.5051305e+00	 1.4848836e-01	  1.2774734e+00 	 2.1571779e-01
     118	 1.4529902e+00	 1.3144474e-01	 1.5071468e+00	 1.4822852e-01	  1.2788335e+00 	 1.9213128e-01


.. parsed-literal::

     119	 1.4551437e+00	 1.3143283e-01	 1.5091491e+00	 1.4824245e-01	  1.2829395e+00 	 2.1282673e-01


.. parsed-literal::

     120	 1.4565786e+00	 1.3138259e-01	 1.5104867e+00	 1.4821062e-01	  1.2860503e+00 	 2.0910621e-01
     121	 1.4586185e+00	 1.3107130e-01	 1.5125204e+00	 1.4807192e-01	  1.2876929e+00 	 1.9259381e-01


.. parsed-literal::

     122	 1.4611580e+00	 1.3060184e-01	 1.5151359e+00	 1.4779364e-01	  1.2869621e+00 	 2.0931792e-01


.. parsed-literal::

     123	 1.4630236e+00	 1.3037410e-01	 1.5170823e+00	 1.4767639e-01	  1.2848747e+00 	 2.1143389e-01
     124	 1.4650254e+00	 1.3002137e-01	 1.5192513e+00	 1.4747996e-01	  1.2802425e+00 	 1.8863249e-01


.. parsed-literal::

     125	 1.4668008e+00	 1.2971073e-01	 1.5211642e+00	 1.4735388e-01	  1.2755225e+00 	 2.1268773e-01


.. parsed-literal::

     126	 1.4693238e+00	 1.2925401e-01	 1.5238331e+00	 1.4714962e-01	  1.2699305e+00 	 2.2021747e-01


.. parsed-literal::

     127	 1.4714632e+00	 1.2894414e-01	 1.5259990e+00	 1.4715629e-01	  1.2666806e+00 	 2.0680571e-01
     128	 1.4729352e+00	 1.2844113e-01	 1.5275700e+00	 1.4704370e-01	  1.2639511e+00 	 1.8943787e-01


.. parsed-literal::

     129	 1.4749296e+00	 1.2838620e-01	 1.5294214e+00	 1.4714110e-01	  1.2649174e+00 	 2.2061729e-01
     130	 1.4763605e+00	 1.2818961e-01	 1.5308388e+00	 1.4718282e-01	  1.2635049e+00 	 1.9725728e-01


.. parsed-literal::

     131	 1.4780048e+00	 1.2789405e-01	 1.5325093e+00	 1.4719027e-01	  1.2603277e+00 	 2.0998192e-01
     132	 1.4802923e+00	 1.2739009e-01	 1.5348583e+00	 1.4715337e-01	  1.2507799e+00 	 2.0123339e-01


.. parsed-literal::

     133	 1.4813130e+00	 1.2650463e-01	 1.5361404e+00	 1.4694319e-01	  1.2310247e+00 	 1.9459891e-01
     134	 1.4837573e+00	 1.2667386e-01	 1.5384350e+00	 1.4692597e-01	  1.2347229e+00 	 1.7452049e-01


.. parsed-literal::

     135	 1.4846290e+00	 1.2660058e-01	 1.5393183e+00	 1.4689694e-01	  1.2316884e+00 	 2.1359372e-01
     136	 1.4861121e+00	 1.2646681e-01	 1.5408374e+00	 1.4684210e-01	  1.2283815e+00 	 1.7530847e-01


.. parsed-literal::

     137	 1.4870741e+00	 1.2595919e-01	 1.5419948e+00	 1.4691917e-01	  1.2072317e+00 	 2.1711922e-01


.. parsed-literal::

     138	 1.4891548e+00	 1.2596766e-01	 1.5439885e+00	 1.4687983e-01	  1.2147012e+00 	 2.1092558e-01


.. parsed-literal::

     139	 1.4902473e+00	 1.2587107e-01	 1.5450688e+00	 1.4689435e-01	  1.2163326e+00 	 2.2380662e-01


.. parsed-literal::

     140	 1.4919483e+00	 1.2561538e-01	 1.5467860e+00	 1.4692653e-01	  1.2140539e+00 	 2.0590425e-01


.. parsed-literal::

     141	 1.4932699e+00	 1.2532718e-01	 1.5482730e+00	 1.4710937e-01	  1.2137038e+00 	 2.1162653e-01


.. parsed-literal::

     142	 1.4954392e+00	 1.2508680e-01	 1.5503326e+00	 1.4705983e-01	  1.2062044e+00 	 2.0431423e-01


.. parsed-literal::

     143	 1.4966667e+00	 1.2498142e-01	 1.5515340e+00	 1.4702611e-01	  1.2033322e+00 	 2.0695782e-01
     144	 1.4979680e+00	 1.2484517e-01	 1.5528458e+00	 1.4700558e-01	  1.2008872e+00 	 2.0251608e-01


.. parsed-literal::

     145	 1.4995618e+00	 1.2468211e-01	 1.5544902e+00	 1.4710561e-01	  1.1991212e+00 	 2.1270990e-01


.. parsed-literal::

     146	 1.5013809e+00	 1.2453997e-01	 1.5563088e+00	 1.4704072e-01	  1.1995800e+00 	 2.1446013e-01


.. parsed-literal::

     147	 1.5028307e+00	 1.2446301e-01	 1.5577678e+00	 1.4702613e-01	  1.1994521e+00 	 2.1362281e-01


.. parsed-literal::

     148	 1.5041499e+00	 1.2429338e-01	 1.5591240e+00	 1.4695836e-01	  1.1980222e+00 	 2.1011519e-01


.. parsed-literal::

     149	 1.5055426e+00	 1.2413571e-01	 1.5605575e+00	 1.4680478e-01	  1.1930560e+00 	 2.4071002e-01
     150	 1.5076064e+00	 1.2384221e-01	 1.5626869e+00	 1.4658197e-01	  1.1853714e+00 	 1.7657113e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.14 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa7b47d6470>



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
    CPU times: user 2.21 s, sys: 61 ms, total: 2.27 s
    Wall time: 720 ms


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

