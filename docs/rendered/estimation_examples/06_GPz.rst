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
       1	-3.5503091e-01	 3.2410239e-01	-3.4530931e-01	 3.0657179e-01	[-3.1258590e-01]	 4.6267915e-01


.. parsed-literal::

       2	-2.8296907e-01	 3.1307620e-01	-2.5922879e-01	 2.9692528e-01	[-2.0871974e-01]	 2.3181581e-01


.. parsed-literal::

       3	-2.3796511e-01	 2.9218204e-01	-1.9622286e-01	 2.7440233e-01	[-1.2221097e-01]	 3.0291557e-01
       4	-2.0891837e-01	 2.6839844e-01	-1.6998474e-01	 2.5084515e-01	[-6.8575347e-02]	 1.6932487e-01


.. parsed-literal::

       5	-1.1733310e-01	 2.6066920e-01	-8.3252838e-02	 2.5012185e-01	[-2.2207250e-02]	 1.9980335e-01


.. parsed-literal::

       6	-8.4109622e-02	 2.5470408e-01	-5.2639547e-02	 2.3866063e-01	[ 5.7676990e-03]	 2.0845318e-01
       7	-6.6692621e-02	 2.5226144e-01	-4.2367409e-02	 2.3663750e-01	[ 1.9826257e-02]	 1.9727230e-01


.. parsed-literal::

       8	-5.3350752e-02	 2.4999914e-01	-3.3044014e-02	 2.3427917e-01	[ 3.1735020e-02]	 1.9879270e-01
       9	-3.9055054e-02	 2.4732108e-01	-2.1853476e-02	 2.3231290e-01	[ 4.2586453e-02]	 1.8215299e-01


.. parsed-literal::

      10	-3.3324270e-02	 2.4672317e-01	-1.8371617e-02	 2.3186202e-01	[ 4.5118991e-02]	 2.0726705e-01
      11	-2.6141856e-02	 2.4504693e-01	-1.1676367e-02	 2.2972803e-01	[ 5.3858522e-02]	 1.7689323e-01


.. parsed-literal::

      12	-2.3558990e-02	 2.4453950e-01	-9.3334230e-03	 2.2897544e-01	[ 5.7266673e-02]	 2.0936203e-01
      13	-1.9682420e-02	 2.4379761e-01	-5.5873603e-03	 2.2767305e-01	[ 6.1953413e-02]	 1.9545293e-01


.. parsed-literal::

      14	-9.6291385e-03	 2.4175595e-01	 5.5241540e-03	 2.2402524e-01	[ 8.3617466e-02]	 2.0563173e-01


.. parsed-literal::

      15	 1.5970931e-01	 2.3231444e-01	 1.8622515e-01	 2.1303128e-01	[ 2.3924071e-01]	 3.3222461e-01


.. parsed-literal::

      16	 2.5177318e-01	 2.2651002e-01	 2.8153365e-01	 2.0766100e-01	[ 3.4693194e-01]	 2.1870542e-01


.. parsed-literal::

      17	 3.2128522e-01	 2.2002288e-01	 3.5316133e-01	 2.0160993e-01	[ 4.2573643e-01]	 2.0447278e-01
      18	 3.9477254e-01	 2.1405385e-01	 4.2834553e-01	 1.9586367e-01	[ 5.0545417e-01]	 1.8228745e-01


.. parsed-literal::

      19	 4.8667038e-01	 2.1118210e-01	 5.2197476e-01	 1.9235317e-01	[ 5.9639136e-01]	 1.8644428e-01


.. parsed-literal::

      20	 5.6824166e-01	 2.1009261e-01	 6.0911757e-01	 1.9190528e-01	[ 6.5729989e-01]	 2.1037102e-01


.. parsed-literal::

      21	 6.0256531e-01	 2.0736318e-01	 6.4110986e-01	 1.8861624e-01	[ 6.7394644e-01]	 3.2757497e-01
      22	 6.3524230e-01	 2.0375343e-01	 6.7379847e-01	 1.8479781e-01	[ 7.1261914e-01]	 1.9901013e-01


.. parsed-literal::

      23	 6.5903364e-01	 2.1027590e-01	 6.9564714e-01	 1.8979776e-01	[ 7.4670656e-01]	 2.0326495e-01


.. parsed-literal::

      24	 7.1238886e-01	 2.0588605e-01	 7.4972731e-01	 1.8554303e-01	[ 7.9871417e-01]	 2.1803045e-01


.. parsed-literal::

      25	 7.5919504e-01	 2.0092030e-01	 7.9867348e-01	 1.8265013e-01	[ 8.2548310e-01]	 2.0401311e-01
      26	 8.0652657e-01	 2.0013771e-01	 8.4761273e-01	 1.8239648e-01	[ 8.6925767e-01]	 1.8383074e-01


.. parsed-literal::

      27	 8.4288855e-01	 2.0068892e-01	 8.8468277e-01	 1.8164636e-01	[ 9.0068564e-01]	 1.6583824e-01


.. parsed-literal::

      28	 8.7331414e-01	 2.0305355e-01	 9.1539990e-01	 1.8224962e-01	[ 9.3587654e-01]	 2.1047330e-01
      29	 8.9118047e-01	 2.0671280e-01	 9.3499935e-01	 1.8437598e-01	[ 9.6260558e-01]	 1.9824719e-01


.. parsed-literal::

      30	 9.1878961e-01	 2.0333625e-01	 9.6242990e-01	 1.8247890e-01	[ 9.8803893e-01]	 2.0720196e-01
      31	 9.3526671e-01	 1.9886786e-01	 9.7888506e-01	 1.8018682e-01	[ 1.0000796e+00]	 1.9847727e-01


.. parsed-literal::

      32	 9.5485734e-01	 1.9608738e-01	 9.9862432e-01	 1.7923335e-01	[ 1.0185215e+00]	 2.1488714e-01


.. parsed-literal::

      33	 9.7449049e-01	 1.9356910e-01	 1.0188558e+00	 1.7763920e-01	[ 1.0357977e+00]	 2.0749760e-01
      34	 9.8984348e-01	 1.9177468e-01	 1.0348394e+00	 1.7553978e-01	[ 1.0519115e+00]	 1.9007730e-01


.. parsed-literal::

      35	 1.0050092e+00	 1.8873501e-01	 1.0509351e+00	 1.7280615e-01	[ 1.0715884e+00]	 2.1139741e-01


.. parsed-literal::

      36	 1.0195155e+00	 1.8655702e-01	 1.0657311e+00	 1.6989594e-01	[ 1.0847802e+00]	 2.1443510e-01


.. parsed-literal::

      37	 1.0360341e+00	 1.8394565e-01	 1.0826056e+00	 1.6727217e-01	[ 1.0975182e+00]	 2.0665073e-01
      38	 1.0493542e+00	 1.7846072e-01	 1.0969332e+00	 1.6224118e-01	[ 1.1052060e+00]	 1.9024754e-01


.. parsed-literal::

      39	 1.0640422e+00	 1.7693103e-01	 1.1113132e+00	 1.6065421e-01	[ 1.1205671e+00]	 2.0928717e-01


.. parsed-literal::

      40	 1.0715426e+00	 1.7635742e-01	 1.1183753e+00	 1.6013783e-01	[ 1.1267517e+00]	 2.1061873e-01


.. parsed-literal::

      41	 1.0828365e+00	 1.7433627e-01	 1.1297349e+00	 1.5839991e-01	[ 1.1383487e+00]	 2.1063399e-01


.. parsed-literal::

      42	 1.1017136e+00	 1.7033444e-01	 1.1486378e+00	 1.5598619e-01	[ 1.1512141e+00]	 2.2024608e-01
      43	 1.1113276e+00	 1.6922610e-01	 1.1582626e+00	 1.5426337e-01	[ 1.1677130e+00]	 1.9854712e-01


.. parsed-literal::

      44	 1.1241487e+00	 1.6818853e-01	 1.1708537e+00	 1.5373079e-01	[ 1.1781735e+00]	 2.0588279e-01


.. parsed-literal::

      45	 1.1375728e+00	 1.6640486e-01	 1.1846195e+00	 1.5277013e-01	[ 1.1859129e+00]	 2.1196437e-01
      46	 1.1498639e+00	 1.6515485e-01	 1.1972842e+00	 1.5164002e-01	[ 1.1956893e+00]	 1.9013667e-01


.. parsed-literal::

      47	 1.1692747e+00	 1.6168201e-01	 1.2170624e+00	 1.4848940e-01	[ 1.2119860e+00]	 2.0518088e-01
      48	 1.1825163e+00	 1.5870046e-01	 1.2314154e+00	 1.4631044e-01	[ 1.2230154e+00]	 2.0341396e-01


.. parsed-literal::

      49	 1.1952521e+00	 1.5821112e-01	 1.2436591e+00	 1.4607212e-01	[ 1.2331798e+00]	 1.9733977e-01


.. parsed-literal::

      50	 1.2065876e+00	 1.5671475e-01	 1.2552023e+00	 1.4498528e-01	[ 1.2391406e+00]	 2.1368718e-01


.. parsed-literal::

      51	 1.2208798e+00	 1.5451223e-01	 1.2698950e+00	 1.4373178e-01	[ 1.2428998e+00]	 2.0654917e-01


.. parsed-literal::

      52	 1.2330817e+00	 1.5147002e-01	 1.2830091e+00	 1.4149795e-01	[ 1.2495966e+00]	 2.1411657e-01


.. parsed-literal::

      53	 1.2447771e+00	 1.5040685e-01	 1.2948196e+00	 1.4159081e-01	[ 1.2548093e+00]	 2.0732689e-01


.. parsed-literal::

      54	 1.2515033e+00	 1.5010045e-01	 1.3013542e+00	 1.4117023e-01	[ 1.2694044e+00]	 2.1403599e-01


.. parsed-literal::

      55	 1.2619008e+00	 1.4923495e-01	 1.3116372e+00	 1.4167213e-01	[ 1.2862912e+00]	 2.1596861e-01


.. parsed-literal::

      56	 1.2735033e+00	 1.4714645e-01	 1.3236822e+00	 1.4015921e-01	[ 1.2975372e+00]	 2.1842003e-01


.. parsed-literal::

      57	 1.2850446e+00	 1.4502808e-01	 1.3356343e+00	 1.3775778e-01	[ 1.3100340e+00]	 2.2359157e-01
      58	 1.2963720e+00	 1.4276799e-01	 1.3467357e+00	 1.3449250e-01	[ 1.3111820e+00]	 1.8447089e-01


.. parsed-literal::

      59	 1.3056785e+00	 1.4176725e-01	 1.3562631e+00	 1.3254016e-01	  1.3095773e+00 	 1.8029380e-01


.. parsed-literal::

      60	 1.3143823e+00	 1.4022914e-01	 1.3654106e+00	 1.2993494e-01	  1.2976207e+00 	 2.0618486e-01
      61	 1.3229172e+00	 1.3987387e-01	 1.3738787e+00	 1.2913772e-01	  1.3029821e+00 	 1.8006039e-01


.. parsed-literal::

      62	 1.3294952e+00	 1.3914480e-01	 1.3804408e+00	 1.2869412e-01	  1.3079204e+00 	 2.0594311e-01


.. parsed-literal::

      63	 1.3408937e+00	 1.3785652e-01	 1.3920855e+00	 1.2722443e-01	[ 1.3236062e+00]	 2.1316314e-01


.. parsed-literal::

      64	 1.3454878e+00	 1.3651406e-01	 1.3969604e+00	 1.2601368e-01	  1.3138597e+00 	 2.0343065e-01


.. parsed-literal::

      65	 1.3538123e+00	 1.3615465e-01	 1.4050414e+00	 1.2563115e-01	[ 1.3336006e+00]	 2.1524429e-01
      66	 1.3579504e+00	 1.3566757e-01	 1.4093885e+00	 1.2477720e-01	[ 1.3369209e+00]	 1.7459536e-01


.. parsed-literal::

      67	 1.3632488e+00	 1.3490943e-01	 1.4148724e+00	 1.2395417e-01	[ 1.3411946e+00]	 1.7770696e-01


.. parsed-literal::

      68	 1.3698876e+00	 1.3362874e-01	 1.4217780e+00	 1.2227450e-01	[ 1.3434237e+00]	 2.1114326e-01


.. parsed-literal::

      69	 1.3763630e+00	 1.3305364e-01	 1.4283707e+00	 1.2159584e-01	[ 1.3503681e+00]	 2.0401144e-01


.. parsed-literal::

      70	 1.3806574e+00	 1.3284963e-01	 1.4325423e+00	 1.2189611e-01	[ 1.3553539e+00]	 2.1251059e-01


.. parsed-literal::

      71	 1.3859418e+00	 1.3254659e-01	 1.4378724e+00	 1.2117260e-01	  1.3547603e+00 	 2.0996881e-01
      72	 1.3913211e+00	 1.3208450e-01	 1.4434864e+00	 1.2101571e-01	[ 1.3587428e+00]	 1.7922759e-01


.. parsed-literal::

      73	 1.3961774e+00	 1.3192320e-01	 1.4482834e+00	 1.2038004e-01	  1.3581989e+00 	 2.0616794e-01


.. parsed-literal::

      74	 1.4010796e+00	 1.3176225e-01	 1.4533471e+00	 1.1967545e-01	  1.3580075e+00 	 2.0110917e-01


.. parsed-literal::

      75	 1.4058775e+00	 1.3134513e-01	 1.4581928e+00	 1.1892891e-01	[ 1.3588428e+00]	 2.1371317e-01
      76	 1.4102794e+00	 1.3084455e-01	 1.4629053e+00	 1.1810567e-01	[ 1.3687892e+00]	 1.8135238e-01


.. parsed-literal::

      77	 1.4145909e+00	 1.3045936e-01	 1.4671130e+00	 1.1784091e-01	[ 1.3696391e+00]	 1.7296672e-01
      78	 1.4172946e+00	 1.3022112e-01	 1.4697648e+00	 1.1771907e-01	[ 1.3755240e+00]	 1.9805956e-01


.. parsed-literal::

      79	 1.4225793e+00	 1.2972794e-01	 1.4750554e+00	 1.1742755e-01	[ 1.3842061e+00]	 2.1452236e-01


.. parsed-literal::

      80	 1.4262559e+00	 1.2915320e-01	 1.4787570e+00	 1.1707821e-01	[ 1.3959336e+00]	 2.0903182e-01


.. parsed-literal::

      81	 1.4305844e+00	 1.2895026e-01	 1.4830278e+00	 1.1683146e-01	  1.3948871e+00 	 2.1392345e-01
      82	 1.4335218e+00	 1.2875474e-01	 1.4859722e+00	 1.1679118e-01	  1.3840845e+00 	 1.7578006e-01


.. parsed-literal::

      83	 1.4361427e+00	 1.2849749e-01	 1.4887039e+00	 1.1663088e-01	  1.3800618e+00 	 1.9816279e-01
      84	 1.4412146e+00	 1.2806048e-01	 1.4939734e+00	 1.1640469e-01	  1.3698313e+00 	 1.7942452e-01


.. parsed-literal::

      85	 1.4439052e+00	 1.2736223e-01	 1.4969143e+00	 1.1695799e-01	  1.3643230e+00 	 1.9852424e-01
      86	 1.4486115e+00	 1.2740813e-01	 1.5014811e+00	 1.1655507e-01	  1.3760293e+00 	 2.0819902e-01


.. parsed-literal::

      87	 1.4506404e+00	 1.2734398e-01	 1.5034468e+00	 1.1650671e-01	  1.3819530e+00 	 1.7964482e-01


.. parsed-literal::

      88	 1.4537882e+00	 1.2716830e-01	 1.5066280e+00	 1.1661339e-01	  1.3841946e+00 	 2.1387696e-01
      89	 1.4574433e+00	 1.2692238e-01	 1.5104413e+00	 1.1692516e-01	  1.3840569e+00 	 1.8469143e-01


.. parsed-literal::

      90	 1.4616365e+00	 1.2667557e-01	 1.5146888e+00	 1.1707033e-01	  1.3778238e+00 	 2.0554304e-01


.. parsed-literal::

      91	 1.4650260e+00	 1.2650907e-01	 1.5181389e+00	 1.1722214e-01	  1.3730022e+00 	 2.0580435e-01
      92	 1.4681837e+00	 1.2630367e-01	 1.5213962e+00	 1.1732749e-01	  1.3702175e+00 	 1.9001579e-01


.. parsed-literal::

      93	 1.4706841e+00	 1.2588005e-01	 1.5240931e+00	 1.1794128e-01	  1.3671265e+00 	 2.0642352e-01


.. parsed-literal::

      94	 1.4740115e+00	 1.2579594e-01	 1.5273692e+00	 1.1782260e-01	  1.3730476e+00 	 2.1719551e-01
      95	 1.4758672e+00	 1.2574905e-01	 1.5291884e+00	 1.1788514e-01	  1.3770608e+00 	 1.8118262e-01


.. parsed-literal::

      96	 1.4791285e+00	 1.2563613e-01	 1.5324851e+00	 1.1818389e-01	  1.3779147e+00 	 2.0264792e-01
      97	 1.4812964e+00	 1.2527201e-01	 1.5348027e+00	 1.1860609e-01	  1.3736186e+00 	 1.9737339e-01


.. parsed-literal::

      98	 1.4846831e+00	 1.2509322e-01	 1.5382111e+00	 1.1867818e-01	  1.3746727e+00 	 2.1822166e-01
      99	 1.4861613e+00	 1.2499895e-01	 1.5396556e+00	 1.1850768e-01	  1.3724359e+00 	 1.9324231e-01


.. parsed-literal::

     100	 1.4881942e+00	 1.2477905e-01	 1.5417577e+00	 1.1843768e-01	  1.3682144e+00 	 1.9833159e-01


.. parsed-literal::

     101	 1.4905458e+00	 1.2452636e-01	 1.5443016e+00	 1.1864824e-01	  1.3626484e+00 	 2.1384215e-01


.. parsed-literal::

     102	 1.4931094e+00	 1.2433806e-01	 1.5469244e+00	 1.1845300e-01	  1.3602442e+00 	 2.1618986e-01


.. parsed-literal::

     103	 1.4951072e+00	 1.2426898e-01	 1.5489398e+00	 1.1851497e-01	  1.3640766e+00 	 2.0751548e-01


.. parsed-literal::

     104	 1.4973189e+00	 1.2422914e-01	 1.5511718e+00	 1.1860280e-01	  1.3673156e+00 	 2.0820498e-01
     105	 1.4998742e+00	 1.2409885e-01	 1.5538344e+00	 1.1869530e-01	  1.3672707e+00 	 1.7577100e-01


.. parsed-literal::

     106	 1.5027450e+00	 1.2403786e-01	 1.5566696e+00	 1.1889684e-01	  1.3702805e+00 	 1.7938089e-01


.. parsed-literal::

     107	 1.5043055e+00	 1.2397826e-01	 1.5582338e+00	 1.1881274e-01	  1.3676765e+00 	 2.0811033e-01


.. parsed-literal::

     108	 1.5059468e+00	 1.2387908e-01	 1.5599361e+00	 1.1895737e-01	  1.3666685e+00 	 2.0206738e-01
     109	 1.5079250e+00	 1.2369164e-01	 1.5619477e+00	 1.1903213e-01	  1.3648631e+00 	 1.8182778e-01


.. parsed-literal::

     110	 1.5097247e+00	 1.2355042e-01	 1.5637302e+00	 1.1901730e-01	  1.3665125e+00 	 2.1615314e-01


.. parsed-literal::

     111	 1.5114928e+00	 1.2335213e-01	 1.5654924e+00	 1.1910953e-01	  1.3711254e+00 	 2.1337938e-01
     112	 1.5130750e+00	 1.2319599e-01	 1.5670371e+00	 1.1902329e-01	  1.3714574e+00 	 1.8777966e-01


.. parsed-literal::

     113	 1.5148042e+00	 1.2307589e-01	 1.5687409e+00	 1.1899199e-01	  1.3697416e+00 	 2.0124865e-01


.. parsed-literal::

     114	 1.5163715e+00	 1.2294601e-01	 1.5703677e+00	 1.1900390e-01	  1.3641445e+00 	 2.1505952e-01


.. parsed-literal::

     115	 1.5178093e+00	 1.2286459e-01	 1.5717961e+00	 1.1911186e-01	  1.3618028e+00 	 2.1445441e-01


.. parsed-literal::

     116	 1.5185782e+00	 1.2283777e-01	 1.5725729e+00	 1.1915765e-01	  1.3614247e+00 	 2.0668578e-01


.. parsed-literal::

     117	 1.5207000e+00	 1.2270421e-01	 1.5747945e+00	 1.1937273e-01	  1.3586750e+00 	 2.0199394e-01
     118	 1.5217554e+00	 1.2259523e-01	 1.5760003e+00	 1.1948336e-01	  1.3486577e+00 	 2.0367002e-01


.. parsed-literal::

     119	 1.5236948e+00	 1.2249963e-01	 1.5779135e+00	 1.1950492e-01	  1.3528565e+00 	 2.1759939e-01
     120	 1.5249537e+00	 1.2242312e-01	 1.5791610e+00	 1.1943090e-01	  1.3541876e+00 	 1.8137980e-01


.. parsed-literal::

     121	 1.5261312e+00	 1.2236730e-01	 1.5803395e+00	 1.1935393e-01	  1.3518576e+00 	 2.0873356e-01


.. parsed-literal::

     122	 1.5284569e+00	 1.2226026e-01	 1.5826905e+00	 1.1943317e-01	  1.3424541e+00 	 2.1275854e-01
     123	 1.5291795e+00	 1.2206257e-01	 1.5835948e+00	 1.1957283e-01	  1.3178129e+00 	 1.8118882e-01


.. parsed-literal::

     124	 1.5320106e+00	 1.2205909e-01	 1.5863463e+00	 1.1980479e-01	  1.3212056e+00 	 2.1112180e-01


.. parsed-literal::

     125	 1.5329368e+00	 1.2205966e-01	 1.5872716e+00	 1.1996722e-01	  1.3218444e+00 	 2.0591617e-01


.. parsed-literal::

     126	 1.5340813e+00	 1.2204898e-01	 1.5884919e+00	 1.2035576e-01	  1.3169124e+00 	 2.0325160e-01


.. parsed-literal::

     127	 1.5350121e+00	 1.2197733e-01	 1.5895089e+00	 1.2076068e-01	  1.3093618e+00 	 2.1785212e-01


.. parsed-literal::

     128	 1.5359106e+00	 1.2191057e-01	 1.5903722e+00	 1.2063208e-01	  1.3066574e+00 	 2.1270657e-01


.. parsed-literal::

     129	 1.5374854e+00	 1.2173909e-01	 1.5919402e+00	 1.2051461e-01	  1.2941361e+00 	 2.1856880e-01


.. parsed-literal::

     130	 1.5382846e+00	 1.2160072e-01	 1.5927449e+00	 1.2055971e-01	  1.2772051e+00 	 2.1970940e-01
     131	 1.5392201e+00	 1.2154041e-01	 1.5936490e+00	 1.2054529e-01	  1.2728650e+00 	 2.0294070e-01


.. parsed-literal::

     132	 1.5399722e+00	 1.2148625e-01	 1.5944005e+00	 1.2068055e-01	  1.2696173e+00 	 2.1516657e-01


.. parsed-literal::

     133	 1.5408617e+00	 1.2140503e-01	 1.5952832e+00	 1.2081163e-01	  1.2652918e+00 	 2.0620775e-01


.. parsed-literal::

     134	 1.5425308e+00	 1.2122880e-01	 1.5969352e+00	 1.2108589e-01	  1.2534984e+00 	 2.1247172e-01
     135	 1.5440593e+00	 1.2088415e-01	 1.5985592e+00	 1.2139623e-01	  1.2157625e+00 	 1.7438436e-01


.. parsed-literal::

     136	 1.5457272e+00	 1.2078748e-01	 1.6002047e+00	 1.2152077e-01	  1.2104438e+00 	 1.8996549e-01


.. parsed-literal::

     137	 1.5466055e+00	 1.2075215e-01	 1.6010802e+00	 1.2137471e-01	  1.2069149e+00 	 2.0373392e-01


.. parsed-literal::

     138	 1.5473496e+00	 1.2055569e-01	 1.6019197e+00	 1.2136313e-01	  1.1839577e+00 	 2.0924854e-01


.. parsed-literal::

     139	 1.5482912e+00	 1.2050695e-01	 1.6028789e+00	 1.2132079e-01	  1.1769325e+00 	 2.0410800e-01


.. parsed-literal::

     140	 1.5491233e+00	 1.2043508e-01	 1.6037161e+00	 1.2139338e-01	  1.1688422e+00 	 2.0911932e-01


.. parsed-literal::

     141	 1.5505757e+00	 1.2022026e-01	 1.6052465e+00	 1.2158459e-01	  1.1417448e+00 	 2.0995069e-01
     142	 1.5512972e+00	 1.2000908e-01	 1.6060293e+00	 1.2187479e-01	  1.1165883e+00 	 1.9886756e-01


.. parsed-literal::

     143	 1.5523725e+00	 1.1996601e-01	 1.6070670e+00	 1.2188826e-01	  1.1161264e+00 	 2.0541239e-01


.. parsed-literal::

     144	 1.5534604e+00	 1.1987246e-01	 1.6081530e+00	 1.2201406e-01	  1.1136984e+00 	 2.1603465e-01
     145	 1.5543516e+00	 1.1978342e-01	 1.6090591e+00	 1.2218818e-01	  1.1155565e+00 	 1.8972349e-01


.. parsed-literal::

     146	 1.5556064e+00	 1.1953966e-01	 1.6104469e+00	 1.2265907e-01	  1.1197232e+00 	 1.9957781e-01


.. parsed-literal::

     147	 1.5569783e+00	 1.1951822e-01	 1.6117832e+00	 1.2270430e-01	  1.1201146e+00 	 2.0408154e-01


.. parsed-literal::

     148	 1.5575691e+00	 1.1945980e-01	 1.6124150e+00	 1.2280546e-01	  1.1111028e+00 	 2.0896745e-01
     149	 1.5583590e+00	 1.1940340e-01	 1.6132616e+00	 1.2290760e-01	  1.0990475e+00 	 2.0499325e-01


.. parsed-literal::

     150	 1.5595985e+00	 1.1924175e-01	 1.6146696e+00	 1.2344857e-01	  1.0728985e+00 	 1.8498778e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.11 s, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa820304e20>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.05 s, sys: 38 ms, total: 2.09 s
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

