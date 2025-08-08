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
       1	-3.3691487e-01	 3.1856810e-01	-3.2721430e-01	 3.2801007e-01	[-3.4605923e-01]	 4.6179628e-01


.. parsed-literal::

       2	-2.6448539e-01	 3.0710006e-01	-2.4002491e-01	 3.1774491e-01	[-2.7308003e-01]	 2.3315597e-01


.. parsed-literal::

       3	-2.2335935e-01	 2.8900132e-01	-1.8387685e-01	 2.9802241e-01	[-2.2366294e-01]	 2.9413247e-01
       4	-1.8262171e-01	 2.6395121e-01	-1.4084881e-01	 2.7078547e-01	[-1.8346644e-01]	 1.9224977e-01


.. parsed-literal::

       5	-1.0436683e-01	 2.5605486e-01	-6.9915488e-02	 2.6395742e-01	[-1.1194490e-01]	 2.1603847e-01


.. parsed-literal::

       6	-6.5113112e-02	 2.4979265e-01	-3.3119073e-02	 2.5749750e-01	[-6.2499811e-02]	 2.1082020e-01


.. parsed-literal::

       7	-4.9896907e-02	 2.4770444e-01	-2.3865087e-02	 2.5528122e-01	[-5.4458238e-02]	 2.0492792e-01
       8	-3.2223461e-02	 2.4470785e-01	-1.1300405e-02	 2.5218257e-01	[-4.2085168e-02]	 1.8916297e-01


.. parsed-literal::

       9	-1.8305221e-02	 2.4220444e-01	-7.9989644e-04	 2.4987036e-01	[-3.4223899e-02]	 2.1967220e-01


.. parsed-literal::

      10	-1.0457270e-02	 2.4108639e-01	 4.5900068e-03	 2.4817433e-01	[-2.4459065e-02]	 2.1768498e-01


.. parsed-literal::

      11	-4.7354955e-03	 2.3996398e-01	 9.8309358e-03	 2.4711430e-01	[-2.0771779e-02]	 2.1003675e-01
      12	-1.7363542e-03	 2.3945204e-01	 1.2409379e-02	 2.4668334e-01	[-1.8486062e-02]	 2.0658708e-01


.. parsed-literal::

      13	 1.7451970e-03	 2.3881632e-01	 1.5755843e-02	 2.4584447e-01	[-1.4943503e-02]	 2.1045852e-01


.. parsed-literal::

      14	 1.6222951e-02	 2.3627292e-01	 3.1178971e-02	 2.4339420e-01	[ 1.2720446e-03]	 2.2010565e-01


.. parsed-literal::

      15	 7.5652087e-02	 2.2518967e-01	 9.4378782e-02	 2.3184213e-01	[ 7.5411011e-02]	 3.3184314e-01


.. parsed-literal::

      16	 1.1036629e-01	 2.2407958e-01	 1.3074710e-01	 2.2888782e-01	[ 1.1704926e-01]	 2.1777630e-01


.. parsed-literal::

      17	 2.1719042e-01	 2.1986644e-01	 2.4073708e-01	 2.2399955e-01	[ 2.2716611e-01]	 2.0998907e-01


.. parsed-literal::

      18	 2.9217434e-01	 2.1565542e-01	 3.2199904e-01	 2.2168567e-01	[ 2.9497367e-01]	 2.0597482e-01


.. parsed-literal::

      19	 3.3303831e-01	 2.1212209e-01	 3.6469907e-01	 2.2194185e-01	[ 3.2418244e-01]	 2.2305489e-01


.. parsed-literal::

      20	 3.8418254e-01	 2.0717818e-01	 4.1683768e-01	 2.1591570e-01	[ 3.7695710e-01]	 2.1320415e-01
      21	 4.5837637e-01	 2.0059399e-01	 4.9196637e-01	 2.0756870e-01	[ 4.5958320e-01]	 2.0016813e-01


.. parsed-literal::

      22	 5.6578820e-01	 1.9568504e-01	 6.0139791e-01	 2.0201821e-01	[ 5.8337412e-01]	 2.0816326e-01


.. parsed-literal::

      23	 5.7813684e-01	 1.9997574e-01	 6.2037497e-01	 2.0321133e-01	[ 6.6170768e-01]	 2.0917082e-01


.. parsed-literal::

      24	 6.5565669e-01	 1.9186028e-01	 6.9335303e-01	 1.9797842e-01	[ 7.0015306e-01]	 2.0660472e-01
      25	 6.8299792e-01	 1.9049262e-01	 7.2050961e-01	 1.9633664e-01	[ 7.2543119e-01]	 1.7317986e-01


.. parsed-literal::

      26	 7.1047845e-01	 1.9034086e-01	 7.4768585e-01	 1.9473764e-01	[ 7.4607270e-01]	 3.2295656e-01
      27	 7.4002260e-01	 1.8945711e-01	 7.7749874e-01	 1.9257105e-01	[ 7.7355072e-01]	 1.7442513e-01


.. parsed-literal::

      28	 7.7297539e-01	 1.8918485e-01	 8.1120990e-01	 1.9240447e-01	[ 8.1389056e-01]	 2.0929766e-01


.. parsed-literal::

      29	 7.9514607e-01	 1.8557464e-01	 8.3354959e-01	 1.8976231e-01	[ 8.2926254e-01]	 2.1149325e-01
      30	 8.1535721e-01	 1.8785867e-01	 8.5313883e-01	 1.9222903e-01	[ 8.4161407e-01]	 2.0135593e-01


.. parsed-literal::

      31	 8.3387145e-01	 1.9042898e-01	 8.7172759e-01	 1.9368253e-01	[ 8.6188897e-01]	 2.1847701e-01


.. parsed-literal::

      32	 8.5707617e-01	 1.8641625e-01	 8.9556733e-01	 1.9042695e-01	[ 8.8288866e-01]	 2.0230627e-01
      33	 8.8615766e-01	 1.8362202e-01	 9.2654740e-01	 1.8871405e-01	[ 9.0340164e-01]	 1.9548345e-01


.. parsed-literal::

      34	 9.0353710e-01	 1.8031074e-01	 9.4477494e-01	 1.8562871e-01	[ 9.2003596e-01]	 2.1050620e-01


.. parsed-literal::

      35	 9.1662755e-01	 1.7924620e-01	 9.5778157e-01	 1.8469138e-01	[ 9.3233770e-01]	 2.2458792e-01


.. parsed-literal::

      36	 9.4006602e-01	 1.7833774e-01	 9.8154355e-01	 1.8437439e-01	[ 9.5003232e-01]	 2.1978521e-01


.. parsed-literal::

      37	 9.5609466e-01	 1.7721556e-01	 9.9881525e-01	 1.8373978e-01	[ 9.6182911e-01]	 2.0775485e-01


.. parsed-literal::

      38	 9.7145366e-01	 1.7568156e-01	 1.0142548e+00	 1.8234924e-01	[ 9.7404737e-01]	 2.1918082e-01


.. parsed-literal::

      39	 9.9079248e-01	 1.7401414e-01	 1.0340855e+00	 1.8067133e-01	[ 9.8849379e-01]	 2.0675230e-01


.. parsed-literal::

      40	 1.0062121e+00	 1.7295491e-01	 1.0501092e+00	 1.7936246e-01	[ 1.0023801e+00]	 2.1555805e-01
      41	 1.0295563e+00	 1.7225197e-01	 1.0749220e+00	 1.7875457e-01	[ 1.0201505e+00]	 1.7423606e-01


.. parsed-literal::

      42	 1.0454267e+00	 1.7069272e-01	 1.0908639e+00	 1.7682098e-01	[ 1.0339320e+00]	 2.0545363e-01


.. parsed-literal::

      43	 1.0592181e+00	 1.6899852e-01	 1.1045568e+00	 1.7448134e-01	[ 1.0481981e+00]	 2.2226453e-01


.. parsed-literal::

      44	 1.0712842e+00	 1.6855313e-01	 1.1172393e+00	 1.7416827e-01	[ 1.0577738e+00]	 2.1047854e-01


.. parsed-literal::

      45	 1.0837413e+00	 1.6694194e-01	 1.1306643e+00	 1.7216581e-01	[ 1.0658337e+00]	 2.1249676e-01


.. parsed-literal::

      46	 1.0927604e+00	 1.6672341e-01	 1.1398043e+00	 1.7170646e-01	[ 1.0701529e+00]	 2.0606041e-01


.. parsed-literal::

      47	 1.1005820e+00	 1.6587537e-01	 1.1474816e+00	 1.7065858e-01	[ 1.0775970e+00]	 2.1362734e-01


.. parsed-literal::

      48	 1.1090412e+00	 1.6476184e-01	 1.1559185e+00	 1.6956853e-01	[ 1.0843567e+00]	 2.0408130e-01
      49	 1.1175496e+00	 1.6343342e-01	 1.1646521e+00	 1.6824160e-01	[ 1.0866030e+00]	 1.8644333e-01


.. parsed-literal::

      50	 1.1286597e+00	 1.6142008e-01	 1.1762854e+00	 1.6685192e-01	[ 1.0877571e+00]	 2.1428633e-01
      51	 1.1375400e+00	 1.5997491e-01	 1.1852694e+00	 1.6549040e-01	[ 1.0928362e+00]	 1.8768406e-01


.. parsed-literal::

      52	 1.1455320e+00	 1.5927284e-01	 1.1934759e+00	 1.6467144e-01	[ 1.0997588e+00]	 2.0961761e-01


.. parsed-literal::

      53	 1.1569202e+00	 1.5762823e-01	 1.2055144e+00	 1.6279169e-01	[ 1.1086137e+00]	 2.1708250e-01


.. parsed-literal::

      54	 1.1624348e+00	 1.5699335e-01	 1.2113044e+00	 1.6201186e-01	[ 1.1131538e+00]	 2.1602440e-01


.. parsed-literal::

      55	 1.1701086e+00	 1.5623279e-01	 1.2187821e+00	 1.6113807e-01	[ 1.1199275e+00]	 2.1068072e-01


.. parsed-literal::

      56	 1.1777834e+00	 1.5497021e-01	 1.2264598e+00	 1.5973566e-01	[ 1.1249214e+00]	 2.1669197e-01


.. parsed-literal::

      57	 1.1837249e+00	 1.5421887e-01	 1.2323886e+00	 1.5899365e-01	[ 1.1288288e+00]	 2.1972704e-01


.. parsed-literal::

      58	 1.1989040e+00	 1.5219584e-01	 1.2482121e+00	 1.5696584e-01	[ 1.1345782e+00]	 2.1099997e-01


.. parsed-literal::

      59	 1.2039273e+00	 1.5151274e-01	 1.2539948e+00	 1.5586910e-01	[ 1.1350989e+00]	 2.2684646e-01


.. parsed-literal::

      60	 1.2180711e+00	 1.5117606e-01	 1.2676112e+00	 1.5616590e-01	[ 1.1493682e+00]	 2.1151066e-01


.. parsed-literal::

      61	 1.2249992e+00	 1.5031984e-01	 1.2747755e+00	 1.5558514e-01	[ 1.1535538e+00]	 2.1149397e-01


.. parsed-literal::

      62	 1.2332905e+00	 1.5010408e-01	 1.2833428e+00	 1.5598418e-01	[ 1.1563318e+00]	 2.0684600e-01


.. parsed-literal::

      63	 1.2429571e+00	 1.4892268e-01	 1.2934978e+00	 1.5577144e-01	[ 1.1584852e+00]	 2.1288943e-01


.. parsed-literal::

      64	 1.2525106e+00	 1.4834980e-01	 1.3029624e+00	 1.5553085e-01	[ 1.1629992e+00]	 2.1583652e-01
      65	 1.2596317e+00	 1.4729036e-01	 1.3100811e+00	 1.5442359e-01	[ 1.1671741e+00]	 1.8238950e-01


.. parsed-literal::

      66	 1.2713614e+00	 1.4550575e-01	 1.3220876e+00	 1.5238772e-01	[ 1.1756877e+00]	 2.1348095e-01


.. parsed-literal::

      67	 1.2724322e+00	 1.4503966e-01	 1.3238536e+00	 1.5151812e-01	  1.1696555e+00 	 2.1664143e-01


.. parsed-literal::

      68	 1.2839387e+00	 1.4464122e-01	 1.3348692e+00	 1.5116543e-01	[ 1.1872729e+00]	 2.0830083e-01


.. parsed-literal::

      69	 1.2878922e+00	 1.4453083e-01	 1.3388596e+00	 1.5112575e-01	[ 1.1929530e+00]	 2.1068311e-01


.. parsed-literal::

      70	 1.2936571e+00	 1.4428670e-01	 1.3448120e+00	 1.5096982e-01	[ 1.1999433e+00]	 2.1208549e-01


.. parsed-literal::

      71	 1.3012540e+00	 1.4366872e-01	 1.3525634e+00	 1.5058527e-01	[ 1.2088461e+00]	 2.0833063e-01
      72	 1.3037196e+00	 1.4283142e-01	 1.3556132e+00	 1.5013092e-01	  1.2007339e+00 	 1.8041444e-01


.. parsed-literal::

      73	 1.3152826e+00	 1.4223840e-01	 1.3667692e+00	 1.4951469e-01	[ 1.2177267e+00]	 2.1480608e-01
      74	 1.3195769e+00	 1.4179342e-01	 1.3709399e+00	 1.4899429e-01	[ 1.2230107e+00]	 1.8110538e-01


.. parsed-literal::

      75	 1.3264021e+00	 1.4066057e-01	 1.3778485e+00	 1.4762091e-01	[ 1.2290612e+00]	 2.1125221e-01


.. parsed-literal::

      76	 1.3346374e+00	 1.3992767e-01	 1.3861850e+00	 1.4614439e-01	[ 1.2359660e+00]	 2.1091199e-01


.. parsed-literal::

      77	 1.3396022e+00	 1.3870144e-01	 1.3918791e+00	 1.4436121e-01	  1.2306408e+00 	 2.1331954e-01


.. parsed-literal::

      78	 1.3479890e+00	 1.3890737e-01	 1.3999848e+00	 1.4435411e-01	[ 1.2412282e+00]	 2.1208167e-01
      79	 1.3524263e+00	 1.3916800e-01	 1.4044574e+00	 1.4459961e-01	[ 1.2457665e+00]	 1.9536710e-01


.. parsed-literal::

      80	 1.3584982e+00	 1.3926653e-01	 1.4107814e+00	 1.4460664e-01	[ 1.2510553e+00]	 2.1838117e-01


.. parsed-literal::

      81	 1.3622986e+00	 1.3963214e-01	 1.4148204e+00	 1.4551343e-01	[ 1.2510957e+00]	 2.1341491e-01


.. parsed-literal::

      82	 1.3676545e+00	 1.3906083e-01	 1.4200439e+00	 1.4494866e-01	[ 1.2561346e+00]	 2.2220564e-01


.. parsed-literal::

      83	 1.3717054e+00	 1.3837346e-01	 1.4241350e+00	 1.4420750e-01	[ 1.2583718e+00]	 2.1305943e-01


.. parsed-literal::

      84	 1.3761024e+00	 1.3773389e-01	 1.4286542e+00	 1.4362241e-01	[ 1.2593761e+00]	 2.2148490e-01


.. parsed-literal::

      85	 1.3826936e+00	 1.3716790e-01	 1.4355938e+00	 1.4353680e-01	  1.2544284e+00 	 2.0445561e-01
      86	 1.3870175e+00	 1.3685689e-01	 1.4401053e+00	 1.4371904e-01	  1.2489317e+00 	 1.8355203e-01


.. parsed-literal::

      87	 1.3912838e+00	 1.3690998e-01	 1.4441296e+00	 1.4379011e-01	  1.2569605e+00 	 2.1016717e-01


.. parsed-literal::

      88	 1.3950251e+00	 1.3695449e-01	 1.4479119e+00	 1.4398217e-01	[ 1.2601729e+00]	 2.0935059e-01


.. parsed-literal::

      89	 1.4004272e+00	 1.3674224e-01	 1.4533953e+00	 1.4426293e-01	[ 1.2609349e+00]	 2.1138501e-01


.. parsed-literal::

      90	 1.4026787e+00	 1.3631039e-01	 1.4558993e+00	 1.4367065e-01	  1.2542669e+00 	 2.0338535e-01
      91	 1.4098809e+00	 1.3604974e-01	 1.4629276e+00	 1.4396031e-01	[ 1.2616424e+00]	 1.9517994e-01


.. parsed-literal::

      92	 1.4132136e+00	 1.3575229e-01	 1.4662539e+00	 1.4385555e-01	[ 1.2617594e+00]	 1.9808578e-01


.. parsed-literal::

      93	 1.4171230e+00	 1.3537620e-01	 1.4702924e+00	 1.4361380e-01	  1.2595483e+00 	 2.0327234e-01


.. parsed-literal::

      94	 1.4226690e+00	 1.3495374e-01	 1.4759865e+00	 1.4351276e-01	  1.2574384e+00 	 2.2170448e-01
      95	 1.4264257e+00	 1.3459962e-01	 1.4799272e+00	 1.4262470e-01	  1.2547346e+00 	 1.9445014e-01


.. parsed-literal::

      96	 1.4305329e+00	 1.3449517e-01	 1.4838485e+00	 1.4271680e-01	  1.2613745e+00 	 2.0767903e-01


.. parsed-literal::

      97	 1.4330619e+00	 1.3435211e-01	 1.4862896e+00	 1.4263586e-01	[ 1.2656376e+00]	 2.0664406e-01


.. parsed-literal::

      98	 1.4367603e+00	 1.3382814e-01	 1.4900876e+00	 1.4221711e-01	  1.2643973e+00 	 2.1523643e-01
      99	 1.4407417e+00	 1.3348109e-01	 1.4941149e+00	 1.4194783e-01	  1.2650475e+00 	 1.9307828e-01


.. parsed-literal::

     100	 1.4437960e+00	 1.3323507e-01	 1.4972493e+00	 1.4170229e-01	  1.2650941e+00 	 2.0681763e-01


.. parsed-literal::

     101	 1.4468503e+00	 1.3296767e-01	 1.5005048e+00	 1.4159562e-01	  1.2646937e+00 	 2.1306181e-01


.. parsed-literal::

     102	 1.4495735e+00	 1.3286433e-01	 1.5032961e+00	 1.4143222e-01	[ 1.2684462e+00]	 2.0446992e-01
     103	 1.4519528e+00	 1.3278323e-01	 1.5056467e+00	 1.4146003e-01	[ 1.2720428e+00]	 1.9685507e-01


.. parsed-literal::

     104	 1.4557369e+00	 1.3260193e-01	 1.5095286e+00	 1.4190305e-01	[ 1.2731802e+00]	 2.1787906e-01


.. parsed-literal::

     105	 1.4573916e+00	 1.3248588e-01	 1.5112567e+00	 1.4158695e-01	  1.2729152e+00 	 2.0411158e-01


.. parsed-literal::

     106	 1.4594380e+00	 1.3243813e-01	 1.5132655e+00	 1.4160181e-01	[ 1.2748952e+00]	 2.2211885e-01


.. parsed-literal::

     107	 1.4621776e+00	 1.3233367e-01	 1.5160355e+00	 1.4141224e-01	[ 1.2756303e+00]	 2.1832061e-01


.. parsed-literal::

     108	 1.4638388e+00	 1.3225780e-01	 1.5177203e+00	 1.4115296e-01	[ 1.2768201e+00]	 2.2120142e-01


.. parsed-literal::

     109	 1.4673524e+00	 1.3179574e-01	 1.5214749e+00	 1.4017440e-01	  1.2732602e+00 	 2.0666504e-01


.. parsed-literal::

     110	 1.4697182e+00	 1.3179946e-01	 1.5239458e+00	 1.3992616e-01	  1.2732327e+00 	 2.1411347e-01
     111	 1.4714574e+00	 1.3175301e-01	 1.5255690e+00	 1.4003935e-01	  1.2740939e+00 	 1.9466519e-01


.. parsed-literal::

     112	 1.4731171e+00	 1.3171528e-01	 1.5272058e+00	 1.4004118e-01	  1.2717378e+00 	 1.7883182e-01
     113	 1.4750353e+00	 1.3166171e-01	 1.5291493e+00	 1.3997146e-01	  1.2663919e+00 	 1.8565869e-01


.. parsed-literal::

     114	 1.4771663e+00	 1.3172339e-01	 1.5312649e+00	 1.3991562e-01	  1.2625288e+00 	 2.1904230e-01
     115	 1.4789503e+00	 1.3174356e-01	 1.5330284e+00	 1.3979442e-01	  1.2617030e+00 	 1.8103814e-01


.. parsed-literal::

     116	 1.4806766e+00	 1.3174721e-01	 1.5347839e+00	 1.3975435e-01	  1.2600565e+00 	 2.1319556e-01


.. parsed-literal::

     117	 1.4824074e+00	 1.3170692e-01	 1.5365306e+00	 1.3975522e-01	  1.2599624e+00 	 2.1227956e-01


.. parsed-literal::

     118	 1.4840839e+00	 1.3162394e-01	 1.5382112e+00	 1.3970067e-01	  1.2604299e+00 	 2.1582389e-01


.. parsed-literal::

     119	 1.4862434e+00	 1.3150527e-01	 1.5404576e+00	 1.3963810e-01	  1.2559046e+00 	 2.1492958e-01


.. parsed-literal::

     120	 1.4878294e+00	 1.3142013e-01	 1.5420819e+00	 1.3943529e-01	  1.2520111e+00 	 2.1423101e-01


.. parsed-literal::

     121	 1.4897446e+00	 1.3138309e-01	 1.5440140e+00	 1.3920026e-01	  1.2448207e+00 	 2.7278852e-01


.. parsed-literal::

     122	 1.4914232e+00	 1.3134802e-01	 1.5457135e+00	 1.3907898e-01	  1.2365667e+00 	 2.2482109e-01


.. parsed-literal::

     123	 1.4930625e+00	 1.3134050e-01	 1.5473079e+00	 1.3883892e-01	  1.2326081e+00 	 2.0602608e-01
     124	 1.4942725e+00	 1.3126212e-01	 1.5484786e+00	 1.3879064e-01	  1.2341008e+00 	 1.9742084e-01


.. parsed-literal::

     125	 1.4965292e+00	 1.3112295e-01	 1.5507395e+00	 1.3865345e-01	  1.2338221e+00 	 2.0636916e-01
     126	 1.4975060e+00	 1.3087138e-01	 1.5518952e+00	 1.3848213e-01	  1.2238901e+00 	 1.7832732e-01


.. parsed-literal::

     127	 1.4997284e+00	 1.3088826e-01	 1.5540480e+00	 1.3843627e-01	  1.2271943e+00 	 2.0325708e-01


.. parsed-literal::

     128	 1.5010855e+00	 1.3088303e-01	 1.5554440e+00	 1.3832112e-01	  1.2228577e+00 	 2.1550655e-01
     129	 1.5027806e+00	 1.3083083e-01	 1.5572145e+00	 1.3815694e-01	  1.2154618e+00 	 1.9934464e-01


.. parsed-literal::

     130	 1.5049403e+00	 1.3067048e-01	 1.5594193e+00	 1.3794893e-01	  1.2063917e+00 	 2.0256066e-01
     131	 1.5071322e+00	 1.3045739e-01	 1.5616606e+00	 1.3779469e-01	  1.1941893e+00 	 1.8577981e-01


.. parsed-literal::

     132	 1.5084955e+00	 1.3034568e-01	 1.5629741e+00	 1.3784625e-01	  1.1968468e+00 	 2.0546174e-01


.. parsed-literal::

     133	 1.5098974e+00	 1.3016482e-01	 1.5643443e+00	 1.3782888e-01	  1.1926940e+00 	 2.2178078e-01


.. parsed-literal::

     134	 1.5113836e+00	 1.3004588e-01	 1.5658787e+00	 1.3784161e-01	  1.1908684e+00 	 2.1084857e-01


.. parsed-literal::

     135	 1.5128659e+00	 1.2999711e-01	 1.5673544e+00	 1.3785238e-01	  1.1825574e+00 	 2.1826267e-01


.. parsed-literal::

     136	 1.5147046e+00	 1.2987240e-01	 1.5692534e+00	 1.3762944e-01	  1.1607093e+00 	 2.0821357e-01
     137	 1.5160916e+00	 1.2982827e-01	 1.5706523e+00	 1.3762106e-01	  1.1518093e+00 	 1.8802738e-01


.. parsed-literal::

     138	 1.5177345e+00	 1.2969919e-01	 1.5722978e+00	 1.3749462e-01	  1.1421125e+00 	 2.0581317e-01
     139	 1.5196459e+00	 1.2931356e-01	 1.5742695e+00	 1.3712249e-01	  1.1237517e+00 	 1.9829822e-01


.. parsed-literal::

     140	 1.5207473e+00	 1.2925843e-01	 1.5753717e+00	 1.3697286e-01	  1.1189568e+00 	 1.9679284e-01


.. parsed-literal::

     141	 1.5217653e+00	 1.2919167e-01	 1.5763484e+00	 1.3698298e-01	  1.1213131e+00 	 2.0705032e-01


.. parsed-literal::

     142	 1.5227914e+00	 1.2907110e-01	 1.5773723e+00	 1.3694142e-01	  1.1197462e+00 	 2.0528984e-01
     143	 1.5240812e+00	 1.2892721e-01	 1.5787046e+00	 1.3685788e-01	  1.1146435e+00 	 2.0819569e-01


.. parsed-literal::

     144	 1.5250540e+00	 1.2867481e-01	 1.5797720e+00	 1.3679812e-01	  1.1102061e+00 	 1.9660115e-01


.. parsed-literal::

     145	 1.5262559e+00	 1.2867253e-01	 1.5809767e+00	 1.3677166e-01	  1.1069982e+00 	 2.0224929e-01


.. parsed-literal::

     146	 1.5271642e+00	 1.2863746e-01	 1.5819170e+00	 1.3675641e-01	  1.1025713e+00 	 2.0945001e-01
     147	 1.5283467e+00	 1.2855850e-01	 1.5831518e+00	 1.3674903e-01	  1.0987020e+00 	 1.8663549e-01


.. parsed-literal::

     148	 1.5302017e+00	 1.2837077e-01	 1.5851306e+00	 1.3686199e-01	  1.0821852e+00 	 2.1038055e-01


.. parsed-literal::

     149	 1.5309429e+00	 1.2820691e-01	 1.5860648e+00	 1.3698353e-01	  1.0833814e+00 	 2.1153474e-01


.. parsed-literal::

     150	 1.5324813e+00	 1.2820356e-01	 1.5874389e+00	 1.3695960e-01	  1.0807464e+00 	 2.1595573e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.13 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f29544d3eb0>



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
    CPU times: user 2.21 s, sys: 44 ms, total: 2.25 s
    Wall time: 713 ms


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

