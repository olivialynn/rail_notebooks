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
       1	-3.4509079e-01	 3.2058099e-01	-3.3533554e-01	 3.2155218e-01	[-3.3522093e-01]	 4.5648766e-01


.. parsed-literal::

       2	-2.7126649e-01	 3.0925589e-01	-2.4705705e-01	 3.1323654e-01	[-2.4992665e-01]	 2.2760105e-01


.. parsed-literal::

       3	-2.2567442e-01	 2.8858194e-01	-1.8372647e-01	 2.9040644e-01	[-1.8354567e-01]	 3.3254075e-01


.. parsed-literal::

       4	-1.8385790e-01	 2.7021139e-01	-1.3540553e-01	 2.7017580e-01	[-1.3125844e-01]	 3.0095387e-01
       5	-1.3274872e-01	 2.5753872e-01	-1.0383678e-01	 2.5830854e-01	[-9.2650868e-02]	 1.9796491e-01


.. parsed-literal::

       6	-7.3862992e-02	 2.5357770e-01	-4.6771342e-02	 2.5136689e-01	[-3.6428724e-02]	 2.0491028e-01


.. parsed-literal::

       7	-5.5246214e-02	 2.4961215e-01	-3.2161766e-02	 2.4838263e-01	[-2.3200742e-02]	 2.0643163e-01
       8	-4.4278007e-02	 2.4779730e-01	-2.3999545e-02	 2.4604116e-01	[-1.3965486e-02]	 1.9531417e-01


.. parsed-literal::

       9	-2.8992242e-02	 2.4494650e-01	-1.1704620e-02	 2.4256531e-01	[-4.0685367e-04]	 2.1606469e-01


.. parsed-literal::

      10	-1.8884560e-02	 2.4290582e-01	-3.4890402e-03	 2.4059686e-01	[ 8.0959677e-03]	 2.0462394e-01


.. parsed-literal::

      11	-1.3661035e-02	 2.4201497e-01	 9.5322620e-04	 2.3975817e-01	[ 1.2459648e-02]	 2.0814466e-01
      12	-9.1571984e-03	 2.4110638e-01	 5.0725386e-03	 2.3912308e-01	[ 1.5380347e-02]	 1.7170882e-01


.. parsed-literal::

      13	-4.2727136e-03	 2.4006989e-01	 1.0038979e-02	 2.3856837e-01	[ 1.8521327e-02]	 2.1005940e-01


.. parsed-literal::

      14	 7.9606676e-02	 2.2787626e-01	 9.8708689e-02	 2.3521807e-01	[ 9.7577604e-02]	 2.9706144e-01


.. parsed-literal::

      15	 1.1344189e-01	 2.2726212e-01	 1.3556189e-01	 2.2659314e-01	[ 1.3736970e-01]	 2.1018958e-01


.. parsed-literal::

      16	 1.7868567e-01	 2.2069316e-01	 2.0464731e-01	 2.2075553e-01	[ 2.0802303e-01]	 2.0901608e-01
      17	 2.8773698e-01	 2.1480025e-01	 3.1799730e-01	 2.1658937e-01	[ 3.2152683e-01]	 1.7422891e-01


.. parsed-literal::

      18	 3.5216124e-01	 2.1078035e-01	 3.8541709e-01	 2.1125991e-01	[ 3.8369187e-01]	 2.0097804e-01


.. parsed-literal::

      19	 4.0348682e-01	 2.0993262e-01	 4.3716914e-01	 2.0922581e-01	[ 4.3184554e-01]	 2.1208000e-01


.. parsed-literal::

      20	 4.5308130e-01	 2.0573715e-01	 4.8781906e-01	 2.0657369e-01	[ 4.7686992e-01]	 2.0952296e-01


.. parsed-literal::

      21	 5.4259736e-01	 2.0344025e-01	 5.8146644e-01	 2.0819443e-01	[ 5.6087068e-01]	 2.0869732e-01


.. parsed-literal::

      22	 6.0488085e-01	 2.0293237e-01	 6.4425783e-01	 2.0162032e-01	[ 6.1062121e-01]	 2.2198701e-01
      23	 6.4733465e-01	 1.9615751e-01	 6.8690323e-01	 1.9322119e-01	[ 6.4805393e-01]	 1.9699860e-01


.. parsed-literal::

      24	 6.8673529e-01	 1.9282007e-01	 7.2667492e-01	 1.8945248e-01	[ 6.9560578e-01]	 1.9736671e-01


.. parsed-literal::

      25	 7.1656785e-01	 1.8903919e-01	 7.5486501e-01	 1.8629975e-01	[ 7.2338147e-01]	 2.0314884e-01
      26	 7.4297561e-01	 1.8856039e-01	 7.8239836e-01	 1.8488573e-01	[ 7.4888145e-01]	 1.9727349e-01


.. parsed-literal::

      27	 7.7128414e-01	 1.9688094e-01	 8.1333173e-01	 1.9755261e-01	[ 7.6964333e-01]	 2.0258021e-01
      28	 8.1432734e-01	 1.9127226e-01	 8.5543560e-01	 1.9076493e-01	[ 8.1151624e-01]	 1.8727803e-01


.. parsed-literal::

      29	 8.4452095e-01	 1.8937071e-01	 8.8531406e-01	 1.9038856e-01	[ 8.3746014e-01]	 2.1100664e-01
      30	 8.8856989e-01	 1.8620197e-01	 9.2995943e-01	 1.8914555e-01	[ 8.8084749e-01]	 1.7115211e-01


.. parsed-literal::

      31	 9.0891227e-01	 1.8461575e-01	 9.5242999e-01	 1.8541050e-01	[ 8.9124610e-01]	 1.8348336e-01
      32	 9.4071868e-01	 1.8205539e-01	 9.8325200e-01	 1.8072366e-01	[ 9.3544891e-01]	 1.7602873e-01


.. parsed-literal::

      33	 9.6100312e-01	 1.7934165e-01	 1.0036527e+00	 1.7735863e-01	[ 9.5799198e-01]	 1.9346333e-01


.. parsed-literal::

      34	 9.9100047e-01	 1.7425619e-01	 1.0350772e+00	 1.7176815e-01	[ 9.9078340e-01]	 2.1031618e-01


.. parsed-literal::

      35	 1.0094882e+00	 1.7244548e-01	 1.0544730e+00	 1.7079526e-01	[ 1.0105852e+00]	 2.0876813e-01


.. parsed-literal::

      36	 1.0247530e+00	 1.7091885e-01	 1.0699854e+00	 1.6973428e-01	[ 1.0257334e+00]	 2.1197200e-01
      37	 1.0414159e+00	 1.6972689e-01	 1.0871501e+00	 1.6956801e-01	[ 1.0375118e+00]	 1.8859124e-01


.. parsed-literal::

      38	 1.0559401e+00	 1.6799240e-01	 1.1024444e+00	 1.6889821e-01	[ 1.0383178e+00]	 2.0622659e-01
      39	 1.0692741e+00	 1.6679652e-01	 1.1160633e+00	 1.6724107e-01	[ 1.0462475e+00]	 1.9361496e-01


.. parsed-literal::

      40	 1.0821820e+00	 1.6479684e-01	 1.1289624e+00	 1.6484224e-01	[ 1.0568316e+00]	 2.0135808e-01
      41	 1.1017255e+00	 1.6086539e-01	 1.1486491e+00	 1.6061539e-01	[ 1.0681896e+00]	 1.7541003e-01


.. parsed-literal::

      42	 1.1133896e+00	 1.5922597e-01	 1.1598831e+00	 1.5815846e-01	[ 1.0919260e+00]	 2.0871210e-01
      43	 1.1236675e+00	 1.5768457e-01	 1.1702119e+00	 1.5616226e-01	[ 1.0990160e+00]	 1.8244219e-01


.. parsed-literal::

      44	 1.1379914e+00	 1.5527956e-01	 1.1848594e+00	 1.5273172e-01	[ 1.1052974e+00]	 2.0044065e-01


.. parsed-literal::

      45	 1.1470323e+00	 1.5443605e-01	 1.1940907e+00	 1.5162557e-01	[ 1.1119883e+00]	 2.0693564e-01
      46	 1.1673753e+00	 1.5275229e-01	 1.2149395e+00	 1.4964895e-01	[ 1.1309162e+00]	 1.8481064e-01


.. parsed-literal::

      47	 1.1783105e+00	 1.5217302e-01	 1.2262809e+00	 1.4945335e-01	[ 1.1395869e+00]	 2.1200991e-01


.. parsed-literal::

      48	 1.1881340e+00	 1.5190739e-01	 1.2361429e+00	 1.4933987e-01	[ 1.1495503e+00]	 2.0928955e-01
      49	 1.2043373e+00	 1.5171180e-01	 1.2527441e+00	 1.4989682e-01	[ 1.1636420e+00]	 1.9919276e-01


.. parsed-literal::

      50	 1.2163584e+00	 1.5111381e-01	 1.2652802e+00	 1.4887983e-01	[ 1.1778841e+00]	 2.0129132e-01
      51	 1.2285582e+00	 1.4945960e-01	 1.2774488e+00	 1.4777220e-01	[ 1.1851807e+00]	 1.9849658e-01


.. parsed-literal::

      52	 1.2409582e+00	 1.4694489e-01	 1.2899897e+00	 1.4542596e-01	[ 1.1923660e+00]	 2.0726562e-01
      53	 1.2510907e+00	 1.4531590e-01	 1.3002717e+00	 1.4379140e-01	[ 1.1951581e+00]	 2.0160699e-01


.. parsed-literal::

      54	 1.2637135e+00	 1.4323479e-01	 1.3131601e+00	 1.4150288e-01	[ 1.2000453e+00]	 1.9510412e-01
      55	 1.2757555e+00	 1.4262021e-01	 1.3252472e+00	 1.4087727e-01	[ 1.2083326e+00]	 1.6952562e-01


.. parsed-literal::

      56	 1.2885030e+00	 1.4142652e-01	 1.3383131e+00	 1.3931075e-01	[ 1.2099249e+00]	 1.8291283e-01
      57	 1.3022437e+00	 1.4055800e-01	 1.3524609e+00	 1.3745430e-01	[ 1.2169549e+00]	 2.0021343e-01


.. parsed-literal::

      58	 1.3136119e+00	 1.4018121e-01	 1.3639760e+00	 1.3693209e-01	[ 1.2193853e+00]	 2.0183015e-01
      59	 1.3216644e+00	 1.3944950e-01	 1.3722228e+00	 1.3636723e-01	[ 1.2226448e+00]	 2.0253086e-01


.. parsed-literal::

      60	 1.3308879e+00	 1.3897813e-01	 1.3821165e+00	 1.3602246e-01	  1.2183455e+00 	 2.1394205e-01
      61	 1.3374993e+00	 1.3822847e-01	 1.3884004e+00	 1.3514155e-01	[ 1.2318824e+00]	 1.9729352e-01


.. parsed-literal::

      62	 1.3471509e+00	 1.3698712e-01	 1.3981444e+00	 1.3376494e-01	[ 1.2382624e+00]	 2.0742440e-01
      63	 1.3588819e+00	 1.3515777e-01	 1.4097850e+00	 1.3189434e-01	[ 1.2424995e+00]	 1.8224525e-01


.. parsed-literal::

      64	 1.3644048e+00	 1.3396045e-01	 1.4166620e+00	 1.3092347e-01	  1.2133788e+00 	 2.0475578e-01


.. parsed-literal::

      65	 1.3769276e+00	 1.3312145e-01	 1.4284161e+00	 1.2987066e-01	  1.2317661e+00 	 2.1542406e-01


.. parsed-literal::

      66	 1.3823109e+00	 1.3284563e-01	 1.4336745e+00	 1.2959501e-01	  1.2377146e+00 	 2.0131707e-01


.. parsed-literal::

      67	 1.3902770e+00	 1.3195004e-01	 1.4420094e+00	 1.2829074e-01	  1.2343894e+00 	 2.1541333e-01


.. parsed-literal::

      68	 1.3981260e+00	 1.3137643e-01	 1.4497826e+00	 1.2745232e-01	  1.2414959e+00 	 2.0506048e-01
      69	 1.4049562e+00	 1.3085365e-01	 1.4567572e+00	 1.2660504e-01	[ 1.2450484e+00]	 1.9986200e-01


.. parsed-literal::

      70	 1.4122501e+00	 1.3038853e-01	 1.4642898e+00	 1.2581681e-01	[ 1.2487628e+00]	 2.0070481e-01


.. parsed-literal::

      71	 1.4185755e+00	 1.2989573e-01	 1.4706954e+00	 1.2542526e-01	[ 1.2539084e+00]	 2.1493340e-01


.. parsed-literal::

      72	 1.4259252e+00	 1.2962766e-01	 1.4781380e+00	 1.2520498e-01	[ 1.2591910e+00]	 2.1703196e-01


.. parsed-literal::

      73	 1.4322520e+00	 1.2922917e-01	 1.4844996e+00	 1.2478985e-01	[ 1.2673958e+00]	 2.0785451e-01


.. parsed-literal::

      74	 1.4380155e+00	 1.2857383e-01	 1.4903646e+00	 1.2445750e-01	[ 1.2739194e+00]	 2.0881844e-01


.. parsed-literal::

      75	 1.4430166e+00	 1.2848248e-01	 1.4953939e+00	 1.2426479e-01	[ 1.2783145e+00]	 2.0845127e-01


.. parsed-literal::

      76	 1.4468588e+00	 1.2817479e-01	 1.4993558e+00	 1.2398652e-01	[ 1.2805641e+00]	 2.1244216e-01


.. parsed-literal::

      77	 1.4518941e+00	 1.2776086e-01	 1.5045818e+00	 1.2360280e-01	[ 1.2816214e+00]	 2.1666908e-01
      78	 1.4561184e+00	 1.2751530e-01	 1.5089213e+00	 1.2319382e-01	  1.2809250e+00 	 1.9947886e-01


.. parsed-literal::

      79	 1.4609186e+00	 1.2738756e-01	 1.5137147e+00	 1.2302954e-01	[ 1.2844650e+00]	 2.0554304e-01
      80	 1.4645011e+00	 1.2716621e-01	 1.5172654e+00	 1.2282107e-01	  1.2842633e+00 	 1.8971634e-01


.. parsed-literal::

      81	 1.4690186e+00	 1.2693323e-01	 1.5219128e+00	 1.2251317e-01	  1.2795532e+00 	 2.0102453e-01


.. parsed-literal::

      82	 1.4732991e+00	 1.2633588e-01	 1.5263298e+00	 1.2210615e-01	  1.2770525e+00 	 2.1329141e-01


.. parsed-literal::

      83	 1.4778075e+00	 1.2619147e-01	 1.5307294e+00	 1.2191143e-01	  1.2808959e+00 	 2.0664263e-01
      84	 1.4809911e+00	 1.2611737e-01	 1.5339154e+00	 1.2175780e-01	[ 1.2847174e+00]	 1.8346906e-01


.. parsed-literal::

      85	 1.4842119e+00	 1.2593458e-01	 1.5372849e+00	 1.2147760e-01	  1.2814085e+00 	 2.0680523e-01


.. parsed-literal::

      86	 1.4859439e+00	 1.2579376e-01	 1.5392928e+00	 1.2067354e-01	  1.2794419e+00 	 2.1486354e-01


.. parsed-literal::

      87	 1.4905311e+00	 1.2552493e-01	 1.5438148e+00	 1.2040784e-01	  1.2800192e+00 	 2.1555138e-01


.. parsed-literal::

      88	 1.4921312e+00	 1.2538582e-01	 1.5453969e+00	 1.2022596e-01	  1.2801074e+00 	 2.1773553e-01
      89	 1.4954618e+00	 1.2511767e-01	 1.5488811e+00	 1.1969330e-01	  1.2781204e+00 	 1.8696451e-01


.. parsed-literal::

      90	 1.4983254e+00	 1.2482633e-01	 1.5518825e+00	 1.1904891e-01	  1.2725198e+00 	 2.1616721e-01
      91	 1.5015192e+00	 1.2492738e-01	 1.5550955e+00	 1.1880769e-01	  1.2757369e+00 	 1.8111777e-01


.. parsed-literal::

      92	 1.5039149e+00	 1.2503604e-01	 1.5575330e+00	 1.1868270e-01	  1.2756686e+00 	 2.0194721e-01


.. parsed-literal::

      93	 1.5067988e+00	 1.2508202e-01	 1.5604750e+00	 1.1840110e-01	  1.2698387e+00 	 2.0560694e-01


.. parsed-literal::

      94	 1.5076020e+00	 1.2495409e-01	 1.5614331e+00	 1.1788546e-01	  1.2604122e+00 	 2.0370603e-01


.. parsed-literal::

      95	 1.5119189e+00	 1.2485124e-01	 1.5655619e+00	 1.1770564e-01	  1.2601211e+00 	 2.0996642e-01
      96	 1.5134558e+00	 1.2471043e-01	 1.5670522e+00	 1.1749429e-01	  1.2579681e+00 	 1.6791701e-01


.. parsed-literal::

      97	 1.5157394e+00	 1.2447725e-01	 1.5693039e+00	 1.1709586e-01	  1.2528951e+00 	 1.7040753e-01


.. parsed-literal::

      98	 1.5188716e+00	 1.2428064e-01	 1.5724768e+00	 1.1671018e-01	  1.2437924e+00 	 2.0324349e-01


.. parsed-literal::

      99	 1.5205756e+00	 1.2411000e-01	 1.5742698e+00	 1.1622075e-01	  1.2359426e+00 	 2.8048539e-01
     100	 1.5230932e+00	 1.2411664e-01	 1.5768752e+00	 1.1619848e-01	  1.2281706e+00 	 1.8360519e-01


.. parsed-literal::

     101	 1.5248022e+00	 1.2413223e-01	 1.5786651e+00	 1.1626667e-01	  1.2236662e+00 	 1.9898343e-01


.. parsed-literal::

     102	 1.5270141e+00	 1.2412638e-01	 1.5810702e+00	 1.1637583e-01	  1.2110073e+00 	 2.0452929e-01
     103	 1.5289874e+00	 1.2396574e-01	 1.5831313e+00	 1.1618694e-01	  1.2020670e+00 	 1.8608975e-01


.. parsed-literal::

     104	 1.5306521e+00	 1.2374281e-01	 1.5847526e+00	 1.1600625e-01	  1.2002404e+00 	 2.0602012e-01


.. parsed-literal::

     105	 1.5329537e+00	 1.2335696e-01	 1.5870386e+00	 1.1567249e-01	  1.1879335e+00 	 2.1894002e-01
     106	 1.5344750e+00	 1.2309323e-01	 1.5885672e+00	 1.1557057e-01	  1.1714980e+00 	 1.9436097e-01


.. parsed-literal::

     107	 1.5360849e+00	 1.2307109e-01	 1.5901722e+00	 1.1561968e-01	  1.1628119e+00 	 2.1062064e-01
     108	 1.5388291e+00	 1.2305812e-01	 1.5930136e+00	 1.1575962e-01	  1.1305848e+00 	 1.9023609e-01


.. parsed-literal::

     109	 1.5403536e+00	 1.2306304e-01	 1.5945548e+00	 1.1583496e-01	  1.1213038e+00 	 1.8411279e-01


.. parsed-literal::

     110	 1.5428516e+00	 1.2294308e-01	 1.5971087e+00	 1.1586040e-01	  1.0967472e+00 	 2.0564675e-01
     111	 1.5444174e+00	 1.2278084e-01	 1.5987072e+00	 1.1575635e-01	  1.0916272e+00 	 1.6816330e-01


.. parsed-literal::

     112	 1.5461053e+00	 1.2262672e-01	 1.6003273e+00	 1.1579207e-01	  1.0951619e+00 	 1.7496681e-01


.. parsed-literal::

     113	 1.5479816e+00	 1.2238837e-01	 1.6021600e+00	 1.1584497e-01	  1.0944007e+00 	 2.1058655e-01
     114	 1.5496307e+00	 1.2219157e-01	 1.6038346e+00	 1.1596405e-01	  1.0872992e+00 	 1.9566989e-01


.. parsed-literal::

     115	 1.5511121e+00	 1.2184314e-01	 1.6053851e+00	 1.1604403e-01	  1.0769757e+00 	 2.1372366e-01
     116	 1.5528203e+00	 1.2179876e-01	 1.6070995e+00	 1.1610916e-01	  1.0711891e+00 	 1.9641542e-01


.. parsed-literal::

     117	 1.5540463e+00	 1.2172982e-01	 1.6083570e+00	 1.1612865e-01	  1.0653825e+00 	 2.1345687e-01


.. parsed-literal::

     118	 1.5559212e+00	 1.2151949e-01	 1.6102927e+00	 1.1625023e-01	  1.0574915e+00 	 2.1288466e-01


.. parsed-literal::

     119	 1.5564678e+00	 1.2129080e-01	 1.6109878e+00	 1.1648866e-01	  1.0571111e+00 	 2.1020150e-01
     120	 1.5591258e+00	 1.2109028e-01	 1.6135304e+00	 1.1645694e-01	  1.0568584e+00 	 1.9762397e-01


.. parsed-literal::

     121	 1.5599829e+00	 1.2100042e-01	 1.6143375e+00	 1.1643401e-01	  1.0601326e+00 	 1.8531728e-01


.. parsed-literal::

     122	 1.5614534e+00	 1.2081074e-01	 1.6157626e+00	 1.1645521e-01	  1.0642214e+00 	 2.1455908e-01


.. parsed-literal::

     123	 1.5628723e+00	 1.2061002e-01	 1.6171563e+00	 1.1643603e-01	  1.0623050e+00 	 2.0637107e-01
     124	 1.5647128e+00	 1.2051115e-01	 1.6189662e+00	 1.1644395e-01	  1.0615901e+00 	 2.0578790e-01


.. parsed-literal::

     125	 1.5658611e+00	 1.2043498e-01	 1.6201521e+00	 1.1637910e-01	  1.0550125e+00 	 1.8002987e-01
     126	 1.5667300e+00	 1.2036297e-01	 1.6210677e+00	 1.1624379e-01	  1.0487425e+00 	 1.8207645e-01


.. parsed-literal::

     127	 1.5684825e+00	 1.2016173e-01	 1.6229463e+00	 1.1601962e-01	  1.0372908e+00 	 2.0378351e-01


.. parsed-literal::

     128	 1.5696968e+00	 1.1999577e-01	 1.6242388e+00	 1.1578634e-01	  1.0317616e+00 	 2.9292107e-01
     129	 1.5711403e+00	 1.1985827e-01	 1.6257427e+00	 1.1572843e-01	  1.0291226e+00 	 1.9792604e-01


.. parsed-literal::

     130	 1.5726800e+00	 1.1971689e-01	 1.6273225e+00	 1.1571419e-01	  1.0368230e+00 	 2.0848680e-01


.. parsed-literal::

     131	 1.5746607e+00	 1.1955745e-01	 1.6293556e+00	 1.1591665e-01	  1.0462554e+00 	 2.1548414e-01


.. parsed-literal::

     132	 1.5754395e+00	 1.1938064e-01	 1.6302857e+00	 1.1616114e-01	  1.0721376e+00 	 2.1496487e-01
     133	 1.5776604e+00	 1.1935542e-01	 1.6323768e+00	 1.1627767e-01	  1.0676702e+00 	 1.8226910e-01


.. parsed-literal::

     134	 1.5784223e+00	 1.1937221e-01	 1.6331090e+00	 1.1635631e-01	  1.0621876e+00 	 2.0103335e-01


.. parsed-literal::

     135	 1.5795421e+00	 1.1928903e-01	 1.6343242e+00	 1.1651261e-01	  1.0555568e+00 	 2.1232438e-01


.. parsed-literal::

     136	 1.5810011e+00	 1.1916842e-01	 1.6359428e+00	 1.1676192e-01	  1.0473903e+00 	 2.1780992e-01


.. parsed-literal::

     137	 1.5822677e+00	 1.1908212e-01	 1.6372598e+00	 1.1679240e-01	  1.0425859e+00 	 2.1571255e-01
     138	 1.5834687e+00	 1.1896897e-01	 1.6385198e+00	 1.1676861e-01	  1.0448454e+00 	 1.9769669e-01


.. parsed-literal::

     139	 1.5848310e+00	 1.1885897e-01	 1.6398876e+00	 1.1667200e-01	  1.0479424e+00 	 2.0022821e-01


.. parsed-literal::

     140	 1.5863560e+00	 1.1873997e-01	 1.6414116e+00	 1.1657996e-01	  1.0551440e+00 	 2.1513557e-01


.. parsed-literal::

     141	 1.5873647e+00	 1.1867858e-01	 1.6423519e+00	 1.1654509e-01	  1.0575223e+00 	 2.1595955e-01


.. parsed-literal::

     142	 1.5884450e+00	 1.1865403e-01	 1.6433695e+00	 1.1647000e-01	  1.0561325e+00 	 2.0488191e-01


.. parsed-literal::

     143	 1.5897398e+00	 1.1860901e-01	 1.6446617e+00	 1.1642230e-01	  1.0508479e+00 	 2.1438503e-01
     144	 1.5908355e+00	 1.1857510e-01	 1.6458015e+00	 1.1633054e-01	  1.0443141e+00 	 1.9175386e-01


.. parsed-literal::

     145	 1.5918775e+00	 1.1855284e-01	 1.6468487e+00	 1.1631716e-01	  1.0414323e+00 	 1.9829917e-01
     146	 1.5927127e+00	 1.1850686e-01	 1.6477339e+00	 1.1633824e-01	  1.0376113e+00 	 1.8624330e-01


.. parsed-literal::

     147	 1.5939051e+00	 1.1840759e-01	 1.6490034e+00	 1.1636239e-01	  1.0279464e+00 	 2.1534014e-01
     148	 1.5947828e+00	 1.1830030e-01	 1.6500634e+00	 1.1642833e-01	  1.0030626e+00 	 1.8141508e-01


.. parsed-literal::

     149	 1.5959318e+00	 1.1824021e-01	 1.6511428e+00	 1.1639439e-01	  1.0049116e+00 	 1.8402004e-01
     150	 1.5963781e+00	 1.1821876e-01	 1.6515582e+00	 1.1634376e-01	  1.0042043e+00 	 1.8961358e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.06 s, total: 2min 3s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff2e2cdd790>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.84 s, sys: 33 ms, total: 1.87 s
    Wall time: 593 ms


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




.. image:: 06_GPz_files/06_GPz_16_1.png


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




.. image:: 06_GPz_files/06_GPz_19_1.png

