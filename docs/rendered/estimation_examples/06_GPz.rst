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
       1	-3.3928608e-01	 3.1915085e-01	-3.2963128e-01	 3.2619891e-01	[-3.4313790e-01]	 4.6241713e-01


.. parsed-literal::

       2	-2.6961253e-01	 3.0918654e-01	-2.4627063e-01	 3.1491605e-01	[-2.6414078e-01]	 2.3683596e-01


.. parsed-literal::

       3	-2.2470206e-01	 2.8802111e-01	-1.8248003e-01	 2.9217138e-01	[-1.9913571e-01]	 2.7684212e-01
       4	-1.8685201e-01	 2.6468454e-01	-1.4493310e-01	 2.6889461e-01	[-1.7147261e-01]	 1.9726729e-01


.. parsed-literal::

       5	-9.6363984e-02	 2.5543604e-01	-6.2721085e-02	 2.5903902e-01	[-7.9819241e-02]	 1.9661164e-01
       6	-6.6432633e-02	 2.5098593e-01	-3.7014138e-02	 2.5440456e-01	[-5.0554537e-02]	 2.0178270e-01


.. parsed-literal::

       7	-4.8153693e-02	 2.4790863e-01	-2.4492936e-02	 2.5110848e-01	[-3.7413427e-02]	 1.8237042e-01
       8	-3.6659160e-02	 2.4603661e-01	-1.6635375e-02	 2.4931810e-01	[-3.0155193e-02]	 1.8691587e-01


.. parsed-literal::

       9	-2.4814373e-02	 2.4389018e-01	-7.4540020e-03	 2.4773374e-01	[-2.3135399e-02]	 2.0968986e-01
      10	-1.4778714e-02	 2.4179398e-01	 5.4605180e-04	 2.4594156e-01	[-1.8074508e-02]	 1.9524527e-01


.. parsed-literal::

      11	-8.5226690e-03	 2.4090486e-01	 5.4508423e-03	 2.4505404e-01	[-1.0949219e-02]	 2.0572805e-01


.. parsed-literal::

      12	-5.4483445e-03	 2.4042396e-01	 8.4269574e-03	 2.4435765e-01	[-7.8307265e-03]	 2.0949030e-01


.. parsed-literal::

      13	-1.7640131e-03	 2.3967162e-01	 1.2203082e-02	 2.4343214e-01	[-3.6131847e-03]	 2.0620275e-01


.. parsed-literal::

      14	 3.0297332e-03	 2.3856060e-01	 1.7905639e-02	 2.4149820e-01	[ 5.3284697e-03]	 2.1615982e-01


.. parsed-literal::

      15	 1.7770648e-01	 2.2356841e-01	 2.0144221e-01	 2.2210485e-01	[ 2.0257965e-01]	 3.2698250e-01


.. parsed-literal::

      16	 2.2030393e-01	 2.2183471e-01	 2.4638281e-01	 2.2115374e-01	[ 2.4446756e-01]	 2.0372176e-01


.. parsed-literal::

      17	 2.9080010e-01	 2.2067314e-01	 3.1891910e-01	 2.1905317e-01	[ 3.2020624e-01]	 2.1474004e-01


.. parsed-literal::

      18	 3.4323991e-01	 2.1543114e-01	 3.7467140e-01	 2.1461973e-01	[ 3.6953787e-01]	 2.0912504e-01
      19	 4.0401396e-01	 2.1173691e-01	 4.3702265e-01	 2.0860041e-01	[ 4.3655259e-01]	 1.7854714e-01


.. parsed-literal::

      20	 4.9805244e-01	 2.0976477e-01	 5.3337255e-01	 2.0725891e-01	[ 5.2776172e-01]	 1.7573500e-01
      21	 5.8710145e-01	 2.0485965e-01	 6.2431944e-01	 2.0279207e-01	[ 6.2051972e-01]	 1.8218756e-01


.. parsed-literal::

      22	 6.3929530e-01	 1.9631032e-01	 6.7880965e-01	 1.9380012e-01	[ 6.6788780e-01]	 2.0352697e-01
      23	 6.8366797e-01	 1.8957114e-01	 7.2061654e-01	 1.8757004e-01	[ 7.0510147e-01]	 1.8859434e-01


.. parsed-literal::

      24	 7.3312806e-01	 1.8861512e-01	 7.7065504e-01	 1.8644931e-01	[ 7.5367932e-01]	 2.0147777e-01


.. parsed-literal::

      25	 7.6985705e-01	 1.9119646e-01	 8.0752752e-01	 1.8761207e-01	[ 7.9683961e-01]	 3.1743479e-01
      26	 7.9744742e-01	 1.8862141e-01	 8.3598943e-01	 1.8606289e-01	[ 8.2352177e-01]	 1.7226505e-01


.. parsed-literal::

      27	 8.2287422e-01	 1.8118599e-01	 8.6188477e-01	 1.7889557e-01	[ 8.4595388e-01]	 2.0942140e-01


.. parsed-literal::

      28	 8.4669313e-01	 1.7860177e-01	 8.8571296e-01	 1.7670725e-01	[ 8.7316163e-01]	 2.0886040e-01


.. parsed-literal::

      29	 8.7412964e-01	 1.7609751e-01	 9.1367444e-01	 1.7490416e-01	[ 9.0405664e-01]	 2.0925069e-01
      30	 8.9670175e-01	 1.7376692e-01	 9.3831243e-01	 1.7395753e-01	[ 9.2722669e-01]	 1.8278623e-01


.. parsed-literal::

      31	 9.1913844e-01	 1.7232724e-01	 9.6063669e-01	 1.7230680e-01	[ 9.4358995e-01]	 2.0409036e-01
      32	 9.2875667e-01	 1.7138148e-01	 9.7017123e-01	 1.7151798e-01	[ 9.5133281e-01]	 1.9868207e-01


.. parsed-literal::

      33	 9.4559788e-01	 1.7058472e-01	 9.8740222e-01	 1.7144255e-01	[ 9.6575327e-01]	 2.0810628e-01


.. parsed-literal::

      34	 9.6786074e-01	 1.7054242e-01	 1.0108579e+00	 1.7335140e-01	[ 9.7887386e-01]	 2.1362448e-01
      35	 9.8849097e-01	 1.6981416e-01	 1.0319893e+00	 1.7304487e-01	[ 9.9798578e-01]	 1.9700122e-01


.. parsed-literal::

      36	 1.0031634e+00	 1.6896848e-01	 1.0469517e+00	 1.7298519e-01	[ 1.0102992e+00]	 2.0748401e-01
      37	 1.0223829e+00	 1.6727971e-01	 1.0676100e+00	 1.7296491e-01	[ 1.0236136e+00]	 1.7874956e-01


.. parsed-literal::

      38	 1.0351563e+00	 1.6644598e-01	 1.0813737e+00	 1.7345750e-01	[ 1.0297330e+00]	 2.1699595e-01


.. parsed-literal::

      39	 1.0466967e+00	 1.6533136e-01	 1.0928167e+00	 1.7266855e-01	[ 1.0379961e+00]	 2.0838952e-01


.. parsed-literal::

      40	 1.0566752e+00	 1.6390889e-01	 1.1029133e+00	 1.7111592e-01	[ 1.0445477e+00]	 2.0606112e-01
      41	 1.0689264e+00	 1.6171737e-01	 1.1154166e+00	 1.6866917e-01	[ 1.0554832e+00]	 1.9673538e-01


.. parsed-literal::

      42	 1.0801870e+00	 1.5843455e-01	 1.1270947e+00	 1.6494235e-01	  1.0532171e+00 	 2.1353507e-01
      43	 1.0929717e+00	 1.5692308e-01	 1.1394018e+00	 1.6380038e-01	[ 1.0637910e+00]	 1.6289902e-01


.. parsed-literal::

      44	 1.1045563e+00	 1.5577376e-01	 1.1510196e+00	 1.6307479e-01	[ 1.0690021e+00]	 2.0215011e-01


.. parsed-literal::

      45	 1.1197371e+00	 1.5383351e-01	 1.1661911e+00	 1.6222818e-01	[ 1.0762655e+00]	 2.0401025e-01
      46	 1.1384678e+00	 1.5266928e-01	 1.1857236e+00	 1.6356816e-01	[ 1.0839791e+00]	 1.8659878e-01


.. parsed-literal::

      47	 1.1539313e+00	 1.5201048e-01	 1.2010461e+00	 1.6245938e-01	[ 1.1031012e+00]	 1.7035890e-01


.. parsed-literal::

      48	 1.1622085e+00	 1.5193856e-01	 1.2094007e+00	 1.6212678e-01	[ 1.1101053e+00]	 2.0468354e-01
      49	 1.1785980e+00	 1.5189654e-01	 1.2268321e+00	 1.6025981e-01	[ 1.1176937e+00]	 1.9809890e-01


.. parsed-literal::

      50	 1.1858546e+00	 1.5238164e-01	 1.2340905e+00	 1.6290240e-01	[ 1.1269957e+00]	 2.1240377e-01
      51	 1.1949079e+00	 1.5068205e-01	 1.2428540e+00	 1.6067088e-01	[ 1.1344202e+00]	 1.9842267e-01


.. parsed-literal::

      52	 1.2030269e+00	 1.4960910e-01	 1.2514566e+00	 1.5953416e-01	[ 1.1368715e+00]	 2.0908666e-01
      53	 1.2098592e+00	 1.4871440e-01	 1.2586648e+00	 1.5786974e-01	[ 1.1404798e+00]	 1.8472767e-01


.. parsed-literal::

      54	 1.2220022e+00	 1.4728037e-01	 1.2712047e+00	 1.5653127e-01	[ 1.1533686e+00]	 2.0756078e-01
      55	 1.2319388e+00	 1.4711804e-01	 1.2814518e+00	 1.5519954e-01	[ 1.1587908e+00]	 1.7766356e-01


.. parsed-literal::

      56	 1.2422743e+00	 1.4605493e-01	 1.2916425e+00	 1.5440373e-01	[ 1.1689844e+00]	 1.7678070e-01


.. parsed-literal::

      57	 1.2511424e+00	 1.4546386e-01	 1.3006294e+00	 1.5402096e-01	[ 1.1757424e+00]	 2.0707464e-01


.. parsed-literal::

      58	 1.2592573e+00	 1.4506630e-01	 1.3087890e+00	 1.5393644e-01	[ 1.1818116e+00]	 2.0610952e-01


.. parsed-literal::

      59	 1.2693116e+00	 1.4393668e-01	 1.3192554e+00	 1.5198017e-01	[ 1.1911380e+00]	 2.0568991e-01


.. parsed-literal::

      60	 1.2816989e+00	 1.4367925e-01	 1.3317402e+00	 1.5285956e-01	[ 1.2007098e+00]	 2.0856166e-01
      61	 1.2879299e+00	 1.4333395e-01	 1.3379978e+00	 1.5282431e-01	[ 1.2053318e+00]	 1.9396138e-01


.. parsed-literal::

      62	 1.2972963e+00	 1.4254051e-01	 1.3480358e+00	 1.5226928e-01	[ 1.2122449e+00]	 1.9854689e-01


.. parsed-literal::

      63	 1.3049059e+00	 1.4209280e-01	 1.3555692e+00	 1.5229778e-01	[ 1.2198084e+00]	 2.0634937e-01


.. parsed-literal::

      64	 1.3097804e+00	 1.4179928e-01	 1.3602793e+00	 1.5195109e-01	[ 1.2261744e+00]	 2.0231724e-01
      65	 1.3206804e+00	 1.4099892e-01	 1.3712537e+00	 1.5091838e-01	[ 1.2386840e+00]	 1.7620087e-01


.. parsed-literal::

      66	 1.3282865e+00	 1.4055973e-01	 1.3788997e+00	 1.5105144e-01	[ 1.2472872e+00]	 2.0753145e-01


.. parsed-literal::

      67	 1.3349722e+00	 1.4035630e-01	 1.3856926e+00	 1.5102671e-01	[ 1.2540991e+00]	 2.1270800e-01
      68	 1.3414005e+00	 1.4010335e-01	 1.3922726e+00	 1.5159208e-01	[ 1.2588564e+00]	 1.7431521e-01


.. parsed-literal::

      69	 1.3494597e+00	 1.4033091e-01	 1.4009243e+00	 1.5279864e-01	  1.2547994e+00 	 1.7705250e-01
      70	 1.3554938e+00	 1.4031175e-01	 1.4070583e+00	 1.5318870e-01	[ 1.2615729e+00]	 1.7528462e-01


.. parsed-literal::

      71	 1.3623988e+00	 1.4007806e-01	 1.4139499e+00	 1.5226162e-01	[ 1.2693032e+00]	 2.0422459e-01
      72	 1.3708006e+00	 1.3974919e-01	 1.4225254e+00	 1.5059116e-01	[ 1.2793032e+00]	 1.9062686e-01


.. parsed-literal::

      73	 1.3765301e+00	 1.3896203e-01	 1.4282423e+00	 1.4787118e-01	[ 1.2812848e+00]	 2.1346378e-01


.. parsed-literal::

      74	 1.3828493e+00	 1.3867990e-01	 1.4345649e+00	 1.4732457e-01	[ 1.2882053e+00]	 2.1600819e-01


.. parsed-literal::

      75	 1.3871594e+00	 1.3827803e-01	 1.4390837e+00	 1.4683932e-01	[ 1.2885029e+00]	 2.2059822e-01


.. parsed-literal::

      76	 1.3927579e+00	 1.3758468e-01	 1.4448472e+00	 1.4586544e-01	  1.2865307e+00 	 2.1211004e-01


.. parsed-literal::

      77	 1.3979903e+00	 1.3674125e-01	 1.4504855e+00	 1.4493511e-01	[ 1.2898108e+00]	 2.1388578e-01
      78	 1.4036812e+00	 1.3622936e-01	 1.4560071e+00	 1.4419020e-01	  1.2857918e+00 	 2.0561910e-01


.. parsed-literal::

      79	 1.4062790e+00	 1.3613932e-01	 1.4584587e+00	 1.4402816e-01	[ 1.2899265e+00]	 2.1478724e-01
      80	 1.4104602e+00	 1.3577213e-01	 1.4626891e+00	 1.4329956e-01	[ 1.2953340e+00]	 1.8937826e-01


.. parsed-literal::

      81	 1.4133411e+00	 1.3522450e-01	 1.4656376e+00	 1.4318332e-01	[ 1.2982889e+00]	 1.8561745e-01


.. parsed-literal::

      82	 1.4178901e+00	 1.3481678e-01	 1.4702407e+00	 1.4217121e-01	[ 1.3003029e+00]	 2.0122123e-01
      83	 1.4222760e+00	 1.3433422e-01	 1.4748339e+00	 1.4129686e-01	  1.2979731e+00 	 1.9396400e-01


.. parsed-literal::

      84	 1.4253121e+00	 1.3394863e-01	 1.4780134e+00	 1.4092835e-01	  1.2954642e+00 	 2.1285677e-01
      85	 1.4327508e+00	 1.3338952e-01	 1.4857883e+00	 1.4000427e-01	  1.2866735e+00 	 1.6829348e-01


.. parsed-literal::

      86	 1.4360341e+00	 1.3315974e-01	 1.4891511e+00	 1.3960814e-01	  1.2856686e+00 	 3.2145858e-01
      87	 1.4402668e+00	 1.3301402e-01	 1.4933663e+00	 1.3910723e-01	  1.2839747e+00 	 2.0079803e-01


.. parsed-literal::

      88	 1.4446222e+00	 1.3292556e-01	 1.4976609e+00	 1.3821214e-01	  1.2813557e+00 	 1.7708230e-01


.. parsed-literal::

      89	 1.4478085e+00	 1.3274447e-01	 1.5008499e+00	 1.3675461e-01	  1.2873765e+00 	 2.1453571e-01


.. parsed-literal::

      90	 1.4512968e+00	 1.3244007e-01	 1.5042950e+00	 1.3655803e-01	  1.2871085e+00 	 2.0945978e-01


.. parsed-literal::

      91	 1.4540233e+00	 1.3209807e-01	 1.5071540e+00	 1.3589415e-01	  1.2847733e+00 	 2.0936489e-01


.. parsed-literal::

      92	 1.4565284e+00	 1.3176219e-01	 1.5097734e+00	 1.3523403e-01	  1.2815565e+00 	 2.0611429e-01


.. parsed-literal::

      93	 1.4592965e+00	 1.3152137e-01	 1.5127003e+00	 1.3416557e-01	  1.2794188e+00 	 2.1073484e-01
      94	 1.4617828e+00	 1.3146388e-01	 1.5151594e+00	 1.3354590e-01	  1.2774042e+00 	 1.7071581e-01


.. parsed-literal::

      95	 1.4641771e+00	 1.3148097e-01	 1.5174306e+00	 1.3325705e-01	  1.2812943e+00 	 1.9820833e-01
      96	 1.4662160e+00	 1.3164253e-01	 1.5194169e+00	 1.3249272e-01	  1.2815274e+00 	 1.7917657e-01


.. parsed-literal::

      97	 1.4689121e+00	 1.3172659e-01	 1.5221058e+00	 1.3184208e-01	  1.2842177e+00 	 2.0973253e-01


.. parsed-literal::

      98	 1.4716390e+00	 1.3171667e-01	 1.5250134e+00	 1.3101199e-01	  1.2817817e+00 	 2.2009897e-01


.. parsed-literal::

      99	 1.4741165e+00	 1.3141085e-01	 1.5275451e+00	 1.3088499e-01	  1.2817586e+00 	 2.0682120e-01


.. parsed-literal::

     100	 1.4762655e+00	 1.3106293e-01	 1.5297558e+00	 1.3097700e-01	  1.2780283e+00 	 2.1020341e-01
     101	 1.4783660e+00	 1.3059557e-01	 1.5319618e+00	 1.3103226e-01	  1.2656173e+00 	 2.0296836e-01


.. parsed-literal::

     102	 1.4806385e+00	 1.3049629e-01	 1.5341607e+00	 1.3117065e-01	  1.2618186e+00 	 1.7960024e-01


.. parsed-literal::

     103	 1.4820237e+00	 1.3041064e-01	 1.5354905e+00	 1.3092513e-01	  1.2611193e+00 	 2.0598388e-01
     104	 1.4845963e+00	 1.3011991e-01	 1.5380623e+00	 1.3064664e-01	  1.2587935e+00 	 2.0230603e-01


.. parsed-literal::

     105	 1.4853550e+00	 1.2971967e-01	 1.5389902e+00	 1.2989567e-01	  1.2620151e+00 	 2.1376395e-01
     106	 1.4878859e+00	 1.2959991e-01	 1.5414970e+00	 1.3018567e-01	  1.2621104e+00 	 1.7814064e-01


.. parsed-literal::

     107	 1.4890272e+00	 1.2944095e-01	 1.5427229e+00	 1.3021584e-01	  1.2623801e+00 	 2.0653367e-01
     108	 1.4909162e+00	 1.2910601e-01	 1.5447375e+00	 1.3016935e-01	  1.2662542e+00 	 1.7783761e-01


.. parsed-literal::

     109	 1.4914140e+00	 1.2887792e-01	 1.5454631e+00	 1.2944648e-01	  1.2684002e+00 	 2.1204233e-01
     110	 1.4940440e+00	 1.2855919e-01	 1.5479766e+00	 1.2952223e-01	  1.2754295e+00 	 1.8829727e-01


.. parsed-literal::

     111	 1.4952292e+00	 1.2844080e-01	 1.5490800e+00	 1.2942580e-01	  1.2784647e+00 	 2.0414996e-01
     112	 1.4969275e+00	 1.2819878e-01	 1.5507000e+00	 1.2918746e-01	  1.2832398e+00 	 1.7954063e-01


.. parsed-literal::

     113	 1.4993318e+00	 1.2784003e-01	 1.5530818e+00	 1.2897291e-01	  1.2895636e+00 	 1.7596555e-01
     114	 1.5013426e+00	 1.2725039e-01	 1.5551522e+00	 1.2888605e-01	  1.2956114e+00 	 1.7378259e-01


.. parsed-literal::

     115	 1.5037053e+00	 1.2729387e-01	 1.5575254e+00	 1.2890381e-01	  1.2957087e+00 	 2.1222162e-01
     116	 1.5048684e+00	 1.2722554e-01	 1.5587811e+00	 1.2896367e-01	  1.2957124e+00 	 1.8199968e-01


.. parsed-literal::

     117	 1.5066471e+00	 1.2701398e-01	 1.5606458e+00	 1.2890194e-01	  1.2950699e+00 	 2.1220469e-01
     118	 1.5079044e+00	 1.2658063e-01	 1.5620843e+00	 1.2854952e-01	[ 1.3032347e+00]	 1.7339277e-01


.. parsed-literal::

     119	 1.5103035e+00	 1.2631284e-01	 1.5644083e+00	 1.2863243e-01	  1.3023818e+00 	 1.8504739e-01
     120	 1.5112630e+00	 1.2622786e-01	 1.5652855e+00	 1.2857533e-01	[ 1.3042995e+00]	 2.0165801e-01


.. parsed-literal::

     121	 1.5131272e+00	 1.2588126e-01	 1.5670773e+00	 1.2844414e-01	[ 1.3094829e+00]	 2.0450687e-01
     122	 1.5144493e+00	 1.2563419e-01	 1.5684155e+00	 1.2882838e-01	[ 1.3178565e+00]	 1.5954995e-01


.. parsed-literal::

     123	 1.5166694e+00	 1.2514886e-01	 1.5706665e+00	 1.2844009e-01	[ 1.3212497e+00]	 2.1579838e-01
     124	 1.5177827e+00	 1.2514024e-01	 1.5718190e+00	 1.2839476e-01	[ 1.3214744e+00]	 1.8578100e-01


.. parsed-literal::

     125	 1.5192707e+00	 1.2504080e-01	 1.5734090e+00	 1.2840907e-01	[ 1.3240345e+00]	 2.0792890e-01


.. parsed-literal::

     126	 1.5201190e+00	 1.2492801e-01	 1.5744173e+00	 1.2823971e-01	[ 1.3243788e+00]	 2.0940375e-01
     127	 1.5216844e+00	 1.2477468e-01	 1.5759701e+00	 1.2821660e-01	[ 1.3281037e+00]	 1.7749715e-01


.. parsed-literal::

     128	 1.5230106e+00	 1.2453289e-01	 1.5772859e+00	 1.2816156e-01	[ 1.3307362e+00]	 2.1709967e-01


.. parsed-literal::

     129	 1.5242247e+00	 1.2430052e-01	 1.5785025e+00	 1.2812390e-01	[ 1.3312936e+00]	 2.0992780e-01


.. parsed-literal::

     130	 1.5252629e+00	 1.2402864e-01	 1.5795504e+00	 1.2820757e-01	  1.3292748e+00 	 3.2490087e-01


.. parsed-literal::

     131	 1.5267225e+00	 1.2380965e-01	 1.5810281e+00	 1.2827434e-01	  1.3268190e+00 	 2.0884395e-01


.. parsed-literal::

     132	 1.5279265e+00	 1.2368577e-01	 1.5822578e+00	 1.2837346e-01	  1.3240071e+00 	 2.1055508e-01


.. parsed-literal::

     133	 1.5291632e+00	 1.2357098e-01	 1.5835456e+00	 1.2856985e-01	  1.3202445e+00 	 2.1458673e-01
     134	 1.5301666e+00	 1.2347144e-01	 1.5846615e+00	 1.2851932e-01	  1.3173517e+00 	 1.7160726e-01


.. parsed-literal::

     135	 1.5314153e+00	 1.2343981e-01	 1.5859133e+00	 1.2865958e-01	  1.3175265e+00 	 2.1556354e-01
     136	 1.5326837e+00	 1.2335722e-01	 1.5872446e+00	 1.2872345e-01	  1.3181505e+00 	 1.9161940e-01


.. parsed-literal::

     137	 1.5337864e+00	 1.2330552e-01	 1.5884047e+00	 1.2882882e-01	  1.3176178e+00 	 2.0379949e-01
     138	 1.5350196e+00	 1.2312346e-01	 1.5897663e+00	 1.2900907e-01	  1.3135681e+00 	 2.0510483e-01


.. parsed-literal::

     139	 1.5359447e+00	 1.2304979e-01	 1.5906828e+00	 1.2910398e-01	  1.3104284e+00 	 1.9420147e-01
     140	 1.5372117e+00	 1.2291768e-01	 1.5919756e+00	 1.2939438e-01	  1.3022696e+00 	 1.9859815e-01


.. parsed-literal::

     141	 1.5387241e+00	 1.2274061e-01	 1.5935239e+00	 1.2954911e-01	  1.2949825e+00 	 1.9218302e-01


.. parsed-literal::

     142	 1.5395069e+00	 1.2250581e-01	 1.5944787e+00	 1.3034725e-01	  1.2695802e+00 	 2.0812488e-01


.. parsed-literal::

     143	 1.5412929e+00	 1.2243837e-01	 1.5961534e+00	 1.2987664e-01	  1.2804927e+00 	 2.1017051e-01


.. parsed-literal::

     144	 1.5418314e+00	 1.2238406e-01	 1.5967024e+00	 1.2974739e-01	  1.2826103e+00 	 2.1388030e-01
     145	 1.5428399e+00	 1.2223372e-01	 1.5977998e+00	 1.2964350e-01	  1.2800926e+00 	 2.0617127e-01


.. parsed-literal::

     146	 1.5439764e+00	 1.2207296e-01	 1.5989927e+00	 1.2959957e-01	  1.2777454e+00 	 2.1360302e-01
     147	 1.5449162e+00	 1.2197306e-01	 1.5999748e+00	 1.2981398e-01	  1.2721844e+00 	 2.0008731e-01


.. parsed-literal::

     148	 1.5468138e+00	 1.2180755e-01	 1.6019910e+00	 1.3048360e-01	  1.2612358e+00 	 2.1711206e-01


.. parsed-literal::

     149	 1.5481239e+00	 1.2171347e-01	 1.6033847e+00	 1.3111487e-01	  1.2539446e+00 	 2.0944524e-01


.. parsed-literal::

     150	 1.5490232e+00	 1.2168075e-01	 1.6043202e+00	 1.3157207e-01	  1.2527694e+00 	 2.1621943e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.07 s, total: 2min 3s
    Wall time: 31 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd53c664e50>



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
    CPU times: user 1.74 s, sys: 62.9 ms, total: 1.8 s
    Wall time: 568 ms


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

