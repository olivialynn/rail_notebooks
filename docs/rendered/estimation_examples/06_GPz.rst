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
       1	-3.4472523e-01	 3.2033121e-01	-3.3501014e-01	 3.2096826e-01	[-3.3644153e-01]	 4.6262169e-01


.. parsed-literal::

       2	-2.6961606e-01	 3.0857117e-01	-2.4457624e-01	 3.0938021e-01	[-2.4736041e-01]	 2.2868419e-01


.. parsed-literal::

       3	-2.2580107e-01	 2.8887271e-01	-1.8390256e-01	 2.8868534e-01	[-1.8590955e-01]	 2.7858377e-01
       4	-1.9517244e-01	 2.6488791e-01	-1.5523118e-01	 2.6344805e-01	[-1.4972545e-01]	 1.7939639e-01


.. parsed-literal::

       5	-1.0411122e-01	 2.5690320e-01	-6.7522301e-02	 2.5505939e-01	[-6.0831276e-02]	 2.0358038e-01


.. parsed-literal::

       6	-7.3012486e-02	 2.5205643e-01	-4.1080599e-02	 2.4967081e-01	[-3.3087889e-02]	 2.0770597e-01
       7	-5.3458021e-02	 2.4894464e-01	-2.8653230e-02	 2.4664938e-01	[-2.0297816e-02]	 2.0617747e-01


.. parsed-literal::

       8	-4.2999758e-02	 2.4731732e-01	-2.1856856e-02	 2.4510196e-01	[-1.3655250e-02]	 2.0911360e-01
       9	-2.9922812e-02	 2.4498161e-01	-1.1953150e-02	 2.4281577e-01	[-3.9923878e-03]	 1.9411826e-01


.. parsed-literal::

      10	-1.7372241e-02	 2.4244585e-01	-1.6656618e-03	 2.4053224e-01	[ 5.6607514e-03]	 2.1048307e-01


.. parsed-literal::

      11	-1.6560294e-02	 2.4200368e-01	-2.2118074e-03	 2.4078403e-01	  1.4461196e-03 	 2.0507145e-01
      12	-9.1915493e-03	 2.4122352e-01	 4.8945354e-03	 2.4019986e-01	[ 8.1982156e-03]	 1.8178844e-01


.. parsed-literal::

      13	-6.7586505e-03	 2.4070041e-01	 7.2369520e-03	 2.3991856e-01	[ 9.6571227e-03]	 2.1104646e-01


.. parsed-literal::

      14	-3.7105971e-03	 2.4003142e-01	 1.0449934e-02	 2.3967956e-01	[ 1.1094709e-02]	 2.0850921e-01


.. parsed-literal::

      15	 5.3706675e-03	 2.3827088e-01	 2.0604250e-02	 2.3863533e-01	[ 1.8407152e-02]	 2.1250081e-01


.. parsed-literal::

      16	 1.0034661e-01	 2.2628781e-01	 1.2184410e-01	 2.2457916e-01	[ 1.2821502e-01]	 2.0816994e-01


.. parsed-literal::

      17	 1.2317925e-01	 2.2437021e-01	 1.4543734e-01	 2.2309405e-01	[ 1.5044486e-01]	 3.1995487e-01
      18	 1.7491277e-01	 2.2073002e-01	 1.9773597e-01	 2.2051852e-01	[ 2.0051045e-01]	 1.9473958e-01


.. parsed-literal::

      19	 3.0358061e-01	 2.1802230e-01	 3.3264391e-01	 2.1732125e-01	[ 3.3316249e-01]	 2.1044135e-01


.. parsed-literal::

      20	 3.4709284e-01	 2.1706434e-01	 3.7667461e-01	 2.1871603e-01	[ 3.6433275e-01]	 2.1223283e-01


.. parsed-literal::

      21	 4.1134059e-01	 2.1418756e-01	 4.4375619e-01	 2.1793718e-01	[ 4.2566355e-01]	 2.0974398e-01


.. parsed-literal::

      22	 4.8943839e-01	 2.1264961e-01	 5.2385093e-01	 2.1734579e-01	[ 5.1226691e-01]	 2.2161269e-01


.. parsed-literal::

      23	 5.4260134e-01	 2.1057358e-01	 5.7948108e-01	 2.1432262e-01	[ 5.6989061e-01]	 2.1038032e-01


.. parsed-literal::

      24	 5.7890026e-01	 2.0626096e-01	 6.1800951e-01	 2.1102829e-01	[ 6.1190921e-01]	 2.0851636e-01


.. parsed-literal::

      25	 6.2058215e-01	 2.0298605e-01	 6.6027260e-01	 2.0767169e-01	[ 6.5933686e-01]	 2.1233916e-01


.. parsed-literal::

      26	 6.6791162e-01	 1.9989891e-01	 7.0695239e-01	 2.0501858e-01	[ 7.0402371e-01]	 2.0559597e-01


.. parsed-literal::

      27	 7.1427091e-01	 2.0050463e-01	 7.5200845e-01	 2.0605475e-01	[ 7.3065544e-01]	 2.1011114e-01


.. parsed-literal::

      28	 7.3772092e-01	 1.9997732e-01	 7.7386011e-01	 2.0329820e-01	[ 7.5519197e-01]	 2.0968819e-01


.. parsed-literal::

      29	 7.6527160e-01	 1.9798903e-01	 8.0358844e-01	 2.0308448e-01	[ 7.7551164e-01]	 2.1357512e-01
      30	 7.8487933e-01	 1.9667225e-01	 8.2435372e-01	 2.0272372e-01	[ 7.9770294e-01]	 1.8460608e-01


.. parsed-literal::

      31	 8.0850370e-01	 1.9714137e-01	 8.4836277e-01	 2.0350964e-01	[ 8.2215492e-01]	 1.8244529e-01


.. parsed-literal::

      32	 8.3944010e-01	 2.0366789e-01	 8.8128543e-01	 2.1157795e-01	[ 8.4134804e-01]	 2.0628452e-01


.. parsed-literal::

      33	 8.6119585e-01	 2.0271227e-01	 9.0235068e-01	 2.1159286e-01	[ 8.5898776e-01]	 2.1079135e-01
      34	 8.7863800e-01	 1.9909010e-01	 9.2011053e-01	 2.0842321e-01	[ 8.7908940e-01]	 1.7430067e-01


.. parsed-literal::

      35	 8.9588274e-01	 1.9716012e-01	 9.3783196e-01	 2.0683183e-01	[ 9.0065623e-01]	 2.1479845e-01
      36	 9.1535573e-01	 1.9622435e-01	 9.5823579e-01	 2.0640725e-01	[ 9.2550039e-01]	 2.0131254e-01


.. parsed-literal::

      37	 9.3388370e-01	 1.9751079e-01	 9.7823821e-01	 2.0748837e-01	[ 9.4331143e-01]	 2.1022916e-01


.. parsed-literal::

      38	 9.5544012e-01	 1.9676188e-01	 9.9924263e-01	 2.0632402e-01	[ 9.6738890e-01]	 2.0791769e-01


.. parsed-literal::

      39	 9.6465794e-01	 1.9554769e-01	 1.0084660e+00	 2.0512711e-01	[ 9.7778590e-01]	 2.1977139e-01


.. parsed-literal::

      40	 9.8249591e-01	 1.9406495e-01	 1.0269621e+00	 2.0314610e-01	[ 9.9804396e-01]	 2.1826077e-01
      41	 9.9512039e-01	 1.9353688e-01	 1.0401649e+00	 2.0193356e-01	[ 1.0004672e+00]	 1.8069029e-01


.. parsed-literal::

      42	 1.0071732e+00	 1.9170463e-01	 1.0524215e+00	 1.9958216e-01	[ 1.0128560e+00]	 2.0587993e-01


.. parsed-literal::

      43	 1.0162727e+00	 1.9127150e-01	 1.0617584e+00	 1.9894771e-01	[ 1.0222499e+00]	 2.0240593e-01
      44	 1.0284797e+00	 1.8993916e-01	 1.0742416e+00	 1.9765098e-01	[ 1.0333187e+00]	 1.8731809e-01


.. parsed-literal::

      45	 1.0415613e+00	 1.8991060e-01	 1.0881878e+00	 1.9797923e-01	[ 1.0454974e+00]	 1.7997360e-01


.. parsed-literal::

      46	 1.0537573e+00	 1.8786560e-01	 1.1003418e+00	 1.9571552e-01	[ 1.0541109e+00]	 2.1261787e-01


.. parsed-literal::

      47	 1.0612439e+00	 1.8676733e-01	 1.1079388e+00	 1.9463995e-01	[ 1.0616376e+00]	 2.0557070e-01
      48	 1.0722992e+00	 1.8405606e-01	 1.1193456e+00	 1.9193195e-01	[ 1.0744304e+00]	 1.9911170e-01


.. parsed-literal::

      49	 1.0790157e+00	 1.8208863e-01	 1.1266781e+00	 1.8968973e-01	[ 1.0855247e+00]	 2.0640278e-01
      50	 1.0875267e+00	 1.8112052e-01	 1.1349254e+00	 1.8874334e-01	[ 1.0937832e+00]	 1.8669772e-01


.. parsed-literal::

      51	 1.0944328e+00	 1.8021002e-01	 1.1418999e+00	 1.8770761e-01	[ 1.0982166e+00]	 1.8797994e-01


.. parsed-literal::

      52	 1.1016383e+00	 1.7902826e-01	 1.1492671e+00	 1.8637187e-01	[ 1.1026392e+00]	 2.1399784e-01


.. parsed-literal::

      53	 1.1110173e+00	 1.7555255e-01	 1.1587875e+00	 1.8176726e-01	[ 1.1088162e+00]	 2.1599364e-01


.. parsed-literal::

      54	 1.1195773e+00	 1.7473169e-01	 1.1676894e+00	 1.8027665e-01	[ 1.1168332e+00]	 2.1166754e-01


.. parsed-literal::

      55	 1.1266344e+00	 1.7371688e-01	 1.1748505e+00	 1.7901883e-01	[ 1.1250150e+00]	 2.0460296e-01


.. parsed-literal::

      56	 1.1361824e+00	 1.7237504e-01	 1.1847157e+00	 1.7696720e-01	[ 1.1357508e+00]	 2.1247435e-01


.. parsed-literal::

      57	 1.1469597e+00	 1.7079110e-01	 1.1958786e+00	 1.7434262e-01	[ 1.1467106e+00]	 2.1045399e-01


.. parsed-literal::

      58	 1.1578597e+00	 1.6892659e-01	 1.2073722e+00	 1.7173608e-01	[ 1.1543369e+00]	 2.1475005e-01
      59	 1.1663273e+00	 1.6844641e-01	 1.2158778e+00	 1.7144997e-01	[ 1.1614718e+00]	 1.8624091e-01


.. parsed-literal::

      60	 1.1792230e+00	 1.6628345e-01	 1.2293700e+00	 1.6949675e-01	[ 1.1709360e+00]	 1.9608521e-01
      61	 1.1864332e+00	 1.6514548e-01	 1.2368953e+00	 1.6834771e-01	[ 1.1742386e+00]	 1.7968822e-01


.. parsed-literal::

      62	 1.1934558e+00	 1.6407936e-01	 1.2436868e+00	 1.6674611e-01	[ 1.1828914e+00]	 2.1211314e-01


.. parsed-literal::

      63	 1.2013477e+00	 1.6213131e-01	 1.2516359e+00	 1.6378087e-01	[ 1.1902754e+00]	 2.0209217e-01


.. parsed-literal::

      64	 1.2076610e+00	 1.6101310e-01	 1.2581038e+00	 1.6247316e-01	[ 1.1983617e+00]	 2.1770787e-01
      65	 1.2137465e+00	 1.5889965e-01	 1.2645363e+00	 1.5914232e-01	[ 1.2034620e+00]	 1.7637038e-01


.. parsed-literal::

      66	 1.2209697e+00	 1.5861861e-01	 1.2716514e+00	 1.5927235e-01	[ 1.2100704e+00]	 1.8659258e-01


.. parsed-literal::

      67	 1.2256109e+00	 1.5912212e-01	 1.2760875e+00	 1.6028463e-01	[ 1.2156979e+00]	 2.1677113e-01
      68	 1.2336466e+00	 1.5800013e-01	 1.2842181e+00	 1.5880359e-01	[ 1.2204307e+00]	 2.0374155e-01


.. parsed-literal::

      69	 1.2410559e+00	 1.5761838e-01	 1.2916627e+00	 1.5740194e-01	[ 1.2229758e+00]	 2.0555043e-01


.. parsed-literal::

      70	 1.2477872e+00	 1.5647053e-01	 1.2982642e+00	 1.5600084e-01	[ 1.2301200e+00]	 2.1141863e-01


.. parsed-literal::

      71	 1.2553226e+00	 1.5500861e-01	 1.3058782e+00	 1.5407757e-01	[ 1.2375672e+00]	 2.0782924e-01


.. parsed-literal::

      72	 1.2635033e+00	 1.5429203e-01	 1.3142270e+00	 1.5338324e-01	[ 1.2464463e+00]	 2.1278310e-01


.. parsed-literal::

      73	 1.2688663e+00	 1.5322053e-01	 1.3200442e+00	 1.5126032e-01	[ 1.2574017e+00]	 2.0322275e-01


.. parsed-literal::

      74	 1.2761726e+00	 1.5396554e-01	 1.3271347e+00	 1.5292231e-01	[ 1.2645297e+00]	 2.0569992e-01


.. parsed-literal::

      75	 1.2809507e+00	 1.5403859e-01	 1.3320092e+00	 1.5296473e-01	[ 1.2674345e+00]	 2.0738506e-01


.. parsed-literal::

      76	 1.2865698e+00	 1.5392469e-01	 1.3377935e+00	 1.5265935e-01	[ 1.2717919e+00]	 2.0677590e-01


.. parsed-literal::

      77	 1.2953588e+00	 1.5305035e-01	 1.3469160e+00	 1.5136342e-01	[ 1.2748016e+00]	 2.1659064e-01
      78	 1.3026888e+00	 1.5176867e-01	 1.3547045e+00	 1.4988069e-01	  1.2731551e+00 	 1.9562078e-01


.. parsed-literal::

      79	 1.3086805e+00	 1.5153156e-01	 1.3604649e+00	 1.4984391e-01	[ 1.2804494e+00]	 2.0415378e-01


.. parsed-literal::

      80	 1.3145075e+00	 1.5096474e-01	 1.3663785e+00	 1.4948023e-01	[ 1.2827813e+00]	 2.1600604e-01


.. parsed-literal::

      81	 1.3218806e+00	 1.5030805e-01	 1.3739478e+00	 1.4915977e-01	[ 1.2842897e+00]	 2.1947098e-01


.. parsed-literal::

      82	 1.3280243e+00	 1.4833242e-01	 1.3805774e+00	 1.4698201e-01	  1.2784058e+00 	 2.1735239e-01


.. parsed-literal::

      83	 1.3373301e+00	 1.4880171e-01	 1.3897532e+00	 1.4813336e-01	[ 1.2867182e+00]	 2.0584655e-01


.. parsed-literal::

      84	 1.3422704e+00	 1.4827983e-01	 1.3947468e+00	 1.4750712e-01	[ 1.2885549e+00]	 2.0952702e-01


.. parsed-literal::

      85	 1.3490630e+00	 1.4745558e-01	 1.4017498e+00	 1.4629299e-01	[ 1.2897510e+00]	 2.0894623e-01
      86	 1.3525927e+00	 1.4686347e-01	 1.4055403e+00	 1.4509110e-01	  1.2809168e+00 	 1.9924641e-01


.. parsed-literal::

      87	 1.3586418e+00	 1.4680931e-01	 1.4113984e+00	 1.4526245e-01	  1.2894429e+00 	 1.9722438e-01
      88	 1.3620491e+00	 1.4672899e-01	 1.4147377e+00	 1.4542262e-01	[ 1.2917461e+00]	 1.7507052e-01


.. parsed-literal::

      89	 1.3658270e+00	 1.4664475e-01	 1.4184665e+00	 1.4563808e-01	[ 1.2932568e+00]	 2.0955634e-01


.. parsed-literal::

      90	 1.3731036e+00	 1.4608327e-01	 1.4258845e+00	 1.4543775e-01	  1.2924589e+00 	 2.2523403e-01


.. parsed-literal::

      91	 1.3769359e+00	 1.4675675e-01	 1.4296548e+00	 1.4679724e-01	[ 1.3007653e+00]	 3.2545376e-01


.. parsed-literal::

      92	 1.3818123e+00	 1.4626733e-01	 1.4345797e+00	 1.4632728e-01	[ 1.3038101e+00]	 2.1229386e-01


.. parsed-literal::

      93	 1.3864788e+00	 1.4579847e-01	 1.4393828e+00	 1.4578771e-01	[ 1.3102989e+00]	 2.1887016e-01
      94	 1.3904629e+00	 1.4568917e-01	 1.4434955e+00	 1.4581969e-01	[ 1.3140728e+00]	 1.9932747e-01


.. parsed-literal::

      95	 1.3941057e+00	 1.4551032e-01	 1.4471638e+00	 1.4561228e-01	[ 1.3176734e+00]	 1.9135261e-01


.. parsed-literal::

      96	 1.3975336e+00	 1.4541395e-01	 1.4506521e+00	 1.4575590e-01	[ 1.3210332e+00]	 2.1596742e-01


.. parsed-literal::

      97	 1.4003833e+00	 1.4508117e-01	 1.4535253e+00	 1.4542966e-01	  1.3179487e+00 	 2.1812344e-01


.. parsed-literal::

      98	 1.4040249e+00	 1.4463594e-01	 1.4572409e+00	 1.4509907e-01	  1.3154442e+00 	 2.1245050e-01


.. parsed-literal::

      99	 1.4091900e+00	 1.4319602e-01	 1.4627081e+00	 1.4407381e-01	  1.3091887e+00 	 2.0094252e-01


.. parsed-literal::

     100	 1.4124040e+00	 1.4264536e-01	 1.4661028e+00	 1.4360015e-01	  1.3148325e+00 	 2.1656013e-01


.. parsed-literal::

     101	 1.4153978e+00	 1.4244963e-01	 1.4689262e+00	 1.4339559e-01	  1.3199381e+00 	 2.1688581e-01


.. parsed-literal::

     102	 1.4176778e+00	 1.4201091e-01	 1.4711941e+00	 1.4307029e-01	[ 1.3212339e+00]	 2.0912814e-01
     103	 1.4212787e+00	 1.4127976e-01	 1.4748252e+00	 1.4258393e-01	  1.3206233e+00 	 2.0653462e-01


.. parsed-literal::

     104	 1.4261305e+00	 1.4029798e-01	 1.4797538e+00	 1.4226551e-01	  1.3154647e+00 	 2.1134973e-01


.. parsed-literal::

     105	 1.4304389e+00	 1.3970789e-01	 1.4842013e+00	 1.4223484e-01	  1.3094884e+00 	 2.1982670e-01


.. parsed-literal::

     106	 1.4331334e+00	 1.3971543e-01	 1.4868841e+00	 1.4236964e-01	  1.3091173e+00 	 2.1031642e-01
     107	 1.4358118e+00	 1.3941358e-01	 1.4896593e+00	 1.4231756e-01	  1.3075000e+00 	 1.8825436e-01


.. parsed-literal::

     108	 1.4376035e+00	 1.3935249e-01	 1.4915420e+00	 1.4215553e-01	  1.3070372e+00 	 1.9268203e-01


.. parsed-literal::

     109	 1.4398740e+00	 1.3908318e-01	 1.4937835e+00	 1.4187761e-01	  1.3098509e+00 	 2.1393418e-01


.. parsed-literal::

     110	 1.4427632e+00	 1.3835331e-01	 1.4967522e+00	 1.4106911e-01	  1.3120541e+00 	 2.1707392e-01


.. parsed-literal::

     111	 1.4446902e+00	 1.3793401e-01	 1.4987222e+00	 1.4047334e-01	  1.3137583e+00 	 2.1627355e-01
     112	 1.4480762e+00	 1.3718176e-01	 1.5022754e+00	 1.3939063e-01	  1.3193960e+00 	 1.8797183e-01


.. parsed-literal::

     113	 1.4504100e+00	 1.3659322e-01	 1.5046775e+00	 1.3846495e-01	  1.3158639e+00 	 2.0635271e-01


.. parsed-literal::

     114	 1.4525851e+00	 1.3681241e-01	 1.5067274e+00	 1.3881205e-01	  1.3209914e+00 	 2.0232463e-01


.. parsed-literal::

     115	 1.4546315e+00	 1.3680392e-01	 1.5087846e+00	 1.3898907e-01	[ 1.3221911e+00]	 2.1796536e-01


.. parsed-literal::

     116	 1.4565523e+00	 1.3661734e-01	 1.5107551e+00	 1.3890514e-01	[ 1.3226091e+00]	 2.1604013e-01


.. parsed-literal::

     117	 1.4594535e+00	 1.3617098e-01	 1.5138359e+00	 1.3856034e-01	[ 1.3241794e+00]	 2.0359206e-01
     118	 1.4628494e+00	 1.3564538e-01	 1.5172679e+00	 1.3793454e-01	[ 1.3276814e+00]	 1.9777441e-01


.. parsed-literal::

     119	 1.4650607e+00	 1.3541035e-01	 1.5194029e+00	 1.3759146e-01	[ 1.3308178e+00]	 2.0660305e-01


.. parsed-literal::

     120	 1.4679080e+00	 1.3491173e-01	 1.5222460e+00	 1.3688145e-01	[ 1.3350939e+00]	 2.0744562e-01
     121	 1.4685562e+00	 1.3468357e-01	 1.5229950e+00	 1.3626544e-01	[ 1.3366111e+00]	 1.8966794e-01


.. parsed-literal::

     122	 1.4705916e+00	 1.3466685e-01	 1.5249438e+00	 1.3647009e-01	[ 1.3383910e+00]	 1.8443227e-01
     123	 1.4716330e+00	 1.3461172e-01	 1.5259940e+00	 1.3650545e-01	[ 1.3387347e+00]	 1.7443013e-01


.. parsed-literal::

     124	 1.4732522e+00	 1.3449382e-01	 1.5276562e+00	 1.3645110e-01	[ 1.3393820e+00]	 2.0391870e-01


.. parsed-literal::

     125	 1.4761483e+00	 1.3427837e-01	 1.5306601e+00	 1.3626789e-01	[ 1.3413499e+00]	 2.1776414e-01


.. parsed-literal::

     126	 1.4781738e+00	 1.3399113e-01	 1.5328176e+00	 1.3600608e-01	  1.3397752e+00 	 3.2190657e-01
     127	 1.4804979e+00	 1.3384514e-01	 1.5351976e+00	 1.3576246e-01	[ 1.3418269e+00]	 1.9378805e-01


.. parsed-literal::

     128	 1.4824930e+00	 1.3372325e-01	 1.5371746e+00	 1.3562752e-01	[ 1.3425079e+00]	 2.0926428e-01
     129	 1.4844217e+00	 1.3367468e-01	 1.5391073e+00	 1.3553614e-01	[ 1.3435588e+00]	 1.9595289e-01


.. parsed-literal::

     130	 1.4868692e+00	 1.3331341e-01	 1.5415720e+00	 1.3540403e-01	  1.3386540e+00 	 1.8554306e-01
     131	 1.4892083e+00	 1.3331930e-01	 1.5438902e+00	 1.3554191e-01	  1.3397833e+00 	 1.8778443e-01


.. parsed-literal::

     132	 1.4918383e+00	 1.3314246e-01	 1.5465753e+00	 1.3572003e-01	  1.3374508e+00 	 1.9661784e-01
     133	 1.4936926e+00	 1.3300427e-01	 1.5484812e+00	 1.3574759e-01	  1.3370403e+00 	 1.7591929e-01


.. parsed-literal::

     134	 1.4953860e+00	 1.3278590e-01	 1.5501773e+00	 1.3571248e-01	  1.3362301e+00 	 2.0882010e-01
     135	 1.4973958e+00	 1.3245853e-01	 1.5521751e+00	 1.3558147e-01	  1.3347515e+00 	 2.0005870e-01


.. parsed-literal::

     136	 1.4988707e+00	 1.3232904e-01	 1.5536297e+00	 1.3548954e-01	  1.3328890e+00 	 2.0479727e-01
     137	 1.5006104e+00	 1.3222059e-01	 1.5553273e+00	 1.3541097e-01	  1.3336245e+00 	 1.9484115e-01


.. parsed-literal::

     138	 1.5019979e+00	 1.3217609e-01	 1.5567391e+00	 1.3542793e-01	  1.3321693e+00 	 1.8859148e-01


.. parsed-literal::

     139	 1.5037476e+00	 1.3210852e-01	 1.5585617e+00	 1.3548018e-01	  1.3336170e+00 	 2.0856500e-01


.. parsed-literal::

     140	 1.5054623e+00	 1.3212397e-01	 1.5603321e+00	 1.3558269e-01	  1.3352091e+00 	 2.0188498e-01


.. parsed-literal::

     141	 1.5072622e+00	 1.3201489e-01	 1.5622264e+00	 1.3564208e-01	  1.3337425e+00 	 2.0515966e-01


.. parsed-literal::

     142	 1.5088173e+00	 1.3201807e-01	 1.5637959e+00	 1.3561910e-01	  1.3388755e+00 	 2.1133256e-01


.. parsed-literal::

     143	 1.5101310e+00	 1.3191080e-01	 1.5650776e+00	 1.3545227e-01	  1.3386289e+00 	 2.1683097e-01


.. parsed-literal::

     144	 1.5120598e+00	 1.3180166e-01	 1.5669539e+00	 1.3527613e-01	  1.3386271e+00 	 2.1545482e-01
     145	 1.5132305e+00	 1.3145001e-01	 1.5681130e+00	 1.3499429e-01	  1.3355737e+00 	 1.8817353e-01


.. parsed-literal::

     146	 1.5148041e+00	 1.3145940e-01	 1.5696394e+00	 1.3496057e-01	  1.3389937e+00 	 2.1974134e-01
     147	 1.5158582e+00	 1.3142613e-01	 1.5706966e+00	 1.3499902e-01	  1.3407975e+00 	 1.9035363e-01


.. parsed-literal::

     148	 1.5172165e+00	 1.3135695e-01	 1.5720456e+00	 1.3508088e-01	  1.3420502e+00 	 2.1162868e-01
     149	 1.5192481e+00	 1.3106679e-01	 1.5740886e+00	 1.3505766e-01	  1.3419583e+00 	 1.7521048e-01


.. parsed-literal::

     150	 1.5209442e+00	 1.3102175e-01	 1.5757555e+00	 1.3537476e-01	  1.3422270e+00 	 1.9819832e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.06 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fb9282872b0>



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
    CPU times: user 1.79 s, sys: 44 ms, total: 1.83 s
    Wall time: 580 ms


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

