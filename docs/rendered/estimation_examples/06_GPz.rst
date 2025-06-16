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
       1	-3.5149403e-01	 3.2277953e-01	-3.4180835e-01	 3.1137546e-01	[-3.2126131e-01]	 4.6533394e-01


.. parsed-literal::

       2	-2.7848287e-01	 3.1160063e-01	-2.5409961e-01	 3.0202517e-01	[-2.2587168e-01]	 2.4050379e-01


.. parsed-literal::

       3	-2.3477450e-01	 2.9129495e-01	-1.9333696e-01	 2.8197525e-01	[-1.5761834e-01]	 2.8183603e-01


.. parsed-literal::

       4	-1.9499997e-01	 2.6651741e-01	-1.5472577e-01	 2.5817992e-01	[-1.0592540e-01]	 2.0981359e-01


.. parsed-literal::

       5	-1.0465254e-01	 2.5711255e-01	-7.1204888e-02	 2.5034410e-01	[-4.0104852e-02]	 2.1469545e-01


.. parsed-literal::

       6	-7.0965385e-02	 2.5195940e-01	-4.1320552e-02	 2.4663550e-01	[-1.9715341e-02]	 2.1050835e-01


.. parsed-literal::

       7	-5.3309379e-02	 2.4916850e-01	-2.9708808e-02	 2.4425686e-01	[-9.5739159e-03]	 2.0829868e-01
       8	-4.1122982e-02	 2.4709424e-01	-2.1054136e-02	 2.4269767e-01	[-2.8246051e-03]	 1.9733405e-01


.. parsed-literal::

       9	-2.7617016e-02	 2.4451435e-01	-1.0154723e-02	 2.4147391e-01	[ 3.7327688e-03]	 2.0391250e-01
      10	-1.6836732e-02	 2.4261961e-01	-1.6596095e-03	 2.4177369e-01	[ 3.9877393e-03]	 1.8712473e-01


.. parsed-literal::

      11	-1.3930562e-02	 2.4215110e-01	 3.7717917e-04	 2.4249190e-01	  1.6950984e-03 	 2.0389175e-01
      12	-1.0912920e-02	 2.4149448e-01	 3.3965736e-03	 2.4147652e-01	[ 5.8650976e-03]	 1.8131590e-01


.. parsed-literal::

      13	-8.5079628e-03	 2.4096807e-01	 5.9453222e-03	 2.4036540e-01	[ 1.0241304e-02]	 2.1096945e-01


.. parsed-literal::

      14	-5.6900393e-03	 2.4034842e-01	 9.0688229e-03	 2.3982239e-01	[ 1.2739526e-02]	 2.1465087e-01
      15	 1.0194442e-02	 2.3741529e-01	 2.6556476e-02	 2.3913316e-01	[ 2.0503097e-02]	 1.9986629e-01


.. parsed-literal::

      16	 7.9100558e-02	 2.2985161e-01	 1.0223836e-01	 2.2920158e-01	[ 1.0181648e-01]	 1.9351172e-01
      17	 1.0110153e-01	 2.2540308e-01	 1.2402901e-01	 2.3019028e-01	[ 1.0865889e-01]	 1.9740534e-01


.. parsed-literal::

      18	 1.9280996e-01	 2.1682730e-01	 2.1933418e-01	 2.2124967e-01	[ 2.0284234e-01]	 2.0518589e-01
      19	 3.5550714e-01	 2.1131389e-01	 3.8826915e-01	 2.0979038e-01	[ 3.7747416e-01]	 1.8823385e-01


.. parsed-literal::

      20	 4.1613957e-01	 2.1310331e-01	 4.4975449e-01	 2.1063784e-01	[ 4.3778167e-01]	 2.0090437e-01


.. parsed-literal::

      21	 4.9925559e-01	 2.0391194e-01	 5.3487467e-01	 2.0367797e-01	[ 5.1279874e-01]	 2.1213841e-01


.. parsed-literal::

      22	 5.5310435e-01	 1.9571868e-01	 5.9302538e-01	 1.9729629e-01	[ 5.6751997e-01]	 2.0606875e-01
      23	 6.1858453e-01	 1.9264431e-01	 6.5735954e-01	 1.9527307e-01	[ 6.2906862e-01]	 1.9541311e-01


.. parsed-literal::

      24	 6.4999269e-01	 1.8883580e-01	 6.8876431e-01	 1.9173089e-01	[ 6.6252929e-01]	 2.0525408e-01


.. parsed-literal::

      25	 7.1212803e-01	 1.8579473e-01	 7.5169580e-01	 1.9205410e-01	[ 7.0989701e-01]	 2.0356297e-01
      26	 7.7392758e-01	 1.8549388e-01	 8.1306400e-01	 1.9241140e-01	[ 7.6445656e-01]	 1.8175602e-01


.. parsed-literal::

      27	 8.0847410e-01	 1.8908659e-01	 8.4722491e-01	 1.9631268e-01	[ 7.8124309e-01]	 2.0985699e-01
      28	 8.3927677e-01	 1.8745412e-01	 8.7886331e-01	 1.9421535e-01	[ 8.0515577e-01]	 1.7793465e-01


.. parsed-literal::

      29	 8.7379336e-01	 1.9148616e-01	 9.1476953e-01	 1.9718747e-01	[ 8.2599575e-01]	 2.0898557e-01
      30	 9.0230607e-01	 1.8593326e-01	 9.4425353e-01	 1.9189333e-01	[ 8.5093746e-01]	 1.9814992e-01


.. parsed-literal::

      31	 9.2291736e-01	 1.8406720e-01	 9.6445555e-01	 1.8924245e-01	[ 8.7634084e-01]	 2.2279310e-01


.. parsed-literal::

      32	 9.4533859e-01	 1.7914438e-01	 9.8726436e-01	 1.8255411e-01	[ 9.0116916e-01]	 2.0938325e-01
      33	 9.6455879e-01	 1.7499941e-01	 1.0067602e+00	 1.7720288e-01	[ 9.1177215e-01]	 1.7728472e-01


.. parsed-literal::

      34	 9.8914581e-01	 1.7342343e-01	 1.0316744e+00	 1.7476580e-01	[ 9.3335422e-01]	 1.8759799e-01


.. parsed-literal::

      35	 1.0060399e+00	 1.7259472e-01	 1.0493223e+00	 1.7397755e-01	[ 9.4899662e-01]	 2.0838976e-01


.. parsed-literal::

      36	 1.0304567e+00	 1.7128350e-01	 1.0749968e+00	 1.7263680e-01	[ 9.7563506e-01]	 2.1318150e-01


.. parsed-literal::

      37	 1.0506119e+00	 1.6823798e-01	 1.0956478e+00	 1.6848511e-01	[ 1.0064880e+00]	 2.1500182e-01
      38	 1.0711508e+00	 1.6475963e-01	 1.1167125e+00	 1.6445389e-01	[ 1.0313603e+00]	 1.9470620e-01


.. parsed-literal::

      39	 1.0858532e+00	 1.6265317e-01	 1.1316850e+00	 1.6274755e-01	[ 1.0460947e+00]	 2.1863103e-01


.. parsed-literal::

      40	 1.1095938e+00	 1.5998442e-01	 1.1559554e+00	 1.5984723e-01	[ 1.0653423e+00]	 2.2210741e-01


.. parsed-literal::

      41	 1.1210752e+00	 1.5829372e-01	 1.1679111e+00	 1.5771116e-01	[ 1.0837731e+00]	 2.1493101e-01


.. parsed-literal::

      42	 1.1379642e+00	 1.5713814e-01	 1.1843988e+00	 1.5696940e-01	[ 1.0942657e+00]	 2.0648670e-01


.. parsed-literal::

      43	 1.1494621e+00	 1.5665017e-01	 1.1958651e+00	 1.5670621e-01	[ 1.0999658e+00]	 2.0438957e-01


.. parsed-literal::

      44	 1.1623614e+00	 1.5534123e-01	 1.2091013e+00	 1.5497396e-01	[ 1.1097839e+00]	 2.1087050e-01


.. parsed-literal::

      45	 1.1738678e+00	 1.5130339e-01	 1.2215380e+00	 1.5011418e-01	[ 1.1162345e+00]	 2.1632719e-01


.. parsed-literal::

      46	 1.1894496e+00	 1.5048910e-01	 1.2369173e+00	 1.4777456e-01	[ 1.1370863e+00]	 2.1747398e-01
      47	 1.1948043e+00	 1.5011750e-01	 1.2422782e+00	 1.4742401e-01	[ 1.1402410e+00]	 1.9904757e-01


.. parsed-literal::

      48	 1.2099894e+00	 1.4889652e-01	 1.2579661e+00	 1.4647581e-01	[ 1.1445464e+00]	 2.0743632e-01
      49	 1.2265612e+00	 1.4664788e-01	 1.2746370e+00	 1.4422008e-01	[ 1.1540621e+00]	 1.8969893e-01


.. parsed-literal::

      50	 1.2382242e+00	 1.4661853e-01	 1.2875435e+00	 1.4421733e-01	[ 1.1542887e+00]	 2.0957184e-01
      51	 1.2450338e+00	 1.4584922e-01	 1.2944425e+00	 1.4289782e-01	[ 1.1626904e+00]	 1.9030714e-01


.. parsed-literal::

      52	 1.2550994e+00	 1.4480298e-01	 1.3047915e+00	 1.4097560e-01	[ 1.1750229e+00]	 1.7981553e-01


.. parsed-literal::

      53	 1.2647046e+00	 1.4407873e-01	 1.3143758e+00	 1.4032614e-01	[ 1.1862067e+00]	 2.2084522e-01


.. parsed-literal::

      54	 1.2757260e+00	 1.4333587e-01	 1.3259496e+00	 1.3852651e-01	[ 1.1957307e+00]	 2.1273422e-01


.. parsed-literal::

      55	 1.2885270e+00	 1.4308125e-01	 1.3387255e+00	 1.4012264e-01	[ 1.1978699e+00]	 2.1595693e-01


.. parsed-literal::

      56	 1.2968862e+00	 1.4323603e-01	 1.3468550e+00	 1.4051206e-01	[ 1.2049506e+00]	 2.1091509e-01


.. parsed-literal::

      57	 1.3064061e+00	 1.4325642e-01	 1.3567809e+00	 1.4033838e-01	[ 1.2068597e+00]	 2.1004725e-01
      58	 1.3155998e+00	 1.4460704e-01	 1.3658789e+00	 1.4065307e-01	[ 1.2152912e+00]	 1.7928743e-01


.. parsed-literal::

      59	 1.3262594e+00	 1.4364352e-01	 1.3769251e+00	 1.3923093e-01	[ 1.2188102e+00]	 2.1140170e-01
      60	 1.3319240e+00	 1.4309270e-01	 1.3829362e+00	 1.3858963e-01	[ 1.2190421e+00]	 1.8900943e-01


.. parsed-literal::

      61	 1.3388545e+00	 1.4257772e-01	 1.3900816e+00	 1.3812865e-01	[ 1.2217067e+00]	 2.1126771e-01


.. parsed-literal::

      62	 1.3480153e+00	 1.4252613e-01	 1.3997199e+00	 1.3828510e-01	  1.2196388e+00 	 2.1314430e-01


.. parsed-literal::

      63	 1.3556919e+00	 1.4337191e-01	 1.4073656e+00	 1.3879326e-01	[ 1.2285701e+00]	 2.1093178e-01
      64	 1.3625448e+00	 1.4252308e-01	 1.4138667e+00	 1.3832517e-01	[ 1.2379234e+00]	 1.8935347e-01


.. parsed-literal::

      65	 1.3675858e+00	 1.4228805e-01	 1.4191297e+00	 1.3774925e-01	[ 1.2403567e+00]	 2.1599245e-01


.. parsed-literal::

      66	 1.3736227e+00	 1.4212286e-01	 1.4255268e+00	 1.3718733e-01	[ 1.2447152e+00]	 2.1246409e-01


.. parsed-literal::

      67	 1.3810625e+00	 1.4136048e-01	 1.4331506e+00	 1.3575895e-01	[ 1.2509313e+00]	 2.1153235e-01


.. parsed-literal::

      68	 1.3867061e+00	 1.4142137e-01	 1.4388650e+00	 1.3492830e-01	[ 1.2579323e+00]	 2.0938206e-01


.. parsed-literal::

      69	 1.3920870e+00	 1.4063099e-01	 1.4440446e+00	 1.3443870e-01	  1.2578961e+00 	 2.2135496e-01
      70	 1.3957039e+00	 1.4042592e-01	 1.4475955e+00	 1.3432457e-01	  1.2577206e+00 	 2.0855618e-01


.. parsed-literal::

      71	 1.4007456e+00	 1.3997112e-01	 1.4527427e+00	 1.3395675e-01	  1.2535030e+00 	 2.0781875e-01


.. parsed-literal::

      72	 1.4041381e+00	 1.3984213e-01	 1.4560460e+00	 1.3368017e-01	  1.2558483e+00 	 2.1498418e-01


.. parsed-literal::

      73	 1.4073537e+00	 1.3960379e-01	 1.4591763e+00	 1.3330452e-01	[ 1.2595062e+00]	 2.0827866e-01
      74	 1.4129687e+00	 1.3902795e-01	 1.4648003e+00	 1.3229924e-01	[ 1.2644354e+00]	 1.9854832e-01


.. parsed-literal::

      75	 1.4171968e+00	 1.3848526e-01	 1.4691086e+00	 1.3155637e-01	[ 1.2657191e+00]	 2.1989083e-01


.. parsed-literal::

      76	 1.4225870e+00	 1.3780160e-01	 1.4748143e+00	 1.2974608e-01	[ 1.2683671e+00]	 2.0956230e-01


.. parsed-literal::

      77	 1.4268794e+00	 1.3691627e-01	 1.4790499e+00	 1.2929757e-01	  1.2637591e+00 	 2.0438075e-01
      78	 1.4295343e+00	 1.3693969e-01	 1.4816152e+00	 1.2950848e-01	  1.2648183e+00 	 1.8475819e-01


.. parsed-literal::

      79	 1.4329758e+00	 1.3682695e-01	 1.4851124e+00	 1.2958448e-01	  1.2621505e+00 	 2.0759583e-01
      80	 1.4365324e+00	 1.3636917e-01	 1.4888956e+00	 1.2954280e-01	  1.2538149e+00 	 1.8848944e-01


.. parsed-literal::

      81	 1.4404117e+00	 1.3625409e-01	 1.4928970e+00	 1.2966397e-01	  1.2497258e+00 	 2.1613479e-01
      82	 1.4442891e+00	 1.3586666e-01	 1.4968634e+00	 1.2928612e-01	  1.2504082e+00 	 2.0838785e-01


.. parsed-literal::

      83	 1.4475566e+00	 1.3582550e-01	 1.5003009e+00	 1.2916164e-01	  1.2483387e+00 	 2.1741199e-01


.. parsed-literal::

      84	 1.4507363e+00	 1.3603493e-01	 1.5035417e+00	 1.2885942e-01	  1.2564753e+00 	 2.0255613e-01
      85	 1.4536688e+00	 1.3623763e-01	 1.5065718e+00	 1.2863644e-01	  1.2587025e+00 	 1.8890357e-01


.. parsed-literal::

      86	 1.4575505e+00	 1.3634662e-01	 1.5106241e+00	 1.2834965e-01	  1.2607668e+00 	 2.1420670e-01


.. parsed-literal::

      87	 1.4597707e+00	 1.3651475e-01	 1.5129208e+00	 1.2825270e-01	  1.2594437e+00 	 3.0284953e-01


.. parsed-literal::

      88	 1.4623939e+00	 1.3619518e-01	 1.5155273e+00	 1.2801827e-01	  1.2584485e+00 	 2.1560073e-01


.. parsed-literal::

      89	 1.4653334e+00	 1.3572231e-01	 1.5184960e+00	 1.2772100e-01	  1.2564489e+00 	 2.0653820e-01
      90	 1.4679832e+00	 1.3553136e-01	 1.5211745e+00	 1.2752806e-01	  1.2524294e+00 	 2.0086098e-01


.. parsed-literal::

      91	 1.4714169e+00	 1.3520160e-01	 1.5246356e+00	 1.2707844e-01	  1.2512100e+00 	 1.9387507e-01


.. parsed-literal::

      92	 1.4747499e+00	 1.3491277e-01	 1.5280271e+00	 1.2650400e-01	  1.2483445e+00 	 2.0165968e-01
      93	 1.4768955e+00	 1.3485902e-01	 1.5302562e+00	 1.2629915e-01	  1.2485007e+00 	 2.0654750e-01


.. parsed-literal::

      94	 1.4790928e+00	 1.3466475e-01	 1.5324209e+00	 1.2605104e-01	  1.2493097e+00 	 2.1393180e-01


.. parsed-literal::

      95	 1.4807326e+00	 1.3447457e-01	 1.5340791e+00	 1.2584713e-01	  1.2497017e+00 	 2.1063924e-01


.. parsed-literal::

      96	 1.4827363e+00	 1.3407786e-01	 1.5362090e+00	 1.2558393e-01	  1.2439189e+00 	 2.1970701e-01


.. parsed-literal::

      97	 1.4849755e+00	 1.3396218e-01	 1.5384914e+00	 1.2548313e-01	  1.2439085e+00 	 2.1827412e-01


.. parsed-literal::

      98	 1.4871007e+00	 1.3389469e-01	 1.5406816e+00	 1.2543269e-01	  1.2422244e+00 	 2.1223354e-01


.. parsed-literal::

      99	 1.4888791e+00	 1.3386260e-01	 1.5425152e+00	 1.2548584e-01	  1.2413955e+00 	 2.1246767e-01
     100	 1.4905460e+00	 1.3398706e-01	 1.5442411e+00	 1.2562631e-01	  1.2393222e+00 	 2.0543885e-01


.. parsed-literal::

     101	 1.4921355e+00	 1.3391795e-01	 1.5458327e+00	 1.2556053e-01	  1.2390525e+00 	 1.7799973e-01
     102	 1.4947137e+00	 1.3387191e-01	 1.5484643e+00	 1.2548586e-01	  1.2343241e+00 	 1.7873836e-01


.. parsed-literal::

     103	 1.4965235e+00	 1.3391478e-01	 1.5504213e+00	 1.2549501e-01	  1.2260529e+00 	 2.0801806e-01


.. parsed-literal::

     104	 1.4988321e+00	 1.3398916e-01	 1.5527608e+00	 1.2552833e-01	  1.2202567e+00 	 2.1864223e-01


.. parsed-literal::

     105	 1.5020415e+00	 1.3402731e-01	 1.5560598e+00	 1.2562260e-01	  1.2080174e+00 	 2.0775986e-01


.. parsed-literal::

     106	 1.5038468e+00	 1.3400782e-01	 1.5579445e+00	 1.2565861e-01	  1.2026490e+00 	 2.0640469e-01
     107	 1.5059740e+00	 1.3396448e-01	 1.5600333e+00	 1.2570786e-01	  1.2001228e+00 	 2.0015812e-01


.. parsed-literal::

     108	 1.5079063e+00	 1.3386995e-01	 1.5619572e+00	 1.2577809e-01	  1.1966726e+00 	 2.0571733e-01


.. parsed-literal::

     109	 1.5092918e+00	 1.3372758e-01	 1.5633978e+00	 1.2580279e-01	  1.1930335e+00 	 2.1580982e-01
     110	 1.5109424e+00	 1.3357542e-01	 1.5650683e+00	 1.2575906e-01	  1.1905792e+00 	 1.8571687e-01


.. parsed-literal::

     111	 1.5127687e+00	 1.3349969e-01	 1.5669525e+00	 1.2573076e-01	  1.1875956e+00 	 2.1039581e-01


.. parsed-literal::

     112	 1.5142322e+00	 1.3336570e-01	 1.5684864e+00	 1.2561665e-01	  1.1894919e+00 	 2.0956016e-01


.. parsed-literal::

     113	 1.5155584e+00	 1.3323267e-01	 1.5697871e+00	 1.2541974e-01	  1.1921761e+00 	 2.0656443e-01


.. parsed-literal::

     114	 1.5168083e+00	 1.3309991e-01	 1.5710052e+00	 1.2520824e-01	  1.1954894e+00 	 2.0431566e-01


.. parsed-literal::

     115	 1.5188273e+00	 1.3286273e-01	 1.5729548e+00	 1.2487313e-01	  1.2010518e+00 	 2.0917797e-01


.. parsed-literal::

     116	 1.5207893e+00	 1.3274587e-01	 1.5749147e+00	 1.2447767e-01	  1.2073886e+00 	 2.1746302e-01


.. parsed-literal::

     117	 1.5229080e+00	 1.3245217e-01	 1.5769853e+00	 1.2414819e-01	  1.2117948e+00 	 2.0815444e-01


.. parsed-literal::

     118	 1.5243416e+00	 1.3225070e-01	 1.5784346e+00	 1.2404167e-01	  1.2119652e+00 	 2.0265293e-01


.. parsed-literal::

     119	 1.5256315e+00	 1.3206732e-01	 1.5797994e+00	 1.2391792e-01	  1.2081441e+00 	 2.2047973e-01


.. parsed-literal::

     120	 1.5271276e+00	 1.3178328e-01	 1.5814196e+00	 1.2364675e-01	  1.2055833e+00 	 2.0691156e-01


.. parsed-literal::

     121	 1.5287571e+00	 1.3155193e-01	 1.5831101e+00	 1.2339944e-01	  1.2040372e+00 	 2.0373511e-01
     122	 1.5299970e+00	 1.3138764e-01	 1.5844820e+00	 1.2309134e-01	  1.1996685e+00 	 1.7407370e-01


.. parsed-literal::

     123	 1.5314284e+00	 1.3123535e-01	 1.5858806e+00	 1.2295245e-01	  1.2022745e+00 	 1.7738342e-01
     124	 1.5321603e+00	 1.3125051e-01	 1.5865742e+00	 1.2294703e-01	  1.2031410e+00 	 2.0895267e-01


.. parsed-literal::

     125	 1.5340021e+00	 1.3123898e-01	 1.5884101e+00	 1.2287957e-01	  1.2025778e+00 	 1.9519067e-01
     126	 1.5347765e+00	 1.3126032e-01	 1.5892224e+00	 1.2269697e-01	  1.1991814e+00 	 1.9193268e-01


.. parsed-literal::

     127	 1.5359424e+00	 1.3120526e-01	 1.5903778e+00	 1.2268526e-01	  1.1992403e+00 	 2.0649433e-01
     128	 1.5368656e+00	 1.3108332e-01	 1.5913526e+00	 1.2257009e-01	  1.1978432e+00 	 1.8308854e-01


.. parsed-literal::

     129	 1.5378472e+00	 1.3091615e-01	 1.5924097e+00	 1.2236546e-01	  1.1960059e+00 	 2.0876670e-01


.. parsed-literal::

     130	 1.5395634e+00	 1.3056801e-01	 1.5942583e+00	 1.2196458e-01	  1.1925056e+00 	 2.0504379e-01


.. parsed-literal::

     131	 1.5407733e+00	 1.3027907e-01	 1.5955990e+00	 1.2163946e-01	  1.1885854e+00 	 3.3657789e-01


.. parsed-literal::

     132	 1.5425964e+00	 1.2993067e-01	 1.5975466e+00	 1.2120710e-01	  1.1837931e+00 	 2.1181917e-01


.. parsed-literal::

     133	 1.5436585e+00	 1.2983738e-01	 1.5986029e+00	 1.2109119e-01	  1.1821692e+00 	 2.0609689e-01


.. parsed-literal::

     134	 1.5447147e+00	 1.2987121e-01	 1.5996500e+00	 1.2109169e-01	  1.1796579e+00 	 2.1548867e-01


.. parsed-literal::

     135	 1.5457397e+00	 1.2992139e-01	 1.6006672e+00	 1.2108728e-01	  1.1789638e+00 	 2.0553422e-01


.. parsed-literal::

     136	 1.5467408e+00	 1.3002583e-01	 1.6016709e+00	 1.2110391e-01	  1.1786186e+00 	 2.0712042e-01


.. parsed-literal::

     137	 1.5476778e+00	 1.2995887e-01	 1.6026234e+00	 1.2100730e-01	  1.1788888e+00 	 2.1549916e-01
     138	 1.5486921e+00	 1.2985926e-01	 1.6036487e+00	 1.2085759e-01	  1.1781027e+00 	 1.8065095e-01


.. parsed-literal::

     139	 1.5501799e+00	 1.2966143e-01	 1.6051685e+00	 1.2060895e-01	  1.1757866e+00 	 2.1809459e-01


.. parsed-literal::

     140	 1.5510156e+00	 1.2966957e-01	 1.6059833e+00	 1.2048301e-01	  1.1719155e+00 	 3.1333613e-01
     141	 1.5519388e+00	 1.2962019e-01	 1.6069212e+00	 1.2033777e-01	  1.1698882e+00 	 2.0121503e-01


.. parsed-literal::

     142	 1.5533794e+00	 1.2967134e-01	 1.6083909e+00	 1.2019564e-01	  1.1660701e+00 	 2.0917130e-01


.. parsed-literal::

     143	 1.5546652e+00	 1.2976423e-01	 1.6097043e+00	 1.2011431e-01	  1.1634613e+00 	 2.1983624e-01


.. parsed-literal::

     144	 1.5558786e+00	 1.2991905e-01	 1.6109835e+00	 1.1996475e-01	  1.1521206e+00 	 2.1211958e-01
     145	 1.5571072e+00	 1.2982900e-01	 1.6121496e+00	 1.1996454e-01	  1.1554579e+00 	 2.0231915e-01


.. parsed-literal::

     146	 1.5577935e+00	 1.2973127e-01	 1.6128163e+00	 1.1994937e-01	  1.1559138e+00 	 2.1661758e-01
     147	 1.5585927e+00	 1.2955508e-01	 1.6136277e+00	 1.1989493e-01	  1.1533449e+00 	 1.8271852e-01


.. parsed-literal::

     148	 1.5590647e+00	 1.2947432e-01	 1.6141915e+00	 1.1997044e-01	  1.1444632e+00 	 2.2215986e-01
     149	 1.5599869e+00	 1.2939326e-01	 1.6150769e+00	 1.1988239e-01	  1.1435686e+00 	 1.9051886e-01


.. parsed-literal::

     150	 1.5606787e+00	 1.2932831e-01	 1.6157904e+00	 1.1984233e-01	  1.1390703e+00 	 1.8491864e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.19 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fdc9ca892d0>



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
    CPU times: user 1.83 s, sys: 46 ms, total: 1.88 s
    Wall time: 630 ms


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

