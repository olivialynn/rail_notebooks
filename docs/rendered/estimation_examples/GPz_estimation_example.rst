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
       1	-3.6048894e-01	 3.2561270e-01	-3.5076925e-01	 2.9898537e-01	[-3.0389794e-01]	 4.6661925e-01


.. parsed-literal::

       2	-2.8482741e-01	 3.1318997e-01	-2.5930125e-01	 2.8923447e-01	[-1.9106715e-01]	 2.3308110e-01


.. parsed-literal::

       3	-2.4009247e-01	 2.9244318e-01	-1.9676052e-01	 2.7505950e-01	[-1.2905969e-01]	 2.9916167e-01


.. parsed-literal::

       4	-1.9678880e-01	 2.7328005e-01	-1.4630530e-01	 2.6354593e-01	[-9.8302772e-02]	 3.2092261e-01


.. parsed-literal::

       5	-1.4723799e-01	 2.6028401e-01	-1.1762920e-01	 2.4874873e-01	[-5.3757983e-02]	 2.0131111e-01


.. parsed-literal::

       6	-8.3887062e-02	 2.5606546e-01	-5.6272818e-02	 2.4375061e-01	[-1.0424151e-02]	 2.0348310e-01
       7	-6.5149143e-02	 2.5195210e-01	-4.1523537e-02	 2.3916912e-01	[ 4.3610768e-03]	 1.9739723e-01


.. parsed-literal::

       8	-5.3647025e-02	 2.5004008e-01	-3.2917525e-02	 2.3725135e-01	[ 1.5600111e-02]	 2.1094608e-01


.. parsed-literal::

       9	-3.6495576e-02	 2.4677722e-01	-1.9164964e-02	 2.3478555e-01	[ 2.6721635e-02]	 2.1662712e-01
      10	-2.5655098e-02	 2.4463478e-01	-1.0289781e-02	 2.3384781e-01	[ 3.3833904e-02]	 1.9675255e-01


.. parsed-literal::

      11	-2.0025944e-02	 2.4365646e-01	-5.5567237e-03	 2.3331648e-01	[ 3.5739883e-02]	 2.1715021e-01
      12	-1.6234234e-02	 2.4290545e-01	-2.0777330e-03	 2.3307395e-01	[ 3.6364645e-02]	 1.8466401e-01


.. parsed-literal::

      13	-1.1577789e-02	 2.4196136e-01	 2.5152933e-03	 2.3260811e-01	[ 3.8892764e-02]	 2.0173788e-01


.. parsed-literal::

      14	 6.2819123e-02	 2.2891056e-01	 8.1868059e-02	 2.2198267e-01	[ 1.0492987e-01]	 3.2416081e-01


.. parsed-literal::

      15	 7.9360113e-02	 2.2633439e-01	 1.0059279e-01	 2.2030377e-01	[ 1.1284108e-01]	 2.9990602e-01
      16	 1.6656658e-01	 2.2025941e-01	 1.9131519e-01	 2.1736965e-01	[ 1.9924315e-01]	 1.8581700e-01


.. parsed-literal::

      17	 2.0689292e-01	 2.1645650e-01	 2.3876979e-01	 2.1289739e-01	[ 2.5826808e-01]	 2.0434189e-01


.. parsed-literal::

      18	 2.9354253e-01	 2.1343237e-01	 3.2527383e-01	 2.1155372e-01	[ 3.3459640e-01]	 2.0772314e-01
      19	 3.2186577e-01	 2.1057400e-01	 3.5370185e-01	 2.1103373e-01	[ 3.5922772e-01]	 1.9430041e-01


.. parsed-literal::

      20	 3.6368493e-01	 2.0736018e-01	 3.9712135e-01	 2.0826143e-01	[ 4.0346075e-01]	 2.0246077e-01


.. parsed-literal::

      21	 4.6913183e-01	 2.0641034e-01	 5.0237927e-01	 2.0828896e-01	[ 5.1010290e-01]	 2.1418262e-01


.. parsed-literal::

      22	 5.5733979e-01	 2.0521632e-01	 5.9396630e-01	 2.0719973e-01	[ 6.0031199e-01]	 2.0347452e-01
      23	 6.2753494e-01	 1.9546303e-01	 6.6671750e-01	 2.0081820e-01	[ 6.7398445e-01]	 1.9891644e-01


.. parsed-literal::

      24	 6.6539586e-01	 1.9191618e-01	 7.0392710e-01	 1.9550957e-01	[ 7.1461286e-01]	 2.1131849e-01
      25	 7.0005833e-01	 1.8978489e-01	 7.3825315e-01	 1.9090835e-01	[ 7.5338023e-01]	 1.9677758e-01


.. parsed-literal::

      26	 7.4736093e-01	 1.8476234e-01	 7.8601633e-01	 1.8468220e-01	[ 7.9563927e-01]	 2.1231699e-01
      27	 7.8878864e-01	 1.8926617e-01	 8.2795369e-01	 1.8656363e-01	[ 8.3267857e-01]	 1.9024277e-01


.. parsed-literal::

      28	 8.1918415e-01	 1.8953535e-01	 8.5874421e-01	 1.8614007e-01	[ 8.6375260e-01]	 2.1008134e-01
      29	 8.4775774e-01	 1.8998667e-01	 8.8815421e-01	 1.8759652e-01	[ 8.9193762e-01]	 1.9684219e-01


.. parsed-literal::

      30	 8.7967934e-01	 1.8593377e-01	 9.2095598e-01	 1.8463817e-01	[ 9.2818043e-01]	 2.0945024e-01


.. parsed-literal::

      31	 9.0591878e-01	 1.8089037e-01	 9.4699535e-01	 1.8312730e-01	[ 9.5172474e-01]	 2.1777058e-01


.. parsed-literal::

      32	 9.2705505e-01	 1.7531595e-01	 9.6898141e-01	 1.7605005e-01	[ 9.7167449e-01]	 2.0415163e-01


.. parsed-literal::

      33	 9.5204589e-01	 1.7134755e-01	 9.9422420e-01	 1.7279623e-01	[ 1.0012532e+00]	 2.2239757e-01
      34	 9.7697847e-01	 1.6693167e-01	 1.0199808e+00	 1.6774838e-01	[ 1.0216979e+00]	 1.8335652e-01


.. parsed-literal::

      35	 9.9989004e-01	 1.6430601e-01	 1.0443367e+00	 1.6471344e-01	[ 1.0419709e+00]	 2.0602632e-01


.. parsed-literal::

      36	 1.0150675e+00	 1.6291808e-01	 1.0605633e+00	 1.6183367e-01	[ 1.0573411e+00]	 2.0977783e-01


.. parsed-literal::

      37	 1.0287946e+00	 1.6141211e-01	 1.0744258e+00	 1.6106212e-01	[ 1.0678672e+00]	 2.1403933e-01
      38	 1.0430788e+00	 1.5936290e-01	 1.0888148e+00	 1.5983351e-01	[ 1.0823212e+00]	 1.9846511e-01


.. parsed-literal::

      39	 1.0626851e+00	 1.5662794e-01	 1.1087979e+00	 1.5713253e-01	[ 1.0981796e+00]	 1.9966316e-01
      40	 1.0730758e+00	 1.5608101e-01	 1.1198223e+00	 1.5771970e-01	[ 1.1063376e+00]	 1.8418694e-01


.. parsed-literal::

      41	 1.0855394e+00	 1.5537476e-01	 1.1316998e+00	 1.5687515e-01	[ 1.1142927e+00]	 2.0882273e-01


.. parsed-literal::

      42	 1.0950072e+00	 1.5435788e-01	 1.1412940e+00	 1.5475042e-01	[ 1.1209123e+00]	 2.0407200e-01


.. parsed-literal::

      43	 1.1102623e+00	 1.5265959e-01	 1.1568798e+00	 1.5105723e-01	[ 1.1293177e+00]	 2.1854138e-01


.. parsed-literal::

      44	 1.1272274e+00	 1.5045320e-01	 1.1739036e+00	 1.4629658e-01	[ 1.1471646e+00]	 2.0149350e-01


.. parsed-literal::

      45	 1.1412135e+00	 1.4921375e-01	 1.1879842e+00	 1.4400976e-01	[ 1.1572029e+00]	 2.1369290e-01


.. parsed-literal::

      46	 1.1521913e+00	 1.4837724e-01	 1.1990757e+00	 1.4286487e-01	[ 1.1697810e+00]	 2.4020243e-01


.. parsed-literal::

      47	 1.1653080e+00	 1.4756654e-01	 1.2128638e+00	 1.4108542e-01	[ 1.1824730e+00]	 2.0629764e-01
      48	 1.1737932e+00	 1.4649609e-01	 1.2219401e+00	 1.3993859e-01	[ 1.1963409e+00]	 1.8705654e-01


.. parsed-literal::

      49	 1.1819918e+00	 1.4606080e-01	 1.2300513e+00	 1.3973976e-01	[ 1.2009548e+00]	 2.1994615e-01


.. parsed-literal::

      50	 1.1929125e+00	 1.4504652e-01	 1.2412032e+00	 1.3892578e-01	[ 1.2042141e+00]	 2.2892022e-01
      51	 1.2028338e+00	 1.4409659e-01	 1.2512528e+00	 1.3840018e-01	[ 1.2105807e+00]	 1.9873548e-01


.. parsed-literal::

      52	 1.2212404e+00	 1.4245868e-01	 1.2703419e+00	 1.3711750e-01	[ 1.2205130e+00]	 2.1628404e-01
      53	 1.2239577e+00	 1.4132808e-01	 1.2735915e+00	 1.3711035e-01	[ 1.2248319e+00]	 1.8368101e-01


.. parsed-literal::

      54	 1.2414677e+00	 1.4062955e-01	 1.2902856e+00	 1.3637636e-01	[ 1.2401429e+00]	 2.1994829e-01


.. parsed-literal::

      55	 1.2481526e+00	 1.4030293e-01	 1.2971008e+00	 1.3637423e-01	[ 1.2423405e+00]	 2.3529053e-01


.. parsed-literal::

      56	 1.2583005e+00	 1.3949165e-01	 1.3075384e+00	 1.3638098e-01	[ 1.2470183e+00]	 2.1723533e-01


.. parsed-literal::

      57	 1.2684009e+00	 1.3847153e-01	 1.3179467e+00	 1.3586254e-01	[ 1.2523050e+00]	 2.1136928e-01


.. parsed-literal::

      58	 1.2799565e+00	 1.3678420e-01	 1.3299178e+00	 1.3473183e-01	[ 1.2626408e+00]	 2.0326018e-01


.. parsed-literal::

      59	 1.2887852e+00	 1.3633339e-01	 1.3386381e+00	 1.3406274e-01	[ 1.2711030e+00]	 2.1398950e-01


.. parsed-literal::

      60	 1.2961964e+00	 1.3602841e-01	 1.3460727e+00	 1.3352864e-01	[ 1.2790640e+00]	 2.0408511e-01


.. parsed-literal::

      61	 1.3032256e+00	 1.3567621e-01	 1.3532903e+00	 1.3351983e-01	[ 1.2840882e+00]	 2.0416355e-01
      62	 1.3111412e+00	 1.3496191e-01	 1.3612049e+00	 1.3286232e-01	[ 1.2961433e+00]	 1.8746781e-01


.. parsed-literal::

      63	 1.3197164e+00	 1.3437346e-01	 1.3699263e+00	 1.3277678e-01	[ 1.3016828e+00]	 1.8546939e-01
      64	 1.3295979e+00	 1.3363832e-01	 1.3800574e+00	 1.3242495e-01	[ 1.3117257e+00]	 1.8381310e-01


.. parsed-literal::

      65	 1.3355675e+00	 1.3334471e-01	 1.3862982e+00	 1.3232059e-01	  1.3080573e+00 	 2.1466255e-01


.. parsed-literal::

      66	 1.3429704e+00	 1.3323684e-01	 1.3935217e+00	 1.3209204e-01	[ 1.3163285e+00]	 2.1479130e-01
      67	 1.3494612e+00	 1.3288207e-01	 1.4000295e+00	 1.3178430e-01	[ 1.3207807e+00]	 1.8395782e-01


.. parsed-literal::

      68	 1.3556950e+00	 1.3276811e-01	 1.4064415e+00	 1.3161521e-01	[ 1.3209831e+00]	 1.8781137e-01
      69	 1.3618921e+00	 1.3181389e-01	 1.4130686e+00	 1.3104685e-01	  1.3209355e+00 	 1.8799806e-01


.. parsed-literal::

      70	 1.3704528e+00	 1.3145785e-01	 1.4215062e+00	 1.3082396e-01	[ 1.3287893e+00]	 2.0504045e-01
      71	 1.3764077e+00	 1.3079751e-01	 1.4275865e+00	 1.3059607e-01	[ 1.3324650e+00]	 1.9289994e-01


.. parsed-literal::

      72	 1.3811142e+00	 1.3011367e-01	 1.4326937e+00	 1.3041756e-01	  1.3304596e+00 	 2.1491170e-01


.. parsed-literal::

      73	 1.3862941e+00	 1.2944563e-01	 1.4379156e+00	 1.3006723e-01	[ 1.3342307e+00]	 2.1602964e-01


.. parsed-literal::

      74	 1.3920985e+00	 1.2888425e-01	 1.4437848e+00	 1.2975897e-01	[ 1.3358647e+00]	 2.0241475e-01
      75	 1.3991866e+00	 1.2839937e-01	 1.4508982e+00	 1.2943409e-01	  1.3352206e+00 	 1.9793630e-01


.. parsed-literal::

      76	 1.4050117e+00	 1.2767923e-01	 1.4568108e+00	 1.2901895e-01	  1.3290831e+00 	 2.0437241e-01
      77	 1.4107253e+00	 1.2762600e-01	 1.4623122e+00	 1.2876530e-01	  1.3358380e+00 	 1.8054771e-01


.. parsed-literal::

      78	 1.4143472e+00	 1.2747235e-01	 1.4659233e+00	 1.2853185e-01	[ 1.3367622e+00]	 2.0251393e-01
      79	 1.4181033e+00	 1.2728309e-01	 1.4697841e+00	 1.2814756e-01	[ 1.3397082e+00]	 1.7980933e-01


.. parsed-literal::

      80	 1.4220071e+00	 1.2707261e-01	 1.4737546e+00	 1.2787451e-01	[ 1.3398247e+00]	 2.1122909e-01


.. parsed-literal::

      81	 1.4263047e+00	 1.2682659e-01	 1.4781270e+00	 1.2771794e-01	  1.3387136e+00 	 2.0267081e-01


.. parsed-literal::

      82	 1.4324119e+00	 1.2650492e-01	 1.4844056e+00	 1.2717409e-01	[ 1.3419109e+00]	 2.1785069e-01
      83	 1.4361478e+00	 1.2639772e-01	 1.4881078e+00	 1.2727174e-01	  1.3391262e+00 	 1.8275547e-01


.. parsed-literal::

      84	 1.4387008e+00	 1.2624835e-01	 1.4905891e+00	 1.2717475e-01	  1.3411626e+00 	 2.0647025e-01


.. parsed-literal::

      85	 1.4426649e+00	 1.2608599e-01	 1.4946295e+00	 1.2690561e-01	  1.3417270e+00 	 2.0428729e-01


.. parsed-literal::

      86	 1.4464799e+00	 1.2591390e-01	 1.4986344e+00	 1.2660476e-01	  1.3378205e+00 	 2.2501254e-01
      87	 1.4506422e+00	 1.2569637e-01	 1.5030858e+00	 1.2613285e-01	  1.3392540e+00 	 1.8435788e-01


.. parsed-literal::

      88	 1.4546649e+00	 1.2551904e-01	 1.5071480e+00	 1.2573801e-01	  1.3409526e+00 	 1.9875550e-01
      89	 1.4578094e+00	 1.2526488e-01	 1.5102368e+00	 1.2555039e-01	[ 1.3440485e+00]	 1.8992686e-01


.. parsed-literal::

      90	 1.4610069e+00	 1.2503485e-01	 1.5134040e+00	 1.2519071e-01	[ 1.3492927e+00]	 2.1250129e-01


.. parsed-literal::

      91	 1.4658753e+00	 1.2454942e-01	 1.5182547e+00	 1.2447446e-01	[ 1.3538846e+00]	 2.1049714e-01
      92	 1.4688316e+00	 1.2445675e-01	 1.5212367e+00	 1.2420643e-01	  1.3533513e+00 	 1.9843388e-01


.. parsed-literal::

      93	 1.4710825e+00	 1.2440816e-01	 1.5234078e+00	 1.2404910e-01	[ 1.3581967e+00]	 2.1527696e-01
      94	 1.4732207e+00	 1.2437466e-01	 1.5255286e+00	 1.2397291e-01	[ 1.3599429e+00]	 1.9349456e-01


.. parsed-literal::

      95	 1.4763500e+00	 1.2425018e-01	 1.5287153e+00	 1.2387839e-01	[ 1.3619856e+00]	 1.9757485e-01
      96	 1.4780894e+00	 1.2414357e-01	 1.5307146e+00	 1.2377172e-01	  1.3550531e+00 	 1.8665814e-01


.. parsed-literal::

      97	 1.4825620e+00	 1.2401500e-01	 1.5351249e+00	 1.2364978e-01	  1.3619793e+00 	 1.7260790e-01


.. parsed-literal::

      98	 1.4841790e+00	 1.2390447e-01	 1.5367234e+00	 1.2353633e-01	[ 1.3637029e+00]	 2.1850491e-01
      99	 1.4867110e+00	 1.2375998e-01	 1.5393225e+00	 1.2327364e-01	[ 1.3645384e+00]	 1.7920446e-01


.. parsed-literal::

     100	 1.4887632e+00	 1.2356075e-01	 1.5415098e+00	 1.2280190e-01	[ 1.3651007e+00]	 2.0579910e-01


.. parsed-literal::

     101	 1.4914272e+00	 1.2351849e-01	 1.5441671e+00	 1.2262373e-01	[ 1.3671202e+00]	 2.1672201e-01


.. parsed-literal::

     102	 1.4933769e+00	 1.2348152e-01	 1.5461408e+00	 1.2246350e-01	[ 1.3681920e+00]	 2.2116733e-01


.. parsed-literal::

     103	 1.4953085e+00	 1.2341461e-01	 1.5481187e+00	 1.2228983e-01	[ 1.3688052e+00]	 2.1914601e-01


.. parsed-literal::

     104	 1.4983887e+00	 1.2311791e-01	 1.5513702e+00	 1.2199669e-01	  1.3667920e+00 	 2.0696521e-01


.. parsed-literal::

     105	 1.5010081e+00	 1.2300893e-01	 1.5540842e+00	 1.2162709e-01	  1.3659516e+00 	 2.1382594e-01


.. parsed-literal::

     106	 1.5027155e+00	 1.2284428e-01	 1.5556953e+00	 1.2164514e-01	  1.3682382e+00 	 2.3289609e-01


.. parsed-literal::

     107	 1.5050257e+00	 1.2258118e-01	 1.5579421e+00	 1.2159138e-01	  1.3656715e+00 	 2.0797038e-01


.. parsed-literal::

     108	 1.5070505e+00	 1.2226827e-01	 1.5599434e+00	 1.2140962e-01	  1.3651478e+00 	 2.0951200e-01


.. parsed-literal::

     109	 1.5091443e+00	 1.2214855e-01	 1.5620084e+00	 1.2134775e-01	  1.3603052e+00 	 2.2337961e-01


.. parsed-literal::

     110	 1.5114840e+00	 1.2204228e-01	 1.5643601e+00	 1.2122085e-01	  1.3554776e+00 	 2.0820951e-01


.. parsed-literal::

     111	 1.5132446e+00	 1.2187875e-01	 1.5661426e+00	 1.2114659e-01	  1.3530302e+00 	 2.1941733e-01


.. parsed-literal::

     112	 1.5153379e+00	 1.2160766e-01	 1.5682862e+00	 1.2110191e-01	  1.3540382e+00 	 2.2090578e-01


.. parsed-literal::

     113	 1.5171953e+00	 1.2136485e-01	 1.5702076e+00	 1.2109223e-01	  1.3532449e+00 	 2.1441317e-01


.. parsed-literal::

     114	 1.5186991e+00	 1.2116894e-01	 1.5717080e+00	 1.2113258e-01	  1.3549102e+00 	 2.1038437e-01


.. parsed-literal::

     115	 1.5211006e+00	 1.2089971e-01	 1.5741463e+00	 1.2103154e-01	  1.3546948e+00 	 2.2599673e-01


.. parsed-literal::

     116	 1.5230042e+00	 1.2068672e-01	 1.5761113e+00	 1.2111542e-01	  1.3509972e+00 	 2.1196628e-01


.. parsed-literal::

     117	 1.5244713e+00	 1.2068524e-01	 1.5775458e+00	 1.2101223e-01	  1.3510357e+00 	 2.1577573e-01
     118	 1.5263392e+00	 1.2068435e-01	 1.5793998e+00	 1.2091726e-01	  1.3495260e+00 	 1.9772792e-01


.. parsed-literal::

     119	 1.5279608e+00	 1.2058518e-01	 1.5810826e+00	 1.2099476e-01	  1.3440481e+00 	 2.1363878e-01


.. parsed-literal::

     120	 1.5299310e+00	 1.2050311e-01	 1.5830942e+00	 1.2099727e-01	  1.3434452e+00 	 2.0831561e-01


.. parsed-literal::

     121	 1.5315012e+00	 1.2034648e-01	 1.5847201e+00	 1.2108183e-01	  1.3415193e+00 	 2.1097255e-01


.. parsed-literal::

     122	 1.5330620e+00	 1.2021555e-01	 1.5863497e+00	 1.2112580e-01	  1.3435331e+00 	 2.1020675e-01


.. parsed-literal::

     123	 1.5345486e+00	 1.2016517e-01	 1.5878568e+00	 1.2112085e-01	  1.3355528e+00 	 2.2516775e-01


.. parsed-literal::

     124	 1.5358085e+00	 1.2021028e-01	 1.5890709e+00	 1.2097621e-01	  1.3355534e+00 	 2.2258997e-01


.. parsed-literal::

     125	 1.5366101e+00	 1.2044290e-01	 1.5898484e+00	 1.2066229e-01	  1.3301080e+00 	 2.1369600e-01


.. parsed-literal::

     126	 1.5381704e+00	 1.2041159e-01	 1.5913652e+00	 1.2068154e-01	  1.3329669e+00 	 2.0414805e-01


.. parsed-literal::

     127	 1.5387134e+00	 1.2037935e-01	 1.5919052e+00	 1.2070663e-01	  1.3332529e+00 	 2.3149300e-01


.. parsed-literal::

     128	 1.5404663e+00	 1.2029190e-01	 1.5936763e+00	 1.2076141e-01	  1.3307177e+00 	 2.1186757e-01


.. parsed-literal::

     129	 1.5422428e+00	 1.2022338e-01	 1.5954681e+00	 1.2080969e-01	  1.3284523e+00 	 2.3574209e-01


.. parsed-literal::

     130	 1.5440999e+00	 1.2026910e-01	 1.5974722e+00	 1.2101471e-01	  1.3153986e+00 	 2.2494078e-01


.. parsed-literal::

     131	 1.5469104e+00	 1.2024413e-01	 1.6002102e+00	 1.2105359e-01	  1.3226758e+00 	 2.0886493e-01


.. parsed-literal::

     132	 1.5480294e+00	 1.2016273e-01	 1.6012906e+00	 1.2102059e-01	  1.3259794e+00 	 2.1529961e-01


.. parsed-literal::

     133	 1.5490531e+00	 1.2016572e-01	 1.6023138e+00	 1.2100749e-01	  1.3257142e+00 	 2.2829223e-01


.. parsed-literal::

     134	 1.5502467e+00	 1.2001737e-01	 1.6036125e+00	 1.2127130e-01	  1.3146997e+00 	 2.1861482e-01


.. parsed-literal::

     135	 1.5516769e+00	 1.1994076e-01	 1.6050635e+00	 1.2135924e-01	  1.3124478e+00 	 2.2165775e-01


.. parsed-literal::

     136	 1.5530845e+00	 1.1977078e-01	 1.6065314e+00	 1.2156228e-01	  1.3039321e+00 	 2.1591473e-01


.. parsed-literal::

     137	 1.5541119e+00	 1.1966953e-01	 1.6075801e+00	 1.2167140e-01	  1.3014278e+00 	 2.2343087e-01


.. parsed-literal::

     138	 1.5560171e+00	 1.1932757e-01	 1.6095785e+00	 1.2199507e-01	  1.2877316e+00 	 2.0853424e-01


.. parsed-literal::

     139	 1.5574224e+00	 1.1932544e-01	 1.6109646e+00	 1.2196858e-01	  1.2910938e+00 	 2.1422887e-01


.. parsed-literal::

     140	 1.5582370e+00	 1.1931929e-01	 1.6117494e+00	 1.2188219e-01	  1.2949011e+00 	 2.2424316e-01


.. parsed-literal::

     141	 1.5593068e+00	 1.1930380e-01	 1.6128080e+00	 1.2182519e-01	  1.2936617e+00 	 2.0956373e-01


.. parsed-literal::

     142	 1.5606302e+00	 1.1921170e-01	 1.6141786e+00	 1.2191361e-01	  1.2872015e+00 	 2.1031547e-01


.. parsed-literal::

     143	 1.5618952e+00	 1.1918191e-01	 1.6154693e+00	 1.2201393e-01	  1.2794761e+00 	 2.1176696e-01
     144	 1.5629918e+00	 1.1915039e-01	 1.6166088e+00	 1.2219483e-01	  1.2683018e+00 	 1.8553734e-01


.. parsed-literal::

     145	 1.5641135e+00	 1.1916606e-01	 1.6177504e+00	 1.2236026e-01	  1.2605282e+00 	 1.7502880e-01


.. parsed-literal::

     146	 1.5656509e+00	 1.1915359e-01	 1.6193439e+00	 1.2270831e-01	  1.2405047e+00 	 2.0625186e-01


.. parsed-literal::

     147	 1.5669722e+00	 1.1927488e-01	 1.6206345e+00	 1.2276664e-01	  1.2406800e+00 	 2.1667576e-01


.. parsed-literal::

     148	 1.5682609e+00	 1.1924598e-01	 1.6218884e+00	 1.2280689e-01	  1.2392151e+00 	 2.2653818e-01


.. parsed-literal::

     149	 1.5696412e+00	 1.1922077e-01	 1.6232389e+00	 1.2282091e-01	  1.2394089e+00 	 2.1526480e-01


.. parsed-literal::

     150	 1.5708162e+00	 1.1921804e-01	 1.6244155e+00	 1.2293840e-01	  1.2329818e+00 	 2.2196746e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 1min 9s, sys: 57.9 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fdf8d3cda50>



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
    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 771 ms, sys: 968 ms, total: 1.74 s
    Wall time: 535 ms


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_16_1.png


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_19_1.png

