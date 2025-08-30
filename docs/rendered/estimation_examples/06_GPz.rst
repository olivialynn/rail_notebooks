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
       1	-3.3433688e-01	 3.1801048e-01	-3.2470264e-01	 3.3073890e-01	[-3.4966839e-01]	 4.6461940e-01


.. parsed-literal::

       2	-2.6715537e-01	 3.0868382e-01	-2.4469127e-01	 3.1963735e-01	[-2.7858264e-01]	 2.3127961e-01


.. parsed-literal::

       3	-2.2159983e-01	 2.8701819e-01	-1.7980009e-01	 2.9287432e-01	[-2.0576959e-01]	 2.8575039e-01
       4	-1.8515398e-01	 2.6411620e-01	-1.4440905e-01	 2.6890428e-01	[-1.7038860e-01]	 1.9976568e-01


.. parsed-literal::

       5	-9.4362968e-02	 2.5539351e-01	-6.1943092e-02	 2.6037644e-01	[-8.5254257e-02]	 2.1862221e-01


.. parsed-literal::

       6	-6.4407482e-02	 2.5055447e-01	-3.5851241e-02	 2.5674166e-01	[-5.8120473e-02]	 2.0768976e-01


.. parsed-literal::

       7	-4.6497524e-02	 2.4771267e-01	-2.3911627e-02	 2.5429866e-01	[-4.9237437e-02]	 2.1887112e-01
       8	-3.4538413e-02	 2.4565789e-01	-1.5291532e-02	 2.5200347e-01	[-4.0249574e-02]	 1.9684792e-01


.. parsed-literal::

       9	-2.2163175e-02	 2.4331648e-01	-5.2689582e-03	 2.4934369e-01	[-2.9517805e-02]	 2.1699238e-01


.. parsed-literal::

      10	-1.1182590e-02	 2.4113402e-01	 4.2179237e-03	 2.4710687e-01	[-1.9285567e-02]	 2.1282125e-01


.. parsed-literal::

      11	-8.5851959e-03	 2.4058803e-01	 5.4703079e-03	 2.4705861e-01	 -2.2450748e-02 	 2.0688701e-01


.. parsed-literal::

      12	-4.3796816e-03	 2.4006242e-01	 9.5529841e-03	 2.4624907e-01	[-1.4836847e-02]	 2.1560049e-01


.. parsed-literal::

      13	-2.3328828e-03	 2.3962204e-01	 1.1574475e-02	 2.4552550e-01	[-1.1535563e-02]	 2.0646453e-01
      14	 1.6911917e-03	 2.3882549e-01	 1.5797541e-02	 2.4436642e-01	[-5.5374804e-03]	 1.9901752e-01


.. parsed-literal::

      15	 1.0072965e-01	 2.2562200e-01	 1.2033057e-01	 2.3087239e-01	[ 1.0542958e-01]	 3.1560755e-01


.. parsed-literal::

      16	 1.0972719e-01	 2.2376785e-01	 1.3271382e-01	 2.3063392e-01	[ 1.3230433e-01]	 2.0765781e-01
      17	 2.1745322e-01	 2.1854078e-01	 2.4243350e-01	 2.2418531e-01	[ 2.4355618e-01]	 1.7969155e-01


.. parsed-literal::

      18	 2.8419134e-01	 2.1560338e-01	 3.1183980e-01	 2.1925672e-01	[ 3.2549817e-01]	 2.9399323e-01


.. parsed-literal::

      19	 3.2038324e-01	 2.1321732e-01	 3.4998762e-01	 2.1909860e-01	[ 3.5927614e-01]	 2.1136189e-01


.. parsed-literal::

      20	 3.5638522e-01	 2.1254707e-01	 3.8817725e-01	 2.1867867e-01	[ 3.9964056e-01]	 2.0967579e-01


.. parsed-literal::

      21	 4.2109864e-01	 2.1210043e-01	 4.5524111e-01	 2.1459462e-01	[ 4.8013482e-01]	 2.1798873e-01


.. parsed-literal::

      22	 4.9238495e-01	 2.1599363e-01	 5.2762383e-01	 2.1587409e-01	[ 5.5294670e-01]	 2.1031427e-01


.. parsed-literal::

      23	 5.4712373e-01	 2.1281275e-01	 5.8452448e-01	 2.1285730e-01	[ 6.1587531e-01]	 2.0929265e-01


.. parsed-literal::

      24	 5.8105294e-01	 2.0836242e-01	 6.2026766e-01	 2.0823910e-01	[ 6.5271127e-01]	 2.0251894e-01


.. parsed-literal::

      25	 6.2464016e-01	 2.0495483e-01	 6.6403996e-01	 2.0598656e-01	[ 6.9036234e-01]	 2.1156573e-01


.. parsed-literal::

      26	 6.6495951e-01	 2.0336210e-01	 7.0258305e-01	 2.0766212e-01	[ 7.2332025e-01]	 2.1115756e-01


.. parsed-literal::

      27	 6.8198535e-01	 2.0360615e-01	 7.1900856e-01	 2.0822781e-01	[ 7.3418748e-01]	 2.0620823e-01


.. parsed-literal::

      28	 7.0027290e-01	 2.0266199e-01	 7.3812691e-01	 2.0720107e-01	[ 7.5267858e-01]	 2.0804191e-01


.. parsed-literal::

      29	 7.2519631e-01	 2.0281077e-01	 7.6454212e-01	 2.0683332e-01	[ 7.7491825e-01]	 2.1163416e-01
      30	 7.5283269e-01	 2.0434374e-01	 7.9282152e-01	 2.0787317e-01	[ 7.9126731e-01]	 1.7361093e-01


.. parsed-literal::

      31	 7.8548717e-01	 2.0638439e-01	 8.2600867e-01	 2.1264887e-01	[ 8.0777084e-01]	 2.0563507e-01


.. parsed-literal::

      32	 8.1092102e-01	 2.0532552e-01	 8.5091855e-01	 2.1053772e-01	[ 8.1574106e-01]	 2.1058297e-01
      33	 8.2846699e-01	 2.0490537e-01	 8.6970742e-01	 2.1057191e-01	[ 8.3724958e-01]	 1.7083049e-01


.. parsed-literal::

      34	 8.4853921e-01	 2.0244781e-01	 8.8994001e-01	 2.0740081e-01	[ 8.4863769e-01]	 2.1321678e-01


.. parsed-literal::

      35	 8.7480777e-01	 2.0015833e-01	 9.1697654e-01	 2.0342834e-01	[ 8.6516688e-01]	 2.1164370e-01


.. parsed-literal::

      36	 8.9423516e-01	 1.9793038e-01	 9.3693810e-01	 1.9945212e-01	[ 8.7979244e-01]	 2.0931578e-01
      37	 9.1934962e-01	 1.9668737e-01	 9.6283576e-01	 1.9722473e-01	[ 9.0532234e-01]	 1.6998386e-01


.. parsed-literal::

      38	 9.3521091e-01	 1.9531347e-01	 9.7939812e-01	 1.9476841e-01	[ 9.1762151e-01]	 2.0866513e-01


.. parsed-literal::

      39	 9.5610873e-01	 1.9118835e-01	 1.0009881e+00	 1.9081566e-01	[ 9.4472143e-01]	 2.1222186e-01


.. parsed-literal::

      40	 9.7799581e-01	 1.8587259e-01	 1.0235316e+00	 1.8564854e-01	[ 9.6672004e-01]	 2.0782876e-01


.. parsed-literal::

      41	 9.9564231e-01	 1.8037533e-01	 1.0424030e+00	 1.7894477e-01	[ 9.8926799e-01]	 2.0951366e-01
      42	 1.0108232e+00	 1.7790664e-01	 1.0575055e+00	 1.7704409e-01	[ 1.0072659e+00]	 1.6439533e-01


.. parsed-literal::

      43	 1.0304660e+00	 1.7456603e-01	 1.0771971e+00	 1.7358028e-01	[ 1.0236744e+00]	 1.7667174e-01


.. parsed-literal::

      44	 1.0443035e+00	 1.7335461e-01	 1.0917987e+00	 1.7330412e-01	[ 1.0336653e+00]	 2.1270967e-01


.. parsed-literal::

      45	 1.0567467e+00	 1.7124576e-01	 1.1042734e+00	 1.7124901e-01	[ 1.0436213e+00]	 2.0292974e-01


.. parsed-literal::

      46	 1.0676427e+00	 1.7037146e-01	 1.1149493e+00	 1.7101036e-01	[ 1.0477994e+00]	 2.0385623e-01
      47	 1.0836897e+00	 1.6685736e-01	 1.1313898e+00	 1.6855857e-01	[ 1.0728977e+00]	 1.8582916e-01


.. parsed-literal::

      48	 1.1062881e+00	 1.6051159e-01	 1.1539531e+00	 1.6539240e-01	[ 1.0980807e+00]	 2.1837425e-01


.. parsed-literal::

      49	 1.1192347e+00	 1.6028848e-01	 1.1675222e+00	 1.6704724e-01	[ 1.1107974e+00]	 2.0565224e-01


.. parsed-literal::

      50	 1.1313481e+00	 1.5691975e-01	 1.1798012e+00	 1.6340461e-01	[ 1.1171006e+00]	 2.1038818e-01
      51	 1.1454301e+00	 1.5534070e-01	 1.1939905e+00	 1.6077933e-01	  1.1165135e+00 	 2.0817256e-01


.. parsed-literal::

      52	 1.1587929e+00	 1.5280951e-01	 1.2078140e+00	 1.5765883e-01	  1.1153005e+00 	 2.0967174e-01
      53	 1.1743476e+00	 1.5155000e-01	 1.2242889e+00	 1.5513044e-01	  1.1062755e+00 	 1.9026852e-01


.. parsed-literal::

      54	 1.1869230e+00	 1.5038048e-01	 1.2368358e+00	 1.5384272e-01	  1.1129606e+00 	 2.0952559e-01


.. parsed-literal::

      55	 1.2006441e+00	 1.4997650e-01	 1.2508385e+00	 1.5319601e-01	[ 1.1175434e+00]	 2.1313596e-01


.. parsed-literal::

      56	 1.2133358e+00	 1.4838676e-01	 1.2635377e+00	 1.5082016e-01	[ 1.1310519e+00]	 2.1407223e-01


.. parsed-literal::

      57	 1.2259177e+00	 1.4681546e-01	 1.2760284e+00	 1.4957554e-01	[ 1.1613256e+00]	 2.1141982e-01


.. parsed-literal::

      58	 1.2350131e+00	 1.4563121e-01	 1.2852372e+00	 1.4837688e-01	[ 1.1755350e+00]	 2.0869875e-01


.. parsed-literal::

      59	 1.2479426e+00	 1.4141195e-01	 1.2991579e+00	 1.4457937e-01	[ 1.1840773e+00]	 2.0968914e-01


.. parsed-literal::

      60	 1.2561472e+00	 1.4049059e-01	 1.3069583e+00	 1.4360621e-01	[ 1.2121967e+00]	 2.0952559e-01


.. parsed-literal::

      61	 1.2648469e+00	 1.4009191e-01	 1.3152920e+00	 1.4306683e-01	  1.2121500e+00 	 2.0858765e-01
      62	 1.2727809e+00	 1.3911102e-01	 1.3235347e+00	 1.4234706e-01	  1.2070528e+00 	 1.9696808e-01


.. parsed-literal::

      63	 1.2814670e+00	 1.3767631e-01	 1.3322778e+00	 1.4146355e-01	  1.2078603e+00 	 2.0892382e-01
      64	 1.2889708e+00	 1.3704988e-01	 1.3401388e+00	 1.4188929e-01	[ 1.2170943e+00]	 1.8905091e-01


.. parsed-literal::

      65	 1.2973898e+00	 1.3637739e-01	 1.3483987e+00	 1.4141776e-01	[ 1.2289115e+00]	 2.0775199e-01


.. parsed-literal::

      66	 1.3022624e+00	 1.3588715e-01	 1.3533270e+00	 1.4092280e-01	[ 1.2350118e+00]	 2.1081638e-01


.. parsed-literal::

      67	 1.3131012e+00	 1.3490696e-01	 1.3644901e+00	 1.4059529e-01	[ 1.2427270e+00]	 2.0870781e-01


.. parsed-literal::

      68	 1.3218774e+00	 1.3239159e-01	 1.3736789e+00	 1.3913500e-01	  1.2411484e+00 	 2.0567727e-01


.. parsed-literal::

      69	 1.3319496e+00	 1.3248943e-01	 1.3835804e+00	 1.4041920e-01	[ 1.2467510e+00]	 2.0973825e-01


.. parsed-literal::

      70	 1.3389690e+00	 1.3229770e-01	 1.3905302e+00	 1.4081986e-01	[ 1.2513608e+00]	 2.1301818e-01
      71	 1.3481169e+00	 1.3227039e-01	 1.4001660e+00	 1.4296836e-01	  1.2463967e+00 	 1.9605041e-01


.. parsed-literal::

      72	 1.3551787e+00	 1.3098417e-01	 1.4074385e+00	 1.4133712e-01	[ 1.2619611e+00]	 2.0896339e-01


.. parsed-literal::

      73	 1.3617445e+00	 1.3070808e-01	 1.4138224e+00	 1.4174716e-01	  1.2615217e+00 	 2.0299554e-01


.. parsed-literal::

      74	 1.3674439e+00	 1.3028800e-01	 1.4194788e+00	 1.4185715e-01	  1.2613524e+00 	 2.1809506e-01


.. parsed-literal::

      75	 1.3748680e+00	 1.2937790e-01	 1.4269900e+00	 1.4036831e-01	  1.2605499e+00 	 2.1383691e-01


.. parsed-literal::

      76	 1.3823849e+00	 1.2882526e-01	 1.4346037e+00	 1.3965608e-01	  1.2568406e+00 	 2.0677447e-01


.. parsed-literal::

      77	 1.3879009e+00	 1.2859793e-01	 1.4401372e+00	 1.3900926e-01	  1.2570387e+00 	 2.0380235e-01
      78	 1.3952405e+00	 1.2858377e-01	 1.4478199e+00	 1.3834400e-01	  1.2524492e+00 	 2.0610285e-01


.. parsed-literal::

      79	 1.3998611e+00	 1.2838224e-01	 1.4524821e+00	 1.3780472e-01	  1.2556457e+00 	 2.0388007e-01
      80	 1.4048589e+00	 1.2824378e-01	 1.4574219e+00	 1.3704334e-01	[ 1.2655689e+00]	 1.8008876e-01


.. parsed-literal::

      81	 1.4091381e+00	 1.2810144e-01	 1.4620108e+00	 1.3803998e-01	[ 1.2678513e+00]	 2.1674299e-01


.. parsed-literal::

      82	 1.4126033e+00	 1.2788718e-01	 1.4655190e+00	 1.3772595e-01	[ 1.2702077e+00]	 2.0177245e-01


.. parsed-literal::

      83	 1.4170182e+00	 1.2763273e-01	 1.4701627e+00	 1.3742421e-01	[ 1.2719326e+00]	 2.1241117e-01


.. parsed-literal::

      84	 1.4219396e+00	 1.2729589e-01	 1.4752515e+00	 1.3724462e-01	[ 1.2788769e+00]	 2.0504093e-01
      85	 1.4256862e+00	 1.2739649e-01	 1.4792297e+00	 1.3802064e-01	[ 1.2849466e+00]	 1.9900131e-01


.. parsed-literal::

      86	 1.4299532e+00	 1.2727592e-01	 1.4833389e+00	 1.3787899e-01	[ 1.2917235e+00]	 2.0465899e-01
      87	 1.4347649e+00	 1.2737735e-01	 1.4881369e+00	 1.3807423e-01	[ 1.2991139e+00]	 1.9051695e-01


.. parsed-literal::

      88	 1.4394583e+00	 1.2750501e-01	 1.4927957e+00	 1.3782842e-01	[ 1.3049391e+00]	 2.1405172e-01


.. parsed-literal::

      89	 1.4425997e+00	 1.2765649e-01	 1.4959645e+00	 1.3803957e-01	[ 1.3077686e+00]	 2.9251361e-01
      90	 1.4461291e+00	 1.2779744e-01	 1.4994847e+00	 1.3751879e-01	[ 1.3110637e+00]	 1.8177509e-01


.. parsed-literal::

      91	 1.4486483e+00	 1.2768958e-01	 1.5020317e+00	 1.3694840e-01	  1.3105092e+00 	 2.0832682e-01


.. parsed-literal::

      92	 1.4526135e+00	 1.2748235e-01	 1.5062029e+00	 1.3629331e-01	  1.3073273e+00 	 2.2021198e-01
      93	 1.4552948e+00	 1.2739490e-01	 1.5090947e+00	 1.3572379e-01	  1.2992335e+00 	 2.0290279e-01


.. parsed-literal::

      94	 1.4583730e+00	 1.2711733e-01	 1.5121387e+00	 1.3579879e-01	  1.3025487e+00 	 1.7712975e-01


.. parsed-literal::

      95	 1.4608975e+00	 1.2721273e-01	 1.5146458e+00	 1.3634206e-01	  1.3062822e+00 	 2.0716262e-01


.. parsed-literal::

      96	 1.4635079e+00	 1.2713940e-01	 1.5172692e+00	 1.3650330e-01	  1.3072334e+00 	 2.1285486e-01


.. parsed-literal::

      97	 1.4687755e+00	 1.2699042e-01	 1.5226803e+00	 1.3687602e-01	  1.3069923e+00 	 2.1272302e-01


.. parsed-literal::

      98	 1.4705695e+00	 1.2722877e-01	 1.5246680e+00	 1.3696526e-01	  1.3061661e+00 	 2.0942783e-01


.. parsed-literal::

      99	 1.4742952e+00	 1.2685781e-01	 1.5283573e+00	 1.3670931e-01	  1.3091457e+00 	 2.0933557e-01


.. parsed-literal::

     100	 1.4758876e+00	 1.2672213e-01	 1.5299960e+00	 1.3641636e-01	  1.3088232e+00 	 2.0210791e-01


.. parsed-literal::

     101	 1.4785344e+00	 1.2657788e-01	 1.5327342e+00	 1.3607137e-01	  1.3076617e+00 	 2.1082997e-01


.. parsed-literal::

     102	 1.4827346e+00	 1.2647138e-01	 1.5370190e+00	 1.3559278e-01	  1.3033066e+00 	 2.1463943e-01


.. parsed-literal::

     103	 1.4842648e+00	 1.2623906e-01	 1.5388176e+00	 1.3551479e-01	  1.2838665e+00 	 2.1682620e-01


.. parsed-literal::

     104	 1.4880829e+00	 1.2622060e-01	 1.5423967e+00	 1.3538515e-01	  1.2928444e+00 	 2.1942949e-01
     105	 1.4894628e+00	 1.2622876e-01	 1.5437436e+00	 1.3554508e-01	  1.2929931e+00 	 1.7971420e-01


.. parsed-literal::

     106	 1.4920809e+00	 1.2604742e-01	 1.5464540e+00	 1.3577818e-01	  1.2842940e+00 	 1.6985035e-01
     107	 1.4936187e+00	 1.2588408e-01	 1.5482216e+00	 1.3620672e-01	  1.2706144e+00 	 2.0806432e-01


.. parsed-literal::

     108	 1.4957756e+00	 1.2582290e-01	 1.5502695e+00	 1.3619743e-01	  1.2701242e+00 	 1.8357778e-01
     109	 1.4969955e+00	 1.2576304e-01	 1.5514548e+00	 1.3626262e-01	  1.2676593e+00 	 1.8286586e-01


.. parsed-literal::

     110	 1.4986228e+00	 1.2569522e-01	 1.5530423e+00	 1.3646395e-01	  1.2653196e+00 	 2.1081018e-01


.. parsed-literal::

     111	 1.5015337e+00	 1.2569929e-01	 1.5559850e+00	 1.3700814e-01	  1.2589394e+00 	 2.2256684e-01


.. parsed-literal::

     112	 1.5025370e+00	 1.2545323e-01	 1.5572507e+00	 1.3761123e-01	  1.2443651e+00 	 2.2075677e-01
     113	 1.5058451e+00	 1.2548101e-01	 1.5604519e+00	 1.3751070e-01	  1.2488953e+00 	 1.9523764e-01


.. parsed-literal::

     114	 1.5072415e+00	 1.2542778e-01	 1.5618824e+00	 1.3738229e-01	  1.2492679e+00 	 2.1252632e-01


.. parsed-literal::

     115	 1.5092006e+00	 1.2526918e-01	 1.5640005e+00	 1.3733694e-01	  1.2431550e+00 	 2.0411563e-01


.. parsed-literal::

     116	 1.5119804e+00	 1.2503372e-01	 1.5669442e+00	 1.3742353e-01	  1.2332377e+00 	 2.1257973e-01


.. parsed-literal::

     117	 1.5136954e+00	 1.2481003e-01	 1.5687764e+00	 1.3739644e-01	  1.2242745e+00 	 3.2938290e-01


.. parsed-literal::

     118	 1.5158138e+00	 1.2473408e-01	 1.5708778e+00	 1.3755561e-01	  1.2154355e+00 	 2.1273375e-01


.. parsed-literal::

     119	 1.5176198e+00	 1.2474935e-01	 1.5725965e+00	 1.3768583e-01	  1.2102925e+00 	 2.1345925e-01


.. parsed-literal::

     120	 1.5198994e+00	 1.2484708e-01	 1.5748261e+00	 1.3778421e-01	  1.2006729e+00 	 2.1629500e-01


.. parsed-literal::

     121	 1.5224523e+00	 1.2498443e-01	 1.5774183e+00	 1.3775702e-01	  1.1869381e+00 	 2.0446110e-01
     122	 1.5246319e+00	 1.2500338e-01	 1.5796300e+00	 1.3774069e-01	  1.1814323e+00 	 1.9483352e-01


.. parsed-literal::

     123	 1.5267944e+00	 1.2501158e-01	 1.5819017e+00	 1.3772671e-01	  1.1728388e+00 	 2.0423484e-01
     124	 1.5280843e+00	 1.2487275e-01	 1.5833106e+00	 1.3778104e-01	  1.1654156e+00 	 1.9418645e-01


.. parsed-literal::

     125	 1.5297787e+00	 1.2481456e-01	 1.5849672e+00	 1.3781099e-01	  1.1629366e+00 	 1.8175316e-01


.. parsed-literal::

     126	 1.5323581e+00	 1.2470117e-01	 1.5874940e+00	 1.3797607e-01	  1.1492815e+00 	 2.1920681e-01
     127	 1.5339654e+00	 1.2459823e-01	 1.5890911e+00	 1.3796171e-01	  1.1424667e+00 	 1.9813752e-01


.. parsed-literal::

     128	 1.5364373e+00	 1.2437263e-01	 1.5918088e+00	 1.3794420e-01	  1.0991970e+00 	 2.0423865e-01


.. parsed-literal::

     129	 1.5393032e+00	 1.2421965e-01	 1.5947173e+00	 1.3792314e-01	  1.0936852e+00 	 2.1963000e-01
     130	 1.5406797e+00	 1.2418043e-01	 1.5960493e+00	 1.3784252e-01	  1.1016366e+00 	 1.9956470e-01


.. parsed-literal::

     131	 1.5422187e+00	 1.2413047e-01	 1.5976662e+00	 1.3790811e-01	  1.0874629e+00 	 2.0114660e-01


.. parsed-literal::

     132	 1.5440432e+00	 1.2398340e-01	 1.5996736e+00	 1.3803264e-01	  1.0591118e+00 	 2.0994711e-01


.. parsed-literal::

     133	 1.5456528e+00	 1.2390984e-01	 1.6012952e+00	 1.3828552e-01	  1.0302577e+00 	 2.1200562e-01


.. parsed-literal::

     134	 1.5468181e+00	 1.2381354e-01	 1.6024199e+00	 1.3829845e-01	  1.0269906e+00 	 2.1099353e-01
     135	 1.5487291e+00	 1.2365438e-01	 1.6042733e+00	 1.3833286e-01	  1.0209357e+00 	 2.0675468e-01


.. parsed-literal::

     136	 1.5506832e+00	 1.2357988e-01	 1.6062336e+00	 1.3822550e-01	  1.0077289e+00 	 2.0678663e-01


.. parsed-literal::

     137	 1.5528390e+00	 1.2342665e-01	 1.6083987e+00	 1.3830129e-01	  1.0053431e+00 	 2.0602775e-01


.. parsed-literal::

     138	 1.5542815e+00	 1.2340709e-01	 1.6099416e+00	 1.3833651e-01	  9.9430296e-01 	 2.1005607e-01
     139	 1.5558755e+00	 1.2333999e-01	 1.6116856e+00	 1.3834662e-01	  9.8188721e-01 	 1.8168426e-01


.. parsed-literal::

     140	 1.5572338e+00	 1.2333228e-01	 1.6132178e+00	 1.3848075e-01	  9.4813741e-01 	 2.1360064e-01


.. parsed-literal::

     141	 1.5587055e+00	 1.2322368e-01	 1.6146590e+00	 1.3854536e-01	  9.4611269e-01 	 2.1336269e-01


.. parsed-literal::

     142	 1.5597805e+00	 1.2310362e-01	 1.6156503e+00	 1.3859215e-01	  9.4462156e-01 	 2.0983958e-01
     143	 1.5608653e+00	 1.2302792e-01	 1.6166985e+00	 1.3870104e-01	  9.3431031e-01 	 1.8008804e-01


.. parsed-literal::

     144	 1.5625768e+00	 1.2283601e-01	 1.6183786e+00	 1.3900291e-01	  8.8171555e-01 	 2.0919108e-01
     145	 1.5637364e+00	 1.2276975e-01	 1.6195790e+00	 1.3882408e-01	  8.8150520e-01 	 1.7438459e-01


.. parsed-literal::

     146	 1.5645249e+00	 1.2280190e-01	 1.6203702e+00	 1.3881698e-01	  8.7723614e-01 	 2.1251464e-01


.. parsed-literal::

     147	 1.5654095e+00	 1.2279902e-01	 1.6213089e+00	 1.3878473e-01	  8.6439811e-01 	 2.0639873e-01


.. parsed-literal::

     148	 1.5666941e+00	 1.2274541e-01	 1.6226874e+00	 1.3872816e-01	  8.3967270e-01 	 2.1020532e-01
     149	 1.5683705e+00	 1.2261757e-01	 1.6244967e+00	 1.3858460e-01	  8.0219604e-01 	 1.8664980e-01


.. parsed-literal::

     150	 1.5696859e+00	 1.2244684e-01	 1.6259975e+00	 1.3850667e-01	  7.4635760e-01 	 2.0720148e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.1 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f288ca36a70>



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
    CPU times: user 2.11 s, sys: 52 ms, total: 2.16 s
    Wall time: 653 ms


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

