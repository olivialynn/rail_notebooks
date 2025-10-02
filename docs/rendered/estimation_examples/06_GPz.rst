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
       1	-3.4012658e-01	 3.1928823e-01	-3.3034690e-01	 3.2567226e-01	[-3.4259177e-01]	 5.1097322e-01


.. parsed-literal::

       2	-2.6604642e-01	 3.0746776e-01	-2.4098383e-01	 3.1495039e-01	[-2.6443573e-01]	 2.5949907e-01


.. parsed-literal::

       3	-2.2164772e-01	 2.8761249e-01	-1.7984708e-01	 2.9726517e-01	[-2.1934196e-01]	 2.9910398e-01


.. parsed-literal::

       4	-1.9415814e-01	 2.6292902e-01	-1.5368947e-01	 2.7198554e-01	[-2.1422204e-01]	 2.0212483e-01
       5	-9.8842070e-02	 2.5507588e-01	-6.1588761e-02	 2.6714406e-01	[-1.2017529e-01]	 1.9578505e-01


.. parsed-literal::

       6	-6.3442841e-02	 2.4909986e-01	-2.9941426e-02	 2.6111874e-01	[-7.7049398e-02]	 1.9861770e-01
       7	-4.4311087e-02	 2.4637229e-01	-1.8558278e-02	 2.5903061e-01	[-7.0895883e-02]	 1.8882155e-01


.. parsed-literal::

       8	-3.2612297e-02	 2.4456400e-01	-1.0964754e-02	 2.5723164e-01	[-6.4357862e-02]	 1.9405985e-01
       9	-1.7794838e-02	 2.4192315e-01	 3.2712084e-04	 2.5436460e-01	[-5.2133871e-02]	 1.9551277e-01


.. parsed-literal::

      10	-8.1909503e-03	 2.4024526e-01	 7.1535404e-03	 2.5286388e-01	[-5.1405368e-02]	 2.0773721e-01
      11	-6.1990283e-04	 2.3894829e-01	 1.3737980e-02	 2.5127376e-01	[-3.9356614e-02]	 2.0487523e-01


.. parsed-literal::

      12	 1.3019780e-03	 2.3858645e-01	 1.5415280e-02	 2.5082277e-01	[-3.7290513e-02]	 2.0620155e-01


.. parsed-literal::

      13	 6.9001517e-03	 2.3755021e-01	 2.0889842e-02	 2.4963567e-01	[-3.2034634e-02]	 2.0345140e-01


.. parsed-literal::

      14	 1.8087758e-02	 2.3522895e-01	 3.3628574e-02	 2.4796909e-01	[-1.9193161e-02]	 2.0331287e-01
      15	 3.9689532e-02	 2.2345470e-01	 5.8845830e-02	 2.3641940e-01	[ 2.7020310e-02]	 1.9701934e-01


.. parsed-literal::

      16	 1.4646308e-01	 2.1945938e-01	 1.6855073e-01	 2.3369638e-01	[ 1.1992539e-01]	 2.0436382e-01


.. parsed-literal::

      17	 1.7632716e-01	 2.1947073e-01	 2.0761376e-01	 2.3417757e-01	  9.8275501e-02 	 2.1818018e-01


.. parsed-literal::

      18	 2.4610456e-01	 2.1718557e-01	 2.7595576e-01	 2.3470565e-01	[ 1.8772935e-01]	 2.0752835e-01


.. parsed-literal::

      19	 2.7580210e-01	 2.1363163e-01	 3.0617776e-01	 2.3131336e-01	[ 2.2926577e-01]	 2.1797872e-01


.. parsed-literal::

      20	 3.1321737e-01	 2.0974275e-01	 3.4443950e-01	 2.2630434e-01	[ 2.7419182e-01]	 2.0753837e-01
      21	 3.9214016e-01	 2.0437853e-01	 4.2534774e-01	 2.2214846e-01	[ 3.5086874e-01]	 2.0762277e-01


.. parsed-literal::

      22	 4.7045952e-01	 2.0466914e-01	 5.0430391e-01	 2.2206371e-01	[ 4.2671273e-01]	 1.8401027e-01
      23	 5.3463259e-01	 1.9945050e-01	 5.7046870e-01	 2.1768082e-01	[ 4.7934323e-01]	 1.8580675e-01


.. parsed-literal::

      24	 5.8619267e-01	 1.9561987e-01	 6.2465092e-01	 2.1309798e-01	[ 5.2075329e-01]	 2.1115899e-01


.. parsed-literal::

      25	 6.2996090e-01	 1.9484624e-01	 6.6950120e-01	 2.1128577e-01	[ 5.7585509e-01]	 2.1067619e-01


.. parsed-literal::

      26	 6.5977566e-01	 1.9160660e-01	 6.9891022e-01	 2.0917647e-01	[ 6.1362820e-01]	 2.1362257e-01
      27	 6.8111837e-01	 1.9008525e-01	 7.2034499e-01	 2.0782333e-01	[ 6.3302372e-01]	 1.9557691e-01


.. parsed-literal::

      28	 7.1587047e-01	 1.9415567e-01	 7.5354820e-01	 2.1108531e-01	[ 6.7547645e-01]	 1.8751764e-01


.. parsed-literal::

      29	 7.5025413e-01	 1.9150995e-01	 7.8816530e-01	 2.1102571e-01	[ 7.1286939e-01]	 2.0587468e-01


.. parsed-literal::

      30	 7.8054525e-01	 1.9188648e-01	 8.2007818e-01	 2.1395029e-01	[ 7.3609248e-01]	 2.0764661e-01


.. parsed-literal::

      31	 7.9965684e-01	 1.9981607e-01	 8.3950135e-01	 2.2086284e-01	[ 7.7330455e-01]	 2.0918751e-01
      32	 8.1534384e-01	 1.9775537e-01	 8.5663379e-01	 2.1982622e-01	[ 7.8818852e-01]	 2.0210004e-01


.. parsed-literal::

      33	 8.3118579e-01	 2.0055525e-01	 8.7155861e-01	 2.2220013e-01	[ 8.0971772e-01]	 1.8849063e-01


.. parsed-literal::

      34	 8.4775458e-01	 1.9754152e-01	 8.8883218e-01	 2.1887810e-01	[ 8.2350736e-01]	 2.0339394e-01


.. parsed-literal::

      35	 8.6601796e-01	 1.9586866e-01	 9.0780826e-01	 2.1718894e-01	[ 8.4245586e-01]	 2.1348190e-01


.. parsed-literal::

      36	 8.9544673e-01	 1.9114961e-01	 9.3869442e-01	 2.1216380e-01	[ 8.6476295e-01]	 2.2424459e-01


.. parsed-literal::

      37	 9.2056122e-01	 1.8751161e-01	 9.6466238e-01	 2.0781335e-01	[ 8.9346871e-01]	 2.1234560e-01
      38	 9.4235342e-01	 1.8214901e-01	 9.8670949e-01	 2.0326880e-01	[ 9.0752397e-01]	 1.9823790e-01


.. parsed-literal::

      39	 9.6276360e-01	 1.7564736e-01	 1.0080485e+00	 1.9684612e-01	[ 9.1245090e-01]	 2.0311475e-01


.. parsed-literal::

      40	 9.7209595e-01	 1.7417951e-01	 1.0166798e+00	 1.9482542e-01	[ 9.2487113e-01]	 2.1329904e-01


.. parsed-literal::

      41	 9.9211928e-01	 1.7111572e-01	 1.0368014e+00	 1.9295098e-01	[ 9.4088313e-01]	 2.0902419e-01
      42	 1.0046923e+00	 1.6967225e-01	 1.0500085e+00	 1.9165892e-01	[ 9.5080876e-01]	 1.7270517e-01


.. parsed-literal::

      43	 1.0259100e+00	 1.6786570e-01	 1.0726352e+00	 1.9026748e-01	[ 9.6649732e-01]	 1.9194126e-01


.. parsed-literal::

      44	 1.0348235e+00	 1.6768215e-01	 1.0827006e+00	 1.8914676e-01	[ 9.7283468e-01]	 2.0411015e-01


.. parsed-literal::

      45	 1.0494302e+00	 1.6698059e-01	 1.0966477e+00	 1.8933079e-01	[ 9.8729677e-01]	 2.0135593e-01


.. parsed-literal::

      46	 1.0574114e+00	 1.6548983e-01	 1.1047316e+00	 1.8814925e-01	[ 9.9306605e-01]	 2.0542479e-01


.. parsed-literal::

      47	 1.0665430e+00	 1.6370794e-01	 1.1143005e+00	 1.8693837e-01	[ 9.9544840e-01]	 2.0827985e-01


.. parsed-literal::

      48	 1.0749350e+00	 1.6289253e-01	 1.1235529e+00	 1.8530660e-01	[ 9.9905936e-01]	 2.0725703e-01


.. parsed-literal::

      49	 1.0844282e+00	 1.6181990e-01	 1.1329121e+00	 1.8466505e-01	[ 1.0039583e+00]	 2.0741725e-01
      50	 1.0919976e+00	 1.6116767e-01	 1.1403859e+00	 1.8421888e-01	[ 1.0096592e+00]	 1.9310260e-01


.. parsed-literal::

      51	 1.1007909e+00	 1.6000518e-01	 1.1490706e+00	 1.8297028e-01	[ 1.0187419e+00]	 2.0391989e-01


.. parsed-literal::

      52	 1.1137059e+00	 1.5620513e-01	 1.1619613e+00	 1.7906009e-01	[ 1.0378507e+00]	 2.1804953e-01


.. parsed-literal::

      53	 1.1261506e+00	 1.5387800e-01	 1.1742989e+00	 1.7518831e-01	[ 1.0535999e+00]	 2.0450473e-01


.. parsed-literal::

      54	 1.1337126e+00	 1.5271719e-01	 1.1816165e+00	 1.7414549e-01	[ 1.0640218e+00]	 2.0194483e-01


.. parsed-literal::

      55	 1.1465382e+00	 1.4937957e-01	 1.1948073e+00	 1.7083245e-01	[ 1.0792991e+00]	 2.0603609e-01


.. parsed-literal::

      56	 1.1565189e+00	 1.4807899e-01	 1.2049709e+00	 1.6974113e-01	[ 1.0852155e+00]	 2.0520616e-01


.. parsed-literal::

      57	 1.1669211e+00	 1.4604499e-01	 1.2155866e+00	 1.6792417e-01	[ 1.0960088e+00]	 2.0682430e-01
      58	 1.1805735e+00	 1.4279424e-01	 1.2299983e+00	 1.6559793e-01	[ 1.1052624e+00]	 1.8504119e-01


.. parsed-literal::

      59	 1.1839135e+00	 1.4182563e-01	 1.2338071e+00	 1.6555212e-01	  1.1052337e+00 	 1.9169140e-01
      60	 1.1922615e+00	 1.4153133e-01	 1.2417539e+00	 1.6521873e-01	[ 1.1150135e+00]	 1.9438362e-01


.. parsed-literal::

      61	 1.1968910e+00	 1.4078358e-01	 1.2464465e+00	 1.6445390e-01	[ 1.1183445e+00]	 2.0590854e-01
      62	 1.2027839e+00	 1.4000768e-01	 1.2526243e+00	 1.6385106e-01	  1.1178071e+00 	 2.0447350e-01


.. parsed-literal::

      63	 1.2104055e+00	 1.3860067e-01	 1.2606073e+00	 1.6227020e-01	  1.1155858e+00 	 1.9856501e-01


.. parsed-literal::

      64	 1.2169835e+00	 1.3768508e-01	 1.2674212e+00	 1.6123060e-01	  1.1126953e+00 	 2.1227646e-01
      65	 1.2232060e+00	 1.3775814e-01	 1.2735398e+00	 1.6078801e-01	[ 1.1208256e+00]	 1.9838572e-01


.. parsed-literal::

      66	 1.2310336e+00	 1.3672733e-01	 1.2814718e+00	 1.5962442e-01	[ 1.1299309e+00]	 2.1777248e-01


.. parsed-literal::

      67	 1.2389078e+00	 1.3590531e-01	 1.2895088e+00	 1.5845967e-01	[ 1.1410986e+00]	 2.1145797e-01


.. parsed-literal::

      68	 1.2469089e+00	 1.3491272e-01	 1.2975781e+00	 1.5767966e-01	[ 1.1490441e+00]	 2.0417833e-01


.. parsed-literal::

      69	 1.2571070e+00	 1.3370819e-01	 1.3079439e+00	 1.5636904e-01	[ 1.1595610e+00]	 2.1197939e-01
      70	 1.2634790e+00	 1.3358493e-01	 1.3144370e+00	 1.5604263e-01	[ 1.1625005e+00]	 1.9315338e-01


.. parsed-literal::

      71	 1.2714266e+00	 1.3374601e-01	 1.3220669e+00	 1.5606000e-01	[ 1.1756634e+00]	 2.0610285e-01


.. parsed-literal::

      72	 1.2772078e+00	 1.3327533e-01	 1.3279654e+00	 1.5554250e-01	[ 1.1786264e+00]	 2.2269750e-01
      73	 1.2843793e+00	 1.3292251e-01	 1.3353344e+00	 1.5542995e-01	[ 1.1827660e+00]	 1.9911289e-01


.. parsed-literal::

      74	 1.2910529e+00	 1.3238192e-01	 1.3424898e+00	 1.5545927e-01	  1.1788710e+00 	 1.9057488e-01
      75	 1.2993091e+00	 1.3176363e-01	 1.3506953e+00	 1.5540063e-01	[ 1.1841549e+00]	 1.9868660e-01


.. parsed-literal::

      76	 1.3050693e+00	 1.3160238e-01	 1.3564760e+00	 1.5555367e-01	[ 1.1859883e+00]	 1.9418097e-01


.. parsed-literal::

      77	 1.3123424e+00	 1.3141149e-01	 1.3639502e+00	 1.5555606e-01	[ 1.1865300e+00]	 2.0986176e-01


.. parsed-literal::

      78	 1.3166264e+00	 1.3130194e-01	 1.3684204e+00	 1.5597398e-01	[ 1.1899697e+00]	 2.0420289e-01


.. parsed-literal::

      79	 1.3238935e+00	 1.3123464e-01	 1.3755305e+00	 1.5586965e-01	[ 1.1968238e+00]	 2.0470381e-01


.. parsed-literal::

      80	 1.3289684e+00	 1.3075099e-01	 1.3805751e+00	 1.5537855e-01	[ 1.2001752e+00]	 2.1079969e-01


.. parsed-literal::

      81	 1.3340790e+00	 1.3055781e-01	 1.3857305e+00	 1.5518864e-01	[ 1.2031854e+00]	 2.1282935e-01


.. parsed-literal::

      82	 1.3394890e+00	 1.2943185e-01	 1.3914794e+00	 1.5423777e-01	  1.1989160e+00 	 2.0627570e-01


.. parsed-literal::

      83	 1.3456405e+00	 1.2958966e-01	 1.3976380e+00	 1.5455271e-01	[ 1.2055765e+00]	 2.0552659e-01


.. parsed-literal::

      84	 1.3494453e+00	 1.2957394e-01	 1.4015193e+00	 1.5468810e-01	[ 1.2058634e+00]	 2.1291423e-01
      85	 1.3546384e+00	 1.2943647e-01	 1.4068504e+00	 1.5468243e-01	[ 1.2062413e+00]	 1.9652390e-01


.. parsed-literal::

      86	 1.3603298e+00	 1.2901468e-01	 1.4129463e+00	 1.5399326e-01	[ 1.2100876e+00]	 2.0900607e-01
      87	 1.3673624e+00	 1.2839217e-01	 1.4199264e+00	 1.5352424e-01	[ 1.2150062e+00]	 1.9090676e-01


.. parsed-literal::

      88	 1.3707937e+00	 1.2805310e-01	 1.4233058e+00	 1.5316912e-01	[ 1.2219097e+00]	 1.9848275e-01


.. parsed-literal::

      89	 1.3752520e+00	 1.2756954e-01	 1.4278429e+00	 1.5258519e-01	[ 1.2260297e+00]	 2.1223187e-01


.. parsed-literal::

      90	 1.3811965e+00	 1.2686489e-01	 1.4340860e+00	 1.5182615e-01	[ 1.2346503e+00]	 2.1235800e-01


.. parsed-literal::

      91	 1.3870422e+00	 1.2634732e-01	 1.4400497e+00	 1.5118279e-01	[ 1.2361650e+00]	 2.1384001e-01
      92	 1.3917568e+00	 1.2613967e-01	 1.4447836e+00	 1.5097736e-01	[ 1.2398107e+00]	 1.9674301e-01


.. parsed-literal::

      93	 1.3972202e+00	 1.2586672e-01	 1.4501821e+00	 1.5064502e-01	[ 1.2445732e+00]	 2.1455002e-01


.. parsed-literal::

      94	 1.4011951e+00	 1.2569563e-01	 1.4541793e+00	 1.5028174e-01	  1.2438965e+00 	 2.0775747e-01


.. parsed-literal::

      95	 1.4048503e+00	 1.2518402e-01	 1.4578735e+00	 1.4959272e-01	[ 1.2456595e+00]	 2.1075273e-01


.. parsed-literal::

      96	 1.4079784e+00	 1.2487461e-01	 1.4610004e+00	 1.4907801e-01	[ 1.2490040e+00]	 2.0513535e-01
      97	 1.4117353e+00	 1.2453823e-01	 1.4648727e+00	 1.4845102e-01	[ 1.2510419e+00]	 1.9866395e-01


.. parsed-literal::

      98	 1.4169278e+00	 1.2410107e-01	 1.4703163e+00	 1.4790893e-01	[ 1.2566543e+00]	 2.0495129e-01


.. parsed-literal::

      99	 1.4211581e+00	 1.2392998e-01	 1.4746401e+00	 1.4740664e-01	[ 1.2591093e+00]	 2.0796895e-01


.. parsed-literal::

     100	 1.4242144e+00	 1.2391127e-01	 1.4776746e+00	 1.4753646e-01	  1.2578757e+00 	 2.2071886e-01
     101	 1.4285232e+00	 1.2387645e-01	 1.4820811e+00	 1.4776262e-01	  1.2504045e+00 	 2.0004201e-01


.. parsed-literal::

     102	 1.4321911e+00	 1.2366715e-01	 1.4859915e+00	 1.4775664e-01	  1.2387852e+00 	 1.9774318e-01


.. parsed-literal::

     103	 1.4366020e+00	 1.2355272e-01	 1.4904207e+00	 1.4754791e-01	  1.2330304e+00 	 2.1137667e-01
     104	 1.4400235e+00	 1.2328096e-01	 1.4938596e+00	 1.4697782e-01	  1.2309008e+00 	 1.9923878e-01


.. parsed-literal::

     105	 1.4430408e+00	 1.2308292e-01	 1.4968846e+00	 1.4666277e-01	  1.2334826e+00 	 1.9522738e-01


.. parsed-literal::

     106	 1.4473896e+00	 1.2269569e-01	 1.5013452e+00	 1.4626852e-01	  1.2283446e+00 	 2.0674920e-01


.. parsed-literal::

     107	 1.4509996e+00	 1.2258570e-01	 1.5049182e+00	 1.4640287e-01	  1.2398952e+00 	 2.1527743e-01
     108	 1.4536620e+00	 1.2250451e-01	 1.5075306e+00	 1.4646758e-01	  1.2421672e+00 	 1.9535041e-01


.. parsed-literal::

     109	 1.4570550e+00	 1.2234464e-01	 1.5110323e+00	 1.4654562e-01	  1.2396232e+00 	 2.0823908e-01


.. parsed-literal::

     110	 1.4597287e+00	 1.2204351e-01	 1.5138705e+00	 1.4626717e-01	  1.2371183e+00 	 2.1272326e-01


.. parsed-literal::

     111	 1.4628251e+00	 1.2188277e-01	 1.5169574e+00	 1.4616066e-01	  1.2399482e+00 	 2.1156025e-01


.. parsed-literal::

     112	 1.4657563e+00	 1.2171566e-01	 1.5199562e+00	 1.4590715e-01	  1.2380667e+00 	 2.0404077e-01


.. parsed-literal::

     113	 1.4676910e+00	 1.2164321e-01	 1.5219143e+00	 1.4590560e-01	  1.2380724e+00 	 2.0819616e-01
     114	 1.4702202e+00	 1.2167440e-01	 1.5244889e+00	 1.4577104e-01	  1.2365945e+00 	 1.8833303e-01


.. parsed-literal::

     115	 1.4735319e+00	 1.2158716e-01	 1.5279138e+00	 1.4573178e-01	  1.2322463e+00 	 2.0328474e-01
     116	 1.4758922e+00	 1.2150237e-01	 1.5303932e+00	 1.4553584e-01	  1.2285293e+00 	 1.9728780e-01


.. parsed-literal::

     117	 1.4785999e+00	 1.2131139e-01	 1.5330393e+00	 1.4562531e-01	  1.2304856e+00 	 2.1436810e-01
     118	 1.4821447e+00	 1.2103767e-01	 1.5366106e+00	 1.4588807e-01	  1.2282283e+00 	 1.9198751e-01


.. parsed-literal::

     119	 1.4841716e+00	 1.2097497e-01	 1.5386082e+00	 1.4611791e-01	  1.2266044e+00 	 2.1437478e-01


.. parsed-literal::

     120	 1.4876710e+00	 1.2092759e-01	 1.5422318e+00	 1.4688234e-01	  1.2091993e+00 	 2.0565104e-01


.. parsed-literal::

     121	 1.4903392e+00	 1.2091650e-01	 1.5449386e+00	 1.4735475e-01	  1.2005508e+00 	 2.0376658e-01


.. parsed-literal::

     122	 1.4923256e+00	 1.2086158e-01	 1.5469138e+00	 1.4729671e-01	  1.1981766e+00 	 2.1168113e-01


.. parsed-literal::

     123	 1.4949009e+00	 1.2073149e-01	 1.5496305e+00	 1.4731926e-01	  1.1944324e+00 	 2.1241331e-01


.. parsed-literal::

     124	 1.4961406e+00	 1.2073898e-01	 1.5509745e+00	 1.4754849e-01	  1.1882009e+00 	 2.1801686e-01


.. parsed-literal::

     125	 1.4977124e+00	 1.2065967e-01	 1.5525039e+00	 1.4754576e-01	  1.1932750e+00 	 2.1077299e-01
     126	 1.4996672e+00	 1.2057512e-01	 1.5544679e+00	 1.4773553e-01	  1.1984504e+00 	 2.0250130e-01


.. parsed-literal::

     127	 1.5010939e+00	 1.2055979e-01	 1.5558829e+00	 1.4784230e-01	  1.2001949e+00 	 2.1134496e-01


.. parsed-literal::

     128	 1.5028896e+00	 1.2051087e-01	 1.5578182e+00	 1.4789227e-01	  1.1966031e+00 	 2.1091318e-01


.. parsed-literal::

     129	 1.5055897e+00	 1.2050188e-01	 1.5604025e+00	 1.4802840e-01	  1.1955551e+00 	 2.1594572e-01


.. parsed-literal::

     130	 1.5065591e+00	 1.2054253e-01	 1.5613567e+00	 1.4810799e-01	  1.1921877e+00 	 2.1423221e-01


.. parsed-literal::

     131	 1.5089020e+00	 1.2061363e-01	 1.5637353e+00	 1.4823124e-01	  1.1846512e+00 	 2.0509958e-01


.. parsed-literal::

     132	 1.5099526e+00	 1.2090792e-01	 1.5649530e+00	 1.4847039e-01	  1.1648940e+00 	 2.0822835e-01
     133	 1.5120552e+00	 1.2077345e-01	 1.5669712e+00	 1.4838099e-01	  1.1724519e+00 	 1.8010592e-01


.. parsed-literal::

     134	 1.5132479e+00	 1.2069995e-01	 1.5681734e+00	 1.4827227e-01	  1.1747874e+00 	 1.8396592e-01


.. parsed-literal::

     135	 1.5147737e+00	 1.2066494e-01	 1.5697537e+00	 1.4815387e-01	  1.1728231e+00 	 2.0713019e-01


.. parsed-literal::

     136	 1.5171102e+00	 1.2067240e-01	 1.5721649e+00	 1.4803948e-01	  1.1687245e+00 	 2.0629501e-01


.. parsed-literal::

     137	 1.5183791e+00	 1.2067451e-01	 1.5735191e+00	 1.4803231e-01	  1.1583693e+00 	 3.3405924e-01


.. parsed-literal::

     138	 1.5201087e+00	 1.2072527e-01	 1.5752630e+00	 1.4801795e-01	  1.1539773e+00 	 2.2987056e-01


.. parsed-literal::

     139	 1.5214903e+00	 1.2074630e-01	 1.5766830e+00	 1.4804744e-01	  1.1485702e+00 	 2.1316099e-01
     140	 1.5228437e+00	 1.2071069e-01	 1.5781069e+00	 1.4795478e-01	  1.1421448e+00 	 1.9553971e-01


.. parsed-literal::

     141	 1.5242074e+00	 1.2069028e-01	 1.5795336e+00	 1.4791735e-01	  1.1346581e+00 	 2.1630549e-01


.. parsed-literal::

     142	 1.5253544e+00	 1.2063725e-01	 1.5806939e+00	 1.4784702e-01	  1.1309029e+00 	 2.0545721e-01
     143	 1.5270135e+00	 1.2055990e-01	 1.5824601e+00	 1.4762419e-01	  1.1211071e+00 	 1.9901538e-01


.. parsed-literal::

     144	 1.5279270e+00	 1.2047306e-01	 1.5834643e+00	 1.4766192e-01	  1.1117129e+00 	 2.0635557e-01


.. parsed-literal::

     145	 1.5290938e+00	 1.2045992e-01	 1.5845692e+00	 1.4761548e-01	  1.1131951e+00 	 2.0684481e-01
     146	 1.5302301e+00	 1.2045634e-01	 1.5857066e+00	 1.4762762e-01	  1.1099766e+00 	 1.9962859e-01


.. parsed-literal::

     147	 1.5310332e+00	 1.2043955e-01	 1.5864968e+00	 1.4768370e-01	  1.1101167e+00 	 2.1123219e-01
     148	 1.5330316e+00	 1.2038790e-01	 1.5886295e+00	 1.4803650e-01	  1.1009758e+00 	 1.8409538e-01


.. parsed-literal::

     149	 1.5341906e+00	 1.2050879e-01	 1.5898161e+00	 1.4840666e-01	  1.1058947e+00 	 2.0881891e-01
     150	 1.5353108e+00	 1.2039660e-01	 1.5908323e+00	 1.4830078e-01	  1.1062184e+00 	 1.9582677e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.12 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f280c7adb70>



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
    CPU times: user 1.67 s, sys: 62 ms, total: 1.73 s
    Wall time: 546 ms


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

