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
       1	-3.4634392e-01	 3.2144198e-01	-3.3666735e-01	 3.1698764e-01	[-3.2855505e-01]	 4.6049595e-01


.. parsed-literal::

       2	-2.7557232e-01	 3.1093502e-01	-2.5188632e-01	 3.0695070e-01	[-2.4016422e-01]	 2.2713351e-01


.. parsed-literal::

       3	-2.3001734e-01	 2.8949230e-01	-1.8746739e-01	 2.8423469e-01	[-1.6911788e-01]	 3.0151320e-01
       4	-1.9722348e-01	 2.6545083e-01	-1.5621984e-01	 2.6380734e-01	[-1.4894090e-01]	 1.6554832e-01


.. parsed-literal::

       5	-1.0484162e-01	 2.5749091e-01	-7.0634372e-02	 2.5623902e-01	[-6.6871428e-02]	 2.1043539e-01


.. parsed-literal::

       6	-7.1641021e-02	 2.5182893e-01	-4.1053672e-02	 2.4954581e-01	[-3.2939499e-02]	 2.1107078e-01
       7	-5.4358699e-02	 2.4932823e-01	-3.0250132e-02	 2.4716306e-01	[-2.1948369e-02]	 1.9472170e-01


.. parsed-literal::

       8	-4.1025248e-02	 2.4706240e-01	-2.0806486e-02	 2.4492971e-01	[-1.2458438e-02]	 2.0723343e-01
       9	-2.6293621e-02	 2.4427162e-01	-8.9350024e-03	 2.4307313e-01	[-4.1883674e-03]	 1.8824530e-01


.. parsed-literal::

      10	-1.5869364e-02	 2.4237309e-01	-5.7285976e-04	 2.4235932e-01	[-2.7368648e-04]	 1.8725443e-01


.. parsed-literal::

      11	-1.1736171e-02	 2.4173351e-01	 2.6077422e-03	 2.4192512e-01	[ 2.6465047e-03]	 2.1133447e-01


.. parsed-literal::

      12	-9.0840308e-03	 2.4114317e-01	 5.0911046e-03	 2.4135357e-01	[ 5.1544568e-03]	 2.1121049e-01
      13	-5.6319840e-03	 2.4049763e-01	 8.3857897e-03	 2.4054193e-01	[ 9.3008333e-03]	 1.9974518e-01


.. parsed-literal::

      14	-1.3303911e-04	 2.3932234e-01	 1.4406682e-02	 2.3941804e-01	[ 1.5636650e-02]	 1.8405581e-01


.. parsed-literal::

      15	 8.2479698e-02	 2.2548697e-01	 1.0048116e-01	 2.2369907e-01	[ 1.0665664e-01]	 4.1808033e-01


.. parsed-literal::

      16	 1.5187976e-01	 2.2160396e-01	 1.7514337e-01	 2.2079524e-01	[ 1.7562676e-01]	 2.0330811e-01


.. parsed-literal::

      17	 2.5963991e-01	 2.1988593e-01	 2.8886999e-01	 2.1591669e-01	[ 2.9301468e-01]	 2.1602392e-01


.. parsed-literal::

      18	 3.2831404e-01	 2.1136685e-01	 3.5936816e-01	 2.0769573e-01	[ 3.6051461e-01]	 2.0880103e-01


.. parsed-literal::

      19	 3.7677624e-01	 2.0776014e-01	 4.0868023e-01	 2.0567076e-01	[ 4.0225570e-01]	 2.1401954e-01


.. parsed-literal::

      20	 4.2923923e-01	 2.0335291e-01	 4.6210273e-01	 2.0115721e-01	[ 4.5287265e-01]	 2.0680213e-01


.. parsed-literal::

      21	 5.1960206e-01	 1.9908319e-01	 5.5406720e-01	 1.9684831e-01	[ 5.4170818e-01]	 2.1295834e-01


.. parsed-literal::

      22	 6.0967227e-01	 1.9616911e-01	 6.4903623e-01	 1.9820966e-01	[ 6.3310105e-01]	 2.1970749e-01


.. parsed-literal::

      23	 6.4068472e-01	 1.9571179e-01	 6.7751525e-01	 1.9389217e-01	[ 6.7176141e-01]	 2.1514392e-01


.. parsed-literal::

      24	 6.8875493e-01	 1.9295122e-01	 7.2750656e-01	 1.9131422e-01	[ 7.1767793e-01]	 2.1285939e-01


.. parsed-literal::

      25	 7.1641726e-01	 1.9282995e-01	 7.5430918e-01	 1.9146403e-01	[ 7.3231183e-01]	 2.2027278e-01


.. parsed-literal::

      26	 7.5708972e-01	 1.8832675e-01	 7.9400420e-01	 1.8718651e-01	[ 7.7847747e-01]	 2.1181083e-01


.. parsed-literal::

      27	 7.8880106e-01	 1.8763801e-01	 8.2601134e-01	 1.8478631e-01	[ 8.1506500e-01]	 2.0540380e-01


.. parsed-literal::

      28	 8.1709883e-01	 1.8749031e-01	 8.5501552e-01	 1.8366051e-01	[ 8.5261372e-01]	 2.0528245e-01


.. parsed-literal::

      29	 8.5193930e-01	 1.8400963e-01	 8.9170695e-01	 1.8060612e-01	[ 8.8664650e-01]	 2.1093106e-01


.. parsed-literal::

      30	 8.8622142e-01	 1.7811032e-01	 9.2673027e-01	 1.7543587e-01	[ 9.2168918e-01]	 2.0833611e-01


.. parsed-literal::

      31	 9.0755665e-01	 1.7628492e-01	 9.4857778e-01	 1.7452499e-01	[ 9.4189277e-01]	 2.1375155e-01


.. parsed-literal::

      32	 9.3570605e-01	 1.7445479e-01	 9.7802605e-01	 1.7329103e-01	[ 9.6811656e-01]	 2.1534848e-01


.. parsed-literal::

      33	 9.4927365e-01	 1.7430522e-01	 9.9198353e-01	 1.7194721e-01	[ 9.8302400e-01]	 2.0093822e-01
      34	 9.6201919e-01	 1.7292528e-01	 1.0046949e+00	 1.7225577e-01	[ 9.9544597e-01]	 2.1071291e-01


.. parsed-literal::

      35	 9.7360249e-01	 1.7145505e-01	 1.0163444e+00	 1.7137218e-01	[ 1.0052846e+00]	 2.0198417e-01


.. parsed-literal::

      36	 9.9080163e-01	 1.6939865e-01	 1.0341302e+00	 1.6924594e-01	[ 1.0188669e+00]	 2.0363736e-01


.. parsed-literal::

      37	 1.0026498e+00	 1.6953381e-01	 1.0475779e+00	 1.6926133e-01	[ 1.0229765e+00]	 2.0752192e-01
      38	 1.0198681e+00	 1.6737779e-01	 1.0645148e+00	 1.6733495e-01	[ 1.0416575e+00]	 1.9719052e-01


.. parsed-literal::

      39	 1.0299010e+00	 1.6669054e-01	 1.0747140e+00	 1.6634264e-01	[ 1.0524547e+00]	 2.1452069e-01
      40	 1.0411522e+00	 1.6629187e-01	 1.0865035e+00	 1.6544434e-01	[ 1.0617721e+00]	 1.8749046e-01


.. parsed-literal::

      41	 1.0557240e+00	 1.6772447e-01	 1.1022604e+00	 1.6627862e-01	[ 1.0685778e+00]	 1.9953418e-01


.. parsed-literal::

      42	 1.0646996e+00	 1.6618146e-01	 1.1116707e+00	 1.6415161e-01	[ 1.0751309e+00]	 2.0646167e-01


.. parsed-literal::

      43	 1.0698164e+00	 1.6530598e-01	 1.1164481e+00	 1.6347093e-01	[ 1.0803115e+00]	 2.0609641e-01
      44	 1.0785573e+00	 1.6393660e-01	 1.1250557e+00	 1.6200270e-01	[ 1.0877198e+00]	 1.9805455e-01


.. parsed-literal::

      45	 1.0875975e+00	 1.6224665e-01	 1.1345467e+00	 1.5867703e-01	[ 1.0923364e+00]	 2.1597576e-01


.. parsed-literal::

      46	 1.0971713e+00	 1.6047336e-01	 1.1439128e+00	 1.5724289e-01	[ 1.1089217e+00]	 2.0742798e-01


.. parsed-literal::

      47	 1.1036121e+00	 1.6020083e-01	 1.1503822e+00	 1.5656023e-01	[ 1.1162162e+00]	 2.1110582e-01


.. parsed-literal::

      48	 1.1139906e+00	 1.5936440e-01	 1.1610958e+00	 1.5537996e-01	[ 1.1297843e+00]	 2.0634699e-01


.. parsed-literal::

      49	 1.1246007e+00	 1.5880687e-01	 1.1720136e+00	 1.5401744e-01	[ 1.1462848e+00]	 2.1632028e-01


.. parsed-literal::

      50	 1.1341088e+00	 1.5811105e-01	 1.1816048e+00	 1.5327464e-01	[ 1.1592195e+00]	 2.1157455e-01


.. parsed-literal::

      51	 1.1413022e+00	 1.5721101e-01	 1.1887420e+00	 1.5277141e-01	[ 1.1664932e+00]	 2.0345020e-01


.. parsed-literal::

      52	 1.1522531e+00	 1.5558812e-01	 1.1999652e+00	 1.5076059e-01	[ 1.1770604e+00]	 2.0968127e-01


.. parsed-literal::

      53	 1.1544450e+00	 1.5450305e-01	 1.2032446e+00	 1.4839547e-01	  1.1760741e+00 	 2.0870519e-01
      54	 1.1679939e+00	 1.5329941e-01	 1.2164284e+00	 1.4715356e-01	[ 1.1886063e+00]	 1.9405437e-01


.. parsed-literal::

      55	 1.1719115e+00	 1.5280613e-01	 1.2203837e+00	 1.4666053e-01	[ 1.1918814e+00]	 2.0770741e-01


.. parsed-literal::

      56	 1.1790018e+00	 1.5181579e-01	 1.2277714e+00	 1.4535241e-01	[ 1.1963986e+00]	 2.0285344e-01


.. parsed-literal::

      57	 1.1878563e+00	 1.5087380e-01	 1.2368636e+00	 1.4372384e-01	[ 1.2029906e+00]	 2.0637727e-01


.. parsed-literal::

      58	 1.1903209e+00	 1.5109196e-01	 1.2404411e+00	 1.4190109e-01	  1.2009032e+00 	 2.0439434e-01


.. parsed-literal::

      59	 1.2027951e+00	 1.5047028e-01	 1.2522485e+00	 1.4131400e-01	[ 1.2147030e+00]	 2.0703220e-01


.. parsed-literal::

      60	 1.2064045e+00	 1.5023012e-01	 1.2557398e+00	 1.4105818e-01	[ 1.2180665e+00]	 2.1527076e-01
      61	 1.2140724e+00	 1.5000577e-01	 1.2633739e+00	 1.4061386e-01	[ 1.2245784e+00]	 1.7191029e-01


.. parsed-literal::

      62	 1.2193500e+00	 1.4990576e-01	 1.2688218e+00	 1.3978191e-01	[ 1.2292476e+00]	 2.0747066e-01


.. parsed-literal::

      63	 1.2257813e+00	 1.4918881e-01	 1.2750512e+00	 1.3970856e-01	[ 1.2343894e+00]	 2.0913911e-01


.. parsed-literal::

      64	 1.2308577e+00	 1.4869633e-01	 1.2802468e+00	 1.3935953e-01	[ 1.2383308e+00]	 2.0933366e-01


.. parsed-literal::

      65	 1.2363394e+00	 1.4814428e-01	 1.2859095e+00	 1.3904715e-01	[ 1.2430883e+00]	 2.1952248e-01


.. parsed-literal::

      66	 1.2430449e+00	 1.4706174e-01	 1.2931742e+00	 1.3854785e-01	[ 1.2464280e+00]	 2.0587826e-01


.. parsed-literal::

      67	 1.2506050e+00	 1.4676477e-01	 1.3008335e+00	 1.3850676e-01	[ 1.2515925e+00]	 2.0410061e-01
      68	 1.2550691e+00	 1.4653354e-01	 1.3052776e+00	 1.3849585e-01	[ 1.2555857e+00]	 1.8307328e-01


.. parsed-literal::

      69	 1.2612289e+00	 1.4551614e-01	 1.3117240e+00	 1.3852305e-01	[ 1.2575078e+00]	 1.9358993e-01
      70	 1.2662925e+00	 1.4530471e-01	 1.3168509e+00	 1.3820264e-01	[ 1.2631057e+00]	 1.9862962e-01


.. parsed-literal::

      71	 1.2697432e+00	 1.4493976e-01	 1.3203122e+00	 1.3794141e-01	[ 1.2649296e+00]	 1.7391276e-01


.. parsed-literal::

      72	 1.2762366e+00	 1.4401927e-01	 1.3271169e+00	 1.3712223e-01	[ 1.2681303e+00]	 2.1788096e-01
      73	 1.2808222e+00	 1.4357210e-01	 1.3318453e+00	 1.3694896e-01	[ 1.2714425e+00]	 1.8124080e-01


.. parsed-literal::

      74	 1.2870280e+00	 1.4305162e-01	 1.3380903e+00	 1.3649472e-01	[ 1.2775843e+00]	 2.0532179e-01


.. parsed-literal::

      75	 1.2946138e+00	 1.4244028e-01	 1.3459842e+00	 1.3614345e-01	[ 1.2849822e+00]	 2.0237470e-01


.. parsed-literal::

      76	 1.3000208e+00	 1.4198600e-01	 1.3513719e+00	 1.3567055e-01	[ 1.2882001e+00]	 2.1338105e-01


.. parsed-literal::

      77	 1.3046776e+00	 1.4181482e-01	 1.3558332e+00	 1.3544405e-01	[ 1.2953221e+00]	 2.1961641e-01


.. parsed-literal::

      78	 1.3087372e+00	 1.4139119e-01	 1.3599805e+00	 1.3517492e-01	[ 1.2989460e+00]	 2.2101498e-01
      79	 1.3138410e+00	 1.4107530e-01	 1.3652687e+00	 1.3480921e-01	[ 1.3009705e+00]	 1.8483448e-01


.. parsed-literal::

      80	 1.3200907e+00	 1.4070956e-01	 1.3718094e+00	 1.3444191e-01	[ 1.3054964e+00]	 2.1214318e-01


.. parsed-literal::

      81	 1.3255560e+00	 1.4070010e-01	 1.3773284e+00	 1.3416801e-01	[ 1.3066042e+00]	 2.0899224e-01


.. parsed-literal::

      82	 1.3289452e+00	 1.4073465e-01	 1.3806875e+00	 1.3387442e-01	[ 1.3101211e+00]	 2.0689440e-01


.. parsed-literal::

      83	 1.3329816e+00	 1.4043995e-01	 1.3847726e+00	 1.3347556e-01	[ 1.3119052e+00]	 2.0515919e-01


.. parsed-literal::

      84	 1.3375684e+00	 1.4026543e-01	 1.3897014e+00	 1.3295103e-01	[ 1.3150042e+00]	 2.0168853e-01
      85	 1.3420605e+00	 1.3982098e-01	 1.3941954e+00	 1.3265947e-01	[ 1.3172191e+00]	 1.8685246e-01


.. parsed-literal::

      86	 1.3456527e+00	 1.3932797e-01	 1.3979473e+00	 1.3248324e-01	  1.3171115e+00 	 2.1006417e-01


.. parsed-literal::

      87	 1.3493593e+00	 1.3899341e-01	 1.4017910e+00	 1.3236239e-01	[ 1.3190018e+00]	 2.0283985e-01
      88	 1.3542739e+00	 1.3843752e-01	 1.4068666e+00	 1.3232133e-01	[ 1.3211298e+00]	 1.9919801e-01


.. parsed-literal::

      89	 1.3590310e+00	 1.3816655e-01	 1.4116789e+00	 1.3212291e-01	[ 1.3249775e+00]	 2.1583605e-01
      90	 1.3620911e+00	 1.3792103e-01	 1.4145790e+00	 1.3207265e-01	[ 1.3297240e+00]	 1.9631529e-01


.. parsed-literal::

      91	 1.3666486e+00	 1.3723540e-01	 1.4190633e+00	 1.3205747e-01	[ 1.3321940e+00]	 2.1695137e-01
      92	 1.3683212e+00	 1.3654096e-01	 1.4209053e+00	 1.3243056e-01	[ 1.3353781e+00]	 1.8578577e-01


.. parsed-literal::

      93	 1.3724277e+00	 1.3647738e-01	 1.4249051e+00	 1.3227060e-01	[ 1.3362908e+00]	 2.0503283e-01


.. parsed-literal::

      94	 1.3752426e+00	 1.3633296e-01	 1.4278333e+00	 1.3222028e-01	  1.3346240e+00 	 2.2009444e-01
      95	 1.3779746e+00	 1.3624741e-01	 1.4306828e+00	 1.3221846e-01	  1.3338785e+00 	 1.9772649e-01


.. parsed-literal::

      96	 1.3830039e+00	 1.3628410e-01	 1.4357644e+00	 1.3225761e-01	  1.3329829e+00 	 2.2373986e-01


.. parsed-literal::

      97	 1.3848226e+00	 1.3629911e-01	 1.4379259e+00	 1.3256290e-01	  1.3343589e+00 	 2.1029067e-01


.. parsed-literal::

      98	 1.3893518e+00	 1.3646617e-01	 1.4421761e+00	 1.3241087e-01	[ 1.3386493e+00]	 2.0932341e-01


.. parsed-literal::

      99	 1.3912066e+00	 1.3619799e-01	 1.4439881e+00	 1.3239070e-01	[ 1.3394123e+00]	 2.0292544e-01
     100	 1.3948164e+00	 1.3596988e-01	 1.4476918e+00	 1.3245646e-01	[ 1.3405228e+00]	 1.7607045e-01


.. parsed-literal::

     101	 1.3980854e+00	 1.3563482e-01	 1.4511974e+00	 1.3232808e-01	  1.3405126e+00 	 1.8436122e-01
     102	 1.4027197e+00	 1.3541478e-01	 1.4558508e+00	 1.3255766e-01	[ 1.3412065e+00]	 1.8852186e-01


.. parsed-literal::

     103	 1.4048831e+00	 1.3554655e-01	 1.4580119e+00	 1.3259624e-01	[ 1.3432542e+00]	 2.0895100e-01


.. parsed-literal::

     104	 1.4071892e+00	 1.3555247e-01	 1.4603364e+00	 1.3263153e-01	[ 1.3449532e+00]	 2.1894193e-01


.. parsed-literal::

     105	 1.4114857e+00	 1.3551054e-01	 1.4646185e+00	 1.3262955e-01	[ 1.3501778e+00]	 2.0856905e-01


.. parsed-literal::

     106	 1.4151652e+00	 1.3500713e-01	 1.4683611e+00	 1.3270248e-01	[ 1.3507753e+00]	 2.1302938e-01
     107	 1.4181309e+00	 1.3481664e-01	 1.4711998e+00	 1.3249481e-01	[ 1.3537218e+00]	 1.9734907e-01


.. parsed-literal::

     108	 1.4207833e+00	 1.3446612e-01	 1.4738410e+00	 1.3226789e-01	[ 1.3569087e+00]	 1.7324185e-01
     109	 1.4241687e+00	 1.3399895e-01	 1.4773096e+00	 1.3187526e-01	  1.3549303e+00 	 1.8280149e-01


.. parsed-literal::

     110	 1.4282049e+00	 1.3358228e-01	 1.4814384e+00	 1.3140041e-01	[ 1.3601115e+00]	 2.1093059e-01


.. parsed-literal::

     111	 1.4314118e+00	 1.3343043e-01	 1.4847003e+00	 1.3124987e-01	[ 1.3603740e+00]	 2.1365428e-01


.. parsed-literal::

     112	 1.4347525e+00	 1.3335460e-01	 1.4881228e+00	 1.3106175e-01	[ 1.3615344e+00]	 2.0928574e-01


.. parsed-literal::

     113	 1.4370938e+00	 1.3324768e-01	 1.4905205e+00	 1.3106103e-01	[ 1.3658498e+00]	 2.1667528e-01
     114	 1.4397351e+00	 1.3317526e-01	 1.4931563e+00	 1.3105940e-01	[ 1.3663181e+00]	 2.0250225e-01


.. parsed-literal::

     115	 1.4427123e+00	 1.3319899e-01	 1.4961476e+00	 1.3104766e-01	[ 1.3684396e+00]	 2.0339799e-01


.. parsed-literal::

     116	 1.4452872e+00	 1.3337523e-01	 1.4988308e+00	 1.3095997e-01	[ 1.3693052e+00]	 2.1658468e-01


.. parsed-literal::

     117	 1.4484355e+00	 1.3371767e-01	 1.5020780e+00	 1.3113289e-01	[ 1.3694743e+00]	 2.0670223e-01
     118	 1.4508922e+00	 1.3393036e-01	 1.5046722e+00	 1.3105611e-01	[ 1.3707773e+00]	 2.0358539e-01


.. parsed-literal::

     119	 1.4531030e+00	 1.3384961e-01	 1.5070086e+00	 1.3110127e-01	  1.3704935e+00 	 1.8196201e-01
     120	 1.4553356e+00	 1.3360760e-01	 1.5093724e+00	 1.3105138e-01	  1.3689488e+00 	 1.9766641e-01


.. parsed-literal::

     121	 1.4580371e+00	 1.3311588e-01	 1.5122315e+00	 1.3107887e-01	  1.3670565e+00 	 1.9904637e-01
     122	 1.4606463e+00	 1.3270538e-01	 1.5149402e+00	 1.3107340e-01	  1.3615941e+00 	 1.9897342e-01


.. parsed-literal::

     123	 1.4628487e+00	 1.3253544e-01	 1.5170709e+00	 1.3106251e-01	  1.3616544e+00 	 2.1184492e-01
     124	 1.4654397e+00	 1.3227244e-01	 1.5196003e+00	 1.3103603e-01	  1.3610423e+00 	 1.9892097e-01


.. parsed-literal::

     125	 1.4674039e+00	 1.3207250e-01	 1.5216337e+00	 1.3090817e-01	  1.3546081e+00 	 2.1427035e-01
     126	 1.4696833e+00	 1.3172183e-01	 1.5239714e+00	 1.3081698e-01	  1.3541877e+00 	 1.8402219e-01


.. parsed-literal::

     127	 1.4716320e+00	 1.3136965e-01	 1.5260209e+00	 1.3064086e-01	  1.3505177e+00 	 1.8839288e-01


.. parsed-literal::

     128	 1.4731406e+00	 1.3121483e-01	 1.5276257e+00	 1.3059930e-01	  1.3503815e+00 	 2.0972109e-01


.. parsed-literal::

     129	 1.4751349e+00	 1.3108241e-01	 1.5296483e+00	 1.3051513e-01	  1.3468479e+00 	 2.1296167e-01
     130	 1.4770691e+00	 1.3099762e-01	 1.5316885e+00	 1.3055916e-01	  1.3376871e+00 	 1.9843102e-01


.. parsed-literal::

     131	 1.4790060e+00	 1.3101177e-01	 1.5335589e+00	 1.3031267e-01	  1.3357130e+00 	 2.1311307e-01


.. parsed-literal::

     132	 1.4800545e+00	 1.3101497e-01	 1.5345363e+00	 1.3025559e-01	  1.3371869e+00 	 2.0706892e-01
     133	 1.4824029e+00	 1.3096051e-01	 1.5368439e+00	 1.3012462e-01	  1.3380916e+00 	 1.8063951e-01


.. parsed-literal::

     134	 1.4829561e+00	 1.3096721e-01	 1.5375069e+00	 1.3009282e-01	  1.3349009e+00 	 2.1293068e-01


.. parsed-literal::

     135	 1.4849096e+00	 1.3091495e-01	 1.5393866e+00	 1.3009332e-01	  1.3393014e+00 	 2.0570493e-01
     136	 1.4858883e+00	 1.3085588e-01	 1.5403920e+00	 1.3010630e-01	  1.3404783e+00 	 1.9863176e-01


.. parsed-literal::

     137	 1.4873830e+00	 1.3077471e-01	 1.5419489e+00	 1.3014757e-01	  1.3422008e+00 	 2.1542954e-01
     138	 1.4896701e+00	 1.3057602e-01	 1.5443210e+00	 1.3019780e-01	  1.3429326e+00 	 1.7604446e-01


.. parsed-literal::

     139	 1.4910114e+00	 1.3046168e-01	 1.5457504e+00	 1.3017770e-01	  1.3428594e+00 	 3.2431436e-01


.. parsed-literal::

     140	 1.4928033e+00	 1.3020677e-01	 1.5475921e+00	 1.3016847e-01	  1.3404793e+00 	 2.0857000e-01
     141	 1.4941950e+00	 1.2994121e-01	 1.5490055e+00	 1.3013142e-01	  1.3402822e+00 	 1.7773318e-01


.. parsed-literal::

     142	 1.4959387e+00	 1.2954449e-01	 1.5508050e+00	 1.2992906e-01	  1.3352346e+00 	 1.9807553e-01
     143	 1.4972973e+00	 1.2918470e-01	 1.5522543e+00	 1.2996850e-01	  1.3366919e+00 	 1.9956565e-01


.. parsed-literal::

     144	 1.4985912e+00	 1.2902086e-01	 1.5535338e+00	 1.2987909e-01	  1.3372492e+00 	 2.1377826e-01
     145	 1.4998323e+00	 1.2886059e-01	 1.5547859e+00	 1.2981663e-01	  1.3371448e+00 	 1.7712379e-01


.. parsed-literal::

     146	 1.5010885e+00	 1.2868220e-01	 1.5560734e+00	 1.2976057e-01	  1.3373703e+00 	 2.1321726e-01


.. parsed-literal::

     147	 1.5029553e+00	 1.2842749e-01	 1.5580549e+00	 1.2969092e-01	  1.3339353e+00 	 2.0872808e-01
     148	 1.5046180e+00	 1.2825833e-01	 1.5597361e+00	 1.2966776e-01	  1.3327167e+00 	 1.9921207e-01


.. parsed-literal::

     149	 1.5056236e+00	 1.2825488e-01	 1.5607102e+00	 1.2972339e-01	  1.3327893e+00 	 2.1533108e-01


.. parsed-literal::

     150	 1.5070087e+00	 1.2825732e-01	 1.5621009e+00	 1.2983206e-01	  1.3305425e+00 	 2.1414113e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.09 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f0e0f7c52d0>



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
    CPU times: user 1.69 s, sys: 42.9 ms, total: 1.73 s
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

