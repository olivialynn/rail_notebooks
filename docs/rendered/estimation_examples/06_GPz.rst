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
       1	-3.4431454e-01	 3.2100794e-01	-3.3464689e-01	 3.1919158e-01	[-3.3017063e-01]	 4.6536112e-01


.. parsed-literal::

       2	-2.7486489e-01	 3.1065279e-01	-2.5088612e-01	 3.0883380e-01	[-2.4261166e-01]	 2.3146176e-01


.. parsed-literal::

       3	-2.2625618e-01	 2.8676200e-01	-1.8029794e-01	 2.9039491e-01	[-1.8739675e-01]	 2.9478478e-01


.. parsed-literal::

       4	-2.0091045e-01	 2.6459289e-01	-1.5959492e-01	 2.7120406e-01	 -1.9149739e-01 	 2.0883107e-01


.. parsed-literal::

       5	-1.0200618e-01	 2.5741181e-01	-6.7639622e-02	 2.6323940e-01	[-8.3356758e-02]	 2.0929527e-01
       6	-7.2435991e-02	 2.5201728e-01	-4.1723258e-02	 2.5469851e-01	[-5.1040128e-02]	 1.9202542e-01


.. parsed-literal::

       7	-5.2820640e-02	 2.4901722e-01	-2.9289097e-02	 2.5185814e-01	[-3.8415422e-02]	 2.0462775e-01
       8	-4.1676954e-02	 2.4722866e-01	-2.1710665e-02	 2.4986154e-01	[-2.9911297e-02]	 1.8453670e-01


.. parsed-literal::

       9	-2.8759733e-02	 2.4484397e-01	-1.1524418e-02	 2.4623290e-01	[-1.5121524e-02]	 1.9597554e-01


.. parsed-literal::

      10	-1.8407248e-02	 2.4280072e-01	-3.0164496e-03	 2.4229192e-01	[ 2.2435039e-04]	 2.1757483e-01


.. parsed-literal::

      11	-1.2026317e-02	 2.4187872e-01	 2.2609518e-03	 2.4002176e-01	[ 9.6650039e-03]	 2.0707512e-01


.. parsed-literal::

      12	-9.1343767e-03	 2.4122332e-01	 4.9893119e-03	 2.3978279e-01	[ 1.1186167e-02]	 2.0684314e-01


.. parsed-literal::

      13	-3.8243736e-03	 2.4013833e-01	 1.0469691e-02	 2.3984058e-01	[ 1.3392395e-02]	 2.0596385e-01


.. parsed-literal::

      14	 6.1288547e-02	 2.2795935e-01	 7.9037217e-02	 2.3442042e-01	[ 6.4328583e-02]	 2.9801726e-01


.. parsed-literal::

      15	 8.4412622e-02	 2.2514952e-01	 1.0416281e-01	 2.3635278e-01	[ 8.6346725e-02]	 4.3615699e-01


.. parsed-literal::

      16	 1.3538114e-01	 2.2239442e-01	 1.5631332e-01	 2.3382403e-01	[ 1.3771446e-01]	 2.0934105e-01


.. parsed-literal::

      17	 2.1510238e-01	 2.1992011e-01	 2.4116494e-01	 2.2707713e-01	[ 2.1885551e-01]	 2.0956135e-01
      18	 2.7562231e-01	 2.2052637e-01	 3.0646790e-01	 2.2191722e-01	[ 2.8710944e-01]	 2.0207882e-01


.. parsed-literal::

      19	 3.2978288e-01	 2.1642707e-01	 3.6138537e-01	 2.1869168e-01	[ 3.4326042e-01]	 2.0587158e-01


.. parsed-literal::

      20	 3.7735415e-01	 2.1090433e-01	 4.0973519e-01	 2.1622556e-01	[ 3.9122687e-01]	 2.1470284e-01


.. parsed-literal::

      21	 4.4678650e-01	 2.0449328e-01	 4.8056215e-01	 2.0765805e-01	[ 4.6243500e-01]	 2.1461368e-01


.. parsed-literal::

      22	 5.2627174e-01	 2.0624475e-01	 5.6373501e-01	 2.0847138e-01	[ 5.3251872e-01]	 2.0763803e-01


.. parsed-literal::

      23	 5.8244826e-01	 2.0334387e-01	 6.2099274e-01	 2.0484449e-01	[ 5.9814077e-01]	 2.0871615e-01


.. parsed-literal::

      24	 6.2202247e-01	 1.9900012e-01	 6.6000154e-01	 2.0137085e-01	[ 6.3678489e-01]	 2.0978856e-01
      25	 6.5592162e-01	 1.9706011e-01	 6.9356119e-01	 2.0031969e-01	[ 6.7021075e-01]	 1.8848419e-01


.. parsed-literal::

      26	 6.8448642e-01	 1.9719013e-01	 7.2202978e-01	 1.9990600e-01	[ 7.0707603e-01]	 2.1786690e-01


.. parsed-literal::

      27	 7.0803115e-01	 1.9835254e-01	 7.4498684e-01	 2.0149498e-01	[ 7.3120796e-01]	 2.2079420e-01
      28	 7.3716422e-01	 1.9773370e-01	 7.7505182e-01	 2.0184563e-01	[ 7.5800045e-01]	 1.8706775e-01


.. parsed-literal::

      29	 7.6670349e-01	 2.0464356e-01	 8.0627551e-01	 2.1149156e-01	[ 7.8123974e-01]	 2.0799780e-01


.. parsed-literal::

      30	 7.9782757e-01	 2.0881965e-01	 8.3887762e-01	 2.1680099e-01	[ 8.2164896e-01]	 2.0697141e-01
      31	 8.2298161e-01	 2.0571466e-01	 8.6373140e-01	 2.1399362e-01	[ 8.4450186e-01]	 1.9453907e-01


.. parsed-literal::

      32	 8.4921183e-01	 2.0493422e-01	 8.8976786e-01	 2.1258973e-01	[ 8.6483254e-01]	 1.8964934e-01
      33	 8.6539729e-01	 2.0854886e-01	 9.0764614e-01	 2.1757558e-01	[ 8.8173704e-01]	 1.9099474e-01


.. parsed-literal::

      34	 8.9225030e-01	 2.0830024e-01	 9.3485090e-01	 2.1650236e-01	[ 9.0861971e-01]	 2.1669459e-01


.. parsed-literal::

      35	 9.1478140e-01	 2.0361471e-01	 9.5784553e-01	 2.1098277e-01	[ 9.3090071e-01]	 2.0789337e-01
      36	 9.3449358e-01	 1.9832000e-01	 9.7789972e-01	 2.0513404e-01	[ 9.5198925e-01]	 1.9940639e-01


.. parsed-literal::

      37	 9.5526880e-01	 1.9349898e-01	 9.9942969e-01	 2.0077717e-01	[ 9.7564572e-01]	 2.0443892e-01


.. parsed-literal::

      38	 9.7398119e-01	 1.8646437e-01	 1.0188391e+00	 1.9220391e-01	[ 9.9498713e-01]	 2.0990849e-01


.. parsed-literal::

      39	 9.8687373e-01	 1.8430590e-01	 1.0319382e+00	 1.8995270e-01	[ 1.0122744e+00]	 2.0274591e-01


.. parsed-literal::

      40	 9.9693539e-01	 1.8246205e-01	 1.0422958e+00	 1.8794051e-01	[ 1.0225610e+00]	 2.1208000e-01
      41	 1.0182637e+00	 1.7761397e-01	 1.0648620e+00	 1.8273225e-01	[ 1.0423166e+00]	 1.8419075e-01


.. parsed-literal::

      42	 1.0197641e+00	 1.7418536e-01	 1.0672073e+00	 1.7785062e-01	  1.0382184e+00 	 1.8704748e-01


.. parsed-literal::

      43	 1.0426989e+00	 1.7345037e-01	 1.0893557e+00	 1.7830461e-01	[ 1.0630683e+00]	 2.0633960e-01
      44	 1.0509302e+00	 1.7209768e-01	 1.0977497e+00	 1.7733359e-01	[ 1.0694877e+00]	 1.8372846e-01


.. parsed-literal::

      45	 1.0622127e+00	 1.6907368e-01	 1.1093321e+00	 1.7464415e-01	[ 1.0786727e+00]	 2.0142221e-01


.. parsed-literal::

      46	 1.0783974e+00	 1.6480537e-01	 1.1258905e+00	 1.7075724e-01	[ 1.0949047e+00]	 2.1496487e-01


.. parsed-literal::

      47	 1.0871572e+00	 1.5699104e-01	 1.1355364e+00	 1.6377986e-01	[ 1.1049286e+00]	 2.1007538e-01


.. parsed-literal::

      48	 1.1073472e+00	 1.5667447e-01	 1.1554050e+00	 1.6377120e-01	[ 1.1252641e+00]	 2.0669341e-01


.. parsed-literal::

      49	 1.1142425e+00	 1.5688317e-01	 1.1621112e+00	 1.6408448e-01	[ 1.1316559e+00]	 2.1520853e-01


.. parsed-literal::

      50	 1.1254780e+00	 1.5520785e-01	 1.1737665e+00	 1.6266764e-01	[ 1.1420496e+00]	 2.1205997e-01


.. parsed-literal::

      51	 1.1416572e+00	 1.5267270e-01	 1.1909653e+00	 1.6048447e-01	[ 1.1556303e+00]	 2.0760798e-01


.. parsed-literal::

      52	 1.1492787e+00	 1.5373647e-01	 1.1987796e+00	 1.6189782e-01	[ 1.1644141e+00]	 2.1328259e-01
      53	 1.1589766e+00	 1.5264299e-01	 1.2081459e+00	 1.6114122e-01	[ 1.1733412e+00]	 1.7647409e-01


.. parsed-literal::

      54	 1.1703407e+00	 1.5133888e-01	 1.2192663e+00	 1.6029233e-01	[ 1.1831731e+00]	 2.1706915e-01


.. parsed-literal::

      55	 1.1785237e+00	 1.5010626e-01	 1.2274711e+00	 1.5950612e-01	[ 1.1897936e+00]	 2.1245217e-01


.. parsed-literal::

      56	 1.1932722e+00	 1.4709455e-01	 1.2422319e+00	 1.5705694e-01	[ 1.2001892e+00]	 2.0923638e-01


.. parsed-literal::

      57	 1.2039590e+00	 1.4622447e-01	 1.2538696e+00	 1.5643112e-01	[ 1.2042390e+00]	 3.0764604e-01


.. parsed-literal::

      58	 1.2153031e+00	 1.4509348e-01	 1.2656237e+00	 1.5562397e-01	[ 1.2096172e+00]	 2.1150637e-01


.. parsed-literal::

      59	 1.2280496e+00	 1.4358265e-01	 1.2789565e+00	 1.5454179e-01	[ 1.2168272e+00]	 2.0802450e-01
      60	 1.2375836e+00	 1.4285292e-01	 1.2887391e+00	 1.5451122e-01	[ 1.2277650e+00]	 1.8742704e-01


.. parsed-literal::

      61	 1.2472112e+00	 1.4187744e-01	 1.2981057e+00	 1.5399248e-01	[ 1.2380380e+00]	 2.1332073e-01


.. parsed-literal::

      62	 1.2642492e+00	 1.3969725e-01	 1.3153622e+00	 1.5273733e-01	[ 1.2523104e+00]	 2.0819688e-01


.. parsed-literal::

      63	 1.2753808e+00	 1.3892292e-01	 1.3259907e+00	 1.5188165e-01	[ 1.2692803e+00]	 2.1033287e-01


.. parsed-literal::

      64	 1.2861032e+00	 1.3770068e-01	 1.3369907e+00	 1.5015677e-01	[ 1.2794839e+00]	 2.0960426e-01


.. parsed-literal::

      65	 1.2933457e+00	 1.3735394e-01	 1.3444489e+00	 1.4973984e-01	[ 1.2835942e+00]	 2.0400405e-01


.. parsed-literal::

      66	 1.2996412e+00	 1.3719421e-01	 1.3507020e+00	 1.4967099e-01	[ 1.2903868e+00]	 2.0634437e-01
      67	 1.3100828e+00	 1.3578541e-01	 1.3613840e+00	 1.4794424e-01	[ 1.2996185e+00]	 1.9762135e-01


.. parsed-literal::

      68	 1.3198996e+00	 1.3515762e-01	 1.3712192e+00	 1.4779028e-01	[ 1.3095425e+00]	 2.0029140e-01


.. parsed-literal::

      69	 1.3277541e+00	 1.3446153e-01	 1.3792966e+00	 1.4700048e-01	[ 1.3166294e+00]	 2.1651220e-01


.. parsed-literal::

      70	 1.3349733e+00	 1.3372287e-01	 1.3866379e+00	 1.4633730e-01	[ 1.3232085e+00]	 2.0978689e-01


.. parsed-literal::

      71	 1.3400473e+00	 1.3326422e-01	 1.3919046e+00	 1.4578365e-01	[ 1.3249847e+00]	 2.0328927e-01


.. parsed-literal::

      72	 1.3464814e+00	 1.3322616e-01	 1.3985715e+00	 1.4499211e-01	[ 1.3252206e+00]	 2.0307732e-01


.. parsed-literal::

      73	 1.3537472e+00	 1.3292611e-01	 1.4059725e+00	 1.4464586e-01	  1.3247330e+00 	 2.1153975e-01


.. parsed-literal::

      74	 1.3597673e+00	 1.3254387e-01	 1.4122401e+00	 1.4456433e-01	[ 1.3258426e+00]	 2.0416546e-01
      75	 1.3649521e+00	 1.3198686e-01	 1.4174563e+00	 1.4449115e-01	[ 1.3292797e+00]	 1.9089699e-01


.. parsed-literal::

      76	 1.3719437e+00	 1.3138010e-01	 1.4246281e+00	 1.4447785e-01	[ 1.3333878e+00]	 2.0755816e-01


.. parsed-literal::

      77	 1.3774594e+00	 1.3114833e-01	 1.4300494e+00	 1.4424638e-01	[ 1.3389185e+00]	 2.1124363e-01
      78	 1.3830588e+00	 1.3083259e-01	 1.4357784e+00	 1.4331165e-01	[ 1.3424718e+00]	 1.8295431e-01


.. parsed-literal::

      79	 1.3870396e+00	 1.3071916e-01	 1.4397446e+00	 1.4299215e-01	[ 1.3442312e+00]	 1.8343234e-01


.. parsed-literal::

      80	 1.3911844e+00	 1.3044594e-01	 1.4439573e+00	 1.4249127e-01	[ 1.3470209e+00]	 2.0161748e-01


.. parsed-literal::

      81	 1.3952396e+00	 1.3011807e-01	 1.4480374e+00	 1.4232754e-01	[ 1.3481103e+00]	 2.0245171e-01


.. parsed-literal::

      82	 1.3998425e+00	 1.2957317e-01	 1.4527379e+00	 1.4203813e-01	[ 1.3490746e+00]	 2.1014118e-01


.. parsed-literal::

      83	 1.4038660e+00	 1.2906741e-01	 1.4567038e+00	 1.4175869e-01	[ 1.3511094e+00]	 2.1168780e-01
      84	 1.4089804e+00	 1.2862661e-01	 1.4618401e+00	 1.4195289e-01	  1.3467884e+00 	 1.7456818e-01


.. parsed-literal::

      85	 1.4117330e+00	 1.2824424e-01	 1.4645643e+00	 1.4146735e-01	  1.3428805e+00 	 2.1011162e-01


.. parsed-literal::

      86	 1.4156777e+00	 1.2818593e-01	 1.4684944e+00	 1.4158296e-01	  1.3444868e+00 	 2.1402001e-01
      87	 1.4182071e+00	 1.2823947e-01	 1.4710744e+00	 1.4163367e-01	  1.3455938e+00 	 1.9091845e-01


.. parsed-literal::

      88	 1.4212112e+00	 1.2827239e-01	 1.4741414e+00	 1.4163721e-01	  1.3475110e+00 	 2.1705985e-01


.. parsed-literal::

      89	 1.4254443e+00	 1.2827114e-01	 1.4785610e+00	 1.4150505e-01	  1.3496594e+00 	 2.0948958e-01


.. parsed-literal::

      90	 1.4287325e+00	 1.2829070e-01	 1.4818429e+00	 1.4130215e-01	[ 1.3514901e+00]	 2.0289135e-01


.. parsed-literal::

      91	 1.4310892e+00	 1.2799442e-01	 1.4841177e+00	 1.4104096e-01	[ 1.3537453e+00]	 2.0671773e-01
      92	 1.4333410e+00	 1.2779723e-01	 1.4863685e+00	 1.4092329e-01	  1.3534885e+00 	 1.9302201e-01


.. parsed-literal::

      93	 1.4364502e+00	 1.2767135e-01	 1.4895715e+00	 1.4077912e-01	  1.3533637e+00 	 1.8580008e-01


.. parsed-literal::

      94	 1.4400503e+00	 1.2778332e-01	 1.4934017e+00	 1.4079249e-01	  1.3465985e+00 	 2.0165014e-01


.. parsed-literal::

      95	 1.4427667e+00	 1.2768731e-01	 1.4961262e+00	 1.4061120e-01	  1.3490944e+00 	 2.0508432e-01


.. parsed-literal::

      96	 1.4444222e+00	 1.2761954e-01	 1.4977487e+00	 1.4048858e-01	  1.3496437e+00 	 2.2182989e-01


.. parsed-literal::

      97	 1.4475887e+00	 1.2739354e-01	 1.5009858e+00	 1.4020793e-01	  1.3473510e+00 	 2.1478057e-01


.. parsed-literal::

      98	 1.4498095e+00	 1.2711946e-01	 1.5033685e+00	 1.3997410e-01	  1.3393539e+00 	 2.0525670e-01


.. parsed-literal::

      99	 1.4528028e+00	 1.2691965e-01	 1.5062926e+00	 1.3989844e-01	  1.3400485e+00 	 2.1452880e-01


.. parsed-literal::

     100	 1.4547387e+00	 1.2681328e-01	 1.5082275e+00	 1.3994040e-01	  1.3392155e+00 	 2.1875405e-01


.. parsed-literal::

     101	 1.4566940e+00	 1.2672364e-01	 1.5102252e+00	 1.4003528e-01	  1.3364382e+00 	 2.2070503e-01


.. parsed-literal::

     102	 1.4584639e+00	 1.2648886e-01	 1.5121851e+00	 1.3991597e-01	  1.3238660e+00 	 2.0954347e-01


.. parsed-literal::

     103	 1.4616898e+00	 1.2641901e-01	 1.5153863e+00	 1.3997250e-01	  1.3230397e+00 	 2.0902634e-01


.. parsed-literal::

     104	 1.4630541e+00	 1.2639994e-01	 1.5167555e+00	 1.3987895e-01	  1.3229078e+00 	 2.1537066e-01
     105	 1.4650222e+00	 1.2627894e-01	 1.5187760e+00	 1.3963967e-01	  1.3212969e+00 	 1.9679999e-01


.. parsed-literal::

     106	 1.4663987e+00	 1.2629114e-01	 1.5201862e+00	 1.3964589e-01	  1.3205412e+00 	 3.2144952e-01


.. parsed-literal::

     107	 1.4681639e+00	 1.2614790e-01	 1.5219771e+00	 1.3948692e-01	  1.3198781e+00 	 2.1742105e-01


.. parsed-literal::

     108	 1.4706940e+00	 1.2599215e-01	 1.5245568e+00	 1.3931812e-01	  1.3179158e+00 	 2.0708132e-01


.. parsed-literal::

     109	 1.4727124e+00	 1.2588880e-01	 1.5266805e+00	 1.3918599e-01	  1.3147076e+00 	 2.1215868e-01


.. parsed-literal::

     110	 1.4749703e+00	 1.2577989e-01	 1.5290627e+00	 1.3886559e-01	  1.3096016e+00 	 2.1421266e-01
     111	 1.4770248e+00	 1.2567826e-01	 1.5312021e+00	 1.3860470e-01	  1.3056760e+00 	 2.0192838e-01


.. parsed-literal::

     112	 1.4792525e+00	 1.2555491e-01	 1.5335420e+00	 1.3823986e-01	  1.3010942e+00 	 2.1114039e-01


.. parsed-literal::

     113	 1.4808315e+00	 1.2545032e-01	 1.5351312e+00	 1.3803363e-01	  1.3005032e+00 	 2.1814966e-01
     114	 1.4822848e+00	 1.2539482e-01	 1.5365386e+00	 1.3804494e-01	  1.3017713e+00 	 2.0321584e-01


.. parsed-literal::

     115	 1.4843018e+00	 1.2527332e-01	 1.5384807e+00	 1.3806974e-01	  1.3047605e+00 	 2.1064878e-01
     116	 1.4856405e+00	 1.2515810e-01	 1.5398329e+00	 1.3802062e-01	  1.3049447e+00 	 1.7594099e-01


.. parsed-literal::

     117	 1.4875614e+00	 1.2502763e-01	 1.5417820e+00	 1.3790651e-01	  1.3042358e+00 	 2.0417404e-01


.. parsed-literal::

     118	 1.4890536e+00	 1.2480080e-01	 1.5434870e+00	 1.3738459e-01	  1.2985396e+00 	 2.1866226e-01
     119	 1.4913161e+00	 1.2479287e-01	 1.5456876e+00	 1.3747451e-01	  1.3013328e+00 	 1.9602728e-01


.. parsed-literal::

     120	 1.4922736e+00	 1.2476670e-01	 1.5466625e+00	 1.3740425e-01	  1.2997259e+00 	 2.0652461e-01


.. parsed-literal::

     121	 1.4944516e+00	 1.2460461e-01	 1.5488833e+00	 1.3728013e-01	  1.2971183e+00 	 2.0608759e-01


.. parsed-literal::

     122	 1.4960676e+00	 1.2422986e-01	 1.5505874e+00	 1.3723659e-01	  1.2861658e+00 	 2.0537901e-01


.. parsed-literal::

     123	 1.4980273e+00	 1.2413354e-01	 1.5524666e+00	 1.3726148e-01	  1.2888687e+00 	 2.1923351e-01


.. parsed-literal::

     124	 1.4992629e+00	 1.2401173e-01	 1.5536559e+00	 1.3733327e-01	  1.2899735e+00 	 2.0540857e-01


.. parsed-literal::

     125	 1.5006374e+00	 1.2385904e-01	 1.5550348e+00	 1.3746139e-01	  1.2870062e+00 	 2.0925379e-01
     126	 1.5012059e+00	 1.2357117e-01	 1.5558563e+00	 1.3733715e-01	  1.2741540e+00 	 1.9933677e-01


.. parsed-literal::

     127	 1.5035648e+00	 1.2351583e-01	 1.5580793e+00	 1.3743071e-01	  1.2757822e+00 	 2.0919657e-01


.. parsed-literal::

     128	 1.5045023e+00	 1.2344309e-01	 1.5590620e+00	 1.3735572e-01	  1.2714021e+00 	 2.1425104e-01
     129	 1.5059479e+00	 1.2330183e-01	 1.5606271e+00	 1.3715410e-01	  1.2636958e+00 	 1.7133284e-01


.. parsed-literal::

     130	 1.5079966e+00	 1.2309674e-01	 1.5627711e+00	 1.3691431e-01	  1.2546512e+00 	 1.9461322e-01


.. parsed-literal::

     131	 1.5102361e+00	 1.2278708e-01	 1.5651406e+00	 1.3657859e-01	  1.2391616e+00 	 2.1883464e-01


.. parsed-literal::

     132	 1.5114839e+00	 1.2266529e-01	 1.5664060e+00	 1.3653840e-01	  1.2418155e+00 	 2.2289848e-01


.. parsed-literal::

     133	 1.5128643e+00	 1.2257163e-01	 1.5677090e+00	 1.3650960e-01	  1.2429848e+00 	 2.1334505e-01


.. parsed-literal::

     134	 1.5137593e+00	 1.2253884e-01	 1.5685800e+00	 1.3652029e-01	  1.2427208e+00 	 2.0620370e-01


.. parsed-literal::

     135	 1.5152207e+00	 1.2247771e-01	 1.5700801e+00	 1.3649527e-01	  1.2366185e+00 	 2.1055627e-01
     136	 1.5163758e+00	 1.2241767e-01	 1.5713468e+00	 1.3663941e-01	  1.2271425e+00 	 1.9873714e-01


.. parsed-literal::

     137	 1.5179362e+00	 1.2239484e-01	 1.5729347e+00	 1.3653716e-01	  1.2239868e+00 	 2.0908093e-01


.. parsed-literal::

     138	 1.5189346e+00	 1.2238113e-01	 1.5739629e+00	 1.3648096e-01	  1.2226538e+00 	 2.0749474e-01


.. parsed-literal::

     139	 1.5199715e+00	 1.2238373e-01	 1.5750303e+00	 1.3649589e-01	  1.2222261e+00 	 2.1386027e-01


.. parsed-literal::

     140	 1.5215736e+00	 1.2235253e-01	 1.5766943e+00	 1.3644843e-01	  1.2200234e+00 	 2.1997142e-01


.. parsed-literal::

     141	 1.5230720e+00	 1.2231779e-01	 1.5782319e+00	 1.3651575e-01	  1.2223003e+00 	 2.1281838e-01


.. parsed-literal::

     142	 1.5242099e+00	 1.2225301e-01	 1.5793353e+00	 1.3649079e-01	  1.2241046e+00 	 2.1404934e-01
     143	 1.5255933e+00	 1.2217116e-01	 1.5807023e+00	 1.3640331e-01	  1.2245899e+00 	 1.8865824e-01


.. parsed-literal::

     144	 1.5268041e+00	 1.2210339e-01	 1.5819389e+00	 1.3634829e-01	  1.2232926e+00 	 2.1212244e-01
     145	 1.5282836e+00	 1.2201470e-01	 1.5834933e+00	 1.3620041e-01	  1.2189314e+00 	 1.9223142e-01


.. parsed-literal::

     146	 1.5294861e+00	 1.2196523e-01	 1.5847555e+00	 1.3612524e-01	  1.2099631e+00 	 2.1058774e-01


.. parsed-literal::

     147	 1.5304054e+00	 1.2188477e-01	 1.5857088e+00	 1.3599667e-01	  1.2056855e+00 	 2.1289587e-01


.. parsed-literal::

     148	 1.5314504e+00	 1.2178942e-01	 1.5868035e+00	 1.3590868e-01	  1.2014687e+00 	 2.1122336e-01
     149	 1.5324726e+00	 1.2167460e-01	 1.5879039e+00	 1.3570130e-01	  1.1947749e+00 	 1.9080830e-01


.. parsed-literal::

     150	 1.5336120e+00	 1.2155991e-01	 1.5891201e+00	 1.3557247e-01	  1.1883945e+00 	 1.7637801e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.14 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f91186db400>



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
    CPU times: user 1.81 s, sys: 42 ms, total: 1.85 s
    Wall time: 569 ms


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

