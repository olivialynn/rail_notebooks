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
       1	-3.4211886e-01	 3.2050629e-01	-3.3238824e-01	 3.2134872e-01	[-3.3387714e-01]	 4.5919871e-01


.. parsed-literal::

       2	-2.7256693e-01	 3.0994720e-01	-2.4857038e-01	 3.0999319e-01	[-2.4904403e-01]	 2.2857070e-01


.. parsed-literal::

       3	-2.3005520e-01	 2.8997450e-01	-1.8830689e-01	 2.9075754e-01	[-1.8877501e-01]	 2.8510785e-01


.. parsed-literal::

       4	-1.9001079e-01	 2.6684053e-01	-1.4887071e-01	 2.6206616e-01	[-1.2413630e-01]	 2.0702505e-01


.. parsed-literal::

       5	-1.0598183e-01	 2.5742508e-01	-7.3789675e-02	 2.5406038e-01	[-5.7319394e-02]	 2.1786141e-01


.. parsed-literal::

       6	-7.0414566e-02	 2.5200864e-01	-4.1780790e-02	 2.4787853e-01	[-2.7040243e-02]	 2.0499158e-01


.. parsed-literal::

       7	-5.4597738e-02	 2.4959313e-01	-3.1170719e-02	 2.4557292e-01	[-1.6032049e-02]	 2.1151042e-01
       8	-4.1053810e-02	 2.4728293e-01	-2.1495002e-02	 2.4358648e-01	[-7.0792997e-03]	 1.8284678e-01


.. parsed-literal::

       9	-2.9470800e-02	 2.4514466e-01	-1.2377889e-02	 2.4089457e-01	[ 4.6977010e-03]	 2.1380949e-01


.. parsed-literal::

      10	-2.1479106e-02	 2.4401448e-01	-6.5286050e-03	 2.3941134e-01	[ 1.1468779e-02]	 2.0978093e-01
      11	-1.5233997e-02	 2.4256422e-01	-7.5030581e-04	 2.3804221e-01	[ 1.7121859e-02]	 1.8638229e-01


.. parsed-literal::

      12	-1.3031183e-02	 2.4215590e-01	 1.2352061e-03	 2.3754502e-01	[ 1.9837551e-02]	 2.1359348e-01


.. parsed-literal::

      13	-7.6234822e-03	 2.4108301e-01	 6.5674357e-03	 2.3630667e-01	[ 2.4830024e-02]	 2.1105647e-01


.. parsed-literal::

      14	 1.8470872e-01	 2.2630726e-01	 2.1046218e-01	 2.2035548e-01	[ 2.2111026e-01]	 4.4625926e-01


.. parsed-literal::

      15	 2.2439386e-01	 2.2107066e-01	 2.5249473e-01	 2.1829932e-01	[ 2.5156209e-01]	 2.9054523e-01


.. parsed-literal::

      16	 2.8245184e-01	 2.1539727e-01	 3.1161158e-01	 2.1487326e-01	[ 3.0162370e-01]	 2.0668149e-01


.. parsed-literal::

      17	 3.5220480e-01	 2.1060397e-01	 3.8363316e-01	 2.0874488e-01	[ 3.6865525e-01]	 2.0727825e-01


.. parsed-literal::

      18	 4.0753462e-01	 2.0448208e-01	 4.4139473e-01	 2.0285693e-01	[ 4.0894639e-01]	 2.0286083e-01


.. parsed-literal::

      19	 4.8861081e-01	 2.0062434e-01	 5.2392929e-01	 2.0015543e-01	[ 4.7879101e-01]	 2.1107960e-01


.. parsed-literal::

      20	 5.8151093e-01	 1.9977816e-01	 6.1857251e-01	 1.9805352e-01	[ 5.6513661e-01]	 2.1045232e-01
      21	 6.3665822e-01	 1.9177393e-01	 6.7596345e-01	 1.8738070e-01	[ 6.2482398e-01]	 2.0499730e-01


.. parsed-literal::

      22	 6.8055031e-01	 1.8657243e-01	 7.1900350e-01	 1.8097057e-01	[ 6.7141990e-01]	 1.7553878e-01


.. parsed-literal::

      23	 7.1256067e-01	 1.8340575e-01	 7.5074048e-01	 1.7688059e-01	[ 7.0750318e-01]	 2.1153522e-01
      24	 7.5141937e-01	 1.8209741e-01	 7.8919742e-01	 1.7440783e-01	[ 7.5291487e-01]	 1.7369223e-01


.. parsed-literal::

      25	 7.7868614e-01	 1.9103198e-01	 8.1783330e-01	 1.7932906e-01	[ 7.9656439e-01]	 2.0725346e-01
      26	 8.1581807e-01	 1.8559077e-01	 8.5446093e-01	 1.7320676e-01	[ 8.2180287e-01]	 1.9803429e-01


.. parsed-literal::

      27	 8.4326839e-01	 1.8219780e-01	 8.8211363e-01	 1.6986140e-01	[ 8.5048956e-01]	 1.7330217e-01
      28	 8.7926788e-01	 1.7504297e-01	 9.2074284e-01	 1.6501311e-01	[ 8.8097750e-01]	 1.9749188e-01


.. parsed-literal::

      29	 9.0614770e-01	 1.7368715e-01	 9.4702721e-01	 1.6368050e-01	[ 9.0313342e-01]	 2.0506215e-01
      30	 9.2209733e-01	 1.6955395e-01	 9.6283272e-01	 1.6143722e-01	[ 9.1355418e-01]	 1.9599080e-01


.. parsed-literal::

      31	 9.4881519e-01	 1.6669653e-01	 9.8999112e-01	 1.5831945e-01	[ 9.3964552e-01]	 2.0878863e-01


.. parsed-literal::

      32	 9.7192654e-01	 1.6362254e-01	 1.0146726e+00	 1.5441449e-01	[ 9.5797636e-01]	 2.1056795e-01
      33	 9.9109919e-01	 1.6135573e-01	 1.0348558e+00	 1.5226597e-01	[ 9.7870827e-01]	 1.7827201e-01


.. parsed-literal::

      34	 1.0079550e+00	 1.5934845e-01	 1.0521637e+00	 1.5081764e-01	[ 9.9664839e-01]	 2.1369290e-01


.. parsed-literal::

      35	 1.0246566e+00	 1.5774699e-01	 1.0693287e+00	 1.4803043e-01	[ 1.0177986e+00]	 2.1348810e-01
      36	 1.0396461e+00	 1.5714395e-01	 1.0849424e+00	 1.4760419e-01	[ 1.0296701e+00]	 1.9513535e-01


.. parsed-literal::

      37	 1.0529120e+00	 1.5663770e-01	 1.0983274e+00	 1.4680081e-01	[ 1.0391173e+00]	 2.0786071e-01


.. parsed-literal::

      38	 1.0688464e+00	 1.5655336e-01	 1.1148253e+00	 1.4516982e-01	[ 1.0541686e+00]	 2.1044397e-01


.. parsed-literal::

      39	 1.0815940e+00	 1.5623501e-01	 1.1281318e+00	 1.4477388e-01	[ 1.0574076e+00]	 2.0465565e-01


.. parsed-literal::

      40	 1.0916155e+00	 1.5588176e-01	 1.1383680e+00	 1.4396253e-01	[ 1.0666448e+00]	 2.0752549e-01
      41	 1.1014525e+00	 1.5471111e-01	 1.1479660e+00	 1.4376288e-01	[ 1.0762728e+00]	 1.9686341e-01


.. parsed-literal::

      42	 1.1111396e+00	 1.5424765e-01	 1.1579628e+00	 1.4443932e-01	[ 1.0791397e+00]	 1.9679904e-01
      43	 1.1214072e+00	 1.5315294e-01	 1.1679221e+00	 1.4504557e-01	[ 1.0916742e+00]	 1.7439723e-01


.. parsed-literal::

      44	 1.1291791e+00	 1.5263103e-01	 1.1756475e+00	 1.4465942e-01	[ 1.1002471e+00]	 1.9996166e-01


.. parsed-literal::

      45	 1.1484192e+00	 1.5171637e-01	 1.1950390e+00	 1.4378327e-01	[ 1.1170136e+00]	 2.0404863e-01


.. parsed-literal::

      46	 1.1548176e+00	 1.5097905e-01	 1.2015925e+00	 1.4315338e-01	[ 1.1235506e+00]	 3.3960009e-01
      47	 1.1618820e+00	 1.5026141e-01	 1.2086491e+00	 1.4197958e-01	[ 1.1294897e+00]	 1.8005419e-01


.. parsed-literal::

      48	 1.1756052e+00	 1.4899996e-01	 1.2231762e+00	 1.3833232e-01	[ 1.1429704e+00]	 2.1272469e-01
      49	 1.1836727e+00	 1.4792161e-01	 1.2314688e+00	 1.3682140e-01	[ 1.1544874e+00]	 2.0207977e-01


.. parsed-literal::

      50	 1.1931313e+00	 1.4723734e-01	 1.2409611e+00	 1.3670878e-01	[ 1.1655202e+00]	 2.0989418e-01


.. parsed-literal::

      51	 1.2037033e+00	 1.4641686e-01	 1.2518510e+00	 1.3698702e-01	[ 1.1749447e+00]	 2.1003151e-01


.. parsed-literal::

      52	 1.2123945e+00	 1.4557811e-01	 1.2607329e+00	 1.3853474e-01	[ 1.1773394e+00]	 2.0969629e-01


.. parsed-literal::

      53	 1.2223903e+00	 1.4517805e-01	 1.2707838e+00	 1.3885281e-01	[ 1.1807263e+00]	 2.1865273e-01


.. parsed-literal::

      54	 1.2317240e+00	 1.4458322e-01	 1.2804729e+00	 1.3847131e-01	[ 1.1809926e+00]	 2.0959902e-01
      55	 1.2397776e+00	 1.4439114e-01	 1.2887543e+00	 1.3772158e-01	[ 1.1854758e+00]	 1.9260621e-01


.. parsed-literal::

      56	 1.2471433e+00	 1.4423193e-01	 1.2967593e+00	 1.3558968e-01	[ 1.1915192e+00]	 2.0563912e-01


.. parsed-literal::

      57	 1.2566976e+00	 1.4397965e-01	 1.3063011e+00	 1.3531936e-01	[ 1.1969914e+00]	 2.0447803e-01
      58	 1.2616862e+00	 1.4381971e-01	 1.3111205e+00	 1.3524726e-01	[ 1.2006038e+00]	 1.9303823e-01


.. parsed-literal::

      59	 1.2684647e+00	 1.4297582e-01	 1.3182195e+00	 1.3569889e-01	[ 1.2055084e+00]	 2.1168995e-01
      60	 1.2758925e+00	 1.4262787e-01	 1.3258040e+00	 1.3546969e-01	[ 1.2090341e+00]	 1.8883777e-01


.. parsed-literal::

      61	 1.2819740e+00	 1.4211508e-01	 1.3317468e+00	 1.3550653e-01	[ 1.2177762e+00]	 2.1965671e-01


.. parsed-literal::

      62	 1.2892845e+00	 1.4102568e-01	 1.3391730e+00	 1.3468702e-01	[ 1.2251095e+00]	 2.1731734e-01


.. parsed-literal::

      63	 1.2968411e+00	 1.4054218e-01	 1.3469074e+00	 1.3311982e-01	[ 1.2314609e+00]	 2.0221210e-01


.. parsed-literal::

      64	 1.3073354e+00	 1.4040578e-01	 1.3573761e+00	 1.3016834e-01	[ 1.2409117e+00]	 2.0604873e-01


.. parsed-literal::

      65	 1.3160831e+00	 1.3994424e-01	 1.3662230e+00	 1.2908101e-01	[ 1.2441752e+00]	 2.1193790e-01
      66	 1.3238314e+00	 1.3986107e-01	 1.3740819e+00	 1.2837952e-01	[ 1.2483360e+00]	 1.9592404e-01


.. parsed-literal::

      67	 1.3306360e+00	 1.3881620e-01	 1.3808851e+00	 1.2901169e-01	[ 1.2536007e+00]	 2.1483731e-01
      68	 1.3394797e+00	 1.3708461e-01	 1.3899674e+00	 1.3001166e-01	[ 1.2596241e+00]	 1.7588329e-01


.. parsed-literal::

      69	 1.3462878e+00	 1.3593203e-01	 1.3968773e+00	 1.3223419e-01	[ 1.2669707e+00]	 2.1632218e-01


.. parsed-literal::

      70	 1.3527563e+00	 1.3535145e-01	 1.4032766e+00	 1.3128314e-01	[ 1.2675858e+00]	 2.1339083e-01
      71	 1.3613033e+00	 1.3482303e-01	 1.4119654e+00	 1.2992753e-01	  1.2651395e+00 	 1.9576120e-01


.. parsed-literal::

      72	 1.3662425e+00	 1.3437772e-01	 1.4170297e+00	 1.2843654e-01	[ 1.2677772e+00]	 1.8350768e-01
      73	 1.3721516e+00	 1.3399897e-01	 1.4229862e+00	 1.2805189e-01	[ 1.2707569e+00]	 1.8642545e-01


.. parsed-literal::

      74	 1.3804839e+00	 1.3326028e-01	 1.4315755e+00	 1.2733987e-01	[ 1.2717353e+00]	 2.1231866e-01


.. parsed-literal::

      75	 1.3846904e+00	 1.3306960e-01	 1.4358831e+00	 1.2784334e-01	  1.2649887e+00 	 2.1889067e-01


.. parsed-literal::

      76	 1.3896168e+00	 1.3273210e-01	 1.4406685e+00	 1.2695415e-01	[ 1.2732054e+00]	 2.1502280e-01
      77	 1.3939781e+00	 1.3241521e-01	 1.4451550e+00	 1.2655858e-01	[ 1.2752607e+00]	 2.0180964e-01


.. parsed-literal::

      78	 1.3982662e+00	 1.3209677e-01	 1.4496450e+00	 1.2605722e-01	  1.2741416e+00 	 2.1644902e-01
      79	 1.4027541e+00	 1.3167417e-01	 1.4543279e+00	 1.2655660e-01	  1.2659188e+00 	 1.9552279e-01


.. parsed-literal::

      80	 1.4087648e+00	 1.3162946e-01	 1.4603521e+00	 1.2679339e-01	  1.2699230e+00 	 2.0847273e-01
      81	 1.4123561e+00	 1.3172366e-01	 1.4639287e+00	 1.2691025e-01	  1.2710789e+00 	 1.9835401e-01


.. parsed-literal::

      82	 1.4178850e+00	 1.3170136e-01	 1.4695486e+00	 1.2658179e-01	  1.2709144e+00 	 2.0918298e-01
      83	 1.4239671e+00	 1.3142900e-01	 1.4756602e+00	 1.2431913e-01	  1.2708089e+00 	 1.9921708e-01


.. parsed-literal::

      84	 1.4266799e+00	 1.3165139e-01	 1.4785471e+00	 1.2515319e-01	  1.2724076e+00 	 2.0514512e-01


.. parsed-literal::

      85	 1.4304337e+00	 1.3120289e-01	 1.4821058e+00	 1.2383872e-01	[ 1.2810085e+00]	 2.1214676e-01


.. parsed-literal::

      86	 1.4328632e+00	 1.3099621e-01	 1.4845910e+00	 1.2330357e-01	[ 1.2816004e+00]	 2.0987177e-01


.. parsed-literal::

      87	 1.4363620e+00	 1.3078756e-01	 1.4882284e+00	 1.2284334e-01	[ 1.2836968e+00]	 2.1075296e-01


.. parsed-literal::

      88	 1.4412184e+00	 1.3073236e-01	 1.4931342e+00	 1.2427216e-01	[ 1.2893524e+00]	 2.0923328e-01
      89	 1.4443726e+00	 1.3087466e-01	 1.4963551e+00	 1.2417388e-01	[ 1.2966527e+00]	 2.0781660e-01


.. parsed-literal::

      90	 1.4472680e+00	 1.3088461e-01	 1.4991229e+00	 1.2452634e-01	[ 1.2994219e+00]	 1.7663980e-01
      91	 1.4494645e+00	 1.3087986e-01	 1.5013361e+00	 1.2451659e-01	[ 1.2995432e+00]	 1.8042517e-01


.. parsed-literal::

      92	 1.4526678e+00	 1.3067968e-01	 1.5046572e+00	 1.2350429e-01	[ 1.3011209e+00]	 2.0838547e-01
      93	 1.4556238e+00	 1.3044460e-01	 1.5078486e+00	 1.2265551e-01	  1.2948368e+00 	 1.9992638e-01


.. parsed-literal::

      94	 1.4600030e+00	 1.3020302e-01	 1.5122756e+00	 1.2081029e-01	  1.3009367e+00 	 1.7197561e-01
      95	 1.4629391e+00	 1.2999838e-01	 1.5151897e+00	 1.1972717e-01	[ 1.3046764e+00]	 1.8886805e-01


.. parsed-literal::

      96	 1.4657833e+00	 1.2968901e-01	 1.5181932e+00	 1.1928032e-01	  1.3033158e+00 	 1.9736028e-01


.. parsed-literal::

      97	 1.4683978e+00	 1.2930269e-01	 1.5209841e+00	 1.1886457e-01	  1.2983439e+00 	 2.1529627e-01


.. parsed-literal::

      98	 1.4715622e+00	 1.2906297e-01	 1.5241842e+00	 1.2000596e-01	  1.2967662e+00 	 2.1374893e-01


.. parsed-literal::

      99	 1.4739460e+00	 1.2903095e-01	 1.5265404e+00	 1.2095449e-01	  1.2965905e+00 	 2.0915461e-01
     100	 1.4768167e+00	 1.2903335e-01	 1.5294245e+00	 1.2110391e-01	  1.2969068e+00 	 1.6868258e-01


.. parsed-literal::

     101	 1.4795119e+00	 1.2877580e-01	 1.5322891e+00	 1.2042722e-01	  1.2964994e+00 	 2.0932269e-01


.. parsed-literal::

     102	 1.4822546e+00	 1.2880658e-01	 1.5349525e+00	 1.1935302e-01	  1.3008768e+00 	 2.0244002e-01


.. parsed-literal::

     103	 1.4840563e+00	 1.2871329e-01	 1.5367692e+00	 1.1850858e-01	  1.3024822e+00 	 2.1087313e-01
     104	 1.4868490e+00	 1.2852672e-01	 1.5395910e+00	 1.1778207e-01	  1.3033733e+00 	 1.9647241e-01


.. parsed-literal::

     105	 1.4905612e+00	 1.2821663e-01	 1.5433171e+00	 1.1732327e-01	[ 1.3073100e+00]	 1.8945384e-01


.. parsed-literal::

     106	 1.4935358e+00	 1.2799863e-01	 1.5462530e+00	 1.1830740e-01	[ 1.3076200e+00]	 2.1056104e-01


.. parsed-literal::

     107	 1.4958313e+00	 1.2798311e-01	 1.5484790e+00	 1.1877067e-01	[ 1.3089965e+00]	 2.0766854e-01


.. parsed-literal::

     108	 1.4980684e+00	 1.2802122e-01	 1.5507648e+00	 1.1937563e-01	[ 1.3096627e+00]	 2.0748377e-01


.. parsed-literal::

     109	 1.5001617e+00	 1.2814138e-01	 1.5530210e+00	 1.1966335e-01	  1.3065835e+00 	 2.0671821e-01


.. parsed-literal::

     110	 1.5028369e+00	 1.2817561e-01	 1.5558348e+00	 1.1923262e-01	  1.3066727e+00 	 2.1711564e-01


.. parsed-literal::

     111	 1.5052620e+00	 1.2826109e-01	 1.5583610e+00	 1.1874305e-01	  1.3056575e+00 	 2.1113539e-01


.. parsed-literal::

     112	 1.5075045e+00	 1.2823703e-01	 1.5606943e+00	 1.1798242e-01	  1.3040207e+00 	 2.1199274e-01


.. parsed-literal::

     113	 1.5091804e+00	 1.2826308e-01	 1.5625842e+00	 1.1813339e-01	  1.2995234e+00 	 2.0182920e-01


.. parsed-literal::

     114	 1.5111443e+00	 1.2806379e-01	 1.5645191e+00	 1.1804547e-01	  1.2967359e+00 	 2.1022129e-01
     115	 1.5120087e+00	 1.2794598e-01	 1.5653668e+00	 1.1836295e-01	  1.2950071e+00 	 1.7870045e-01


.. parsed-literal::

     116	 1.5137919e+00	 1.2775586e-01	 1.5672229e+00	 1.1892188e-01	  1.2883185e+00 	 1.7243719e-01
     117	 1.5151427e+00	 1.2753101e-01	 1.5689464e+00	 1.1851215e-01	  1.2720063e+00 	 1.9631529e-01


.. parsed-literal::

     118	 1.5180933e+00	 1.2742077e-01	 1.5717795e+00	 1.1917176e-01	  1.2673856e+00 	 2.0969534e-01


.. parsed-literal::

     119	 1.5194676e+00	 1.2738657e-01	 1.5731931e+00	 1.1889165e-01	  1.2653570e+00 	 2.0210624e-01


.. parsed-literal::

     120	 1.5212156e+00	 1.2729492e-01	 1.5749961e+00	 1.1852876e-01	  1.2575522e+00 	 2.0360518e-01
     121	 1.5224041e+00	 1.2721880e-01	 1.5763454e+00	 1.1817810e-01	  1.2475442e+00 	 1.9880557e-01


.. parsed-literal::

     122	 1.5242762e+00	 1.2714925e-01	 1.5780894e+00	 1.1807275e-01	  1.2451121e+00 	 2.0492005e-01
     123	 1.5256404e+00	 1.2703370e-01	 1.5794302e+00	 1.1821329e-01	  1.2395102e+00 	 1.9881177e-01


.. parsed-literal::

     124	 1.5270889e+00	 1.2694003e-01	 1.5808574e+00	 1.1824472e-01	  1.2358763e+00 	 1.8177271e-01
     125	 1.5292539e+00	 1.2673319e-01	 1.5831207e+00	 1.1870514e-01	  1.2262661e+00 	 1.9414282e-01


.. parsed-literal::

     126	 1.5312837e+00	 1.2668903e-01	 1.5850994e+00	 1.1850335e-01	  1.2266496e+00 	 1.9343662e-01


.. parsed-literal::

     127	 1.5325389e+00	 1.2665502e-01	 1.5863298e+00	 1.1845836e-01	  1.2286968e+00 	 2.1262217e-01


.. parsed-literal::

     128	 1.5337664e+00	 1.2664876e-01	 1.5875700e+00	 1.1862529e-01	  1.2306233e+00 	 2.0581007e-01
     129	 1.5350427e+00	 1.2661869e-01	 1.5889067e+00	 1.1889547e-01	  1.2312698e+00 	 1.8298864e-01


.. parsed-literal::

     130	 1.5368244e+00	 1.2660120e-01	 1.5907154e+00	 1.1921629e-01	  1.2322747e+00 	 2.0771027e-01


.. parsed-literal::

     131	 1.5382927e+00	 1.2659050e-01	 1.5922362e+00	 1.1994621e-01	  1.2277990e+00 	 2.0904493e-01


.. parsed-literal::

     132	 1.5391330e+00	 1.2642131e-01	 1.5931269e+00	 1.1948850e-01	  1.2300549e+00 	 2.0222926e-01


.. parsed-literal::

     133	 1.5398205e+00	 1.2641420e-01	 1.5937737e+00	 1.1938365e-01	  1.2286679e+00 	 2.1853685e-01


.. parsed-literal::

     134	 1.5409813e+00	 1.2637221e-01	 1.5949605e+00	 1.1930312e-01	  1.2237202e+00 	 2.1267772e-01


.. parsed-literal::

     135	 1.5420312e+00	 1.2629274e-01	 1.5960821e+00	 1.1900046e-01	  1.2181055e+00 	 2.2101855e-01


.. parsed-literal::

     136	 1.5429365e+00	 1.2623143e-01	 1.5971502e+00	 1.1871998e-01	  1.2179936e+00 	 2.1109462e-01


.. parsed-literal::

     137	 1.5446240e+00	 1.2615607e-01	 1.5987920e+00	 1.1835717e-01	  1.2170413e+00 	 2.1898985e-01


.. parsed-literal::

     138	 1.5457180e+00	 1.2608503e-01	 1.5998756e+00	 1.1805055e-01	  1.2199930e+00 	 2.1219563e-01
     139	 1.5466494e+00	 1.2599432e-01	 1.6008074e+00	 1.1775540e-01	  1.2233839e+00 	 1.7954206e-01


.. parsed-literal::

     140	 1.5484168e+00	 1.2575868e-01	 1.6025989e+00	 1.1716248e-01	  1.2262167e+00 	 2.2651005e-01


.. parsed-literal::

     141	 1.5491778e+00	 1.2537027e-01	 1.6036248e+00	 1.1702785e-01	  1.2268096e+00 	 2.1768355e-01
     142	 1.5512188e+00	 1.2533683e-01	 1.6054738e+00	 1.1690322e-01	  1.2253848e+00 	 1.9060206e-01


.. parsed-literal::

     143	 1.5519371e+00	 1.2531320e-01	 1.6061810e+00	 1.1702513e-01	  1.2232180e+00 	 2.1020365e-01


.. parsed-literal::

     144	 1.5530828e+00	 1.2515092e-01	 1.6074038e+00	 1.1724934e-01	  1.2208828e+00 	 2.1904802e-01


.. parsed-literal::

     145	 1.5542671e+00	 1.2501244e-01	 1.6086365e+00	 1.1757578e-01	  1.2170727e+00 	 2.2278571e-01
     146	 1.5553030e+00	 1.2490888e-01	 1.6096814e+00	 1.1743167e-01	  1.2211741e+00 	 2.0814300e-01


.. parsed-literal::

     147	 1.5563053e+00	 1.2483827e-01	 1.6106820e+00	 1.1734397e-01	  1.2235513e+00 	 2.0058274e-01


.. parsed-literal::

     148	 1.5573072e+00	 1.2472278e-01	 1.6116853e+00	 1.1725768e-01	  1.2227616e+00 	 2.0857143e-01


.. parsed-literal::

     149	 1.5588141e+00	 1.2458023e-01	 1.6132381e+00	 1.1765378e-01	  1.2143767e+00 	 2.1800303e-01


.. parsed-literal::

     150	 1.5597757e+00	 1.2436421e-01	 1.6142215e+00	 1.1782565e-01	  1.2084447e+00 	 2.1641994e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.02 s, total: 2min 5s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f9501e31870>



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
    CPU times: user 1.78 s, sys: 37 ms, total: 1.81 s
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

