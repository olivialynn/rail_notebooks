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
       1	-3.6143438e-01	 3.2641477e-01	-3.5175256e-01	 2.9838588e-01	[-2.9963831e-01]	 4.7577095e-01


.. parsed-literal::

       2	-2.9375095e-01	 3.1710054e-01	-2.7109382e-01	 2.8663022e-01	[-1.8340544e-01]	 2.3429203e-01


.. parsed-literal::

       3	-2.4889778e-01	 2.9618180e-01	-2.0564744e-01	 2.7047322e-01	[-1.0497646e-01]	 2.8638625e-01


.. parsed-literal::

       4	-2.1878983e-01	 2.8085335e-01	-1.7126468e-01	 2.6146125e-01	[-6.4269832e-02]	 3.2418489e-01


.. parsed-literal::

       5	-1.5510516e-01	 2.6418584e-01	-1.1243705e-01	 2.4520744e-01	[-9.9364975e-03]	 2.1360016e-01


.. parsed-literal::

       6	-9.1090352e-02	 2.5671389e-01	-6.1222599e-02	 2.3757897e-01	[ 1.8618140e-02]	 2.1559024e-01


.. parsed-literal::

       7	-7.3464480e-02	 2.5409471e-01	-4.8641566e-02	 2.3267079e-01	[ 2.9163895e-02]	 2.0126724e-01


.. parsed-literal::

       8	-5.7882855e-02	 2.5130856e-01	-3.7696087e-02	 2.3074882e-01	[ 4.0723841e-02]	 2.1165872e-01


.. parsed-literal::

       9	-4.6727254e-02	 2.4924612e-01	-2.9286885e-02	 2.2905431e-01	[ 4.6905613e-02]	 2.1868420e-01


.. parsed-literal::

      10	-3.5697624e-02	 2.4706860e-01	-2.0087780e-02	 2.2720878e-01	[ 5.9109073e-02]	 2.1232772e-01
      11	-2.9250236e-02	 2.4552208e-01	-1.5826805e-02	 2.2769095e-01	  4.6106539e-02 	 1.9801402e-01


.. parsed-literal::

      12	-2.2376645e-02	 2.4478221e-01	-8.8264904e-03	 2.2691160e-01	[ 6.1963912e-02]	 2.1026707e-01
      13	-2.0417539e-02	 2.4437685e-01	-6.7948061e-03	 2.2642408e-01	[ 6.5572624e-02]	 1.9931960e-01


.. parsed-literal::

      14	 1.8671978e-01	 2.2761016e-01	 2.1112062e-01	 2.1533174e-01	[ 2.5986291e-01]	 4.2236328e-01


.. parsed-literal::

      15	 2.1159476e-01	 2.2746454e-01	 2.3757071e-01	 2.1553020e-01	[ 2.7939605e-01]	 2.1276665e-01


.. parsed-literal::

      16	 2.9025849e-01	 2.2212270e-01	 3.1755409e-01	 2.0917165e-01	[ 3.6705484e-01]	 2.1081376e-01


.. parsed-literal::

      17	 3.8758923e-01	 2.1508875e-01	 4.1935313e-01	 2.0095205e-01	[ 4.9224062e-01]	 2.1082425e-01


.. parsed-literal::

      18	 4.3967801e-01	 2.1306586e-01	 4.7306814e-01	 1.9989966e-01	[ 5.4740757e-01]	 2.1064162e-01


.. parsed-literal::

      19	 4.9017978e-01	 2.1054475e-01	 5.2428762e-01	 1.9804956e-01	[ 5.9110150e-01]	 2.1297145e-01


.. parsed-literal::

      20	 5.6399680e-01	 2.0764901e-01	 6.0038370e-01	 1.9415912e-01	[ 6.6918081e-01]	 2.1783948e-01
      21	 6.0685763e-01	 2.0463308e-01	 6.4729958e-01	 1.9179221e-01	[ 7.1634752e-01]	 1.9456053e-01


.. parsed-literal::

      22	 6.4822188e-01	 2.0015352e-01	 6.8715837e-01	 1.8896380e-01	[ 7.4860886e-01]	 2.0958447e-01


.. parsed-literal::

      23	 6.8416273e-01	 1.9667163e-01	 7.2359361e-01	 1.8479469e-01	[ 7.8293664e-01]	 2.2185850e-01
      24	 7.1495569e-01	 2.0149512e-01	 7.5390242e-01	 1.8836400e-01	[ 7.9885594e-01]	 1.9046807e-01


.. parsed-literal::

      25	 7.6005809e-01	 1.9620746e-01	 7.9893376e-01	 1.8409813e-01	[ 8.3865811e-01]	 2.0883441e-01


.. parsed-literal::

      26	 7.9043558e-01	 1.9154636e-01	 8.3020617e-01	 1.8063461e-01	[ 8.6524825e-01]	 2.1608806e-01


.. parsed-literal::

      27	 8.2814351e-01	 1.8987087e-01	 8.6901170e-01	 1.7944782e-01	[ 8.9674919e-01]	 2.1879101e-01


.. parsed-literal::

      28	 8.4371458e-01	 1.9614122e-01	 8.8627876e-01	 1.8206812e-01	[ 9.1515403e-01]	 2.1870208e-01


.. parsed-literal::

      29	 8.7542307e-01	 1.8883528e-01	 9.1863083e-01	 1.7894714e-01	[ 9.4732659e-01]	 2.1544409e-01
      30	 8.8970359e-01	 1.8574670e-01	 9.3239536e-01	 1.7596680e-01	[ 9.6019264e-01]	 1.8251204e-01


.. parsed-literal::

      31	 9.0819449e-01	 1.8299620e-01	 9.5095617e-01	 1.7307458e-01	[ 9.7861576e-01]	 2.0721936e-01


.. parsed-literal::

      32	 9.2927158e-01	 1.7853839e-01	 9.7265158e-01	 1.6730455e-01	[ 9.9782130e-01]	 2.1463633e-01
      33	 9.5303494e-01	 1.7390968e-01	 9.9717150e-01	 1.6285820e-01	[ 1.0141677e+00]	 1.7218471e-01


.. parsed-literal::

      34	 9.7307510e-01	 1.6972176e-01	 1.0176497e+00	 1.5932393e-01	[ 1.0324471e+00]	 1.9980907e-01


.. parsed-literal::

      35	 1.0001596e+00	 1.6546830e-01	 1.0454221e+00	 1.5673660e-01	[ 1.0516124e+00]	 2.1014094e-01
      36	 1.0090528e+00	 1.6594348e-01	 1.0556618e+00	 1.5581139e-01	[ 1.0616381e+00]	 2.0000577e-01


.. parsed-literal::

      37	 1.0287443e+00	 1.6234078e-01	 1.0751318e+00	 1.5418388e-01	[ 1.0734135e+00]	 2.1324682e-01


.. parsed-literal::

      38	 1.0371616e+00	 1.6080944e-01	 1.0835376e+00	 1.5285245e-01	[ 1.0824533e+00]	 2.0568705e-01


.. parsed-literal::

      39	 1.0510662e+00	 1.5843172e-01	 1.0981416e+00	 1.5004440e-01	[ 1.0970582e+00]	 2.1134663e-01
      40	 1.0669302e+00	 1.5581219e-01	 1.1147426e+00	 1.4693099e-01	[ 1.1076772e+00]	 1.9723892e-01


.. parsed-literal::

      41	 1.0767814e+00	 1.5561755e-01	 1.1250085e+00	 1.4458260e-01	[ 1.1084735e+00]	 2.0441604e-01


.. parsed-literal::

      42	 1.0919748e+00	 1.5282766e-01	 1.1396134e+00	 1.4215097e-01	[ 1.1191881e+00]	 2.1315432e-01


.. parsed-literal::

      43	 1.1018252e+00	 1.5160796e-01	 1.1491627e+00	 1.4091552e-01	[ 1.1246642e+00]	 2.0800829e-01


.. parsed-literal::

      44	 1.1147034e+00	 1.5052874e-01	 1.1623032e+00	 1.3942978e-01	[ 1.1270070e+00]	 2.1138144e-01


.. parsed-literal::

      45	 1.1271554e+00	 1.4906148e-01	 1.1749952e+00	 1.3688282e-01	[ 1.1374664e+00]	 2.0431828e-01
      46	 1.1383359e+00	 1.4862970e-01	 1.1861651e+00	 1.3659280e-01	[ 1.1499274e+00]	 1.9812441e-01


.. parsed-literal::

      47	 1.1502686e+00	 1.4810275e-01	 1.1983647e+00	 1.3646364e-01	[ 1.1660355e+00]	 2.1504974e-01


.. parsed-literal::

      48	 1.1631981e+00	 1.4790017e-01	 1.2114381e+00	 1.3663901e-01	[ 1.1819310e+00]	 2.0766282e-01


.. parsed-literal::

      49	 1.1747496e+00	 1.4950174e-01	 1.2236204e+00	 1.3982496e-01	[ 1.1943800e+00]	 2.1029377e-01


.. parsed-literal::

      50	 1.1935189e+00	 1.4813160e-01	 1.2420126e+00	 1.3824744e-01	[ 1.2096422e+00]	 2.0486617e-01


.. parsed-literal::

      51	 1.2006626e+00	 1.4658558e-01	 1.2490254e+00	 1.3685796e-01	[ 1.2125670e+00]	 2.1356010e-01


.. parsed-literal::

      52	 1.2143462e+00	 1.4397499e-01	 1.2630069e+00	 1.3397111e-01	[ 1.2196303e+00]	 2.0190001e-01


.. parsed-literal::

      53	 1.2189757e+00	 1.4299275e-01	 1.2685195e+00	 1.3179336e-01	[ 1.2206590e+00]	 2.1848702e-01


.. parsed-literal::

      54	 1.2314701e+00	 1.4187211e-01	 1.2807617e+00	 1.3103619e-01	[ 1.2353907e+00]	 2.0287609e-01


.. parsed-literal::

      55	 1.2388628e+00	 1.4119846e-01	 1.2881349e+00	 1.3063257e-01	[ 1.2439439e+00]	 2.1367621e-01
      56	 1.2461037e+00	 1.4055100e-01	 1.2956050e+00	 1.3023577e-01	[ 1.2509539e+00]	 1.8872952e-01


.. parsed-literal::

      57	 1.2583556e+00	 1.4015099e-01	 1.3083233e+00	 1.3030034e-01	[ 1.2515984e+00]	 1.9881415e-01


.. parsed-literal::

      58	 1.2625635e+00	 1.3878302e-01	 1.3130683e+00	 1.2892769e-01	  1.2431337e+00 	 2.1427417e-01


.. parsed-literal::

      59	 1.2713516e+00	 1.3885904e-01	 1.3214183e+00	 1.2920964e-01	[ 1.2518697e+00]	 2.1049118e-01


.. parsed-literal::

      60	 1.2758862e+00	 1.3856462e-01	 1.3260958e+00	 1.2915719e-01	  1.2504751e+00 	 2.1982098e-01


.. parsed-literal::

      61	 1.2844894e+00	 1.3829018e-01	 1.3349641e+00	 1.2871063e-01	  1.2482619e+00 	 2.1618605e-01


.. parsed-literal::

      62	 1.2925371e+00	 1.3794085e-01	 1.3431139e+00	 1.2746027e-01	  1.2405004e+00 	 2.2091794e-01
      63	 1.3006595e+00	 1.3799176e-01	 1.3512860e+00	 1.2674420e-01	  1.2418428e+00 	 1.8810487e-01


.. parsed-literal::

      64	 1.3061226e+00	 1.3771317e-01	 1.3568042e+00	 1.2607637e-01	  1.2470043e+00 	 1.9979954e-01
      65	 1.3142304e+00	 1.3715745e-01	 1.3652132e+00	 1.2543091e-01	[ 1.2528123e+00]	 1.7573524e-01


.. parsed-literal::

      66	 1.3235278e+00	 1.3670360e-01	 1.3748650e+00	 1.2548003e-01	  1.2501628e+00 	 1.9841838e-01


.. parsed-literal::

      67	 1.3305527e+00	 1.3632454e-01	 1.3817615e+00	 1.2512220e-01	[ 1.2585421e+00]	 2.0359993e-01
      68	 1.3357817e+00	 1.3612714e-01	 1.3869828e+00	 1.2521043e-01	[ 1.2586843e+00]	 2.0373917e-01


.. parsed-literal::

      69	 1.3412119e+00	 1.3603853e-01	 1.3923204e+00	 1.2491639e-01	[ 1.2664034e+00]	 2.0510340e-01


.. parsed-literal::

      70	 1.3477175e+00	 1.3548429e-01	 1.3989275e+00	 1.2469850e-01	[ 1.2700466e+00]	 2.0814943e-01


.. parsed-literal::

      71	 1.3535771e+00	 1.3528827e-01	 1.4048526e+00	 1.2383664e-01	[ 1.2766016e+00]	 2.1259117e-01


.. parsed-literal::

      72	 1.3587812e+00	 1.3440153e-01	 1.4099180e+00	 1.2311679e-01	[ 1.2852461e+00]	 2.1232915e-01
      73	 1.3640265e+00	 1.3399783e-01	 1.4154467e+00	 1.2242976e-01	  1.2824411e+00 	 1.9592214e-01


.. parsed-literal::

      74	 1.3699095e+00	 1.3378447e-01	 1.4215710e+00	 1.2227073e-01	  1.2772590e+00 	 2.1905255e-01


.. parsed-literal::

      75	 1.3755191e+00	 1.3371050e-01	 1.4272899e+00	 1.2198523e-01	  1.2734569e+00 	 2.2172141e-01
      76	 1.3807186e+00	 1.3354806e-01	 1.4327146e+00	 1.2170156e-01	  1.2664748e+00 	 1.9433236e-01


.. parsed-literal::

      77	 1.3850637e+00	 1.3337989e-01	 1.4370293e+00	 1.2155887e-01	  1.2649989e+00 	 2.0829034e-01


.. parsed-literal::

      78	 1.3900930e+00	 1.3312257e-01	 1.4422484e+00	 1.2116701e-01	  1.2538634e+00 	 2.1457481e-01


.. parsed-literal::

      79	 1.3940068e+00	 1.3316533e-01	 1.4465527e+00	 1.2122176e-01	  1.2359008e+00 	 2.1013761e-01


.. parsed-literal::

      80	 1.3981871e+00	 1.3285826e-01	 1.4507457e+00	 1.2082985e-01	  1.2324847e+00 	 2.0736289e-01


.. parsed-literal::

      81	 1.4016532e+00	 1.3268511e-01	 1.4542449e+00	 1.2062863e-01	  1.2327470e+00 	 2.1105123e-01


.. parsed-literal::

      82	 1.4060580e+00	 1.3248689e-01	 1.4587275e+00	 1.2046331e-01	  1.2287083e+00 	 2.2280836e-01


.. parsed-literal::

      83	 1.4104168e+00	 1.3252927e-01	 1.4632467e+00	 1.1995267e-01	  1.2318851e+00 	 2.1729493e-01


.. parsed-literal::

      84	 1.4160570e+00	 1.3231942e-01	 1.4688010e+00	 1.1992441e-01	  1.2254846e+00 	 2.1241426e-01


.. parsed-literal::

      85	 1.4197526e+00	 1.3215118e-01	 1.4723430e+00	 1.1976078e-01	  1.2289418e+00 	 2.1000409e-01


.. parsed-literal::

      86	 1.4238940e+00	 1.3224453e-01	 1.4764806e+00	 1.1975566e-01	  1.2276407e+00 	 2.1973372e-01


.. parsed-literal::

      87	 1.4272782e+00	 1.3207537e-01	 1.4795784e+00	 1.1914965e-01	  1.2393116e+00 	 2.0230746e-01


.. parsed-literal::

      88	 1.4311241e+00	 1.3218424e-01	 1.4834079e+00	 1.1937009e-01	  1.2435736e+00 	 2.0527196e-01


.. parsed-literal::

      89	 1.4337132e+00	 1.3213769e-01	 1.4861010e+00	 1.1953824e-01	  1.2416746e+00 	 2.1066451e-01


.. parsed-literal::

      90	 1.4376981e+00	 1.3193419e-01	 1.4903640e+00	 1.1988788e-01	  1.2355156e+00 	 2.1467662e-01


.. parsed-literal::

      91	 1.4400576e+00	 1.3159932e-01	 1.4930859e+00	 1.2010657e-01	  1.2250223e+00 	 2.1562815e-01


.. parsed-literal::

      92	 1.4438500e+00	 1.3136924e-01	 1.4967994e+00	 1.2013261e-01	  1.2309009e+00 	 2.2123623e-01


.. parsed-literal::

      93	 1.4459233e+00	 1.3122761e-01	 1.4988099e+00	 1.2006354e-01	  1.2359600e+00 	 2.1145415e-01


.. parsed-literal::

      94	 1.4486040e+00	 1.3106529e-01	 1.5014566e+00	 1.1994083e-01	  1.2417447e+00 	 2.1406460e-01


.. parsed-literal::

      95	 1.4525155e+00	 1.3088596e-01	 1.5054276e+00	 1.1973599e-01	  1.2510061e+00 	 2.1040583e-01


.. parsed-literal::

      96	 1.4549560e+00	 1.3081270e-01	 1.5080575e+00	 1.1956454e-01	  1.2540519e+00 	 2.0402408e-01


.. parsed-literal::

      97	 1.4577627e+00	 1.3069589e-01	 1.5107607e+00	 1.1938645e-01	  1.2574058e+00 	 2.2234535e-01


.. parsed-literal::

      98	 1.4597650e+00	 1.3054958e-01	 1.5128109e+00	 1.1929085e-01	  1.2571800e+00 	 2.1469069e-01


.. parsed-literal::

      99	 1.4617947e+00	 1.3036448e-01	 1.5148584e+00	 1.1914473e-01	  1.2582282e+00 	 2.0846891e-01


.. parsed-literal::

     100	 1.4653949e+00	 1.3006431e-01	 1.5185169e+00	 1.1911465e-01	  1.2561350e+00 	 2.3152947e-01


.. parsed-literal::

     101	 1.4688594e+00	 1.2950972e-01	 1.5218412e+00	 1.1878835e-01	  1.2610546e+00 	 2.0779538e-01


.. parsed-literal::

     102	 1.4707145e+00	 1.2944736e-01	 1.5236386e+00	 1.1890943e-01	  1.2650846e+00 	 2.0489812e-01
     103	 1.4735669e+00	 1.2926666e-01	 1.5264782e+00	 1.1921984e-01	  1.2670190e+00 	 1.9521308e-01


.. parsed-literal::

     104	 1.4760825e+00	 1.2918641e-01	 1.5290953e+00	 1.1985248e-01	  1.2669110e+00 	 1.9070077e-01
     105	 1.4792020e+00	 1.2895250e-01	 1.5322225e+00	 1.1998186e-01	  1.2639820e+00 	 1.9722915e-01


.. parsed-literal::

     106	 1.4823114e+00	 1.2860898e-01	 1.5353740e+00	 1.2007047e-01	  1.2573865e+00 	 2.1422076e-01
     107	 1.4839906e+00	 1.2838407e-01	 1.5370615e+00	 1.2001322e-01	  1.2539230e+00 	 1.9268060e-01


.. parsed-literal::

     108	 1.4872727e+00	 1.2787078e-01	 1.5403308e+00	 1.1969990e-01	  1.2533549e+00 	 2.1931195e-01


.. parsed-literal::

     109	 1.4890962e+00	 1.2756390e-01	 1.5422193e+00	 1.1962734e-01	  1.2460520e+00 	 2.0978427e-01
     110	 1.4912653e+00	 1.2758218e-01	 1.5442642e+00	 1.1937686e-01	  1.2543794e+00 	 1.9110799e-01


.. parsed-literal::

     111	 1.4931034e+00	 1.2753117e-01	 1.5460639e+00	 1.1915957e-01	  1.2590800e+00 	 1.7831922e-01


.. parsed-literal::

     112	 1.4951056e+00	 1.2744762e-01	 1.5480913e+00	 1.1888517e-01	  1.2574739e+00 	 2.1452856e-01
     113	 1.4982163e+00	 1.2732236e-01	 1.5513535e+00	 1.1840414e-01	  1.2477027e+00 	 1.9875479e-01


.. parsed-literal::

     114	 1.5011930e+00	 1.2719007e-01	 1.5543967e+00	 1.1815646e-01	  1.2379020e+00 	 2.1597600e-01


.. parsed-literal::

     115	 1.5028384e+00	 1.2709381e-01	 1.5560684e+00	 1.1823260e-01	  1.2332388e+00 	 2.1142769e-01


.. parsed-literal::

     116	 1.5044052e+00	 1.2694390e-01	 1.5577427e+00	 1.1803949e-01	  1.2221797e+00 	 2.1949530e-01
     117	 1.5063861e+00	 1.2675578e-01	 1.5597729e+00	 1.1815893e-01	  1.2165653e+00 	 1.9780159e-01


.. parsed-literal::

     118	 1.5075364e+00	 1.2671968e-01	 1.5609141e+00	 1.1818610e-01	  1.2180249e+00 	 1.8912411e-01


.. parsed-literal::

     119	 1.5094729e+00	 1.2655673e-01	 1.5629332e+00	 1.1804299e-01	  1.2137694e+00 	 2.1078300e-01


.. parsed-literal::

     120	 1.5119131e+00	 1.2628888e-01	 1.5654718e+00	 1.1794834e-01	  1.2110750e+00 	 2.1280265e-01
     121	 1.5141666e+00	 1.2613039e-01	 1.5678117e+00	 1.1801134e-01	  1.2091010e+00 	 1.9718528e-01


.. parsed-literal::

     122	 1.5153454e+00	 1.2605790e-01	 1.5689912e+00	 1.1800278e-01	  1.2104420e+00 	 2.1416616e-01


.. parsed-literal::

     123	 1.5167534e+00	 1.2590726e-01	 1.5704192e+00	 1.1801759e-01	  1.2081554e+00 	 2.1713638e-01
     124	 1.5184056e+00	 1.2566112e-01	 1.5721361e+00	 1.1796847e-01	  1.2040132e+00 	 1.8885469e-01


.. parsed-literal::

     125	 1.5201299e+00	 1.2546513e-01	 1.5738679e+00	 1.1794046e-01	  1.1996598e+00 	 2.0824409e-01
     126	 1.5217861e+00	 1.2532725e-01	 1.5755108e+00	 1.1788098e-01	  1.1990226e+00 	 1.8938470e-01


.. parsed-literal::

     127	 1.5232090e+00	 1.2509717e-01	 1.5769375e+00	 1.1766554e-01	  1.1978988e+00 	 1.8660307e-01


.. parsed-literal::

     128	 1.5247652e+00	 1.2514693e-01	 1.5784345e+00	 1.1769743e-01	  1.2048216e+00 	 2.0601964e-01


.. parsed-literal::

     129	 1.5258580e+00	 1.2513974e-01	 1.5795218e+00	 1.1763270e-01	  1.2070954e+00 	 2.1052098e-01


.. parsed-literal::

     130	 1.5277520e+00	 1.2510345e-01	 1.5814771e+00	 1.1761846e-01	  1.2103697e+00 	 2.0923042e-01


.. parsed-literal::

     131	 1.5282284e+00	 1.2519072e-01	 1.5820472e+00	 1.1704447e-01	  1.2089502e+00 	 2.1353459e-01
     132	 1.5304390e+00	 1.2501668e-01	 1.5842173e+00	 1.1719462e-01	  1.2098745e+00 	 1.7295218e-01


.. parsed-literal::

     133	 1.5316478e+00	 1.2488124e-01	 1.5854550e+00	 1.1715310e-01	  1.2087058e+00 	 2.1198630e-01


.. parsed-literal::

     134	 1.5327536e+00	 1.2475454e-01	 1.5866114e+00	 1.1695918e-01	  1.2077389e+00 	 2.1204376e-01


.. parsed-literal::

     135	 1.5346270e+00	 1.2458484e-01	 1.5885171e+00	 1.1659785e-01	  1.2095676e+00 	 2.2273707e-01


.. parsed-literal::

     136	 1.5357582e+00	 1.2446150e-01	 1.5896975e+00	 1.1629019e-01	  1.2113529e+00 	 3.1910944e-01


.. parsed-literal::

     137	 1.5373235e+00	 1.2440528e-01	 1.5912378e+00	 1.1598383e-01	  1.2167420e+00 	 2.2154641e-01


.. parsed-literal::

     138	 1.5382547e+00	 1.2436854e-01	 1.5921372e+00	 1.1591239e-01	  1.2188057e+00 	 2.1292639e-01


.. parsed-literal::

     139	 1.5396944e+00	 1.2429158e-01	 1.5935852e+00	 1.1581826e-01	  1.2212247e+00 	 2.1188760e-01
     140	 1.5411563e+00	 1.2415670e-01	 1.5951253e+00	 1.1578206e-01	  1.2150717e+00 	 1.9490838e-01


.. parsed-literal::

     141	 1.5425503e+00	 1.2405867e-01	 1.5965971e+00	 1.1578174e-01	  1.2101144e+00 	 2.0508909e-01


.. parsed-literal::

     142	 1.5440692e+00	 1.2392110e-01	 1.5983037e+00	 1.1583782e-01	  1.2008359e+00 	 2.1517491e-01


.. parsed-literal::

     143	 1.5452039e+00	 1.2385241e-01	 1.5994561e+00	 1.1579276e-01	  1.1956024e+00 	 2.1562004e-01


.. parsed-literal::

     144	 1.5461712e+00	 1.2381695e-01	 1.6004033e+00	 1.1570945e-01	  1.1966343e+00 	 2.1078491e-01


.. parsed-literal::

     145	 1.5483805e+00	 1.2359541e-01	 1.6026290e+00	 1.1554867e-01	  1.1883349e+00 	 2.1424770e-01


.. parsed-literal::

     146	 1.5492236e+00	 1.2358819e-01	 1.6034698e+00	 1.1541562e-01	  1.1966448e+00 	 3.3216286e-01


.. parsed-literal::

     147	 1.5499027e+00	 1.2354437e-01	 1.6041304e+00	 1.1542232e-01	  1.1960660e+00 	 2.0635796e-01
     148	 1.5522212e+00	 1.2333066e-01	 1.6064592e+00	 1.1545182e-01	  1.1852816e+00 	 2.0196915e-01


.. parsed-literal::

     149	 1.5528615e+00	 1.2325654e-01	 1.6071483e+00	 1.1557338e-01	  1.1792590e+00 	 2.0898771e-01
     150	 1.5539162e+00	 1.2321194e-01	 1.6082002e+00	 1.1554278e-01	  1.1762996e+00 	 2.0342207e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.12 s, total: 2min 8s
    Wall time: 32.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fbe3cf47d00>



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
    CPU times: user 2.23 s, sys: 42 ms, total: 2.27 s
    Wall time: 729 ms


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

