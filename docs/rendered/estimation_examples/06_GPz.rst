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
       1	-3.5211446e-01	 3.2346948e-01	-3.4244844e-01	 3.0850342e-01	[-3.1408723e-01]	 4.6129799e-01


.. parsed-literal::

       2	-2.8469550e-01	 3.1416300e-01	-2.6232680e-01	 2.9958298e-01	[-2.1653853e-01]	 2.2924137e-01


.. parsed-literal::

       3	-2.4050196e-01	 2.9256571e-01	-1.9916338e-01	 2.7405580e-01	[-1.2289141e-01]	 2.7796268e-01
       4	-1.9579071e-01	 2.6949125e-01	-1.5413826e-01	 2.4931019e-01	[-4.6663790e-02]	 1.7961860e-01


.. parsed-literal::

       5	-1.1068895e-01	 2.5975635e-01	-8.1245056e-02	 2.4271947e-01	[ 3.5651502e-03]	 2.1587873e-01


.. parsed-literal::

       6	-8.1837458e-02	 2.5517656e-01	-5.4794907e-02	 2.3469324e-01	[ 1.5629808e-02]	 2.5296879e-01
       7	-6.5120840e-02	 2.5242982e-01	-4.2822415e-02	 2.3270031e-01	[ 3.1569729e-02]	 1.8709850e-01


.. parsed-literal::

       8	-5.1495894e-02	 2.4999396e-01	-3.2690421e-02	 2.3092412e-01	[ 4.3375416e-02]	 2.1270418e-01
       9	-4.0264853e-02	 2.4781142e-01	-2.3447431e-02	 2.2864202e-01	[ 5.3254276e-02]	 1.8693757e-01


.. parsed-literal::

      10	-3.1125183e-02	 2.4606397e-01	-1.6179536e-02	 2.2795218e-01	[ 5.9373033e-02]	 2.1358562e-01


.. parsed-literal::

      11	-2.6793703e-02	 2.4524762e-01	-1.2438129e-02	 2.2685881e-01	  5.6997482e-02 	 2.0413947e-01
      12	-2.4271039e-02	 2.4480080e-01	-1.0034422e-02	 2.2623194e-01	[ 6.3495795e-02]	 1.8567038e-01


.. parsed-literal::

      13	-2.0304052e-02	 2.4393229e-01	-5.9529951e-03	 2.2535287e-01	[ 6.9063415e-02]	 2.0333695e-01


.. parsed-literal::

      14	 1.2197190e-01	 2.2453228e-01	 1.4723822e-01	 2.1696456e-01	[ 2.0826187e-01]	 4.3651271e-01


.. parsed-literal::

      15	 1.8379493e-01	 2.3115370e-01	 2.1172156e-01	 2.1618128e-01	[ 2.4213085e-01]	 3.2407999e-01


.. parsed-literal::

      16	 2.4468182e-01	 2.2220696e-01	 2.7306214e-01	 2.1012745e-01	[ 2.9741146e-01]	 2.1156836e-01
      17	 3.1797293e-01	 2.1427529e-01	 3.4896599e-01	 2.0799769e-01	[ 3.6214949e-01]	 1.9688678e-01


.. parsed-literal::

      18	 3.7438275e-01	 2.1156229e-01	 4.0872391e-01	 2.0367528e-01	[ 4.2148205e-01]	 2.1456504e-01


.. parsed-literal::

      19	 4.3437829e-01	 2.0950977e-01	 4.6886468e-01	 2.0276124e-01	[ 4.7239878e-01]	 2.0905924e-01
      20	 4.8260983e-01	 2.0762484e-01	 5.1758180e-01	 2.0189426e-01	[ 5.1195738e-01]	 1.9970155e-01


.. parsed-literal::

      21	 5.4768717e-01	 2.0525485e-01	 5.8497105e-01	 1.9962158e-01	[ 5.5919958e-01]	 2.0173407e-01


.. parsed-literal::

      22	 6.0276820e-01	 2.0253385e-01	 6.4333953e-01	 1.9390493e-01	[ 6.0055671e-01]	 2.0962691e-01
      23	 6.2412413e-01	 2.0099173e-01	 6.6217457e-01	 1.9359793e-01	[ 6.2001650e-01]	 1.7508078e-01


.. parsed-literal::

      24	 6.6665641e-01	 1.9940344e-01	 7.0452733e-01	 1.9260046e-01	[ 6.5769407e-01]	 1.9870687e-01


.. parsed-literal::

      25	 6.9486385e-01	 2.0346241e-01	 7.3332924e-01	 1.9584167e-01	[ 6.8512509e-01]	 2.1123767e-01


.. parsed-literal::

      26	 7.2712767e-01	 2.0312734e-01	 7.6497809e-01	 1.9491983e-01	[ 7.2148579e-01]	 2.1357918e-01
      27	 7.7091399e-01	 2.0438947e-01	 8.0912556e-01	 1.9469190e-01	[ 7.6741488e-01]	 2.0476675e-01


.. parsed-literal::

      28	 7.9497770e-01	 2.1291012e-01	 8.3525909e-01	 2.0093617e-01	[ 8.0007737e-01]	 2.0711565e-01


.. parsed-literal::

      29	 8.2606897e-01	 2.1066495e-01	 8.6700252e-01	 1.9744204e-01	[ 8.2869279e-01]	 2.1713328e-01


.. parsed-literal::

      30	 8.4963906e-01	 2.0757213e-01	 8.9215679e-01	 1.9547413e-01	[ 8.4159821e-01]	 2.0965290e-01


.. parsed-literal::

      31	 8.7050464e-01	 2.0757155e-01	 9.1372821e-01	 1.9644126e-01	[ 8.6482100e-01]	 2.1997452e-01


.. parsed-literal::

      32	 8.9407567e-01	 2.0616050e-01	 9.3836568e-01	 1.9681057e-01	[ 8.9138031e-01]	 2.1203065e-01


.. parsed-literal::

      33	 9.1919309e-01	 2.0430927e-01	 9.6467390e-01	 1.9784412e-01	[ 9.2289624e-01]	 2.1168947e-01


.. parsed-literal::

      34	 9.3956590e-01	 1.9788285e-01	 9.8503724e-01	 1.9051480e-01	[ 9.4920482e-01]	 2.0882154e-01


.. parsed-literal::

      35	 9.5617152e-01	 1.9604267e-01	 1.0021579e+00	 1.8802827e-01	[ 9.6765179e-01]	 2.1255350e-01


.. parsed-literal::

      36	 9.7053517e-01	 1.9178582e-01	 1.0163570e+00	 1.8489883e-01	[ 9.7839864e-01]	 2.1244764e-01


.. parsed-literal::

      37	 9.8706911e-01	 1.8908558e-01	 1.0333200e+00	 1.8288816e-01	[ 9.9237565e-01]	 2.0223761e-01
      38	 9.9565417e-01	 1.8782585e-01	 1.0423980e+00	 1.8226526e-01	[ 1.0011969e+00]	 1.9765306e-01


.. parsed-literal::

      39	 1.0070017e+00	 1.8584485e-01	 1.0544303e+00	 1.8173733e-01	[ 1.0084072e+00]	 2.0843816e-01


.. parsed-literal::

      40	 1.0232864e+00	 1.8229742e-01	 1.0711879e+00	 1.7953324e-01	[ 1.0184325e+00]	 2.0895028e-01


.. parsed-literal::

      41	 1.0398387e+00	 1.7832689e-01	 1.0878363e+00	 1.7760122e-01	[ 1.0241786e+00]	 2.1392989e-01


.. parsed-literal::

      42	 1.0518900e+00	 1.7557646e-01	 1.0994122e+00	 1.7487611e-01	[ 1.0324714e+00]	 2.1667671e-01
      43	 1.0716288e+00	 1.7154548e-01	 1.1188099e+00	 1.7069710e-01	[ 1.0376081e+00]	 1.7103696e-01


.. parsed-literal::

      44	 1.0924015e+00	 1.6819922e-01	 1.1396076e+00	 1.6747412e-01	[ 1.0454823e+00]	 2.0919800e-01


.. parsed-literal::

      45	 1.1097977e+00	 1.6293574e-01	 1.1571977e+00	 1.6392499e-01	  1.0340295e+00 	 2.1154404e-01


.. parsed-literal::

      46	 1.1231795e+00	 1.6192570e-01	 1.1707255e+00	 1.6224515e-01	[ 1.0491899e+00]	 2.0939970e-01


.. parsed-literal::

      47	 1.1348495e+00	 1.5938275e-01	 1.1827983e+00	 1.6015413e-01	[ 1.0579068e+00]	 2.1065688e-01
      48	 1.1531873e+00	 1.5439704e-01	 1.2017875e+00	 1.5696680e-01	[ 1.0739141e+00]	 1.7864609e-01


.. parsed-literal::

      49	 1.1675022e+00	 1.5066324e-01	 1.2171716e+00	 1.5625130e-01	[ 1.0744090e+00]	 2.1819305e-01
      50	 1.1827845e+00	 1.4879818e-01	 1.2320618e+00	 1.5433948e-01	[ 1.0987314e+00]	 1.7105412e-01


.. parsed-literal::

      51	 1.1970054e+00	 1.4775516e-01	 1.2465839e+00	 1.5289221e-01	[ 1.1078110e+00]	 2.0138741e-01


.. parsed-literal::

      52	 1.2109684e+00	 1.4644001e-01	 1.2609445e+00	 1.5224508e-01	[ 1.1239926e+00]	 2.0958900e-01


.. parsed-literal::

      53	 1.2224928e+00	 1.4568624e-01	 1.2727412e+00	 1.5214445e-01	[ 1.1336763e+00]	 2.0968819e-01


.. parsed-literal::

      54	 1.2355557e+00	 1.4424842e-01	 1.2858838e+00	 1.5139967e-01	[ 1.1429320e+00]	 2.1139002e-01


.. parsed-literal::

      55	 1.2448995e+00	 1.4328742e-01	 1.2955325e+00	 1.5170402e-01	[ 1.1468968e+00]	 2.1001935e-01
      56	 1.2570710e+00	 1.4185643e-01	 1.3078099e+00	 1.5134914e-01	[ 1.1558673e+00]	 1.9602394e-01


.. parsed-literal::

      57	 1.2670019e+00	 1.4083740e-01	 1.3180728e+00	 1.5202663e-01	  1.1519176e+00 	 3.2770824e-01
      58	 1.2785447e+00	 1.3950531e-01	 1.3295534e+00	 1.5147035e-01	[ 1.1589211e+00]	 1.8849301e-01


.. parsed-literal::

      59	 1.2909804e+00	 1.3758586e-01	 1.3420134e+00	 1.4991404e-01	[ 1.1668546e+00]	 2.0305800e-01


.. parsed-literal::

      60	 1.2986500e+00	 1.3692652e-01	 1.3498743e+00	 1.5009010e-01	[ 1.1732051e+00]	 2.0906615e-01


.. parsed-literal::

      61	 1.3072563e+00	 1.3624415e-01	 1.3583003e+00	 1.4945096e-01	[ 1.1834809e+00]	 2.0214295e-01
      62	 1.3159698e+00	 1.3549523e-01	 1.3670631e+00	 1.4874091e-01	[ 1.1944914e+00]	 1.8781877e-01


.. parsed-literal::

      63	 1.3239830e+00	 1.3483868e-01	 1.3752062e+00	 1.4808107e-01	[ 1.2043023e+00]	 1.7859554e-01


.. parsed-literal::

      64	 1.3314397e+00	 1.3334780e-01	 1.3830295e+00	 1.4746349e-01	[ 1.2157490e+00]	 2.0664454e-01
      65	 1.3403358e+00	 1.3261155e-01	 1.3919234e+00	 1.4736347e-01	[ 1.2225152e+00]	 1.9858861e-01


.. parsed-literal::

      66	 1.3486291e+00	 1.3164304e-01	 1.4003967e+00	 1.4646238e-01	[ 1.2301836e+00]	 2.1265459e-01


.. parsed-literal::

      67	 1.3562341e+00	 1.3063543e-01	 1.4084376e+00	 1.4672054e-01	[ 1.2351752e+00]	 2.1403956e-01


.. parsed-literal::

      68	 1.3664800e+00	 1.3006510e-01	 1.4188524e+00	 1.4672830e-01	[ 1.2486164e+00]	 2.1257639e-01
      69	 1.3754461e+00	 1.2972984e-01	 1.4279912e+00	 1.4668573e-01	[ 1.2573451e+00]	 1.8230271e-01


.. parsed-literal::

      70	 1.3842429e+00	 1.2943655e-01	 1.4369507e+00	 1.4686095e-01	[ 1.2651925e+00]	 2.0630598e-01


.. parsed-literal::

      71	 1.3926396e+00	 1.2895122e-01	 1.4452899e+00	 1.4622695e-01	[ 1.2689695e+00]	 2.1975064e-01


.. parsed-literal::

      72	 1.3987226e+00	 1.2868416e-01	 1.4513838e+00	 1.4606735e-01	[ 1.2705485e+00]	 2.1677661e-01
      73	 1.4033949e+00	 1.2801031e-01	 1.4561568e+00	 1.4563239e-01	  1.2684100e+00 	 1.9515324e-01


.. parsed-literal::

      74	 1.4080022e+00	 1.2787784e-01	 1.4607470e+00	 1.4515597e-01	[ 1.2718833e+00]	 2.0875049e-01


.. parsed-literal::

      75	 1.4132440e+00	 1.2790067e-01	 1.4659768e+00	 1.4554075e-01	[ 1.2764216e+00]	 2.1306992e-01


.. parsed-literal::

      76	 1.4197667e+00	 1.2767618e-01	 1.4724123e+00	 1.4478067e-01	[ 1.2817582e+00]	 2.0941019e-01


.. parsed-literal::

      77	 1.4253113e+00	 1.2772741e-01	 1.4780385e+00	 1.4526276e-01	[ 1.2818232e+00]	 2.0812488e-01


.. parsed-literal::

      78	 1.4301712e+00	 1.2722588e-01	 1.4828563e+00	 1.4499902e-01	[ 1.2824471e+00]	 2.1455622e-01
      79	 1.4334226e+00	 1.2700709e-01	 1.4860885e+00	 1.4447350e-01	[ 1.2840272e+00]	 2.0164561e-01


.. parsed-literal::

      80	 1.4374508e+00	 1.2673088e-01	 1.4903369e+00	 1.4467068e-01	  1.2783874e+00 	 1.8018317e-01


.. parsed-literal::

      81	 1.4423655e+00	 1.2676019e-01	 1.4955359e+00	 1.4546600e-01	  1.2708286e+00 	 2.1327472e-01
      82	 1.4470668e+00	 1.2678257e-01	 1.5002167e+00	 1.4582691e-01	  1.2697957e+00 	 1.7596698e-01


.. parsed-literal::

      83	 1.4505004e+00	 1.2676130e-01	 1.5035576e+00	 1.4605468e-01	  1.2719287e+00 	 2.0740318e-01
      84	 1.4551933e+00	 1.2642962e-01	 1.5084190e+00	 1.4608932e-01	  1.2693600e+00 	 1.9338012e-01


.. parsed-literal::

      85	 1.4574827e+00	 1.2634940e-01	 1.5110593e+00	 1.4659771e-01	  1.2635594e+00 	 2.0078421e-01


.. parsed-literal::

      86	 1.4610063e+00	 1.2583411e-01	 1.5144834e+00	 1.4612681e-01	  1.2691339e+00 	 2.1380901e-01


.. parsed-literal::

      87	 1.4633067e+00	 1.2544130e-01	 1.5168678e+00	 1.4612170e-01	  1.2702607e+00 	 2.2238970e-01
      88	 1.4659111e+00	 1.2504399e-01	 1.5195123e+00	 1.4585358e-01	  1.2739657e+00 	 1.7475271e-01


.. parsed-literal::

      89	 1.4702884e+00	 1.2469933e-01	 1.5239642e+00	 1.4599631e-01	  1.2787890e+00 	 2.1945477e-01


.. parsed-literal::

      90	 1.4725285e+00	 1.2436163e-01	 1.5262755e+00	 1.4485469e-01	  1.2777036e+00 	 2.1076632e-01


.. parsed-literal::

      91	 1.4755709e+00	 1.2446038e-01	 1.5291744e+00	 1.4508803e-01	  1.2826940e+00 	 2.0449853e-01


.. parsed-literal::

      92	 1.4780066e+00	 1.2459170e-01	 1.5316079e+00	 1.4538549e-01	  1.2832370e+00 	 2.1424913e-01


.. parsed-literal::

      93	 1.4809214e+00	 1.2461170e-01	 1.5345711e+00	 1.4540224e-01	  1.2826140e+00 	 2.1293259e-01


.. parsed-literal::

      94	 1.4831070e+00	 1.2471304e-01	 1.5368309e+00	 1.4550191e-01	  1.2822218e+00 	 3.1914973e-01
      95	 1.4860321e+00	 1.2450760e-01	 1.5397802e+00	 1.4530655e-01	  1.2818092e+00 	 1.7791200e-01


.. parsed-literal::

      96	 1.4890198e+00	 1.2407445e-01	 1.5428106e+00	 1.4497169e-01	  1.2807425e+00 	 2.0957732e-01


.. parsed-literal::

      97	 1.4923185e+00	 1.2356818e-01	 1.5461604e+00	 1.4465296e-01	  1.2802170e+00 	 2.0951986e-01


.. parsed-literal::

      98	 1.4940008e+00	 1.2277153e-01	 1.5479817e+00	 1.4419443e-01	  1.2744447e+00 	 2.1642351e-01


.. parsed-literal::

      99	 1.4972309e+00	 1.2276981e-01	 1.5511406e+00	 1.4425089e-01	  1.2796997e+00 	 2.1302271e-01


.. parsed-literal::

     100	 1.4984100e+00	 1.2274505e-01	 1.5522942e+00	 1.4430188e-01	  1.2816516e+00 	 2.1992946e-01
     101	 1.5008956e+00	 1.2260427e-01	 1.5548383e+00	 1.4432915e-01	  1.2800694e+00 	 1.8372440e-01


.. parsed-literal::

     102	 1.5029553e+00	 1.2225770e-01	 1.5570310e+00	 1.4416620e-01	  1.2805322e+00 	 2.1273279e-01


.. parsed-literal::

     103	 1.5059904e+00	 1.2216788e-01	 1.5600575e+00	 1.4411202e-01	  1.2766960e+00 	 2.2235084e-01


.. parsed-literal::

     104	 1.5080682e+00	 1.2206015e-01	 1.5621812e+00	 1.4404316e-01	  1.2731430e+00 	 2.0961213e-01


.. parsed-literal::

     105	 1.5103292e+00	 1.2191711e-01	 1.5644922e+00	 1.4385617e-01	  1.2718855e+00 	 2.1111298e-01


.. parsed-literal::

     106	 1.5124430e+00	 1.2193997e-01	 1.5667219e+00	 1.4355587e-01	  1.2705482e+00 	 2.1116471e-01


.. parsed-literal::

     107	 1.5148629e+00	 1.2179614e-01	 1.5691171e+00	 1.4339029e-01	  1.2741913e+00 	 2.0992017e-01


.. parsed-literal::

     108	 1.5161612e+00	 1.2170550e-01	 1.5703874e+00	 1.4323980e-01	  1.2765009e+00 	 2.0329070e-01
     109	 1.5184314e+00	 1.2154098e-01	 1.5727130e+00	 1.4298675e-01	  1.2769412e+00 	 1.7060328e-01


.. parsed-literal::

     110	 1.5200861e+00	 1.2131816e-01	 1.5744763e+00	 1.4256343e-01	  1.2723473e+00 	 1.9878054e-01


.. parsed-literal::

     111	 1.5223406e+00	 1.2133845e-01	 1.5767258e+00	 1.4262526e-01	  1.2729049e+00 	 2.2029829e-01


.. parsed-literal::

     112	 1.5237422e+00	 1.2128834e-01	 1.5781028e+00	 1.4262105e-01	  1.2719167e+00 	 2.1978807e-01


.. parsed-literal::

     113	 1.5253631e+00	 1.2121723e-01	 1.5797397e+00	 1.4263957e-01	  1.2696752e+00 	 2.2063661e-01


.. parsed-literal::

     114	 1.5268967e+00	 1.2095413e-01	 1.5813195e+00	 1.4208719e-01	  1.2640020e+00 	 2.1457696e-01


.. parsed-literal::

     115	 1.5288858e+00	 1.2094447e-01	 1.5832952e+00	 1.4220649e-01	  1.2643204e+00 	 2.0802569e-01
     116	 1.5304488e+00	 1.2086389e-01	 1.5849202e+00	 1.4210127e-01	  1.2620459e+00 	 1.9934177e-01


.. parsed-literal::

     117	 1.5319065e+00	 1.2074062e-01	 1.5864330e+00	 1.4192271e-01	  1.2570990e+00 	 1.7836928e-01


.. parsed-literal::

     118	 1.5332826e+00	 1.2053058e-01	 1.5879123e+00	 1.4178482e-01	  1.2520422e+00 	 2.0703101e-01


.. parsed-literal::

     119	 1.5352085e+00	 1.2044456e-01	 1.5897856e+00	 1.4170393e-01	  1.2485455e+00 	 2.1099734e-01


.. parsed-literal::

     120	 1.5363983e+00	 1.2036850e-01	 1.5909444e+00	 1.4169693e-01	  1.2459716e+00 	 2.0847559e-01


.. parsed-literal::

     121	 1.5376315e+00	 1.2028394e-01	 1.5921567e+00	 1.4170386e-01	  1.2451656e+00 	 2.0855093e-01


.. parsed-literal::

     122	 1.5399094e+00	 1.2018759e-01	 1.5944226e+00	 1.4170899e-01	  1.2419188e+00 	 2.0797801e-01


.. parsed-literal::

     123	 1.5411271e+00	 1.2007368e-01	 1.5956745e+00	 1.4172454e-01	  1.2417505e+00 	 3.1718397e-01
     124	 1.5429023e+00	 1.2009587e-01	 1.5974412e+00	 1.4171020e-01	  1.2421757e+00 	 1.8937159e-01


.. parsed-literal::

     125	 1.5447743e+00	 1.2004097e-01	 1.5993373e+00	 1.4159591e-01	  1.2408910e+00 	 2.1698785e-01
     126	 1.5462428e+00	 1.2009479e-01	 1.6008711e+00	 1.4167150e-01	  1.2413291e+00 	 1.9179654e-01


.. parsed-literal::

     127	 1.5479648e+00	 1.1991100e-01	 1.6026168e+00	 1.4152048e-01	  1.2396626e+00 	 2.1562886e-01


.. parsed-literal::

     128	 1.5495520e+00	 1.1974352e-01	 1.6042196e+00	 1.4160053e-01	  1.2376510e+00 	 2.0815063e-01
     129	 1.5515166e+00	 1.1958314e-01	 1.6062036e+00	 1.4173304e-01	  1.2349900e+00 	 1.8679190e-01


.. parsed-literal::

     130	 1.5537053e+00	 1.1945795e-01	 1.6084746e+00	 1.4205079e-01	  1.2280655e+00 	 2.1991253e-01


.. parsed-literal::

     131	 1.5555007e+00	 1.1944430e-01	 1.6102883e+00	 1.4204575e-01	  1.2277924e+00 	 2.0846081e-01


.. parsed-literal::

     132	 1.5567217e+00	 1.1949750e-01	 1.6114965e+00	 1.4192238e-01	  1.2269614e+00 	 2.1937633e-01


.. parsed-literal::

     133	 1.5579406e+00	 1.1963290e-01	 1.6127316e+00	 1.4185529e-01	  1.2266019e+00 	 2.0401382e-01
     134	 1.5593710e+00	 1.1970337e-01	 1.6142070e+00	 1.4178669e-01	  1.2240346e+00 	 1.9016671e-01


.. parsed-literal::

     135	 1.5608333e+00	 1.1982202e-01	 1.6157286e+00	 1.4169836e-01	  1.2186553e+00 	 2.1785784e-01
     136	 1.5620552e+00	 1.1971234e-01	 1.6169515e+00	 1.4173591e-01	  1.2200120e+00 	 1.9516778e-01


.. parsed-literal::

     137	 1.5633558e+00	 1.1954228e-01	 1.6182456e+00	 1.4171245e-01	  1.2204208e+00 	 2.0066619e-01
     138	 1.5649703e+00	 1.1943048e-01	 1.6198914e+00	 1.4172267e-01	  1.2187013e+00 	 1.7674255e-01


.. parsed-literal::

     139	 1.5663304e+00	 1.1915371e-01	 1.6212739e+00	 1.4153938e-01	  1.2151898e+00 	 2.1188664e-01


.. parsed-literal::

     140	 1.5673096e+00	 1.1920350e-01	 1.6222384e+00	 1.4155699e-01	  1.2149306e+00 	 2.0757556e-01
     141	 1.5684130e+00	 1.1924067e-01	 1.6233539e+00	 1.4156742e-01	  1.2119495e+00 	 2.0591331e-01


.. parsed-literal::

     142	 1.5695712e+00	 1.1912523e-01	 1.6245275e+00	 1.4166035e-01	  1.2100068e+00 	 2.1135306e-01


.. parsed-literal::

     143	 1.5711884e+00	 1.1889405e-01	 1.6262044e+00	 1.4173426e-01	  1.2072325e+00 	 2.0803595e-01


.. parsed-literal::

     144	 1.5722040e+00	 1.1861065e-01	 1.6272669e+00	 1.4182066e-01	  1.2023125e+00 	 2.0771885e-01


.. parsed-literal::

     145	 1.5731621e+00	 1.1855513e-01	 1.6281944e+00	 1.4184602e-01	  1.2050408e+00 	 2.1156621e-01


.. parsed-literal::

     146	 1.5741426e+00	 1.1848808e-01	 1.6291756e+00	 1.4183116e-01	  1.2059835e+00 	 2.2015166e-01


.. parsed-literal::

     147	 1.5753857e+00	 1.1844446e-01	 1.6304494e+00	 1.4184161e-01	  1.2053010e+00 	 2.1327662e-01
     148	 1.5774248e+00	 1.1842927e-01	 1.6325460e+00	 1.4187661e-01	  1.2017168e+00 	 1.8491244e-01


.. parsed-literal::

     149	 1.5786862e+00	 1.1846233e-01	 1.6338487e+00	 1.4188126e-01	  1.2001441e+00 	 2.8838468e-01


.. parsed-literal::

     150	 1.5799650e+00	 1.1846497e-01	 1.6351310e+00	 1.4179561e-01	  1.1955625e+00 	 2.0629930e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.21 s, total: 2min 8s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f9b9f4fbd30>



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
    CPU times: user 1.84 s, sys: 44 ms, total: 1.89 s
    Wall time: 634 ms


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

