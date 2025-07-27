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
       1	-3.9058443e-01	 3.2456959e-01	-3.7706891e-01	 3.1089418e-01	[-2.6977498e-01]	 3.4441304e-01
       2	-3.2014094e-01	 3.0893705e-01	-2.9959905e-01	 2.9014179e-01	[-2.5822610e-01]	 1.1531949e-01


.. parsed-literal::

       3	-1.4945697e-01	 2.7030089e-01	-1.1152746e-01	 2.5296315e-01	[-3.7524715e-02]	 3.9478946e-01


.. parsed-literal::

       4	-1.1280944e-01	 2.5904670e-01	-8.1454788e-02	 2.4539138e-01	[-5.6633308e-04]	 3.0954313e-01


.. parsed-literal::

       5	-7.2712267e-02	 2.5313767e-01	-4.6711004e-02	 2.4180201e-01	[ 1.6573035e-02]	 2.2271752e-01


.. parsed-literal::

       6	-5.6796857e-02	 2.5026702e-01	-3.4566379e-02	 2.3300834e-01	[ 3.0109411e-02]	 2.1307349e-01
       7	-4.6253735e-02	 2.4859997e-01	-2.7370810e-02	 2.3219582e-01	[ 3.8638296e-02]	 1.9201875e-01


.. parsed-literal::

       8	-3.3549820e-02	 2.4623065e-01	-1.7737419e-02	 2.3075282e-01	[ 4.5802446e-02]	 2.0361543e-01


.. parsed-literal::

       9	-2.5010942e-02	 2.4453707e-01	-1.1043741e-02	 2.3489267e-01	[ 5.3733431e-02]	 2.0366812e-01


.. parsed-literal::

      10	-1.9023063e-02	 2.4361285e-01	-6.0281779e-03	 2.2899630e-01	[ 6.1097241e-02]	 2.0127726e-01


.. parsed-literal::

      11	-1.4718928e-02	 2.4265886e-01	-1.8576275e-03	 2.2577908e-01	[ 6.7217623e-02]	 2.1493721e-01


.. parsed-literal::

      12	 1.1283276e-02	 2.3704924e-01	 2.4320959e-02	 2.2354546e-01	[ 8.5066850e-02]	 2.1257401e-01


.. parsed-literal::

      13	 4.6292434e-02	 2.3504374e-01	 6.7983521e-02	 2.2454583e-01	[ 1.2661547e-01]	 2.1332479e-01


.. parsed-literal::

      14	 1.3336712e-01	 2.2342760e-01	 1.5762382e-01	 2.1200784e-01	[ 1.9700714e-01]	 2.0815277e-01


.. parsed-literal::

      15	 2.1609082e-01	 2.1999362e-01	 2.4207292e-01	 2.1014671e-01	[ 2.9042465e-01]	 2.1405530e-01
      16	 3.1281508e-01	 2.1522440e-01	 3.4382612e-01	 2.0770740e-01	[ 4.0946855e-01]	 1.9535589e-01


.. parsed-literal::

      17	 4.3116934e-01	 2.1397046e-01	 4.6435332e-01	 1.9584522e-01	[ 5.1640328e-01]	 2.0239377e-01


.. parsed-literal::

      18	 5.1576985e-01	 2.1063058e-01	 5.5030152e-01	 1.9958238e-01	[ 5.6000543e-01]	 2.0556879e-01


.. parsed-literal::

      19	 5.9199530e-01	 2.0127658e-01	 6.2950037e-01	 1.8729251e-01	[ 6.4139853e-01]	 2.1860719e-01
      20	 6.3580509e-01	 1.9382067e-01	 6.7553737e-01	 1.8363987e-01	[ 6.6559114e-01]	 1.7928267e-01


.. parsed-literal::

      21	 6.8101406e-01	 1.8915098e-01	 7.2102172e-01	 1.8204239e-01	[ 6.9516655e-01]	 2.0168424e-01


.. parsed-literal::

      22	 7.3821871e-01	 1.8426034e-01	 7.7850292e-01	 1.7746958e-01	[ 7.5142517e-01]	 2.1393180e-01


.. parsed-literal::

      23	 8.0233568e-01	 1.8268463e-01	 8.4157655e-01	 1.8196420e-01	[ 8.2718208e-01]	 2.0396495e-01


.. parsed-literal::

      24	 8.5834114e-01	 1.7855869e-01	 8.9825859e-01	 1.7933496e-01	[ 8.7654507e-01]	 2.0837212e-01


.. parsed-literal::

      25	 8.9870233e-01	 1.7441102e-01	 9.3897977e-01	 1.8086247e-01	[ 9.0872906e-01]	 2.0102429e-01
      26	 9.3225447e-01	 1.6923940e-01	 9.7257798e-01	 1.8605074e-01	[ 9.4841472e-01]	 2.0373631e-01


.. parsed-literal::

      27	 9.5780567e-01	 1.6720979e-01	 9.9916721e-01	 1.6501646e-01	[ 9.6343420e-01]	 2.0233369e-01


.. parsed-literal::

      28	 9.8387634e-01	 1.6496307e-01	 1.0255394e+00	 1.5728150e-01	[ 9.8503161e-01]	 2.0797348e-01
      29	 1.0074393e+00	 1.6349537e-01	 1.0491710e+00	 1.5210307e-01	[ 1.0042179e+00]	 2.0000935e-01


.. parsed-literal::

      30	 1.0274166e+00	 1.6260173e-01	 1.0701482e+00	 1.4847952e-01	[ 1.0260502e+00]	 2.1187282e-01


.. parsed-literal::

      31	 1.0437313e+00	 1.6148285e-01	 1.0867689e+00	 1.4731668e-01	[ 1.0415905e+00]	 2.0853949e-01


.. parsed-literal::

      32	 1.0525244e+00	 1.6054610e-01	 1.0956342e+00	 1.4689248e-01	[ 1.0477002e+00]	 2.1104670e-01


.. parsed-literal::

      33	 1.0708107e+00	 1.5799763e-01	 1.1144523e+00	 1.4635461e-01	[ 1.0624593e+00]	 2.0888972e-01


.. parsed-literal::

      34	 1.0829603e+00	 1.5492749e-01	 1.1272630e+00	 1.4812741e-01	[ 1.0694150e+00]	 2.0168376e-01
      35	 1.0963553e+00	 1.5381403e-01	 1.1405192e+00	 1.4660482e-01	[ 1.0876574e+00]	 1.9897938e-01


.. parsed-literal::

      36	 1.1043011e+00	 1.5310848e-01	 1.1488339e+00	 1.4511746e-01	[ 1.0944442e+00]	 2.0838785e-01


.. parsed-literal::

      37	 1.1138466e+00	 1.5181679e-01	 1.1586243e+00	 1.4254968e-01	[ 1.1034536e+00]	 2.1085167e-01
      38	 1.1266081e+00	 1.4956921e-01	 1.1718279e+00	 1.3882419e-01	[ 1.1092427e+00]	 1.7512655e-01


.. parsed-literal::

      39	 1.1384610e+00	 1.4786453e-01	 1.1838479e+00	 1.3601300e-01	[ 1.1223154e+00]	 2.0947909e-01
      40	 1.1495229e+00	 1.4636865e-01	 1.1947802e+00	 1.3400186e-01	[ 1.1334691e+00]	 1.7366910e-01


.. parsed-literal::

      41	 1.1677665e+00	 1.4432088e-01	 1.2128906e+00	 1.3077410e-01	[ 1.1545728e+00]	 1.9910812e-01


.. parsed-literal::

      42	 1.1775854e+00	 1.4442149e-01	 1.2229340e+00	 1.2954146e-01	[ 1.1644200e+00]	 2.1286821e-01


.. parsed-literal::

      43	 1.1876476e+00	 1.4383619e-01	 1.2330043e+00	 1.2875624e-01	[ 1.1783555e+00]	 2.1396732e-01
      44	 1.1998172e+00	 1.4307364e-01	 1.2453458e+00	 1.2717898e-01	[ 1.1931424e+00]	 1.8683767e-01


.. parsed-literal::

      45	 1.2107406e+00	 1.4227759e-01	 1.2565334e+00	 1.2663662e-01	[ 1.2025642e+00]	 1.9975114e-01


.. parsed-literal::

      46	 1.2193121e+00	 1.4201788e-01	 1.2654112e+00	 1.2580620e-01	[ 1.2057068e+00]	 2.0369768e-01
      47	 1.2307040e+00	 1.4093451e-01	 1.2768713e+00	 1.2623219e-01	[ 1.2131986e+00]	 1.8878651e-01


.. parsed-literal::

      48	 1.2364195e+00	 1.4063091e-01	 1.2825234e+00	 1.2611875e-01	[ 1.2170492e+00]	 2.0698929e-01


.. parsed-literal::

      49	 1.2477245e+00	 1.3957931e-01	 1.2943405e+00	 1.2562966e-01	[ 1.2201321e+00]	 2.1091843e-01


.. parsed-literal::

      50	 1.2600552e+00	 1.3849588e-01	 1.3070564e+00	 1.2443962e-01	[ 1.2250152e+00]	 2.0443058e-01


.. parsed-literal::

      51	 1.2693529e+00	 1.3784574e-01	 1.3164602e+00	 1.2366060e-01	[ 1.2351908e+00]	 2.0972657e-01


.. parsed-literal::

      52	 1.2779956e+00	 1.3727179e-01	 1.3252931e+00	 1.2278287e-01	[ 1.2427220e+00]	 2.0965338e-01


.. parsed-literal::

      53	 1.2890001e+00	 1.3673424e-01	 1.3367992e+00	 1.2185717e-01	[ 1.2495869e+00]	 2.7796626e-01
      54	 1.2970704e+00	 1.3630670e-01	 1.3449889e+00	 1.2147998e-01	[ 1.2537430e+00]	 1.8646264e-01


.. parsed-literal::

      55	 1.3034518e+00	 1.3618754e-01	 1.3512262e+00	 1.2156894e-01	[ 1.2581723e+00]	 2.1707344e-01


.. parsed-literal::

      56	 1.3116272e+00	 1.3581805e-01	 1.3595697e+00	 1.2182817e-01	[ 1.2594928e+00]	 2.0580554e-01
      57	 1.3174995e+00	 1.3579026e-01	 1.3653477e+00	 1.2173810e-01	[ 1.2659218e+00]	 1.8601990e-01


.. parsed-literal::

      58	 1.3239714e+00	 1.3544379e-01	 1.3717798e+00	 1.2217444e-01	[ 1.2791058e+00]	 1.9960642e-01
      59	 1.3306262e+00	 1.3527925e-01	 1.3783573e+00	 1.2171231e-01	[ 1.2886340e+00]	 1.8454742e-01


.. parsed-literal::

      60	 1.3359503e+00	 1.3508521e-01	 1.3837645e+00	 1.2145325e-01	[ 1.2958909e+00]	 2.0655251e-01


.. parsed-literal::

      61	 1.3427421e+00	 1.3470890e-01	 1.3907121e+00	 1.2140359e-01	[ 1.2981777e+00]	 2.0893931e-01
      62	 1.3460817e+00	 1.3486732e-01	 1.3945549e+00	 1.2183809e-01	  1.2953935e+00 	 2.0186305e-01


.. parsed-literal::

      63	 1.3543407e+00	 1.3431825e-01	 1.4025660e+00	 1.2180017e-01	  1.2941294e+00 	 1.8085694e-01


.. parsed-literal::

      64	 1.3572237e+00	 1.3413859e-01	 1.4054321e+00	 1.2168195e-01	  1.2920338e+00 	 2.0845294e-01
      65	 1.3624839e+00	 1.3397453e-01	 1.4109597e+00	 1.2140462e-01	  1.2832186e+00 	 1.6911316e-01


.. parsed-literal::

      66	 1.3671298e+00	 1.3340130e-01	 1.4159150e+00	 1.2092811e-01	  1.2786664e+00 	 2.1836853e-01
      67	 1.3728220e+00	 1.3324100e-01	 1.4216611e+00	 1.2033806e-01	  1.2891460e+00 	 1.7078876e-01


.. parsed-literal::

      68	 1.3768214e+00	 1.3292809e-01	 1.4257052e+00	 1.2023964e-01	  1.2981390e+00 	 2.1352887e-01
      69	 1.3796281e+00	 1.3269088e-01	 1.4285237e+00	 1.2013145e-01	[ 1.3076120e+00]	 1.8066144e-01


.. parsed-literal::

      70	 1.3837570e+00	 1.3251967e-01	 1.4327289e+00	 1.2045329e-01	[ 1.3116482e+00]	 1.8115449e-01


.. parsed-literal::

      71	 1.3873547e+00	 1.3264171e-01	 1.4365154e+00	 1.2052508e-01	[ 1.3169460e+00]	 2.1239209e-01


.. parsed-literal::

      72	 1.3908195e+00	 1.3276261e-01	 1.4399321e+00	 1.2123959e-01	[ 1.3211352e+00]	 2.1075249e-01


.. parsed-literal::

      73	 1.3926749e+00	 1.3282736e-01	 1.4416637e+00	 1.2096985e-01	[ 1.3251148e+00]	 2.1142817e-01


.. parsed-literal::

      74	 1.3964208e+00	 1.3308044e-01	 1.4454940e+00	 1.2091572e-01	[ 1.3323489e+00]	 2.0521712e-01
      75	 1.3992713e+00	 1.3311923e-01	 1.4485657e+00	 1.2137298e-01	[ 1.3396134e+00]	 1.8077707e-01


.. parsed-literal::

      76	 1.4019067e+00	 1.3304197e-01	 1.4512439e+00	 1.2145836e-01	[ 1.3416561e+00]	 2.1067166e-01


.. parsed-literal::

      77	 1.4045456e+00	 1.3286003e-01	 1.4540415e+00	 1.2186462e-01	[ 1.3427809e+00]	 2.1072769e-01
      78	 1.4072359e+00	 1.3264643e-01	 1.4567860e+00	 1.2216830e-01	  1.3421752e+00 	 1.8333244e-01


.. parsed-literal::

      79	 1.4106466e+00	 1.3228686e-01	 1.4604486e+00	 1.2205014e-01	[ 1.3432896e+00]	 2.1130180e-01
      80	 1.4150342e+00	 1.3200384e-01	 1.4646354e+00	 1.2240390e-01	  1.3409227e+00 	 1.8315649e-01


.. parsed-literal::

      81	 1.4169240e+00	 1.3203746e-01	 1.4664589e+00	 1.2260927e-01	  1.3405285e+00 	 2.1321988e-01
      82	 1.4202886e+00	 1.3190185e-01	 1.4698538e+00	 1.2238994e-01	  1.3361893e+00 	 1.9995093e-01


.. parsed-literal::

      83	 1.4215937e+00	 1.3142063e-01	 1.4713302e+00	 1.2400957e-01	  1.3238983e+00 	 2.0911288e-01


.. parsed-literal::

      84	 1.4249009e+00	 1.3129250e-01	 1.4745837e+00	 1.2292373e-01	  1.3287805e+00 	 2.0589304e-01


.. parsed-literal::

      85	 1.4269641e+00	 1.3100869e-01	 1.4767503e+00	 1.2258354e-01	  1.3289854e+00 	 2.0924568e-01
      86	 1.4291099e+00	 1.3060158e-01	 1.4789681e+00	 1.2256323e-01	  1.3288925e+00 	 1.6779804e-01


.. parsed-literal::

      87	 1.4321572e+00	 1.3015581e-01	 1.4820844e+00	 1.2330277e-01	  1.3339396e+00 	 2.0237279e-01


.. parsed-literal::

      88	 1.4345003e+00	 1.2977210e-01	 1.4843916e+00	 1.2430481e-01	  1.3356175e+00 	 2.0927334e-01


.. parsed-literal::

      89	 1.4359028e+00	 1.2987021e-01	 1.4856860e+00	 1.2382167e-01	  1.3390030e+00 	 2.1449566e-01


.. parsed-literal::

      90	 1.4371066e+00	 1.2986732e-01	 1.4868452e+00	 1.2371777e-01	  1.3406728e+00 	 2.1427560e-01
      91	 1.4391924e+00	 1.2973273e-01	 1.4889609e+00	 1.2351285e-01	  1.3430314e+00 	 1.9905424e-01


.. parsed-literal::

      92	 1.4410670e+00	 1.2953898e-01	 1.4909341e+00	 1.2347826e-01	  1.3403154e+00 	 2.1734786e-01
      93	 1.4429866e+00	 1.2923997e-01	 1.4929608e+00	 1.2340573e-01	  1.3406678e+00 	 1.8657589e-01


.. parsed-literal::

      94	 1.4446252e+00	 1.2899929e-01	 1.4947218e+00	 1.2342508e-01	  1.3397806e+00 	 2.1473527e-01


.. parsed-literal::

      95	 1.4462592e+00	 1.2873458e-01	 1.4964688e+00	 1.2338527e-01	  1.3389063e+00 	 2.1115732e-01


.. parsed-literal::

      96	 1.4492740e+00	 1.2844547e-01	 1.4996307e+00	 1.2333951e-01	  1.3412020e+00 	 2.0857882e-01
      97	 1.4502303e+00	 1.2788134e-01	 1.5008087e+00	 1.2172216e-01	  1.3385680e+00 	 1.8019867e-01


.. parsed-literal::

      98	 1.4527630e+00	 1.2811581e-01	 1.5031521e+00	 1.2268840e-01	[ 1.3444066e+00]	 1.9811869e-01
      99	 1.4535703e+00	 1.2820030e-01	 1.5039043e+00	 1.2275147e-01	[ 1.3465566e+00]	 1.6302085e-01


.. parsed-literal::

     100	 1.4559304e+00	 1.2839106e-01	 1.5062194e+00	 1.2313338e-01	[ 1.3501292e+00]	 2.0472264e-01


.. parsed-literal::

     101	 1.4572614e+00	 1.2883309e-01	 1.5075776e+00	 1.2279229e-01	[ 1.3523282e+00]	 2.0372510e-01
     102	 1.4599327e+00	 1.2851856e-01	 1.5102692e+00	 1.2315506e-01	[ 1.3524392e+00]	 1.7877603e-01


.. parsed-literal::

     103	 1.4612325e+00	 1.2827224e-01	 1.5116549e+00	 1.2317582e-01	  1.3504820e+00 	 2.0670629e-01


.. parsed-literal::

     104	 1.4624327e+00	 1.2806478e-01	 1.5129699e+00	 1.2287292e-01	  1.3493045e+00 	 2.0601201e-01
     105	 1.4633548e+00	 1.2776397e-01	 1.5141384e+00	 1.2276196e-01	  1.3437103e+00 	 1.9690418e-01


.. parsed-literal::

     106	 1.4652363e+00	 1.2774327e-01	 1.5159631e+00	 1.2245870e-01	  1.3480346e+00 	 2.0320916e-01


.. parsed-literal::

     107	 1.4666891e+00	 1.2771738e-01	 1.5174057e+00	 1.2231259e-01	  1.3518013e+00 	 2.1715212e-01
     108	 1.4682841e+00	 1.2762324e-01	 1.5190021e+00	 1.2229191e-01	[ 1.3553791e+00]	 1.9941640e-01


.. parsed-literal::

     109	 1.4705423e+00	 1.2732448e-01	 1.5212644e+00	 1.2190814e-01	[ 1.3600795e+00]	 2.0385528e-01


.. parsed-literal::

     110	 1.4726403e+00	 1.2710634e-01	 1.5233131e+00	 1.2232263e-01	[ 1.3652703e+00]	 2.1939397e-01


.. parsed-literal::

     111	 1.4735820e+00	 1.2701655e-01	 1.5242286e+00	 1.2218582e-01	  1.3646767e+00 	 2.0582795e-01


.. parsed-literal::

     112	 1.4749988e+00	 1.2683253e-01	 1.5256230e+00	 1.2205939e-01	  1.3638725e+00 	 2.0844173e-01


.. parsed-literal::

     113	 1.4762013e+00	 1.2653061e-01	 1.5268146e+00	 1.2145218e-01	  1.3648962e+00 	 2.0584989e-01


.. parsed-literal::

     114	 1.4777206e+00	 1.2635463e-01	 1.5283157e+00	 1.2155975e-01	[ 1.3657407e+00]	 2.0896316e-01


.. parsed-literal::

     115	 1.4789774e+00	 1.2617485e-01	 1.5295977e+00	 1.2162424e-01	[ 1.3670527e+00]	 2.0873952e-01


.. parsed-literal::

     116	 1.4802871e+00	 1.2602005e-01	 1.5309625e+00	 1.2166934e-01	  1.3665512e+00 	 2.1072125e-01


.. parsed-literal::

     117	 1.4818631e+00	 1.2575527e-01	 1.5326398e+00	 1.2174059e-01	  1.3639713e+00 	 2.0792127e-01
     118	 1.4829233e+00	 1.2554964e-01	 1.5338428e+00	 1.2167655e-01	  1.3594775e+00 	 1.9687009e-01


.. parsed-literal::

     119	 1.4844861e+00	 1.2549454e-01	 1.5353755e+00	 1.2164896e-01	  1.3589852e+00 	 2.0289350e-01
     120	 1.4853838e+00	 1.2544444e-01	 1.5362565e+00	 1.2165789e-01	  1.3590071e+00 	 1.7750049e-01


.. parsed-literal::

     121	 1.4864458e+00	 1.2538196e-01	 1.5373109e+00	 1.2172545e-01	  1.3598824e+00 	 2.0594311e-01


.. parsed-literal::

     122	 1.4887107e+00	 1.2531742e-01	 1.5395763e+00	 1.2196078e-01	  1.3628505e+00 	 2.1391344e-01


.. parsed-literal::

     123	 1.4898835e+00	 1.2521434e-01	 1.5407662e+00	 1.2174027e-01	  1.3647632e+00 	 3.2552910e-01


.. parsed-literal::

     124	 1.4915851e+00	 1.2525341e-01	 1.5424400e+00	 1.2191870e-01	[ 1.3682468e+00]	 2.0859098e-01


.. parsed-literal::

     125	 1.4930156e+00	 1.2531064e-01	 1.5438600e+00	 1.2199308e-01	[ 1.3701900e+00]	 2.1049833e-01
     126	 1.4944215e+00	 1.2538542e-01	 1.5452662e+00	 1.2222267e-01	[ 1.3717546e+00]	 1.9868541e-01


.. parsed-literal::

     127	 1.4958270e+00	 1.2546349e-01	 1.5467443e+00	 1.2244891e-01	  1.3705027e+00 	 2.1032214e-01


.. parsed-literal::

     128	 1.4974343e+00	 1.2556838e-01	 1.5483073e+00	 1.2296426e-01	  1.3704801e+00 	 2.1504974e-01


.. parsed-literal::

     129	 1.4984858e+00	 1.2552228e-01	 1.5493383e+00	 1.2311094e-01	  1.3702858e+00 	 2.1518946e-01
     130	 1.5000527e+00	 1.2551216e-01	 1.5509129e+00	 1.2348224e-01	  1.3692950e+00 	 1.7369676e-01


.. parsed-literal::

     131	 1.5016234e+00	 1.2546663e-01	 1.5525081e+00	 1.2376746e-01	[ 1.3724533e+00]	 2.0485950e-01


.. parsed-literal::

     132	 1.5029878e+00	 1.2546430e-01	 1.5538698e+00	 1.2375712e-01	[ 1.3743208e+00]	 2.0978808e-01


.. parsed-literal::

     133	 1.5043424e+00	 1.2548134e-01	 1.5552329e+00	 1.2367289e-01	[ 1.3778388e+00]	 2.0560598e-01


.. parsed-literal::

     134	 1.5054974e+00	 1.2545734e-01	 1.5564263e+00	 1.2355728e-01	[ 1.3795176e+00]	 2.0762753e-01


.. parsed-literal::

     135	 1.5072847e+00	 1.2545834e-01	 1.5583418e+00	 1.2371632e-01	[ 1.3813307e+00]	 2.1895289e-01


.. parsed-literal::

     136	 1.5087914e+00	 1.2543794e-01	 1.5598904e+00	 1.2380323e-01	  1.3802937e+00 	 2.1289015e-01


.. parsed-literal::

     137	 1.5100323e+00	 1.2538271e-01	 1.5611770e+00	 1.2390727e-01	  1.3766659e+00 	 2.1697807e-01
     138	 1.5112349e+00	 1.2538818e-01	 1.5624166e+00	 1.2413413e-01	  1.3741024e+00 	 1.9857454e-01


.. parsed-literal::

     139	 1.5125083e+00	 1.2525456e-01	 1.5638063e+00	 1.2345276e-01	  1.3693596e+00 	 1.9096828e-01


.. parsed-literal::

     140	 1.5142726e+00	 1.2533128e-01	 1.5655548e+00	 1.2374978e-01	  1.3716313e+00 	 2.0897198e-01


.. parsed-literal::

     141	 1.5149887e+00	 1.2531301e-01	 1.5662393e+00	 1.2360085e-01	  1.3738435e+00 	 2.0829368e-01


.. parsed-literal::

     142	 1.5157087e+00	 1.2526726e-01	 1.5669674e+00	 1.2340594e-01	  1.3744683e+00 	 2.0419312e-01


.. parsed-literal::

     143	 1.5173862e+00	 1.2517987e-01	 1.5686751e+00	 1.2331283e-01	  1.3743849e+00 	 2.0390153e-01


.. parsed-literal::

     144	 1.5189212e+00	 1.2500555e-01	 1.5703773e+00	 1.2324377e-01	  1.3645512e+00 	 2.0644593e-01


.. parsed-literal::

     145	 1.5203743e+00	 1.2493512e-01	 1.5718344e+00	 1.2351961e-01	  1.3647823e+00 	 2.1084476e-01
     146	 1.5211882e+00	 1.2492089e-01	 1.5726438e+00	 1.2381795e-01	  1.3630563e+00 	 1.9390297e-01


.. parsed-literal::

     147	 1.5225238e+00	 1.2484535e-01	 1.5740432e+00	 1.2451507e-01	  1.3578743e+00 	 2.1135116e-01


.. parsed-literal::

     148	 1.5236014e+00	 1.2480081e-01	 1.5752551e+00	 1.2518212e-01	  1.3490417e+00 	 2.0941091e-01
     149	 1.5245177e+00	 1.2474351e-01	 1.5761642e+00	 1.2520400e-01	  1.3503776e+00 	 1.9869876e-01


.. parsed-literal::

     150	 1.5256643e+00	 1.2463260e-01	 1.5773514e+00	 1.2518169e-01	  1.3510088e+00 	 2.0525026e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.2 s, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8d84794f40>



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
    CPU times: user 1.72 s, sys: 41 ms, total: 1.76 s
    Wall time: 544 ms


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

