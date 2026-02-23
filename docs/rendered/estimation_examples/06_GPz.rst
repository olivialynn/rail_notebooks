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
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

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
       1	-3.4036463e-01	 3.1988143e-01	-3.3061924e-01	 3.2355949e-01	[-3.3807559e-01]	 4.4030952e-01


.. parsed-literal::

       2	-2.6873588e-01	 3.0849577e-01	-2.4394430e-01	 3.1296271e-01	[-2.5833496e-01]	 2.1782398e-01


.. parsed-literal::

       3	-2.2520563e-01	 2.8869581e-01	-1.8310183e-01	 2.9380399e-01	[-2.0602486e-01]	 2.7859569e-01


.. parsed-literal::

       4	-1.9339367e-01	 2.6410446e-01	-1.5273737e-01	 2.7020993e-01	[-1.8907082e-01]	 2.0479584e-01


.. parsed-literal::

       5	-9.7983373e-02	 2.5493904e-01	-6.0616080e-02	 2.6310006e-01	[-9.9377686e-02]	 2.0784140e-01
       6	-6.6784001e-02	 2.5071577e-01	-3.5093779e-02	 2.6038181e-01	[-6.6375725e-02]	 1.8843699e-01


.. parsed-literal::

       7	-4.5840288e-02	 2.4715965e-01	-2.1462091e-02	 2.5472404e-01	[-5.0861451e-02]	 1.7477322e-01


.. parsed-literal::

       8	-3.5786521e-02	 2.4558335e-01	-1.4865331e-02	 2.5279523e-01	[-4.4094390e-02]	 2.1572971e-01
       9	-2.3440570e-02	 2.4342550e-01	-5.5840484e-03	 2.5052668e-01	[-3.4932187e-02]	 1.9746065e-01


.. parsed-literal::

      10	-1.0289990e-02	 2.4083994e-01	 5.5329844e-03	 2.4796016e-01	[-2.4325898e-02]	 1.8746400e-01
      11	-9.1756247e-03	 2.4103098e-01	 5.4035080e-03	 2.4823381e-01	[-2.2268423e-02]	 1.9569492e-01


.. parsed-literal::

      12	-2.3717918e-03	 2.3965185e-01	 1.1536265e-02	 2.4712974e-01	[-1.9311382e-02]	 1.9739747e-01


.. parsed-literal::

      13	 6.2619633e-04	 2.3898332e-01	 1.4302563e-02	 2.4673080e-01	[-1.6220905e-02]	 2.0216918e-01
      14	 4.1581936e-03	 2.3825519e-01	 1.7988009e-02	 2.4650790e-01	[-1.5340791e-02]	 1.9612765e-01


.. parsed-literal::

      15	 1.2154079e-02	 2.3662856e-01	 2.6805612e-02	 2.4602760e-01	[-1.0527249e-02]	 1.9696856e-01


.. parsed-literal::

      16	 1.5234380e-01	 2.2184838e-01	 1.7493000e-01	 2.2788306e-01	[ 1.5864107e-01]	 4.1207528e-01
      17	 1.9911955e-01	 2.1700267e-01	 2.2142863e-01	 2.2336913e-01	[ 2.0511615e-01]	 1.9843435e-01


.. parsed-literal::

      18	 3.1216828e-01	 2.1348350e-01	 3.3880018e-01	 2.1934057e-01	[ 3.1239404e-01]	 1.8323064e-01


.. parsed-literal::

      19	 3.4617415e-01	 2.1131226e-01	 3.7756367e-01	 2.1720631e-01	[ 3.2492541e-01]	 2.1269608e-01


.. parsed-literal::

      20	 4.0126474e-01	 2.1034875e-01	 4.3306906e-01	 2.1672861e-01	[ 3.8808258e-01]	 2.1343589e-01
      21	 4.5127056e-01	 2.0757367e-01	 4.8400119e-01	 2.1391553e-01	[ 4.3722748e-01]	 1.8797302e-01


.. parsed-literal::

      22	 5.3073350e-01	 2.0634199e-01	 5.6445350e-01	 2.1139758e-01	[ 5.1045848e-01]	 2.1513891e-01


.. parsed-literal::

      23	 6.0231599e-01	 2.0933175e-01	 6.4083570e-01	 2.1427620e-01	[ 5.4442992e-01]	 2.0340037e-01


.. parsed-literal::

      24	 6.5105871e-01	 2.0229053e-01	 6.9080884e-01	 2.0856032e-01	[ 5.7295904e-01]	 2.0881271e-01
      25	 6.8353918e-01	 1.9766873e-01	 7.2131312e-01	 2.0339737e-01	[ 6.1217956e-01]	 1.9814229e-01


.. parsed-literal::

      26	 7.1453857e-01	 1.9386817e-01	 7.5044413e-01	 2.0043496e-01	[ 6.5953065e-01]	 2.0840359e-01
      27	 7.3683788e-01	 1.9190927e-01	 7.7220805e-01	 2.0036172e-01	[ 6.9178052e-01]	 1.9831228e-01


.. parsed-literal::

      28	 7.6051379e-01	 1.9165567e-01	 7.9708918e-01	 2.0101266e-01	[ 7.0195274e-01]	 1.9607663e-01
      29	 7.8696473e-01	 1.9032646e-01	 8.2437073e-01	 1.9978185e-01	[ 7.2518098e-01]	 1.7949104e-01


.. parsed-literal::

      30	 8.1182905e-01	 1.9094660e-01	 8.4975303e-01	 2.0208079e-01	[ 7.4999106e-01]	 2.0112300e-01


.. parsed-literal::

      31	 8.3575901e-01	 1.9343660e-01	 8.7423294e-01	 2.0660233e-01	[ 7.8568878e-01]	 2.0829988e-01
      32	 8.5294329e-01	 1.9284817e-01	 8.9411520e-01	 2.0683956e-01	[ 7.8949733e-01]	 2.0589662e-01


.. parsed-literal::

      33	 8.8194171e-01	 1.9309172e-01	 9.2259739e-01	 2.0571208e-01	[ 8.3484410e-01]	 1.6606998e-01
      34	 8.9922447e-01	 1.8880195e-01	 9.3973084e-01	 1.9897608e-01	[ 8.4549682e-01]	 1.8037009e-01


.. parsed-literal::

      35	 9.1393862e-01	 1.8522364e-01	 9.5394618e-01	 1.9617286e-01	[ 8.6785378e-01]	 1.9959402e-01
      36	 9.3100693e-01	 1.8312914e-01	 9.7167181e-01	 1.9396212e-01	[ 8.8339804e-01]	 1.9565225e-01


.. parsed-literal::

      37	 9.4874099e-01	 1.7963818e-01	 9.9001820e-01	 1.8977088e-01	[ 8.9703998e-01]	 1.6592622e-01
      38	 9.6542415e-01	 1.7741803e-01	 1.0074244e+00	 1.8719285e-01	[ 9.1829454e-01]	 1.9356561e-01


.. parsed-literal::

      39	 9.7754633e-01	 1.7561279e-01	 1.0203234e+00	 1.8460230e-01	[ 9.2942567e-01]	 2.0531797e-01
      40	 9.8890039e-01	 1.7498010e-01	 1.0314527e+00	 1.8400640e-01	[ 9.3984778e-01]	 1.9488788e-01


.. parsed-literal::

      41	 9.9988899e-01	 1.7411489e-01	 1.0427084e+00	 1.8319995e-01	[ 9.4951132e-01]	 1.6115308e-01
      42	 1.0168258e+00	 1.7355352e-01	 1.0602256e+00	 1.8298118e-01	[ 9.6277244e-01]	 1.9935679e-01


.. parsed-literal::

      43	 1.0289812e+00	 1.7477356e-01	 1.0740011e+00	 1.8416723e-01	[ 9.7262443e-01]	 2.0472813e-01


.. parsed-literal::

      44	 1.0429447e+00	 1.7486606e-01	 1.0879068e+00	 1.8500511e-01	[ 9.8500722e-01]	 2.0759130e-01
      45	 1.0494167e+00	 1.7401154e-01	 1.0940435e+00	 1.8384756e-01	[ 9.9642956e-01]	 2.0146894e-01


.. parsed-literal::

      46	 1.0591107e+00	 1.7354422e-01	 1.1041419e+00	 1.8301167e-01	[ 1.0053506e+00]	 2.1136808e-01
      47	 1.0676956e+00	 1.7325948e-01	 1.1135306e+00	 1.8236996e-01	[ 1.0173079e+00]	 1.7327952e-01


.. parsed-literal::

      48	 1.0753672e+00	 1.7270976e-01	 1.1213662e+00	 1.8160701e-01	[ 1.0239060e+00]	 2.0324063e-01
      49	 1.0875886e+00	 1.7118683e-01	 1.1337132e+00	 1.7997287e-01	[ 1.0374193e+00]	 1.9540453e-01


.. parsed-literal::

      50	 1.0989338e+00	 1.6955450e-01	 1.1452699e+00	 1.7815050e-01	[ 1.0537236e+00]	 2.0415711e-01
      51	 1.1077869e+00	 1.6708351e-01	 1.1539447e+00	 1.7558564e-01	[ 1.0611433e+00]	 1.9387007e-01


.. parsed-literal::

      52	 1.1133151e+00	 1.6676590e-01	 1.1593889e+00	 1.7547992e-01	[ 1.0663447e+00]	 1.9778800e-01


.. parsed-literal::

      53	 1.1208728e+00	 1.6477112e-01	 1.1669096e+00	 1.7382694e-01	[ 1.0680070e+00]	 2.0684147e-01
      54	 1.1257853e+00	 1.6449000e-01	 1.1721394e+00	 1.7380213e-01	[ 1.0732789e+00]	 1.9773889e-01


.. parsed-literal::

      55	 1.1306639e+00	 1.6360226e-01	 1.1769968e+00	 1.7322862e-01	[ 1.0765073e+00]	 1.9927192e-01
      56	 1.1406955e+00	 1.6098572e-01	 1.1873345e+00	 1.7178669e-01	[ 1.0831727e+00]	 1.9386363e-01


.. parsed-literal::

      57	 1.1461719e+00	 1.5987814e-01	 1.1928541e+00	 1.7090935e-01	[ 1.0909139e+00]	 2.1374965e-01


.. parsed-literal::

      58	 1.1515441e+00	 1.5610734e-01	 1.1990933e+00	 1.6818711e-01	[ 1.0983055e+00]	 2.0756936e-01


.. parsed-literal::

      59	 1.1610738e+00	 1.5555050e-01	 1.2083868e+00	 1.6661184e-01	[ 1.1120466e+00]	 2.0372963e-01


.. parsed-literal::

      60	 1.1645442e+00	 1.5533941e-01	 1.2118448e+00	 1.6640551e-01	[ 1.1145221e+00]	 2.0497441e-01


.. parsed-literal::

      61	 1.1710543e+00	 1.5409593e-01	 1.2186395e+00	 1.6514670e-01	[ 1.1182928e+00]	 2.0424795e-01
      62	 1.1780186e+00	 1.5242659e-01	 1.2259420e+00	 1.6333753e-01	[ 1.1239740e+00]	 1.7480707e-01


.. parsed-literal::

      63	 1.1891232e+00	 1.4960643e-01	 1.2374393e+00	 1.6052856e-01	[ 1.1390507e+00]	 2.0929670e-01
      64	 1.1939075e+00	 1.4710504e-01	 1.2426852e+00	 1.5820231e-01	[ 1.1492491e+00]	 1.7439890e-01


.. parsed-literal::

      65	 1.2012961e+00	 1.4750798e-01	 1.2497093e+00	 1.5823026e-01	[ 1.1582327e+00]	 2.1253490e-01
      66	 1.2054818e+00	 1.4722224e-01	 1.2538302e+00	 1.5799849e-01	[ 1.1620605e+00]	 1.7190838e-01


.. parsed-literal::

      67	 1.2121632e+00	 1.4629265e-01	 1.2608213e+00	 1.5718051e-01	[ 1.1680904e+00]	 2.0682144e-01


.. parsed-literal::

      68	 1.2172931e+00	 1.4531313e-01	 1.2662187e+00	 1.5623335e-01	  1.1641766e+00 	 2.0459580e-01
      69	 1.2215726e+00	 1.4456695e-01	 1.2706004e+00	 1.5563638e-01	  1.1666897e+00 	 1.7043328e-01


.. parsed-literal::

      70	 1.2260808e+00	 1.4368331e-01	 1.2752471e+00	 1.5485603e-01	  1.1656728e+00 	 1.9894528e-01


.. parsed-literal::

      71	 1.2298845e+00	 1.4338877e-01	 1.2790483e+00	 1.5439784e-01	[ 1.1702489e+00]	 2.0558047e-01
      72	 1.2365683e+00	 1.4254160e-01	 1.2857835e+00	 1.5294991e-01	[ 1.1764449e+00]	 1.9643354e-01


.. parsed-literal::

      73	 1.2417669e+00	 1.4252890e-01	 1.2910235e+00	 1.5250210e-01	[ 1.1875198e+00]	 2.1207738e-01
      74	 1.2464595e+00	 1.4230947e-01	 1.2958277e+00	 1.5200696e-01	[ 1.1938677e+00]	 1.7742944e-01


.. parsed-literal::

      75	 1.2515042e+00	 1.4185921e-01	 1.3012266e+00	 1.5150689e-01	[ 1.1958797e+00]	 3.2250786e-01
      76	 1.2572900e+00	 1.4141852e-01	 1.3073759e+00	 1.5091085e-01	[ 1.2000111e+00]	 1.9823575e-01


.. parsed-literal::

      77	 1.2599482e+00	 1.4110751e-01	 1.3101153e+00	 1.5086813e-01	  1.1990384e+00 	 2.0963550e-01


.. parsed-literal::

      78	 1.2653383e+00	 1.4039057e-01	 1.3159301e+00	 1.5090312e-01	  1.1919043e+00 	 2.0612526e-01
      79	 1.2663104e+00	 1.3982900e-01	 1.3170999e+00	 1.5066710e-01	  1.1922087e+00 	 1.9501615e-01


.. parsed-literal::

      80	 1.2705684e+00	 1.3990092e-01	 1.3210767e+00	 1.5045206e-01	[ 1.2000503e+00]	 1.9768929e-01
      81	 1.2727069e+00	 1.3975213e-01	 1.3232112e+00	 1.5013139e-01	[ 1.2030733e+00]	 1.6256809e-01


.. parsed-literal::

      82	 1.2759489e+00	 1.3948144e-01	 1.3263888e+00	 1.4964850e-01	[ 1.2084020e+00]	 2.0389056e-01
      83	 1.2800903e+00	 1.3907535e-01	 1.3305551e+00	 1.4912509e-01	[ 1.2129289e+00]	 1.7014623e-01


.. parsed-literal::

      84	 1.2821672e+00	 1.3890178e-01	 1.3328357e+00	 1.4873987e-01	  1.2103707e+00 	 2.0953631e-01


.. parsed-literal::

      85	 1.2868767e+00	 1.3862981e-01	 1.3374845e+00	 1.4862468e-01	[ 1.2171601e+00]	 2.1038651e-01


.. parsed-literal::

      86	 1.2890317e+00	 1.3845820e-01	 1.3397489e+00	 1.4853748e-01	[ 1.2184635e+00]	 2.1107173e-01
      87	 1.2926743e+00	 1.3811673e-01	 1.3436119e+00	 1.4813686e-01	[ 1.2209345e+00]	 2.0132709e-01


.. parsed-literal::

      88	 1.2979229e+00	 1.3751702e-01	 1.3491149e+00	 1.4731318e-01	[ 1.2243994e+00]	 1.7429376e-01


.. parsed-literal::

      89	 1.3010788e+00	 1.3691095e-01	 1.3525182e+00	 1.4614332e-01	[ 1.2300476e+00]	 2.9445553e-01


.. parsed-literal::

      90	 1.3046389e+00	 1.3654861e-01	 1.3560568e+00	 1.4550154e-01	[ 1.2338996e+00]	 2.0641017e-01
      91	 1.3068386e+00	 1.3631818e-01	 1.3581436e+00	 1.4515864e-01	[ 1.2362579e+00]	 1.7157817e-01


.. parsed-literal::

      92	 1.3095513e+00	 1.3599864e-01	 1.3607389e+00	 1.4465594e-01	[ 1.2389673e+00]	 2.1168733e-01
      93	 1.3105856e+00	 1.3587734e-01	 1.3616765e+00	 1.4458944e-01	  1.2386468e+00 	 1.9658613e-01


.. parsed-literal::

      94	 1.3136562e+00	 1.3569656e-01	 1.3647516e+00	 1.4437173e-01	[ 1.2413178e+00]	 1.9831824e-01
      95	 1.3151860e+00	 1.3562349e-01	 1.3663433e+00	 1.4433673e-01	[ 1.2420289e+00]	 1.8948960e-01


.. parsed-literal::

      96	 1.3173022e+00	 1.3552851e-01	 1.3685734e+00	 1.4428270e-01	[ 1.2429891e+00]	 2.0293403e-01
      97	 1.3208594e+00	 1.3533124e-01	 1.3723099e+00	 1.4403296e-01	[ 1.2459340e+00]	 1.9860005e-01


.. parsed-literal::

      98	 1.3278167e+00	 1.3476998e-01	 1.3795339e+00	 1.4322045e-01	[ 1.2552473e+00]	 2.1056247e-01


.. parsed-literal::

      99	 1.3316613e+00	 1.3447519e-01	 1.3834954e+00	 1.4286024e-01	[ 1.2558933e+00]	 3.1718540e-01


.. parsed-literal::

     100	 1.3371258e+00	 1.3402138e-01	 1.3890193e+00	 1.4224052e-01	[ 1.2658807e+00]	 2.0813394e-01
     101	 1.3405952e+00	 1.3373491e-01	 1.3924106e+00	 1.4198156e-01	[ 1.2696083e+00]	 2.0014191e-01


.. parsed-literal::

     102	 1.3435121e+00	 1.3341881e-01	 1.3952277e+00	 1.4200478e-01	  1.2684674e+00 	 2.1308565e-01
     103	 1.3462846e+00	 1.3325958e-01	 1.3980176e+00	 1.4193767e-01	[ 1.2709279e+00]	 1.9586802e-01


.. parsed-literal::

     104	 1.3494768e+00	 1.3291235e-01	 1.4014292e+00	 1.4162806e-01	[ 1.2711219e+00]	 2.0865011e-01
     105	 1.3517560e+00	 1.3276039e-01	 1.4039225e+00	 1.4161637e-01	[ 1.2715272e+00]	 1.9384456e-01


.. parsed-literal::

     106	 1.3545449e+00	 1.3231959e-01	 1.4069329e+00	 1.4100863e-01	[ 1.2727177e+00]	 2.0202208e-01


.. parsed-literal::

     107	 1.3578269e+00	 1.3198234e-01	 1.4103870e+00	 1.4058845e-01	[ 1.2737480e+00]	 2.1171045e-01
     108	 1.3618367e+00	 1.3169232e-01	 1.4145159e+00	 1.4021295e-01	[ 1.2804600e+00]	 1.6449285e-01


.. parsed-literal::

     109	 1.3647410e+00	 1.3179641e-01	 1.4173883e+00	 1.4041686e-01	  1.2771814e+00 	 2.1232820e-01


.. parsed-literal::

     110	 1.3667706e+00	 1.3174263e-01	 1.4192903e+00	 1.4047348e-01	  1.2803565e+00 	 2.3918056e-01


.. parsed-literal::

     111	 1.3686456e+00	 1.3177823e-01	 1.4210827e+00	 1.4059627e-01	[ 1.2834706e+00]	 2.1612477e-01
     112	 1.3710619e+00	 1.3169639e-01	 1.4235655e+00	 1.4065908e-01	[ 1.2844812e+00]	 1.7557788e-01


.. parsed-literal::

     113	 1.3742748e+00	 1.3157534e-01	 1.4270112e+00	 1.4065073e-01	  1.2839196e+00 	 2.0669580e-01
     114	 1.3766899e+00	 1.3145305e-01	 1.4297598e+00	 1.4043138e-01	[ 1.2884634e+00]	 2.0331812e-01


.. parsed-literal::

     115	 1.3787646e+00	 1.3131235e-01	 1.4320114e+00	 1.4029245e-01	  1.2829775e+00 	 1.9491220e-01
     116	 1.3799969e+00	 1.3127498e-01	 1.4331460e+00	 1.4016263e-01	  1.2854752e+00 	 1.9171166e-01


.. parsed-literal::

     117	 1.3813240e+00	 1.3120448e-01	 1.4344461e+00	 1.3997153e-01	  1.2868581e+00 	 2.0146966e-01
     118	 1.3832544e+00	 1.3113249e-01	 1.4363777e+00	 1.3981069e-01	  1.2870536e+00 	 1.6721845e-01


.. parsed-literal::

     119	 1.3858789e+00	 1.3112736e-01	 1.4392741e+00	 1.3964120e-01	  1.2818748e+00 	 1.7255259e-01
     120	 1.3888205e+00	 1.3113442e-01	 1.4421953e+00	 1.3958358e-01	  1.2852596e+00 	 1.9259405e-01


.. parsed-literal::

     121	 1.3905399e+00	 1.3122186e-01	 1.4438801e+00	 1.3972575e-01	  1.2853028e+00 	 2.0406294e-01


.. parsed-literal::

     122	 1.3933686e+00	 1.3137089e-01	 1.4466934e+00	 1.3996479e-01	  1.2873309e+00 	 2.0440555e-01
     123	 1.3959058e+00	 1.3159268e-01	 1.4492806e+00	 1.4005960e-01	  1.2861515e+00 	 1.8378544e-01


.. parsed-literal::

     124	 1.3982813e+00	 1.3166059e-01	 1.4516372e+00	 1.4016742e-01	  1.2878120e+00 	 1.9035912e-01


.. parsed-literal::

     125	 1.4023766e+00	 1.3152231e-01	 1.4558821e+00	 1.4011129e-01	  1.2869026e+00 	 2.1380734e-01


.. parsed-literal::

     126	 1.4043543e+00	 1.3155774e-01	 1.4579842e+00	 1.4043821e-01	[ 1.2902120e+00]	 3.1292272e-01


.. parsed-literal::

     127	 1.4062854e+00	 1.3126724e-01	 1.4599935e+00	 1.4034239e-01	  1.2880108e+00 	 2.0985103e-01


.. parsed-literal::

     128	 1.4077216e+00	 1.3098930e-01	 1.4614734e+00	 1.4026845e-01	  1.2876308e+00 	 2.0344472e-01
     129	 1.4095525e+00	 1.3082862e-01	 1.4633758e+00	 1.4036513e-01	  1.2847521e+00 	 1.9965363e-01


.. parsed-literal::

     130	 1.4120169e+00	 1.3063434e-01	 1.4658938e+00	 1.4050249e-01	  1.2828270e+00 	 1.7984462e-01
     131	 1.4140925e+00	 1.3049803e-01	 1.4679507e+00	 1.4053696e-01	  1.2812902e+00 	 1.8784595e-01


.. parsed-literal::

     132	 1.4167489e+00	 1.3035472e-01	 1.4705196e+00	 1.4048201e-01	  1.2809341e+00 	 1.8844128e-01


.. parsed-literal::

     133	 1.4192016e+00	 1.2988084e-01	 1.4728690e+00	 1.4030936e-01	  1.2868938e+00 	 2.0084643e-01


.. parsed-literal::

     134	 1.4217623e+00	 1.2974860e-01	 1.4753538e+00	 1.4035390e-01	  1.2888363e+00 	 2.0846868e-01


.. parsed-literal::

     135	 1.4233858e+00	 1.2961251e-01	 1.4769766e+00	 1.4037818e-01	  1.2899629e+00 	 2.1155930e-01


.. parsed-literal::

     136	 1.4251591e+00	 1.2940761e-01	 1.4787928e+00	 1.4046598e-01	[ 1.2902759e+00]	 2.1328211e-01
     137	 1.4278365e+00	 1.2910490e-01	 1.4815287e+00	 1.4058767e-01	  1.2896474e+00 	 1.8175435e-01


.. parsed-literal::

     138	 1.4290476e+00	 1.2905081e-01	 1.4828135e+00	 1.4082765e-01	[ 1.2925855e+00]	 3.1799912e-01
     139	 1.4304999e+00	 1.2893648e-01	 1.4843087e+00	 1.4095330e-01	  1.2903469e+00 	 1.9644165e-01


.. parsed-literal::

     140	 1.4319615e+00	 1.2889165e-01	 1.4857930e+00	 1.4099415e-01	  1.2903997e+00 	 2.0591998e-01
     141	 1.4344823e+00	 1.2870918e-01	 1.4884090e+00	 1.4100425e-01	  1.2901341e+00 	 1.9620132e-01


.. parsed-literal::

     142	 1.4365418e+00	 1.2869397e-01	 1.4905236e+00	 1.4094865e-01	[ 1.2927317e+00]	 1.9723535e-01


.. parsed-literal::

     143	 1.4382205e+00	 1.2867644e-01	 1.4921683e+00	 1.4090012e-01	[ 1.2945847e+00]	 2.1408963e-01


.. parsed-literal::

     144	 1.4406979e+00	 1.2866362e-01	 1.4946204e+00	 1.4083493e-01	[ 1.2953151e+00]	 2.1136689e-01


.. parsed-literal::

     145	 1.4427193e+00	 1.2871139e-01	 1.4966716e+00	 1.4089648e-01	  1.2928602e+00 	 2.1198463e-01


.. parsed-literal::

     146	 1.4442296e+00	 1.2880196e-01	 1.4982044e+00	 1.4083306e-01	  1.2893713e+00 	 2.8411889e-01
     147	 1.4458952e+00	 1.2887461e-01	 1.4999712e+00	 1.4103434e-01	  1.2842604e+00 	 2.0184255e-01


.. parsed-literal::

     148	 1.4469151e+00	 1.2888445e-01	 1.5010260e+00	 1.4105511e-01	  1.2833644e+00 	 2.1109605e-01
     149	 1.4488258e+00	 1.2893086e-01	 1.5030400e+00	 1.4085737e-01	  1.2770636e+00 	 2.0399857e-01


.. parsed-literal::

     150	 1.4502589e+00	 1.2904312e-01	 1.5045585e+00	 1.4084861e-01	  1.2814034e+00 	 1.7767954e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.09 s, total: 2min 4s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe876a9e010>



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

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 941 ms, sys: 51.9 ms, total: 993 ms
    Wall time: 364 ms


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




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

