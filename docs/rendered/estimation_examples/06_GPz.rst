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
       1	-3.2688359e-01	 3.1569384e-01	-3.1717755e-01	 3.4055800e-01	[-3.6528695e-01]	 4.6095920e-01


.. parsed-literal::

       2	-2.6182347e-01	 3.0733466e-01	-2.4025778e-01	 3.2901268e-01	[-3.0853896e-01]	 2.2942162e-01


.. parsed-literal::

       3	-2.1910751e-01	 2.8842367e-01	-1.7915370e-01	 3.0732255e-01	[-2.5075295e-01]	 2.9894543e-01


.. parsed-literal::

       4	-1.8900193e-01	 2.7323311e-01	-1.4410149e-01	 2.9096049e-01	[-2.1891313e-01]	 2.9062390e-01


.. parsed-literal::

       5	-1.2852059e-01	 2.5725084e-01	-8.8937942e-02	 2.6941558e-01	[-1.4646728e-01]	 2.2076869e-01


.. parsed-literal::

       6	-6.6787760e-02	 2.5063443e-01	-3.7013118e-02	 2.6111242e-01	[-7.9483113e-02]	 2.2114968e-01
       7	-4.8026241e-02	 2.4793582e-01	-2.3916480e-02	 2.5843148e-01	[-6.5537316e-02]	 1.8247056e-01


.. parsed-literal::

       8	-3.1837795e-02	 2.4506779e-01	-1.2881190e-02	 2.5588145e-01	[-5.7266712e-02]	 1.8572998e-01
       9	-2.4848086e-02	 2.4386587e-01	-7.7694639e-03	 2.5446021e-01	[-5.0192678e-02]	 1.9875741e-01


.. parsed-literal::

      10	-1.4567878e-02	 2.4186399e-01	 5.2722506e-04	 2.5144121e-01	[-4.0627696e-02]	 2.0273638e-01
      11	-7.2919605e-03	 2.4025767e-01	 6.2379574e-03	 2.4808633e-01	[-2.2956906e-02]	 1.7216611e-01


.. parsed-literal::

      12	-1.6431167e-03	 2.3960142e-01	 1.2038817e-02	 2.4739608e-01	[-1.9533500e-02]	 2.0618749e-01


.. parsed-literal::

      13	 1.4661199e-03	 2.3896723e-01	 1.5316360e-02	 2.4684032e-01	[-1.8066746e-02]	 2.0363235e-01


.. parsed-literal::

      14	 6.2703333e-03	 2.3788325e-01	 2.0821560e-02	 2.4586759e-01	[-1.3865117e-02]	 2.0365620e-01


.. parsed-literal::

      15	 1.6418337e-01	 2.2771657e-01	 1.8985689e-01	 2.4522058e-01	[ 1.6702073e-01]	 3.2668543e-01


.. parsed-literal::

      16	 1.9529345e-01	 2.2203565e-01	 2.2141314e-01	 2.2484467e-01	[ 2.1794638e-01]	 2.8891468e-01


.. parsed-literal::

      17	 2.2695364e-01	 2.1786217e-01	 2.5429397e-01	 2.2235681e-01	[ 2.4854176e-01]	 2.1534657e-01


.. parsed-literal::

      18	 2.7832407e-01	 2.1335336e-01	 3.0858272e-01	 2.2157536e-01	[ 2.9871107e-01]	 2.1497822e-01


.. parsed-literal::

      19	 3.2493214e-01	 2.0775768e-01	 3.5737671e-01	 2.1460870e-01	[ 3.5168758e-01]	 2.1378517e-01


.. parsed-literal::

      20	 3.7373680e-01	 2.0610957e-01	 4.0673853e-01	 2.1764325e-01	[ 4.0048085e-01]	 2.0215297e-01


.. parsed-literal::

      21	 4.3777815e-01	 2.0510364e-01	 4.7167818e-01	 2.1587020e-01	[ 4.7385476e-01]	 2.1006799e-01


.. parsed-literal::

      22	 5.1480838e-01	 2.0473046e-01	 5.5018103e-01	 2.1025820e-01	[ 5.4887349e-01]	 2.0913672e-01
      23	 5.4158089e-01	 2.0670262e-01	 5.8345609e-01	 2.1301306e-01	[ 5.5474952e-01]	 2.0111108e-01


.. parsed-literal::

      24	 6.1286447e-01	 2.0202069e-01	 6.5124237e-01	 2.0913424e-01	[ 6.2354960e-01]	 1.6867113e-01
      25	 6.4207987e-01	 1.9749870e-01	 6.8035626e-01	 2.0250398e-01	[ 6.4935047e-01]	 1.8360877e-01


.. parsed-literal::

      26	 6.8003228e-01	 1.9547782e-01	 7.1706225e-01	 1.9499815e-01	[ 6.8223853e-01]	 1.7630315e-01


.. parsed-literal::

      27	 7.2407273e-01	 1.9309963e-01	 7.6158556e-01	 1.9464434e-01	[ 7.2024931e-01]	 2.1343446e-01
      28	 7.5933992e-01	 1.8980884e-01	 7.9836270e-01	 1.9016529e-01	[ 7.3817078e-01]	 2.0724249e-01


.. parsed-literal::

      29	 7.8354657e-01	 1.9082442e-01	 8.2259708e-01	 1.8958676e-01	[ 7.6517736e-01]	 2.0805836e-01


.. parsed-literal::

      30	 8.1147492e-01	 1.8769077e-01	 8.5113844e-01	 1.8904957e-01	[ 7.8267051e-01]	 2.0610309e-01


.. parsed-literal::

      31	 8.3959079e-01	 1.8760288e-01	 8.8010097e-01	 1.8939550e-01	[ 8.0931837e-01]	 2.0316458e-01


.. parsed-literal::

      32	 8.6698106e-01	 1.8519530e-01	 9.0844868e-01	 1.8968121e-01	[ 8.1937161e-01]	 2.1027613e-01
      33	 8.8408810e-01	 1.8330495e-01	 9.2552206e-01	 1.8669253e-01	[ 8.4117601e-01]	 2.0002794e-01


.. parsed-literal::

      34	 9.0408314e-01	 1.8091724e-01	 9.4557478e-01	 1.8188775e-01	[ 8.7155255e-01]	 1.9042182e-01
      35	 9.1628107e-01	 1.7932443e-01	 9.5803999e-01	 1.8004800e-01	[ 8.9157701e-01]	 1.8864632e-01


.. parsed-literal::

      36	 9.3012895e-01	 1.7801624e-01	 9.7189954e-01	 1.7836904e-01	[ 9.0419782e-01]	 1.7496228e-01
      37	 9.4491622e-01	 1.7699175e-01	 9.8704407e-01	 1.7735845e-01	[ 9.1946165e-01]	 1.8707252e-01


.. parsed-literal::

      38	 9.6174380e-01	 1.7498795e-01	 1.0046393e+00	 1.7500400e-01	[ 9.3311712e-01]	 2.0064402e-01


.. parsed-literal::

      39	 9.7676109e-01	 1.7145516e-01	 1.0218889e+00	 1.7202491e-01	[ 9.3619061e-01]	 2.0439029e-01


.. parsed-literal::

      40	 9.9773833e-01	 1.6943583e-01	 1.0431624e+00	 1.6908806e-01	[ 9.5416210e-01]	 2.1625376e-01


.. parsed-literal::

      41	 1.0076149e+00	 1.6883002e-01	 1.0529071e+00	 1.6895983e-01	[ 9.6352934e-01]	 2.1447396e-01


.. parsed-literal::

      42	 1.0269713e+00	 1.6770637e-01	 1.0727355e+00	 1.6918273e-01	[ 9.7237774e-01]	 2.1623755e-01


.. parsed-literal::

      43	 1.0427510e+00	 1.6683264e-01	 1.0888204e+00	 1.6854759e-01	[ 9.9490280e-01]	 2.1890116e-01
      44	 1.0557980e+00	 1.6557286e-01	 1.1019602e+00	 1.6602516e-01	[ 1.0094980e+00]	 2.1167755e-01


.. parsed-literal::

      45	 1.0657232e+00	 1.6389551e-01	 1.1122620e+00	 1.6268756e-01	[ 1.0163099e+00]	 2.0781040e-01


.. parsed-literal::

      46	 1.0805159e+00	 1.6197396e-01	 1.1275861e+00	 1.5910801e-01	[ 1.0267620e+00]	 2.1399641e-01
      47	 1.0969878e+00	 1.6088530e-01	 1.1446975e+00	 1.5716883e-01	[ 1.0317964e+00]	 1.9968200e-01


.. parsed-literal::

      48	 1.1088488e+00	 1.6001541e-01	 1.1565415e+00	 1.5756898e-01	[ 1.0432957e+00]	 2.1529245e-01
      49	 1.1171516e+00	 1.5992255e-01	 1.1648162e+00	 1.5780276e-01	[ 1.0538012e+00]	 1.6710973e-01


.. parsed-literal::

      50	 1.1257259e+00	 1.5959516e-01	 1.1738156e+00	 1.5739558e-01	[ 1.0621028e+00]	 2.0259428e-01


.. parsed-literal::

      51	 1.1375950e+00	 1.5879780e-01	 1.1861495e+00	 1.5642236e-01	[ 1.0756004e+00]	 2.1383786e-01


.. parsed-literal::

      52	 1.1488597e+00	 1.5830289e-01	 1.1980178e+00	 1.5674191e-01	[ 1.0885029e+00]	 2.0515895e-01


.. parsed-literal::

      53	 1.1614247e+00	 1.5625636e-01	 1.2104387e+00	 1.5407140e-01	[ 1.1096897e+00]	 2.1278191e-01
      54	 1.1714713e+00	 1.5531222e-01	 1.2204558e+00	 1.5332028e-01	[ 1.1193429e+00]	 1.7980218e-01


.. parsed-literal::

      55	 1.1846825e+00	 1.5469640e-01	 1.2336181e+00	 1.5435932e-01	[ 1.1283510e+00]	 2.0331478e-01
      56	 1.1949137e+00	 1.5457381e-01	 1.2439540e+00	 1.5661138e-01	[ 1.1370908e+00]	 1.9442868e-01


.. parsed-literal::

      57	 1.2080533e+00	 1.5433176e-01	 1.2569315e+00	 1.5724286e-01	[ 1.1486263e+00]	 2.1738434e-01


.. parsed-literal::

      58	 1.2162599e+00	 1.5459625e-01	 1.2653804e+00	 1.5866157e-01	[ 1.1540046e+00]	 2.0898700e-01
      59	 1.2270846e+00	 1.5486794e-01	 1.2766641e+00	 1.5998240e-01	  1.1531876e+00 	 1.7243505e-01


.. parsed-literal::

      60	 1.2358610e+00	 1.5647447e-01	 1.2864791e+00	 1.6416343e-01	  1.1416407e+00 	 1.8059111e-01
      61	 1.2489760e+00	 1.5494607e-01	 1.2994872e+00	 1.6196440e-01	  1.1416472e+00 	 1.9666123e-01


.. parsed-literal::

      62	 1.2556777e+00	 1.5340121e-01	 1.3062474e+00	 1.5966941e-01	  1.1381568e+00 	 2.1521020e-01


.. parsed-literal::

      63	 1.2638217e+00	 1.5235115e-01	 1.3146535e+00	 1.5854326e-01	  1.1336394e+00 	 2.0934820e-01


.. parsed-literal::

      64	 1.2738350e+00	 1.5151297e-01	 1.3250475e+00	 1.5850215e-01	  1.1232790e+00 	 2.1562362e-01
      65	 1.2836879e+00	 1.5031498e-01	 1.3351468e+00	 1.5646760e-01	  1.1241781e+00 	 1.8954206e-01


.. parsed-literal::

      66	 1.2954342e+00	 1.4876523e-01	 1.3472279e+00	 1.5418968e-01	  1.1224894e+00 	 2.0321965e-01
      67	 1.3048423e+00	 1.4696272e-01	 1.3568143e+00	 1.5222565e-01	  1.1046465e+00 	 1.8697166e-01


.. parsed-literal::

      68	 1.3142595e+00	 1.4558220e-01	 1.3659160e+00	 1.5006984e-01	  1.1189910e+00 	 2.1459484e-01


.. parsed-literal::

      69	 1.3207497e+00	 1.4468661e-01	 1.3725670e+00	 1.4937133e-01	  1.1103936e+00 	 2.0157480e-01


.. parsed-literal::

      70	 1.3274202e+00	 1.4398032e-01	 1.3791912e+00	 1.4772678e-01	  1.1131670e+00 	 2.1200871e-01


.. parsed-literal::

      71	 1.3337951e+00	 1.4293390e-01	 1.3857038e+00	 1.4606284e-01	  1.1073545e+00 	 2.0978117e-01
      72	 1.3406296e+00	 1.4184734e-01	 1.3926481e+00	 1.4354293e-01	  1.1252728e+00 	 1.8686819e-01


.. parsed-literal::

      73	 1.3462399e+00	 1.4139218e-01	 1.3982737e+00	 1.4251722e-01	  1.1348664e+00 	 2.0128155e-01


.. parsed-literal::

      74	 1.3553191e+00	 1.4036543e-01	 1.4075750e+00	 1.3987014e-01	  1.1470320e+00 	 2.0571828e-01
      75	 1.3638383e+00	 1.3920683e-01	 1.4163179e+00	 1.3581532e-01	[ 1.1660039e+00]	 1.8639565e-01


.. parsed-literal::

      76	 1.3713555e+00	 1.3843244e-01	 1.4240430e+00	 1.3459342e-01	  1.1634314e+00 	 2.0920706e-01


.. parsed-literal::

      77	 1.3760984e+00	 1.3790699e-01	 1.4288701e+00	 1.3371008e-01	[ 1.1681682e+00]	 2.0739031e-01


.. parsed-literal::

      78	 1.3811781e+00	 1.3756713e-01	 1.4342263e+00	 1.3355804e-01	  1.1649499e+00 	 2.1554542e-01
      79	 1.3870887e+00	 1.3704917e-01	 1.4401235e+00	 1.3325800e-01	[ 1.1791573e+00]	 2.0572710e-01


.. parsed-literal::

      80	 1.3940603e+00	 1.3615469e-01	 1.4472203e+00	 1.3281578e-01	  1.1760201e+00 	 2.1002102e-01


.. parsed-literal::

      81	 1.3989506e+00	 1.3606803e-01	 1.4520263e+00	 1.3416452e-01	  1.1744545e+00 	 2.0169234e-01


.. parsed-literal::

      82	 1.4024177e+00	 1.3555134e-01	 1.4553982e+00	 1.3353427e-01	  1.1752750e+00 	 2.0882201e-01
      83	 1.4066541e+00	 1.3497581e-01	 1.4595950e+00	 1.3291410e-01	  1.1656766e+00 	 1.9725490e-01


.. parsed-literal::

      84	 1.4100953e+00	 1.3423497e-01	 1.4631073e+00	 1.3295671e-01	  1.1325192e+00 	 2.0057178e-01


.. parsed-literal::

      85	 1.4146676e+00	 1.3393031e-01	 1.4677057e+00	 1.3244843e-01	  1.1304947e+00 	 2.1498179e-01


.. parsed-literal::

      86	 1.4185507e+00	 1.3364849e-01	 1.4716853e+00	 1.3210026e-01	  1.1240722e+00 	 2.1891546e-01
      87	 1.4219745e+00	 1.3334797e-01	 1.4751986e+00	 1.3184123e-01	  1.1172440e+00 	 1.9850874e-01


.. parsed-literal::

      88	 1.4292835e+00	 1.3251035e-01	 1.4828011e+00	 1.3186608e-01	  1.0951359e+00 	 1.8230557e-01


.. parsed-literal::

      89	 1.4333885e+00	 1.3225665e-01	 1.4870425e+00	 1.3185333e-01	  1.0848740e+00 	 3.2353997e-01
      90	 1.4368766e+00	 1.3180893e-01	 1.4905057e+00	 1.3146189e-01	  1.0887133e+00 	 1.8593669e-01


.. parsed-literal::

      91	 1.4401407e+00	 1.3138132e-01	 1.4938457e+00	 1.3114729e-01	  1.0843455e+00 	 1.9256020e-01


.. parsed-literal::

      92	 1.4429713e+00	 1.3112015e-01	 1.4966525e+00	 1.3099692e-01	  1.0933578e+00 	 2.0872116e-01


.. parsed-literal::

      93	 1.4464962e+00	 1.3091495e-01	 1.5001063e+00	 1.3102999e-01	  1.1020810e+00 	 2.1359777e-01


.. parsed-literal::

      94	 1.4507663e+00	 1.3062485e-01	 1.5043400e+00	 1.3100607e-01	  1.1155592e+00 	 2.0695543e-01


.. parsed-literal::

      95	 1.4529051e+00	 1.3052042e-01	 1.5064413e+00	 1.3155198e-01	  1.1293356e+00 	 2.0220160e-01


.. parsed-literal::

      96	 1.4553443e+00	 1.3043906e-01	 1.5088139e+00	 1.3134774e-01	  1.1319131e+00 	 2.1786261e-01
      97	 1.4581041e+00	 1.3024497e-01	 1.5116255e+00	 1.3122966e-01	  1.1342229e+00 	 1.8873787e-01


.. parsed-literal::

      98	 1.4601219e+00	 1.3013336e-01	 1.5137220e+00	 1.3124660e-01	  1.1366503e+00 	 2.0202708e-01


.. parsed-literal::

      99	 1.4640657e+00	 1.2982092e-01	 1.5178042e+00	 1.3165215e-01	  1.1387479e+00 	 2.1852446e-01
     100	 1.4657295e+00	 1.2971224e-01	 1.5197537e+00	 1.3154099e-01	  1.1384495e+00 	 1.9668674e-01


.. parsed-literal::

     101	 1.4693567e+00	 1.2959099e-01	 1.5231901e+00	 1.3181242e-01	  1.1429751e+00 	 1.8437481e-01


.. parsed-literal::

     102	 1.4710498e+00	 1.2943241e-01	 1.5248526e+00	 1.3172267e-01	  1.1386024e+00 	 2.1270299e-01
     103	 1.4736544e+00	 1.2911064e-01	 1.5275518e+00	 1.3144201e-01	  1.1248310e+00 	 1.8763924e-01


.. parsed-literal::

     104	 1.4749041e+00	 1.2875567e-01	 1.5289798e+00	 1.3143547e-01	  1.0932126e+00 	 1.8359137e-01


.. parsed-literal::

     105	 1.4779599e+00	 1.2862624e-01	 1.5319772e+00	 1.3103854e-01	  1.0965889e+00 	 2.0634699e-01


.. parsed-literal::

     106	 1.4796756e+00	 1.2849059e-01	 1.5337101e+00	 1.3090801e-01	  1.0932905e+00 	 2.1204138e-01


.. parsed-literal::

     107	 1.4818469e+00	 1.2826002e-01	 1.5359194e+00	 1.3074921e-01	  1.0793080e+00 	 2.0925093e-01


.. parsed-literal::

     108	 1.4846626e+00	 1.2783436e-01	 1.5388693e+00	 1.3098283e-01	  1.0495107e+00 	 2.0856071e-01
     109	 1.4870727e+00	 1.2754538e-01	 1.5413489e+00	 1.3093563e-01	  1.0287650e+00 	 1.8667698e-01


.. parsed-literal::

     110	 1.4886407e+00	 1.2746371e-01	 1.5428718e+00	 1.3090693e-01	  1.0315349e+00 	 2.0698142e-01
     111	 1.4906583e+00	 1.2733563e-01	 1.5449023e+00	 1.3093449e-01	  1.0327287e+00 	 1.9654155e-01


.. parsed-literal::

     112	 1.4925129e+00	 1.2719127e-01	 1.5468274e+00	 1.3092545e-01	  1.0383147e+00 	 1.8881559e-01


.. parsed-literal::

     113	 1.4947605e+00	 1.2709646e-01	 1.5490967e+00	 1.3110967e-01	  1.0466572e+00 	 2.1695375e-01


.. parsed-literal::

     114	 1.4968250e+00	 1.2697382e-01	 1.5512523e+00	 1.3124107e-01	  1.0499020e+00 	 2.1517873e-01


.. parsed-literal::

     115	 1.4984688e+00	 1.2688342e-01	 1.5528797e+00	 1.3144038e-01	  1.0493538e+00 	 2.0828509e-01
     116	 1.4998788e+00	 1.2678689e-01	 1.5542844e+00	 1.3153337e-01	  1.0436930e+00 	 1.8818521e-01


.. parsed-literal::

     117	 1.5029026e+00	 1.2648145e-01	 1.5574037e+00	 1.3189148e-01	  1.0266890e+00 	 2.1443701e-01


.. parsed-literal::

     118	 1.5043285e+00	 1.2634436e-01	 1.5588615e+00	 1.3222902e-01	  1.0212249e+00 	 2.9769874e-01


.. parsed-literal::

     119	 1.5059907e+00	 1.2618663e-01	 1.5605326e+00	 1.3235702e-01	  1.0190993e+00 	 2.1302366e-01


.. parsed-literal::

     120	 1.5080893e+00	 1.2595996e-01	 1.5626380e+00	 1.3244658e-01	  1.0190679e+00 	 2.1626472e-01
     121	 1.5092658e+00	 1.2573549e-01	 1.5638353e+00	 1.3273220e-01	  1.0183883e+00 	 1.8573046e-01


.. parsed-literal::

     122	 1.5108666e+00	 1.2565975e-01	 1.5653854e+00	 1.3254332e-01	  1.0173348e+00 	 2.0654154e-01


.. parsed-literal::

     123	 1.5122065e+00	 1.2554932e-01	 1.5667219e+00	 1.3253201e-01	  1.0124027e+00 	 2.0554376e-01
     124	 1.5131840e+00	 1.2544585e-01	 1.5677163e+00	 1.3246835e-01	  1.0062600e+00 	 1.7752004e-01


.. parsed-literal::

     125	 1.5152433e+00	 1.2518014e-01	 1.5698393e+00	 1.3267428e-01	  9.9204643e-01 	 1.8537760e-01


.. parsed-literal::

     126	 1.5169811e+00	 1.2483080e-01	 1.5716510e+00	 1.3267273e-01	  9.8277882e-01 	 2.1048450e-01


.. parsed-literal::

     127	 1.5185024e+00	 1.2482738e-01	 1.5731214e+00	 1.3286490e-01	  9.8363063e-01 	 2.0883703e-01


.. parsed-literal::

     128	 1.5200952e+00	 1.2475602e-01	 1.5746871e+00	 1.3311953e-01	  9.8761542e-01 	 2.1319222e-01


.. parsed-literal::

     129	 1.5214535e+00	 1.2460089e-01	 1.5760745e+00	 1.3340234e-01	  9.8678716e-01 	 2.0753121e-01


.. parsed-literal::

     130	 1.5232267e+00	 1.2449513e-01	 1.5778933e+00	 1.3361889e-01	  9.8021917e-01 	 2.0534778e-01
     131	 1.5247368e+00	 1.2429977e-01	 1.5794801e+00	 1.3361484e-01	  9.7742121e-01 	 1.7532992e-01


.. parsed-literal::

     132	 1.5260956e+00	 1.2417605e-01	 1.5808834e+00	 1.3358504e-01	  9.7218502e-01 	 1.9987178e-01


.. parsed-literal::

     133	 1.5276710e+00	 1.2403548e-01	 1.5825199e+00	 1.3356241e-01	  9.6518946e-01 	 2.0582247e-01


.. parsed-literal::

     134	 1.5288828e+00	 1.2379400e-01	 1.5839232e+00	 1.3378964e-01	  9.5506682e-01 	 2.0571160e-01
     135	 1.5308622e+00	 1.2380118e-01	 1.5858104e+00	 1.3361322e-01	  9.5086429e-01 	 1.7893982e-01


.. parsed-literal::

     136	 1.5316720e+00	 1.2380798e-01	 1.5865800e+00	 1.3366663e-01	  9.4975358e-01 	 2.0230365e-01


.. parsed-literal::

     137	 1.5332937e+00	 1.2376096e-01	 1.5881735e+00	 1.3367411e-01	  9.3468907e-01 	 2.1283770e-01


.. parsed-literal::

     138	 1.5343769e+00	 1.2373548e-01	 1.5892641e+00	 1.3352424e-01	  9.2814199e-01 	 2.1235561e-01


.. parsed-literal::

     139	 1.5354111e+00	 1.2369451e-01	 1.5902752e+00	 1.3338381e-01	  9.2407311e-01 	 2.0993066e-01
     140	 1.5364375e+00	 1.2362945e-01	 1.5913294e+00	 1.3325187e-01	  9.1632618e-01 	 1.7799544e-01


.. parsed-literal::

     141	 1.5375773e+00	 1.2360285e-01	 1.5925154e+00	 1.3309833e-01	  9.0588182e-01 	 1.7916059e-01
     142	 1.5386821e+00	 1.2353701e-01	 1.5937452e+00	 1.3319840e-01	  8.8453313e-01 	 1.7076993e-01


.. parsed-literal::

     143	 1.5401467e+00	 1.2354558e-01	 1.5951762e+00	 1.3320635e-01	  8.7961621e-01 	 1.9181156e-01


.. parsed-literal::

     144	 1.5410368e+00	 1.2354130e-01	 1.5960451e+00	 1.3333044e-01	  8.7538995e-01 	 2.1282005e-01


.. parsed-literal::

     145	 1.5422073e+00	 1.2351303e-01	 1.5972083e+00	 1.3351875e-01	  8.6226182e-01 	 2.0481586e-01


.. parsed-literal::

     146	 1.5433666e+00	 1.2341810e-01	 1.5984197e+00	 1.3339051e-01	  8.4716681e-01 	 2.0695877e-01
     147	 1.5449211e+00	 1.2339819e-01	 1.5999533e+00	 1.3369070e-01	  8.2442783e-01 	 1.9780564e-01


.. parsed-literal::

     148	 1.5457300e+00	 1.2337733e-01	 1.6007812e+00	 1.3366907e-01	  8.1516634e-01 	 2.0000696e-01


.. parsed-literal::

     149	 1.5468886e+00	 1.2335642e-01	 1.6019994e+00	 1.3363179e-01	  7.9982639e-01 	 2.0818973e-01


.. parsed-literal::

     150	 1.5483675e+00	 1.2336112e-01	 1.6035533e+00	 1.3370088e-01	  7.8378256e-01 	 2.0737076e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.04 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f17b80d3cd0>



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
    CPU times: user 1.81 s, sys: 53 ms, total: 1.86 s
    Wall time: 622 ms


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

